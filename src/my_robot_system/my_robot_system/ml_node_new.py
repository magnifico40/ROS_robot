#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge
from ament_index_python.packages import get_package_share_directory
import cv2
import numpy as np
import os
import time
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

IMG_SIZE = 608
CONFIDENCE_THRESHOLD = 0.3
PARTICULAR_CLASSES = [2, 4, 5, 7, 8, 9, 11, 12, 13, 14, 16, 20, 22]
ALL_CLASSES = [
    'right turn','left turn','puddle','street vendor','obstacle',
    'bad road','garbage bin','chair','pothole','car',
    'motorcycle','pedestrian','fence','gate barrier','roadblock',
    'door','tree','plant pot','drain','stair','pole','zebra cross'
]
CLASSES = [ALL_CLASSES[i] for i in PARTICULAR_CLASSES if i < len(ALL_CLASSES)]

class TensorRTInfer:
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(self.logger, namespace="")
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.stream = cuda.Stream()
        self.out_shapes = []

        for binding in self.engine:
            shape = self.engine.get_binding_shape(binding)
            size = trt.volume(shape) * self.engine.max_batch_size
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(device_mem))
            if self.engine.binding_is_input(binding):
                self.inputs.append({'host': host_mem, 'device': device_mem})
            else:
                self.outputs.append({'host': host_mem, 'device': device_mem})
                self.out_shapes.append(shape)

    def infer(self, image_data):
        np.copyto(self.inputs[0]['host'], image_data.ravel())
        cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out['host'], out['device'], self.stream)
        self.stream.synchronize()
        return [out['host'] for out in self.outputs]

class YOLODetectionNode(Node):
    def __init__(self):
        super().__init__('yolo_detection_node')
        self.bridge = CvBridge()
        pkg_dir = get_package_share_directory('my_robot_system')
        
        model_path = os.path.join(pkg_dir, 'models', 'best.engine')
        if not os.path.exists(model_path):
            raise FileNotFoundError(model_path)
            
        self.model = TensorRTInfer(model_path)
        self.get_logger().info('Model zaladowany.')

        self.subscription = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        self.pub_raw = self.create_publisher(Image, '/ml/detected_image', 10)
        self.pub_jpg = self.create_publisher(CompressedImage, '/ml/detected_image/compressed', 10)
        self.prev_time = time.time()

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception:
            return

        frame = cv2.resize(cv_image, (IMG_SIZE, IMG_SIZE))
        
        img_input = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_input = img_input.transpose((2, 0, 1)).astype(np.float32)
        img_input /= 255.0
        img_input = np.expand_dims(img_input, axis=0)
        img_input = np.ascontiguousarray(img_input)

        start = time.time()
        outputs = self.model.infer(img_input)
        infer_ms = (time.time() - start) * 1000

        out_shape = self.model.out_shapes[0]
        output = np.array(outputs[0]).reshape(out_shape)
        output = output[0].transpose()

        scores_all = np.max(output[:, 4:], axis=1)
        valid_indices = np.where(scores_all > CONFIDENCE_THRESHOLD)[0]

        boxes = []
        scores = []
        class_ids = []

        for i in valid_indices:
            row = output[i]
            score = scores_all[i]
            class_id = np.argmax(row[4:])
            
            x, y, w, h = row[0], row[1], row[2], row[3]
            x1 = int(x - w / 2)
            y1 = int(y - h / 2)
            
            boxes.append([x1, y1, int(w), int(h)])
            scores.append(float(score))
            class_ids.append(int(class_id))

        indices = cv2.dnn.NMSBoxes(boxes, scores, CONFIDENCE_THRESHOLD, 0.45)

        if len(indices) > 0:
            for i in indices.flatten():
                x1, y1, w, h = boxes[i]
                x2 = x1 + w
                y2 = y1 + h
                conf = scores[i]
                cls_id = class_ids[i]
                name = CLASSES[cls_id] if cls_id < len(CLASSES) else f"ID:{cls_id}"

                if name in ('pedestrian','car','motorcycle'):
                    color = (0,0,255)
                elif name in ('pothole','bad road','obstacle'):
                    color = (0,165,255)
                else:
                    color = (0,255,0)

                cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
                cv2.putText(frame, f"{name}: {conf:.2f}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        curr = time.time()
        fps = 1/(curr - self.prev_time) if (curr - self.prev_time) > 0 else 0
        self.prev_time = curr
        cv2.putText(frame, f"FPS:{fps:.1f} Infer:{infer_ms:.0f}ms", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

        out_msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        out_msg.header = msg.header
        self.pub_raw.publish(out_msg)

        ok, enc = cv2.imencode('.jpg', frame)
        if ok:
            c = CompressedImage()
            c.header = msg.header
            c.format = "jpeg"
            c.data = enc.tobytes()
            self.pub_jpg.publish(c)

def main(args=None):
    rclpy.init(args=args)
    node = YOLODetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()