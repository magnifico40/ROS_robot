#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, LaserScan
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import Point
from ament_index_python.packages import get_package_share_directory
import cv2
import numpy as np
import os, yaml
import time
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from typing import Any, Dict, Tuple

# ML data
IMG_SIZE = 416
CONFIDENCE_THRESHOLD = 0.3
PARTICULAR_CLASSES = [2, 4, 5, 7, 8, 9, 11, 12, 13, 14, 16, 20, 22]
CLASSES = ['obstacle', 'grass', 'person']

# Fusion data
config_folder = 'config'
general_configuration_file = 'general_configuration.yaml'
SKIP_RATE = 1

def extract_configuration() -> Dict[str, Any]:
    config_file = os.path.join(
        get_package_share_directory('obstacle_avoidance2'),
        config_folder,
        general_configuration_file
    )
    try:
        with open(config_file, 'r') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        raise FileNotFoundError(general_configuration_file)

def load_extrinsic_matrix(yaml_path: str) -> np.ndarray:
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(yaml_path)
    with open(yaml_path, 'r') as file:
        data_yaml = yaml.safe_load(file)
    return np.array(data_yaml['extrinsic_matrix'], dtype=np.float64)

def load_intrinsic_matrix(yaml_path: str) -> Tuple[np.ndarray, np.ndarray]:
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(yaml_path)
    with open(yaml_path, 'r') as file:
        data_yaml = yaml.safe_load(file)
    camera_matrix = np.array(data_yaml['camera_matrix']['data'], dtype=np.float64)
    dist_coeffs = np.array(
        data_yaml['distortion_coefficients']['data'], dtype=np.float64
    ).reshape((1, -1))
    return camera_matrix, dist_coeffs

def laserscan_to_xyz_array(scan_msg: LaserScan, skip_rate: int) -> np.ndarray:
    ranges = np.array(scan_msg.ranges)
    angles = np.linspace(
        scan_msg.angle_min, scan_msg.angle_max, len(ranges), endpoint=False
    )
    valid_mask = (
        np.isfinite(ranges) &
        (ranges > scan_msg.range_min) &
        (ranges < scan_msg.range_max)
    )
    ranges = ranges[valid_mask]
    angles = angles[valid_mask]

    if skip_rate > 1:
        ranges = ranges[::skip_rate]
        angles = angles[::skip_rate]

    if len(ranges) == 0:
        return np.zeros((0, 3), dtype=np.float64)

    x = ranges * np.cos(angles)
    y = ranges * np.sin(angles)
    z = np.zeros_like(x)
    return np.column_stack((x, y, z)).astype(np.float64)


class TensorRTInfer:
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(self.logger, namespace="")
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.inputs, self.outputs = [], []
        self.stream = cuda.Stream()
        self.out_shapes = []

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = self.engine.get_tensor_shape(name)
            vol = trt.volume(shape)
            if vol < 0:
                vol = abs(vol)
                
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            
            host_mem = cuda.pagelocked_empty(vol, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.inputs.append({'host': host_mem, 'device': device_mem, 'name': name})
            else:
                self.outputs.append({'host': host_mem, 'device': device_mem, 'name': name})
                self.out_shapes.append(shape)

    def infer(self, image_data):
        np.copyto(self.inputs[0]['host'], image_data.ravel())
        cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)
        
        for inp in self.inputs:
            self.context.set_tensor_address(inp['name'], int(inp['device']))
        for out in self.outputs:
            self.context.set_tensor_address(out['name'], int(out['device']))
            
        self.context.execute_async_v3(stream_handle=self.stream.handle)
        
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out['host'], out['device'], self.stream)
        
        self.stream.synchronize()
        return [out['host'] for out in self.outputs]


class MlFusionNode(Node):
    def __init__(self):
        super().__init__('ml_fusion_node')

        pkg_dir = get_package_share_directory('obstacle_avoidance2')
        model_path = os.path.join(pkg_dir, 'models', 'best_new.engine')
        if not os.path.exists(model_path):
            raise FileNotFoundError(model_path)

        self.model = TensorRTInfer(model_path)
        self.get_logger().info('Model loaded')

        config_file = extract_configuration()
        config_dir = config_file['general']['config_folder']

        extrinsic_yaml = os.path.join(config_dir, config_file['general']['camera_extrinsic_calibration'])
        self.T_extrinsic = load_extrinsic_matrix(extrinsic_yaml)

        intrinsic_yaml = os.path.join(config_dir, config_file['general']['camera_intrinsic_calibration'])
        self.T_intrinsic, self.dist_coeffs = load_intrinsic_matrix(intrinsic_yaml)

        #lidar_topic = config_file['lidar']['lidar_topic']
        lidar_topic = '/scan'
        #image_topic = config_file['camera']['image_topic']
        image_topic = '/my_camera/image_raw/compressed'

        qos_sensor = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.create_subscription(LaserScan, lidar_topic, self.lidar_callback, qos_sensor)
        self.create_subscription(CompressedImage, image_topic, self.image_callback, 10)
        self.pub_compressed = self.create_publisher(CompressedImage,'/fusion/detected_image/compressed', 10)
        self.pub_detections = self.create_publisher(Detection2DArray, '/fusion/detections_meters', 10)
        self.pub_markers = self.create_publisher(MarkerArray, '/fusion/detection_markers', 10) #for debug only

        self._jpeg_params = [cv2.IMWRITE_JPEG_QUALITY, 50]
        self.new_scan_msg = None
        self.prev_time = time.time()

    def publish_markers(self, det_array, stamp):
        marker_array = MarkerArray()

        clear = Marker()
        clear.action = Marker.DELETEALL
        marker_array.markers.append(clear)

        robot = Marker()
        robot.header.frame_id = 'base_link'
        robot.header.stamp = stamp
        robot.ns = 'robot'
        robot.id = 9999
        robot.type = Marker.CUBE
        robot.action = Marker.ADD
        robot.pose.position.x = 0.0
        robot.pose.position.y = 0.0
        robot.pose.position.z = 0.0
        robot.pose.orientation.w = 1.0
        robot.scale.x = 0.5  
        robot.scale.y = 0.3  
        robot.scale.z = 0.2
        robot.color.a = 1.0
        robot.color.r = 0.0
        robot.color.g = 0.5
        robot.color.b = 1.0
        robot.lifetime.nanosec = int(1e9)
        marker_array.markers.append(robot)

        arrow = Marker()
        arrow.header.frame_id = 'base_link'
        arrow.header.stamp = stamp
        arrow.ns = 'robot'
        arrow.id = 9998
        arrow.type = Marker.ARROW
        arrow.action = Marker.ADD
        arrow.pose.position.x = 0.0
        arrow.pose.position.y = 0.0
        arrow.pose.position.z = 0.1
        arrow.pose.orientation.w = 1.0
        arrow.scale.x = 0.6  
        arrow.scale.y = 0.05
        arrow.scale.z = 0.05
        arrow.color.a = 1.0
        arrow.color.r = 1.0
        arrow.color.g = 1.0
        arrow.color.b = 0.0
        arrow.lifetime.nanosec = int(1e9)
        marker_array.markers.append(arrow)

        for i, det in enumerate(det_array.detections):
            name = det.results[0].id if det.results else '?'
            cx = det.bbox.center.x   
            cy = det.bbox.center.y   
            sx = det.bbox.size_x     
            sy = det.bbox.size_y     
            conf = det.results[0].score if det.results else 0.0

            box = Marker()
            box.header.frame_id = 'base_link'
            box.header.stamp = stamp
            box.ns = 'detections'
            box.id = i
            box.type = Marker.CUBE
            box.action = Marker.ADD
            box.pose.position.x = cy   
            box.pose.position.y = -cx  
            box.pose.position.z = 0.0
            box.pose.orientation.w = 1.0
            box.scale.x = max(sy, 0.2)  
            box.scale.y = max(sx, 0.2)  
            box.scale.z = 0.3           
            box.color.a = 0.5
            if name in ('pedestrian', 'car', 'motorcycle'):
                box.color.r = 1.0
            elif name in ('pothole', 'bad road', 'obstacle'):
                box.color.r = 1.0
                box.color.g = 0.6
            else:
                box.color.g = 1.0
            box.lifetime.nanosec = int(1e9)
            marker_array.markers.append(box)

            line = Marker()
            line.header.frame_id = 'base_link'
            line.header.stamp = stamp
            line.ns = 'lines'
            line.id = i + 500
            line.type = Marker.LINE_STRIP
            line.action = Marker.ADD
            line.scale.x = 0.02  
            line.color.a = 0.4
            line.color.r = 0.8
            line.color.g = 0.8
            line.color.b = 0.8
            line.lifetime.nanosec = int(1e9)
            p_robot = Point()
            p_robot.x, p_robot.y, p_robot.z = 0.0, 0.0, 0.0
            p_obj = Point()
            p_obj.x = cy
            p_obj.y = -cx
            p_obj.z = 0.0
            line.points.append(p_robot)
            line.points.append(p_obj)
            marker_array.markers.append(line)

            label = Marker()
            label.header.frame_id = 'base_link'
            label.header.stamp = stamp
            label.ns = 'labels'
            label.id = i + 1000
            label.type = Marker.TEXT_VIEW_FACING
            label.action = Marker.ADD
            label.pose.position.x = cy
            label.pose.position.y = -cx
            label.pose.position.z = 0.5
            label.pose.orientation.w = 1.0
            label.scale.z = 0.15
            label.color.a = 1.0
            label.color.r = 1.0
            label.color.g = 1.0
            label.color.b = 1.0
            label.text = f"{name}\ndepth={cy:.1f}m\nside={cx:.1f}m\n{conf:.0%}" 
            label.lifetime.nanosec = int(1e9)
            marker_array.markers.append(label)
            self.pub_markers.publish(marker_array)

    

    def lidar_callback(self, msg):
        self.new_scan_msg = msg

    def image_callback(self, msg):
        np_arr = np.frombuffer(msg.data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            self.get_logger().warn('Brak klatki')
            return

        projected_points = None
        lidar_depths = None
        xyz_cam_front = None

        if self.new_scan_msg is not None:
            xyz_lidar = laserscan_to_xyz_array(self.new_scan_msg, skip_rate=SKIP_RATE)
            if xyz_lidar.shape[0] > 0:
                ones = np.ones((xyz_lidar.shape[0], 1), dtype=np.float64)
                xyz_lidar_h = np.hstack((xyz_lidar, ones))

                xyz_cam = xyz_lidar_h @ self.T_extrinsic.T
                xyz_cam = xyz_cam[:, :3]

                mask = xyz_cam[:, 2] > 0.0
                xyz_cam_front = xyz_cam[mask]

                if xyz_cam_front.shape[0] > 0:
                    rvec = np.zeros((3, 1), dtype=np.float64)
                    tvec = np.zeros((3, 1), dtype=np.float64)
                    img_pts, _ = cv2.projectPoints(
                        xyz_cam_front, rvec, tvec,
                        self.T_intrinsic, self.dist_coeffs
                    )
                    projected_points = img_pts.reshape(-1, 2)
                    lidar_depths = xyz_cam_front[:, 2]

                    for pt in projected_points:
                        u, v = int(pt[0]), int(pt[1])
                        if 0 <= u < frame.shape[1] and 0 <= v < frame.shape[0]:
                            cv2.circle(frame, (u, v), 2, (0, 255, 255), -1)

        resized_frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        img_input = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        img_input = img_input.transpose((2, 0, 1)).astype(np.float32) / 255.0
        img_input = np.ascontiguousarray(np.expand_dims(img_input, axis=0))

        start = time.time()
        outputs = self.model.infer(img_input)
        infer_ms = (time.time() - start) * 1000

        output = np.array(outputs[0]).reshape(self.model.out_shapes[0])[0].transpose()
        scores_all = np.max(output[:, 4:], axis=1)
        valid_indices = np.where(scores_all > CONFIDENCE_THRESHOLD)[0]

        boxes, scores, class_ids = [], [], []
        for i in valid_indices:
            row = output[i]
            x, y, w, h = row[0], row[1], row[2], row[3]
            boxes.append([int(x - w/2), int(y - h/2), int(w), int(h)])
            scores.append(float(scores_all[i]))
            class_ids.append(int(np.argmax(row[4:])))

        indices = cv2.dnn.NMSBoxes(boxes, scores, CONFIDENCE_THRESHOLD, 0.45)

        det_array = Detection2DArray()
        det_array.header = msg.header

        if len(indices) > 0:
            for i in indices.flatten():
                x1, y1, w, h = boxes[i]
                x2, y2 = x1 + w, y1 + h
                conf = scores[i]
                cls_id = class_ids[i]
                name = CLASSES[cls_id] if cls_id < len(CLASSES) else f"ID:{cls_id}"

                color = (
                (0, 0, 255) if name in ('pedestrian', 'car', 'motorcycle') else
                (0, 165, 255) if name in ('pothole', 'bad road', 'obstacle') else 
                (0, 255, 0)
                )
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                distance_str = ""

                if projected_points is not None:
                    u_coords = projected_points[:, 0]
                    v_coords = projected_points[:, 1]
                    box_mask = (
                        (u_coords >= x1) & (u_coords <= x2) &
                        (v_coords >= y1) & (v_coords <= y2)
                    )
                    pts_in_box = xyz_cam_front[box_mask]

                    if len(pts_in_box) >= 3:
                        cx = float(np.mean(pts_in_box[:, 0]))
                        #cy = float(np.median(pts_in_box[:, 2]))
                        cy = float(np.min(pts_in_box[:, 2]))
                        sx = float(np.max(pts_in_box[:, 0]) - np.min(pts_in_box[:, 0]))
                        sy = float(np.max(pts_in_box[:, 1]) - np.min(pts_in_box[:, 1]))

                        distance_str = f" [{cy:.2f}m]"

                        det = Detection2D()
                        det.header = msg.header
                        det.bbox.center.x = cx
                        det.bbox.center.y = cy
                        det.bbox.center.theta = 0.0
                        det.bbox.size_x = max(sx, 0.1)
                        det.bbox.size_y = max(sy, 0.1)

                        hyp = ObjectHypothesisWithPose()
                        hyp.id = name
                        hyp.score = float(conf)
                        det.results.append(hyp)

                        det_array.detections.append(det)

                cv2.putText(frame, f"{name}: {conf:.2f}{distance_str}",
                            (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        self.pub_detections.publish(det_array)

        curr = time.time()
        fps = 1.0 / (curr - self.prev_time + 1e-9)
        self.prev_time = curr
        cv2.putText(frame, f"FPS:{fps:.1f} Infer:{infer_ms:.0f}ms",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        self.publish_markers(det_array, msg.header.stamp) #for debug

        ok, enc = cv2.imencode('.jpg', frame, self._jpeg_params)
        if ok:
            self.get_logger().info("Epub image")
            c = CompressedImage()
            c.header = msg.header
            c.format = 'jpeg'
            c.data = enc.tobytes()
            self.pub_compressed.publish(c)


def main(args=None):
    rclpy.init(args=args)
    node = MlFusionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()