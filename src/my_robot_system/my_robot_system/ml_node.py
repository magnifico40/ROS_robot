#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge
from ament_index_python.packages import get_package_share_directory

import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.backbone_utils import mobilenet_backbone
import numpy as np
import cv2
import os
import time

from memory_profiler import profile

MODEL_FILENAME = 'best_model_map_aft_fix.pth'
IMG_SIZE = 608
CONFIDENCE_THRESHOLD = 0.3

ALL_CLASSES = [
    'right turn', 'left turn', 'puddle', 'street vendor', 'obstacle',
    'bad road', 'garbage bin', 'chair', 'pothole', 'car', 'motorcycle',
    'pedestrian', 'fence', 'gate barrier', 'roadblock', 'door', 'tree',
    'plant pot', 'drain', 'stair', 'pole', 'zebra cross'
]
PARTICULAR_CLASSES_INDICES = [2, 4, 5, 7, 8, 9, 11, 12, 13, 14, 16, 20]
CLASSES = [ALL_CLASSES[i] for i in range(len(ALL_CLASSES)) if i in PARTICULAR_CLASSES_INDICES]

def preprocess_image(image):
    image = image.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    image = (image - mean) / std
    image = np.transpose(image, (2, 0, 1))
    return torch.from_numpy(image).float()

def remap_state_dict_keys(state_dict):
    new_state_dict = {}
    mappings = {
       'backbone.fpn.inner_blocks.0.0': 'backbone.fpn.inner_blocks.0',
        'backbone.fpn.inner_blocks.1.0': 'backbone.fpn.inner_blocks.1',
        'backbone.fpn.layer_blocks.0.0': 'backbone.fpn.layer_blocks.0',
        'backbone.fpn.layer_blocks.1.0': 'backbone.fpn.layer_blocks.1',
        'rpn.head.conv.0.0': 'rpn.head.conv',
        'roi_heads.box_predictor.class_score': 'roi_heads.box_predictor.cls_score',
        'roi_heads.box_predictor.bounding_box_pred': 'roi_heads.box_predictor.bbox_pred'
    }
    for key, value in state_dict.items():
        new_key = key
        for old_prefix, new_prefix in mappings.items():
            if key.startswith(old_prefix):
                new_key = key.replace(old_prefix, new_prefix)
                break
        new_state_dict[new_key] = value
    return new_state_dict

def get_model_jetson(num_classes, img_size=608):
    backbone = mobilenet_backbone(
        backbone_name="mobilenet_v3_large",
        pretrained=True,
        trainable_layers=5,
        fpn=True
    )
    backbone.out_channels = 256

    with torch.no_grad():
        dummy_input = torch.randn(1, 3, img_size, img_size)
        features = backbone(dummy_input)
        feature_map_names = list(features.keys())
        num_feature_maps = len(feature_map_names)

    if img_size >= 600:
        available_sizes = [16, 32, 64, 128, 256, 512]
    else:
        available_sizes = [8, 16, 32, 64, 128, 256]

    selected_sizes = tuple((s,) for s in available_sizes[:num_feature_maps])
    selected_aspect_ratios = ((0.5, 1.0, 2.0),) * num_feature_maps

    anchor_generator = AnchorGenerator(
        sizes=selected_sizes,
        aspect_ratios=selected_aspect_ratios
    )

    roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=feature_map_names,
        output_size=7,
        sampling_ratio=2
    )

    model = FasterRCNN(
        backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,
        box_score_thresh=0.001,
        box_nms_thresh=0.5,
        box_detections_per_img=100,
        rpn_pre_nms_top_n_train=2000,
        rpn_post_nms_top_n_train=2000,
        rpn_pre_nms_top_n_test=1000,
        rpn_post_nms_top_n_test=1000,
        rpn_nms_thresh=0.7,
        rpn_fg_iou_thresh=0.5,
        rpn_bg_iou_thresh=0.3,
        rpn_score_thresh=0.0
    )
    return model

class MLObjectDetectionNode(Node):
    def __init__(self):
        super().__init__('ml_object_detection_node')
        
        self.bridge = CvBridge()
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        
        package_share_directory = get_package_share_directory('my_robot_system')
        model_path = os.path.join(package_share_directory, 'models', MODEL_FILENAME)
        
        try:
            num_classes = len(CLASSES) + 1
            self.model = get_model_jetson(num_classes, img_size=IMG_SIZE)
            
            checkpoint = torch.load(model_path, map_location=self.device)
            state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
            
            state_dict = remap_state_dict_keys(state_dict)
            
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            raise e

        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )
        
        self.publisher_ = self.create_publisher(Image, '/ml/detected_image', 10)
        self.publisher_compressed = self.create_publisher(CompressedImage, '/ml/detected_image/compressed', 10)
        
        self.prev_time = time.time()
    
    @profile
    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception:
            return

        frame_resized = cv2.resize(cv_image, (IMG_SIZE, IMG_SIZE))
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        img_tensor = preprocess_image(frame_rgb).to(self.device).unsqueeze(0)

        with torch.no_grad():
            start_infer = time.time()
            prediction = self.model(img_tensor)[0]
            infer_time = (time.time() - start_infer) * 1000

        for box, label, score in zip(prediction['boxes'], prediction['labels'], prediction['scores']):
            if score > CONFIDENCE_THRESHOLD:
                box = box.cpu().numpy().astype(int)
                x_min, y_min, x_max, y_max = box
                
                label_idx = int(label.cpu()) - 1
                class_name = CLASSES[label_idx] if 0 <= label_idx < len(CLASSES) else f"ID: {label_idx}"
                text = f"{class_name}: {score:.2f}"
                
                if any(x in class_name for x in ['pedestrian', 'car', 'motorcycle']):
                    color = (0, 0, 255)
                elif any(x in class_name for x in ['pothole', 'bad road', 'obstacle']):
                    color = (0, 165, 255)
                else:
                    color = (0, 255, 0)

                cv2.rectangle(frame_resized, (x_min, y_min), (x_max, y_max), color, 2)
                cv2.putText(frame_resized, text, (x_min, y_min - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        curr_time = time.time()
        fps = 1 / (curr_time - self.prev_time) if (curr_time - self.prev_time) > 0 else 0
        self.prev_time = curr_time

        cv2.putText(frame_resized, f"FPS: {fps:.1f} Infer: {infer_time:.1f}ms", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        out_msg = self.bridge.cv2_to_imgmsg(frame_resized, encoding='bgr8')
        out_msg.header = msg.header
        self.publisher_.publish(out_msg)

        success, encoded_img = cv2.imencode('.jpg', frame_resized)
        if success:
            compressed_msg = CompressedImage()
            compressed_msg.header = msg.header
            compressed_msg.format = "jpeg"
            compressed_msg.data = encoded_img.tobytes()
            self.publisher_compressed.publish(compressed_msg)

def main(args=None):
    rclpy.init(args=args)
    node = MLObjectDetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()