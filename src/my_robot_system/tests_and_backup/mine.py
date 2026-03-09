#!/usr/bin/env python3

import os
import time
import yaml
import numpy as np
import cv2
import torch
import torchvision

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, LaserScan, CompressedImage
from cv_bridge import CvBridge
from ament_index_python.packages import get_package_share_directory

from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.backbone_utils import mobilenet_backbone

import yaml, os
from typing import Any, Dict, Tuple



#---------------------------------------------------------------------
#Detection functions

MODEL_FILENAME = 'best_model_map_small.pth'
IMG_SIZE = 608
CONFIDENCE_THRESHOLD = 0.3

ALL_CLASSES = [
    'right turn', 'left turn', 'puddle', 'street vendor', 'obstacle',
    'bad road', 'garbage bin', 'chair', 'pothole', 'car', 'motorcycle',
    'pedestrian', 'fence', 'gate barrier', 'roadblock', 'door', 'tree',
    'plant pot', 'drain', 'stair', 'pole', 'zebra cross'
]

PARTICULAR_CLASSES_INDICES = [2, 4, 5, 7, 8, 9, 11, 12, 13, 14, 16, 20]
CLASSES = [ALL_CLASSES[i] for i in PARTICULAR_CLASSES_INDICES]


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
        backbone_name="mobilenet_v3_small",
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

#---------------------------------------------------------------------
#Calibration functions
config_folder = 'config'
general_configuration_file = 'general_configuration.yaml'
SKIP_RATE = 1 #every skip_rate we skip readings from lidar for efficiency


def extract_configuration() -> Dict[str, Any]:
    """
    Reads configuration file for lidar_camera fusion

    Returns:
        _type_: dict[str]
    """
    config_file = os.path.join(get_package_share_directory('my_robot_system'), config_folder, general_configuration_file)

    try:
        with open(config_file, 'r') as file:
            return  yaml.safe_load(file)
    except FileNotFoundError as e:
        raise FileNotFoundError(general_configuration_file)

def load_extrinsic_matrix(yaml_path: str) -> np.ndarray:
    """
    Loads extrinsic matrix definig camera-lidar positions

    Args:
        yaml_path (str): 

    Raises:
        FileNotFoundError: 

    Returns:
        np.ndarray: extrinsinc_matrix(np.float64)
    """
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(yaml_path)
    
    with open(yaml_path, 'r') as file:
        data_yaml = yaml.safe_load(file)
    matrix = data_yaml['extrinsic_matrix']
    return np.array(matrix, dtype=np.float64)

def load_intrinsic_matrix(yaml_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads camera intrinsic matrix configuration file

    Args:
        yaml_path (str): _description_

    Raises:
        FileNotFoundError: _description_

    Returns:
        Tuple[np.ndarray, np.ndarray]: camera_matrix, distortion_coefficients
    """
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(yaml_path)
    
    with open(yaml_path, 'r') as file:
        data_yaml = yaml.safe_load(file)
    camera_matrix = np.array(data_yaml['camera_matrix']['data'], dtype=np.float64)
    distortion_coefficients = np.array(data_yaml['distortion_coefficients']['data'], dtype=np.float64).reshape((1, -1)) #reshape((1, -1)) - list to 2D
    return camera_matrix, distortion_coefficients

def laserscan_to_xyz_array(scan_msg: LaserScan, skip_rate) -> np.ndarray:
    """
    Changes laserscan from lidar to xyz points coordinate system
    There is an option to take every certain value for better efficiency

    Args:
        scan_msg (LaserScan): LaserScan from lidar
        skip_rate (int, optional): Skip every skip_rate value.  Defaults to 1.

    Returns:
        np.ndarray: xyz points
        #x1, y1, z1 
        #x2, y2, z2 
    """
    ranges = np.array(scan_msg.ranges)
    angles = np.linspace(scan_msg.angle_min, scan_msg.angle_max, len(ranges), endpoint=False)

    #creates true/false table for values that meet following condition:
    valid_mask = np.isfinite(ranges) & (ranges > scan_msg.range_min) & (ranges < scan_msg.range_max)
    ranges = ranges[valid_mask]
    angles = angles[valid_mask]

    if skip_rate > 1:
        ranges = ranges[::skip_rate]
        angles = angles[::skip_rate]
    
    if len(ranges) == 0:
        return np.zeros((0, 3), dtype = np.float32)
    x = ranges * np.cos(angles)
    y = ranges * np.sin(angles)
    z = np.zeros_like(x)

    xyz = np.column_stack((x, y, z), dtype=np.float32) #x, y, z 
    return xyz.astype(np.float64)


#main things happen here
class FusionDetectorNode(Node): 
    def __init__(self):
        super().__init__('fusion_detector_node')
        self.bridge = CvBridge() #sensor_msgs/Image -> opencv(np.ndarray)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        config_file = extract_configuration()
        if config_file is None:
            self.get_logger().error("Error opening configuration file")
            return
        
        #reading yaml configuration for calibration
        config_folder = config_file['general']['config_folder']
        extrinsic_yaml = os.path.join(config_folder, config_file['general']['camera_extrinsic_calibration'])
        self.T_extrinsic = load_extrinsic_matrix(extrinsic_yaml)

        intrinsic_yaml = os.path.join(config_folder, config_file['general']['camera_intrinsic_calibration'])
        self.T_intrinsic, self.distortion_coefficients = load_intrinsic_matrix(intrinsic_yaml)

        lidar_topic = config_file['lidar']['lidar_topic']
        image_topic = config_file['camera']['image_topic']
        output_topic = '/fusion/detected_image'
        output_topic_compressed = '/fusion/detected_image/compressed'

        qos_best_eff = QoSProfile(reliability = ReliabilityPolicy.BEST_EFFORT, #data sent without confirmation, best for high freq. data (lidar)
                                   history=HistoryPolicy.KEEP_LAST, #reject old, KEEP_ALL - process all
                                    depth=1 #queue = 1, if 1 img is in in buffor and 3 another came, only last is processed
                                    )    
        qos_reliable = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,   #data sent with confirmation Acknowledge
                                   history=HistoryPolicy.KEEP_LAST,
                                    depth=1
                                    )
        
        self.create_subscription(LaserScan, lidar_topic, self.lidar_callback, qos_best_eff) #messageType, topic to subscribe, callback - function called when new message is received, QoS
        self.create_subscription(Image, image_topic, self.image_callback, qos_reliable)
        self.pub_image = self.create_publisher(Image, output_topic, 10)
        self.pub_image_compressed = self.create_publisher(CompressedImage, output_topic_compressed, 10)

        self.new_scan_msg = None
        self.prev_time = time.time()

        #reading model for ML
        package_share_directory = get_package_share_directory('my_robot_system')
        model_path = os.path.join(package_share_directory, 'models', MODEL_FILENAME)

        #model load section
        try:
            num_classes = len(CLASSES) + 1
            self.model = get_model_jetson(num_classes, img_size=IMG_SIZE)
            
            checkpoint = torch.load(model_path, map_location=self.device)
            state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
            state_dict = remap_state_dict_keys(state_dict)
            
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()
            self.get_logger().info("Model loaded successfully.")
        except Exception as e:
            self.get_logger().error(f"Failed to load model: {e}")
            raise e
        
    def lidar_callback(self, msg):
        self.new_scan_msg = msg
    
    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception:
            return
        
        h, w = cv_image.shape[0], cv_image.shape[1]
        
        scale_x = w / IMG_SIZE
        scale_y = h / IMG_SIZE
        resized_image = cv2.resize(cv_image, (IMG_SIZE, IMG_SIZE))

        rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
        tensor_image = preprocess_image(rgb_image).to(self.device).unsqueeze(0) #.unsqueeze(0) - added batch Dimension

        #prediction
        with torch.no_grad():
            time_start = time.time()
            prediction = self.model(tensor_image)[0]
            
        projected_points = None
        lidar_depths = None

        if self.new_scan_msg is not None:
            xyz_lidar = laserscan_to_xyz_array(self.new_scan_msg, skip_rate = SKIP_RATE)
            if xyz_lidar.shape[0] > 0: #rows >0
                ones = np.ones((xyz_lidar.shape[0], 1), dtype = np.float64) 
                xyz_lidar_x4 = np.hstack((xyz_lidar, ones)) #x, y, z, 1 -> 4x4

                #extrinsic
                xyz_cam = xyz_lidar_x4 @ self.T_extrinsic.T
                xyz_cam = xyz_cam[:, :3]

                mask = (xyz_cam[:, 2] > 0.0)
                xyz_cam_front = xyz_cam[mask]

                if xyz_cam_front.shape[0] > 0:
                    rvec = np.zeros((3,1), dtype = np.float64) #rotation
                    tvec = np.zeros((3,1), dtype = np.float64)  #translation
                    
                    img_pts, _ = cv2.projectPoints(xyz_cam_front, rvec, tvec, self.T_intrinsic, self.distortion_coefficients)
                    projected_points = img_pts.reshape(-1, 2) #Projected Points - [ux, vy] array
                    lidar_depths = xyz_cam_front[:, 2] #z

                    for point, depth in zip(projected_points, lidar_depths):
                        u, v = int(point[0]), int(point[1])    #x, y
                        if 0 <= u < cv_image.shape[1] and 0 <= v < cv_image.shape[0]:
                            cv2.circle(cv_image, (u, v), 2, (0, 255, 255), -1)

        for box, label, score in zip(prediction['boxes'], prediction['labels'], prediction['scores']):
            if score > CONFIDENCE_THRESHOLD:
                box = box.cpu().numpy().astype(int)    #cpu - VRAM-> RAM, numpy-tensor->ndarray
                x_min = int(box[0] * scale_x)
                y_min = int(box[1] * scale_y)
                x_max = int(box[2] * scale_x)
                y_max = int(box[3] * scale_y)

                label_idx = int(label.cpu()) - 1
                class_name = CLASSES[label_idx] if 0 <= label_idx < len(CLASSES) else f"ID: {label_idx}"

                distance_str = ""
                if projected_points is not None:
                    u_coords = projected_points[:, 0] #x
                    v_coords = projected_points[:, 1] #y

                    box_mask = (u_coords >= x_min) & (u_coords <= x_max) & (v_coords >= y_min) & (v_coords <= y_max) #mask with pixels only containing detected object
                    
                    box_distances = lidar_depths[box_mask]
                    if len(box_distances) > 0:
                        distance = np.median(box_distances)
                        distance_str = f"[{distance:.2f}m]"
                    
                text = f"{class_name}: {score:.2f} {distance_str}"
                color = (0, 255, 0)
                    
                cv2.rectangle(cv_image, (x_min, y_min), (x_max, y_max), color, 2)
                cv2.putText(cv_image, text, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        curr_time = time.time()
        fps = 1 / (curr_time - self.prev_time) if (curr_time - self.prev_time) > 0 else 0
        self.prev_time = curr_time
        cv2.putText(cv_image, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        out_msg = self.bridge.cv2_to_imgmsg(cv_image, encoding='bgr8')
        out_msg.header = msg.header
        self.pub_image.publish(out_msg)

        encoded_img = cv2.imencode('.jpg', cv_image)[1] #compression, 1-take just data, without true/false succes info
        compressed_msg = CompressedImage()
        compressed_msg.header = msg.header
        compressed_msg.format = "jpeg"
        compressed_msg.data = encoded_img.tobytes()
        self.pub_image_compressed.publish(compressed_msg)

def main(args=None):
    rclpy.init(args=args)
    node = FusionDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()