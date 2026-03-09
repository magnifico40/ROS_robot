#!/usr/bin/env python3
#from ros2_camera_lidar_fusion.read_yaml import extract_configuration
import os
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

import cv2
import numpy as np
import yaml

from sensor_msgs.msg import Image, LaserScan, CompressedImage
from cv_bridge import CvBridge

from ros2_camera_lidar_fusion.read_yaml import extract_configuration

def load_extrinsic_matrix(yaml_path: str) -> np.ndarray:
    if not os.path.isfile(yaml_path):
        raise FileNotFoundError(f"No extrinsic file found: {yaml_path}")
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    matrix_list = data['extrinsic_matrix']
    T = np.array(matrix_list, dtype=np.float64)
    return T

def load_camera_calibration(yaml_path: str) -> (np.ndarray, np.ndarray):
    if not os.path.isfile(yaml_path):
        raise FileNotFoundError(f"No camera calibration file: {yaml_path}")
    with open(yaml_path, 'r') as f:
        calib_data = yaml.safe_load(f)
    cam_mat_data = calib_data['camera_matrix']['data']
    camera_matrix = np.array(cam_mat_data, dtype=np.float64)
    dist_data = calib_data['distortion_coefficients']['data']
    dist_coeffs = np.array(dist_data, dtype=np.float64).reshape((1, -1))
    return camera_matrix, dist_coeffs

def laserscan_to_xyz_array(scan_msg: LaserScan, skip_rate: int = 1) -> np.ndarray:
    ranges = np.array(scan_msg.ranges)
    
    angles = np.linspace(scan_msg.angle_min, 
                         scan_msg.angle_max, 
                         len(ranges), 
                         endpoint=False)

    valid_mask = np.isfinite(ranges) & (ranges > scan_msg.range_min) & (ranges < scan_msg.range_max)
    
    ranges = ranges[valid_mask]
    angles = angles[valid_mask]

    if skip_rate > 1:
        ranges = ranges[::skip_rate]
        angles = angles[::skip_rate]

    if len(ranges) == 0:
        return np.zeros((0, 3), dtype=np.float32)

    x = ranges * np.cos(angles)
    y = ranges * np.sin(angles)
    z = np.zeros_like(x)

    points = np.column_stack((x, y, z))
    return points.astype(np.float32)


class LidarCameraProjectionNode(Node):
    def __init__(self):
        super().__init__('lidar_camera_projection_node')
        
        config_file = extract_configuration()
        if config_file is None:
            self.get_logger().error("Failed to extract configuration file.")
            return
        
        config_folder = config_file['general']['config_folder']
        extrinsic_yaml = os.path.join(config_folder, config_file['general']['camera_extrinsic_calibration'])
        self.T_lidar_to_cam = load_extrinsic_matrix(extrinsic_yaml)

        camera_yaml = os.path.join(config_folder, config_file['general']['camera_intrinsic_calibration'])
        self.camera_matrix, self.dist_coeffs = load_camera_calibration(camera_yaml)

        lidar_topic = config_file['lidar']['lidar_topic']
        image_topic = config_file['camera']['image_topic']
        projected_topic = config_file['camera']['projected_topic']

        self.get_logger().info(f"Lidar topic (LaserScan): {lidar_topic}")
        self.get_logger().info(f"Image topic: {image_topic}")

        self.latest_scan_msg = None 

        qos_sensor = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        qos_reliable = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.create_subscription(LaserScan, lidar_topic, self.lidar_callback, qos_sensor)
        self.create_subscription(Image, image_topic, self.image_callback, qos_reliable)

        self.pub_image = self.create_publisher(Image, projected_topic, 10)
        self.pub_compressed = self.create_publisher(CompressedImage, projected_topic + '/compressed', 10)
        
        self.bridge = CvBridge()
        self.skip_rate = 2 

    def lidar_callback(self, msg):
        self.latest_scan_msg = msg

    def image_callback(self, image_msg):
        if self.latest_scan_msg is None:
            return

        try:
            cv_image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"CV Bridge error: {e}")
            return

        xyz_lidar = laserscan_to_xyz_array(self.latest_scan_msg, skip_rate=self.skip_rate)
        
        n_points = xyz_lidar.shape[0]

        xyz_lidar_f64 = xyz_lidar.astype(np.float64)
        ones = np.ones((n_points, 1), dtype=np.float64)
        xyz_lidar_h = np.hstack((xyz_lidar_f64, ones))

        xyz_cam_h = xyz_lidar_h @ self.T_lidar_to_cam.T
        xyz_cam = xyz_cam_h[:, :3]

        mask_in_front = (xyz_cam[:, 2] > 0.0)
        xyz_cam_front = xyz_cam[mask_in_front]
        
        if xyz_cam_front.shape[0] > 0:
            rvec = np.zeros((3,1), dtype=np.float64)
            tvec = np.zeros((3,1), dtype=np.float64)
            
            image_points, _ = cv2.projectPoints(
                xyz_cam_front, rvec, tvec, 
                self.camera_matrix, self.dist_coeffs
            )
            image_points = image_points.reshape(-1, 2)

            h, w = cv_image.shape[:2]
            max_dist = 5.0 
            
            for i, (u, v) in enumerate(image_points):
                u_int = int(u + 0.5)
                v_int = int(v + 0.5)
                
                if 0 <= u_int < w and 0 <= v_int < h:
                    dist = xyz_cam_front[i, 2]
                    ratio = min(dist / max_dist, 1.0)
                    b = int(255 * ratio)
                    g = int(255 * (1 - abs(ratio - 0.5) * 2))
                    r = int(255 * (1 - ratio))
                    color = (b, g, r)
                    
                    cv2.circle(cv_image, (u_int, v_int), 3, color, -1)

        out_msg = self.bridge.cv2_to_imgmsg(cv_image, encoding='bgr8')
        out_msg.header = image_msg.header
        self.pub_image.publish(out_msg)

        success, encoded_img = cv2.imencode('.jpg', cv_image)
        if success:
            compressed_msg = CompressedImage()
            compressed_msg.header = image_msg.header
            compressed_msg.format = "jpeg"
            compressed_msg.data = encoded_img.tobytes()
            self.pub_compressed.publish(compressed_msg)

def main(args=None):
    rclpy.init(args=args)
    node = LidarCameraProjectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()