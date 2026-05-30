#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge
import cv2
import numpy as np

class CSICameraPublisher(Node):
    def __init__(self):
        super().__init__('camera_node')
        #self.cap = cv2.VideoCapture(
         #   'nvarguscamerasrc ! '
         #   'video/x-raw(memory:NVMM), width=1640, height=1232, format=NV12, framerate=30/1 ! '
         #   'nvvidconv flip-method=2 ! '
         #   'video/x-raw, width=608, height=608, format=BGRx ! '
        #    'videoconvert ! '
        #    'video/x-raw, format=BGR ! '
        #    'appsink',
        #    cv2.CAP_GSTREAMER
        #)
        #Wycinamy środek 1232x1232
        self.cap = cv2.VideoCapture(
            'nvarguscamerasrc ! '
            'video/x-raw(memory:NVMM), width=1640, height=1232, format=NV12, framerate=30/1 ! '
            'nvvidconv flip-method=2 left=204 right=1436 top=0 bottom=1232 ! ' 
            'video/x-raw(memory:NVMM), width=608, height=608 ! '
            'nvvidconv ! '
            'video/x-raw, format=BGRx ! '
            'videoconvert ! '
            'video/x-raw, format=BGR ! '
            'appsink',
            cv2.CAP_GSTREAMER
        )

        if not self.cap.isOpened():
            self.get_logger().error('Nie można otworzyć kamery, sprawdź połączenie, lub zresetuj usługę nvargus')
            return 
        
        self.publisher_ = self.create_publisher(Image, '/camera/image_raw', 10)
        self.publisher_compressed = self.create_publisher(CompressedImage, '/camera/image_raw/compressed', 10)
        
        self.bridge = CvBridge()
        
        timer_period = 1.0 / 30.0
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def timer_callback(self):
        ret, frame = self.cap.read()
        if ret:
            current_time = self.get_clock().now().to_msg()
            frame_id = 'camera'

            try:
                ros_image = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
                ros_image.header.stamp = current_time
                ros_image.header.frame_id = frame_id
                self.publisher_.publish(ros_image)
            except Exception as e:
                self.get_logger().error(f"Błąd konwersji RAW: {e}")

            try:
                success, encoded_img = cv2.imencode('.jpg', frame)
                if success:
                    compressed_msg = CompressedImage()
                    compressed_msg.header.stamp = current_time
                    compressed_msg.header.frame_id = frame_id
                    compressed_msg.format = "jpeg"
                    compressed_msg.data = encoded_img.tobytes()
                    self.publisher_compressed.publish(compressed_msg)
            except Exception as e:
                self.get_logger().error(f"Błąd konwersji Compressed: {e}")

        else:
            self.get_logger().warn('Brak klatki z kamery.')

def main(args=None):
    rclpy.init(args=args)
    camera_publisher = CSICameraPublisher()
    try:
        rclpy.spin(camera_publisher)
    except KeyboardInterrupt:
        pass
    finally:
        if camera_publisher.cap.isOpened():
            camera_publisher.cap.release()
        camera_publisher.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()