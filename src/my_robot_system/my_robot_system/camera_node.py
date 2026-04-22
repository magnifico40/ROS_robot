#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
import gi
import numpy as np

gi.require_version('Gst', '1.0')
from gi.repository import Gst

class CSICameraPublisher(Node):
    def __init__(self):
        super().__init__('camera_node')
        self.publisher_ = self.create_publisher(Image, '/camera/image_raw', 10)
        self.publisher_compressed = self.create_publisher(CompressedImage, '/camera/image_raw/compressed', 10)

        Gst.init(None)
        
        pipeline_cmd = (
            "nvarguscamerasrc ! "
            "video/x-raw(memory:NVMM), width=1640, height=1232, format=NV12, framerate=30/1 ! "
            "nvvidconv flip-method=2 left=204 right=1436 top=0 bottom=1232 ! "
            "video/x-raw(memory:NVMM), width=608, height=608, format=NV12 ! "
            "tee name=t "
            "t. ! queue ! nvvidconv ! video/x-raw, format=BGRx ! appsink name=raw_sink max-buffers=1 drop=True "
            "t. ! queue ! nvjpegenc ! appsink name=jpeg_sink max-buffers=1 drop=True"
        )

        self.pipeline = Gst.parse_launch(pipeline_cmd)
        self.raw_sink = self.pipeline.get_by_name("raw_sink")
        self.jpeg_sink = self.pipeline.get_by_name("jpeg_sink")

        self.pipeline.set_state(Gst.State.PLAYING)

        timer_period = 1.0 / 30.0
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def timer_callback(self):
        raw_sample = self.raw_sink.emit("pull-sample")
        jpeg_sample = self.jpeg_sink.emit("pull-sample")

        if raw_sample and jpeg_sample:
            current_time = self.get_clock().now().to_msg()
            frame_id = 'camera'

            raw_buf = raw_sample.get_buffer()
            success, map_info = raw_buf.map(Gst.MapFlags.READ)
            if success:
                raw_data = map_info.data
                frame_bgrx = np.ndarray((608, 608, 4), buffer=raw_data, dtype=np.uint8)
                frame_bgr = frame_bgrx[:, :, :3].copy()
                
                ros_image = Image()
                ros_image.header.stamp = current_time
                ros_image.header.frame_id = frame_id
                ros_image.height = 608
                ros_image.width = 608
                ros_image.encoding = 'bgr8'
                ros_image.is_bigendian = 0
                ros_image.step = 608 * 3
                ros_image.data = frame_bgr.tobytes()
                self.publisher_.publish(ros_image)
                
                raw_buf.unmap(map_info)

            jpeg_buf = jpeg_sample.get_buffer()
            success, map_info = jpeg_buf.map(Gst.MapFlags.READ)
            if success:
                jpeg_data = map_info.data
                
                compressed_msg = CompressedImage()
                compressed_msg.header.stamp = current_time
                compressed_msg.header.frame_id = frame_id
                compressed_msg.format = "jpeg"
                compressed_msg.data = bytes(jpeg_data)
                self.publisher_compressed.publish(compressed_msg)
                
                jpeg_buf.unmap(map_info)
        else:
            self.get_logger().warn('Brak klatki z potoku GStreamer.')

def main(args=None):
    rclpy.init(args=args)
    camera_publisher = CSICameraPublisher()
    try:
        rclpy.spin(camera_publisher)
    except KeyboardInterrupt:
        pass
    finally:
        camera_publisher.pipeline.set_state(Gst.State.NULL)
        camera_publisher.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()