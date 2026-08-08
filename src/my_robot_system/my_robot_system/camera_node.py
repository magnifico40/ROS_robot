import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage

from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import cv2

PIPELINE_JPEG = (
    'nvarguscamerasrc sensor-mode=4 ! '
    'video/x-raw(memory:NVMM), width=1280, height=720, format=NV12, framerate=30/1 ! '
    'nvvidconv flip-method=0 top=0 bottom=720 left=280 right=1000 ! '
    'video/x-raw(memory:NVMM), width=416, height=416 ! '
    'nvjpegenc quality=20 ! ' #build in compression option
    'appsink drop=1 max-buffers=1 sync=false'
)


class CSICameraPublisher(Node):
    def __init__(self):
        super().__init__('camera_node')

        self.cap = cv2.VideoCapture(PIPELINE_JPEG, cv2.CAP_GSTREAMER)
        if not self.cap.isOpened():
            self.get_logger().error('Nie można otworzyć kamery')
            return

        qos_sensor = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        

        self.publisher = self.create_publisher( #real
            CompressedImage,
            '/camera/image_raw/compressed',
            10
        )
        self.timer = self.create_timer(1.0 / 30.0, self.timer_callback)
        self.get_logger().info('Kamera gotowa')

    def timer_callback(self):
        ret, jpeg_buf = self.cap.read()
        if not ret or jpeg_buf is None:
            self.get_logger().warn('Brak klatki')
            return

        msg = CompressedImage()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'camera'
        msg.format = 'jpeg'
        msg.data = jpeg_buf.tobytes()
        self.publisher.publish(msg)

    def destroy_node(self):
        if self.cap.isOpened():
            self.cap.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CSICameraPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()