import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
import math

class OdometryCompassDebug(Node):
    def __init__(self):
        super().__init__('odom_compass_debug')
        self.subscription = self.create_subscription(
            Odometry, 
            '/odometry/global', 
            self.listener_callback, 
            10
        )

    def listener_callback(self, msg):
        q = msg.pose.pose.orientation
        
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw_rad = math.atan2(siny_cosp, cosy_cosp)
        
        yaw_deg = math.degrees(yaw_rad)
        
        compass_heading = (270.0 - yaw_deg) % 360.0
        
        self.get_logger().info(f'Kierunek: {compass_heading:.1f}°')

def main(args=None):
    rclpy.init(args=args)
    node = OdometryCompassDebug()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()