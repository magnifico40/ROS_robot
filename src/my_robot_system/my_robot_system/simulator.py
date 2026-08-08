import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point, Twist
from sensor_msgs.msg import Imu
import math
from pyproj import Proj

class SimpleSimulator(Node):
    def __init__(self):
        super().__init__('simple_simulator')

        self.sub_cmd = self.create_subscription(Twist, '/cmd_vel', self.cb_cmd, 10)
        self.pub_pos = self.create_publisher(Point, '/robot_position', 10)
        self.pub_imu = self.create_publisher(Imu, 'imu/data_raw', 10)

        self.proj = Proj(proj='utm', zone='34', ellps='WGS84', preserve_units=False)

        self.start_lat = 54.614556
        self.start_lon = 18.326639

        self.x, self.y = self.proj(self.start_lon, self.start_lat)
        self.yaw = 0.0

        self.v = 0.0
        self.omega = 0.0

        self.dt = 0.1
        self.timer = self.create_timer(self.dt, self.update_physics)

    def cb_cmd(self, msg):
        self.v = msg.linear.x
        self.omega = msg.angular.z

    def update_physics(self):
        self.x += self.v * math.cos(self.yaw) * self.dt
        self.y += self.v * math.sin(self.yaw) * self.dt
        self.yaw += self.omega * self.dt

        lon, lat = self.proj(self.x, self.y, inverse=True)

        pos_msg = Point()
        pos_msg.x = lat
        pos_msg.y = lon
        self.pub_pos.publish(pos_msg)

        imu_msg = Imu()
        imu_msg.orientation.z = math.sin(self.yaw / 2.0)
        imu_msg.orientation.w = math.cos(self.yaw / 2.0)
        self.pub_imu.publish(imu_msg)

def main(args=None):
    rclpy.init(args=args)
    node = SimpleSimulator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()