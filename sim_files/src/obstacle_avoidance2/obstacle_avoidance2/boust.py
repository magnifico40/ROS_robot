import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan, Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
import math


class Zigzag(Node):
    def __init__(self):
        super().__init__('area_coverage')
        self.subscription = self.create_subscription(
            Odometry, '/odometry/filtered', self.odom_callback, 10)

        self.curr_x = 0.0
        self.curr_y = 0.0
        self.curr_yaw = 0.0

        self.linear_speed = 0.5
        self.angular_speed = 0.4
        self.robot_width = 0.4274

        self.start_x = 0.0
        self.start_y = 0.0
        self.end_x = 6.0
        self.end_y = 5.0

        self.road_points = self.generate_zigzag()
        self.curr_target = 0


        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.get_logger().info('Węzeł zygzaka uruchomiony')
        self.timer = self.create_timer(0.1, self.zigzag_loop)

    def odom_callback(self, msg):
        self.curr_x = msg.pose.pose.position.x
        self.curr_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        self.curr_yaw = math.atan2(siny_cosp, cosy_cosp)

    def generate_zigzag(self):
        road = []
        y = self.start_y
        right_direction = True

        while y <= self.end_y:
            if right_direction:
                road.append((self.start_x, y))
                road.append((self.end_x, y))
            else:
                road.append((self.end_x, y))
                road.append((self.start_x, y))

            y += self.robot_width
            right_direction = not right_direction

        return road


    def zigzag_loop(self):
        twist = Twist()

        #sprawdzamy czy koniec
        if self.curr_target >= len(self.road_points):
            twist.linear.x = 0.0
            twist.angular.z= 0.0
            self.get_logger().info('Zygzak zakończony', once=True)
            self.cmd_vel_pub.publish(twist)
            return

        #dystans do nastepnego punktu
        target_x,target_y = self.road_points[self.curr_target]
        dist = math.sqrt((target_x - self.curr_x)**2+(target_y - self.curr_y) ** 2)


        if dist < 0.1:
            self.curr_target = self.curr_target + 1
            self.get_logger().info(f'Osiągnięto punkt: {target_x}, {target_y}')
            return

        #sprawdzamy kat wzgledem celu i normalizujemy, by obracac sie w najblizsza strone
        dx = target_x - self.curr_x
        dy = target_y - self.curr_y
        angle_to_target = math.atan2(dy, dx)
        angle_error = angle_to_target - self.curr_yaw
        angle_error = math.atan2(math.sin(angle_error), math.cos(angle_error))

        if abs(angle_error) > 0.18: #ok. 10 stopni
            twist.linear.x = 0.05
            twist.angular.z = self.angular_speed * angle_error
        else:
            twist.linear.x = self.linear_speed
            twist.angular.z = self.angular_speed * angle_error


        self.cmd_vel_pub.publish(twist)


def main(args=None):
    rclpy.init(args=args)
    node = Zigzag()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
