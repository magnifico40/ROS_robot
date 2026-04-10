import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan, Imu
from geometry_msgs.msg import Twist
import math

class ObstacleAvoidance(Node):
    def __init__(self):
        super().__init__('obstacle_avoidance')
        
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, qos_profile_sensor_data)

        self.imu_subscription = self.create_subscription(
            Imu, 'imu/data', self.imu_callback, qos_profile_sensor_data)

        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.safe_distance = 0.2    #0.6
        self.critical_distance = 0.1
        self.linear_speed = 0.5
        self.angular_speed = 0.5
        self.corner_reverse_count = 0
        self.max_reverse_cycles = 15
        self.current_yaw = 0.0
        self.KP = 1.0
        self.max_turn_cycles = 15
        self.turn_count = 0

        self.get_logger().info('Węzeł omijania uruchomiony')

    def quaternion_to_yaw(self, q):
        t3 = +2.0 * (q.w * q.z + q.x * q.y)
        t4 = +1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(t3, t4)

    #listens for imu values
    def imu_callback(self, msg):
        self.current_yaw = self.quaternion_to_yaw(msg.orientation)

    def stop_robot(self):
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        self.cmd_vel_pub.publish(twist)

    def scan_callback(self, msg):
        ranges = msg.ranges
        num_readings = len(ranges)

        # Czyszczenie wartości nieskończonych
        ranges = [r if not math.isinf(r) and not math.isnan(r) else msg.range_max for r in ranges]

        sector_size = num_readings // 12
        half_sector = sector_size // 2

        fright_sector = ranges[-sector_size: -half_sector]
        front_sector = ranges[-half_sector:] + ranges[:half_sector]
        fleft_sector = ranges[half_sector: sector_size]
        left_sector = ranges[2 * sector_size: 4 * sector_size]
        bleft_sector = ranges[4 * sector_size: 5 * sector_size]
        back_sector = ranges[5 * sector_size: 7 * sector_size]
        bright_sector = ranges[7 * sector_size: 8 * sector_size]
        right_sector = ranges[8 * sector_size: 10 * sector_size]

        min_left = min(left_sector) if left_sector else msg.range_max
        min_front = min(front_sector) if front_sector else msg.range_max
        min_right = min(right_sector) if right_sector else msg.range_max
        min_fleft = min(fleft_sector) if fleft_sector else msg.range_max
        min_fright = min(fright_sector) if fright_sector else msg.range_max
        min_back = min(back_sector) if back_sector else msg.range_max

        twist = Twist()
        is_blocked_back = min_back < self.critical_distance
        is_blocked_front = min_front < self.critical_distance
        is_blocked_fleft = min_fleft < self.critical_distance
        is_blocked_fright = min_fright < self.critical_distance
        if is_blocked_front:
            self.get_logger().info('Blok przod')
        if is_blocked_fleft or is_blocked_fright:
            self.get_logger().info('Blok fleft/fright')


        if is_blocked_front and (is_blocked_fleft or is_blocked_fright):
            self.get_logger().warn('WYKRYTO RÓG.')
            self.corner_reverse_count = self.max_reverse_cycles
            self.turn_count = self.max_reverse_cycles
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            self.cmd_vel_pub.publish(twist)

        if min_front < self.safe_distance or min_fleft < self.safe_distance or min_fright < self.safe_distance:
            
            
            space_left = min_fleft + min_left
            space_right = min_fright + min_right

            if space_left > space_right:
                twist.angular.z = self.angular_speed
                self.get_logger().info('Przeszkoda -> Obrót w miejscu w LEWO')
            else:
                twist.angular.z = -self.angular_speed
                self.get_logger().info('Przeszkoda -> Obrót w miejscu w PRAWO')
        else:
            # Droga wolna - jedziemy prosto, ewentualnie korygując kąt z IMU
            target_yaw = 0.0
            error = target_yaw - self.current_yaw
            steering_angle = error * self.KP
            steering_angle = max(min(steering_angle, 0.6), -0.6)

            twist.linear.x = self.linear_speed
            twist.angular.z = 0.0
            self.get_logger().info('Droga wolna -> Prosto')

        self.cmd_vel_pub.publish(twist)

def main(args=None):
    rclpy.init(args=args)
    obstacle_avoidance = ObstacleAvoidance()
    try:
        rclpy.spin(obstacle_avoidance)
    except KeyboardInterrupt:
        obstacle_avoidance.get_logger().info('Zamykanie węzła')
    finally:
        obstacle_avoidance.stop_robot()
        obstacle_avoidance.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()