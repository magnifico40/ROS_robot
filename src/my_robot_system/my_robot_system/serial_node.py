#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Imu
import serial
import math

class RcCarDriver(Node):
    def __init__(self):
        super().__init__('rc_car_driver')

        self.declare_parameter('serial_port', '/dev/ttyACM0')
        self.declare_parameter('baud_rate', 115200)
        
        self.declare_parameter('servo_min_us', 1200)
        self.declare_parameter('servo_center_us', 1500)
        self.declare_parameter('servo_max_us', 1800)
        
        self.declare_parameter('motor_max_pwm', 255)

        self.serial_port = self.get_parameter('serial_port').value
        self.baud_rate = self.get_parameter('baud_rate').value
        self.servo_min = self.get_parameter('servo_min_us').value
        self.servo_center = self.get_parameter('servo_center_us').value
        self.servo_max = self.get_parameter('servo_max_us').value
        self.motor_max = self.get_parameter('motor_max_pwm').value

        try:
            self.ser = serial.Serial(self.serial_port, self.baud_rate, timeout=0.1)
            self.get_logger().info(f'Połączono z ESP32 na {self.serial_port}')
        except Exception as e:
            self.get_logger().error(f'Błąd portu szeregowego: {e}')
            exit(1)

        self.subscription = self.create_subscription(
            Twist,
            'cmd_vel',
            self.cmd_vel_callback,
            10)

        self.imu_publisher = self.create_publisher(Imu, 'imu/data', 10)
        
        self.create_timer(0.02, self.read_serial_data)

    def map_value(self, x, in_min, in_max, out_min, out_max):
        return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

    def euler_to_quaternion(self, roll, pitch, yaw):
        qx = math.sin(roll/2) * math.cos(pitch/2) * math.cos(yaw/2) - math.cos(roll/2) * math.sin(pitch/2) * math.sin(yaw/2)
        qy = math.cos(roll/2) * math.sin(pitch/2) * math.cos(yaw/2) + math.sin(roll/2) * math.cos(pitch/2) * math.sin(yaw/2)
        qz = math.cos(roll/2) * math.cos(pitch/2) * math.sin(yaw/2) - math.sin(roll/2) * math.sin(pitch/2) * math.cos(yaw/2)
        qw = math.cos(roll/2) * math.cos(pitch/2) * math.cos(yaw/2) + math.sin(roll/2) * math.sin(pitch/2) * math.sin(yaw/2)
        return [qx, qy, qz, qw]

    def cmd_vel_callback(self, msg):
        linear_x = msg.linear.x
        angular_z = msg.angular.z

        pwm_val = int(linear_x * self.motor_max)
        pwm_val = max(min(pwm_val, 255), -255)

        if angular_z == 0:
            servo_us = self.servo_center
        elif angular_z > 0:
            servo_us = self.map_value(angular_z, 0.0, 1.0, self.servo_center, self.servo_max)
        else:
            servo_us = self.map_value(angular_z, -1.0, 0.0, self.servo_min, self.servo_center)

        servo_us = int(max(min(servo_us, self.servo_max), self.servo_min))

        cmd_str = f"{servo_us},{pwm_val}\n"
        self.ser.write(cmd_str.encode('utf-8'))

    def read_serial_data(self):
        if self.ser.in_waiting > 0:
            try:
                line = self.ser.readline().decode('utf-8').strip()
                if line.startswith("IMU,"):
                    parts = line.split(',')
                    if len(parts) == 4:
                        self.get_logger().info(f'Roll, Pitch, Yaw [deg]: {float(parts[1])}, {float(parts[2])}, {float(parts[3])}')
                        imu_msg = Imu()
                        imu_msg.header.stamp = self.get_clock().now().to_msg()
                        imu_msg.header.frame_id = "imu_link"
                        
                        roll_rad = math.radians(float(parts[1]))
                        pitch_rad = math.radians(float(parts[2]))
                        yaw_rad = math.radians(float(parts[3]))

                        q = self.euler_to_quaternion(roll_rad, pitch_rad, yaw_rad)
                        
                        imu_msg.orientation.x = q[0]
                        imu_msg.orientation.y = q[1]
                        imu_msg.orientation.z = q[2]
                        imu_msg.orientation.w = q[3]
                        
                        self.imu_publisher.publish(imu_msg)
            except Exception as e:
                self.get_logger().warn(f'Błąd parsowania danych IMU: {e}')

def main(args=None):
    rclpy.init(args=args)
    node = RcCarDriver()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()