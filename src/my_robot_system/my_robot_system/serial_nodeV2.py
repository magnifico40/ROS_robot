#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import math
import serial

from geometry_msgs.msg import Twist, Point
from std_msgs.msg import Int8
from sensor_msgs.msg import Imu
from nmea_msgs.msg import Sentence
from rtcm_msgs.msg import Message as RtcmMessage
from nav_msgs.msg import Odometry


class RobotHardwareBridge(Node):
    def __init__(self):
        super().__init__('robot_hardware_bridge')

        self.declare_parameter('serial_port', '/dev/ttyACM0')
        self.declare_parameter('baud_rate', 230400)
        self.declare_parameter('wheel_base_meters', 0.368)
        self.declare_parameter('wheel_radius_meters', 0.099)

        self.serial_port = self.get_parameter('serial_port').value
        self.baud_rate = self.get_parameter('baud_rate').value
        self.wheel_base = self.get_parameter('wheel_base_meters').value
        self.wheel_radius = self.get_parameter('wheel_radius_meters').value

        self.odom_x = 0.0
        self.odom_y = 0.0
        self.odom_yaw = 0.0
        self.last_odom_time = None

        # Serial
        try:
            self.ser = serial.Serial(self.serial_port, self.baud_rate, timeout=0.05)
            self.ser.reset_input_buffer()
            self.serial_buffer = ""
            self.get_logger().info(f'ESP32 on port {self.serial_port}')
        except Exception as e:
            self.get_logger().error(f'Port error: {e}')
            exit(1)

        # Subscriptions
        self.sub_cmd_vel = self.create_subscription(Twist, 'cmd_vel', self.cmd_vel_callback, 10)
        self.sub_rtcm = self.create_subscription(RtcmMessage, '/rtcm', self.rtcm_callback, 10)
        self.sub_mower = self.create_subscription(Int8, '/mower_status', self.mower_callback, 10)

        # Publications
        self.pub_imu = self.create_publisher(Imu, 'imu/data_raw', 10)
        self.pub_nmea = self.create_publisher(Sentence, '/nmea_sentence', 10)
        self.pub_pos = self.create_publisher(Point, '/robot_position', 10)
        self.pub_odom = self.create_publisher(Odometry, '/odom_wheels', 10)

        self.create_timer(0.01, self.read_serial_data)

    def cmd_vel_callback(self, msg):
        if self.ser is None or not self.ser.is_open:
            return

        v = msg.linear.x
        omega = msg.angular.z

        v_left = v - (omega * self.wheel_base / 2.0)
        v_right = v + (omega * self.wheel_base / 2.0)

        cmd_str = f"M,{v_left:.3f},{v_right:.3f}\n"

        try:
            self.ser.write(cmd_str.encode('utf-8'))
        except (OSError, serial.SerialException):
            self.ser.close()

    def rtcm_callback(self, msg):
        if self.ser is None or not self.ser.is_open:
            return

        hex_string = ''.join(f'{b:02x}' for b in msg.message)
        if hex_string:
            rtcm_str = f"RTCM,{hex_string}\n"
            try:
                self.ser.write(rtcm_str.encode('utf-8'))
            except (OSError, serial.SerialException):
                self.ser.close()

    def mower_callback(self, msg):
        if self.ser is None or not self.ser.is_open:
            return

        status = msg.data
        cmd_str = f"BLDC,{status}\n"
        try:
            self.ser.write(cmd_str.encode('utf-8'))
        except (OSError, serial.SerialException):
            self.ser.close()

    def read_serial_data(self):
        # Auto-reconnect
        if self.ser is None or not self.ser.is_open:
            try:
                self.ser = serial.Serial(self.serial_port, self.baud_rate, timeout=0.01)
                self.ser.reset_input_buffer()
                self.serial_buffer = ""
                self.get_logger().info('Reconnected ESP32')
            except Exception:
                return

        try:
            bytes_waiting = self.ser.in_waiting
            if bytes_waiting > 0:
                chunk = self.ser.read(bytes_waiting).decode('utf-8', errors='ignore')
                self.serial_buffer += chunk

                if '\n' in self.serial_buffer:
                    lines = self.serial_buffer.split('\n')
                    self.serial_buffer = lines.pop()

                    latest_imu_line = None
                    latest_odom_line = None

                    for line in lines:
                        line = line.strip()
                        if not line:
                            continue

                        if line.startswith("IMU,"):
                            latest_imu_line = line
                        elif line.startswith("ODOM,"):
                            latest_odom_line = line

                        elif line.startswith("$GN") or line.startswith("$GP"):
                            nmea_msg = Sentence()
                            nmea_msg.header.stamp = self.get_clock().now().to_msg()
                            nmea_msg.header.frame_id = "gps_link"
                            nmea_msg.sentence = line
                            self.pub_nmea.publish(nmea_msg)

                            if "GGA" in line:
                                parts = line.split(',')
                                if len(parts) >= 6 and parts[2] and parts[4]:
                                    try:
                                        lat = float(parts[2][:2]) + (float(parts[2][2:]) / 60.0)
                                        if parts[3] == 'S': lat = -lat
                                        lon = float(parts[4][:3]) + (float(parts[4][3:]) / 60.0)
                                        if parts[5] == 'W': lon = -lon

                                        pos_msg = Point()
                                        pos_msg.x = lat
                                        pos_msg.y = lon

                                        if len(parts) >= 7 and parts[6]:
                                            pos_msg.z = float(parts[6])

                                        self.pub_pos.publish(pos_msg)
                                    except ValueError:
                                        pass

                    # IMU
                    if latest_imu_line:
                        parts = latest_imu_line.split(',')
                        if len(parts) == 11:
                            imu_msg = Imu()
                            imu_msg.header.stamp = self.get_clock().now().to_msg()
                            imu_msg.header.frame_id = "imu_link"

                            # accelerations
                            imu_msg.linear_acceleration.x = float(parts[1])
                            imu_msg.linear_acceleration.y = float(parts[2])
                            imu_msg.linear_acceleration.z = float(parts[3])

                            # gyro
                            imu_msg.angular_velocity.x = float(parts[4])
                            imu_msg.angular_velocity.y = float(parts[5])
                            imu_msg.angular_velocity.z = float(parts[6])

                            # quaternions
                            imu_msg.orientation.w = float(parts[7])
                            imu_msg.orientation.x = float(parts[8])
                            imu_msg.orientation.y = float(parts[9])
                            imu_msg.orientation.z = float(parts[10])

                            # matrix EKF
                            imu_msg.orientation_covariance[0] = 0.01
                            imu_msg.orientation_covariance[4] = 0.01
                            imu_msg.orientation_covariance[8] = 0.01

                            imu_msg.angular_velocity_covariance[0] = 0.05
                            imu_msg.angular_velocity_covariance[4] = 0.05
                            imu_msg.angular_velocity_covariance[8] = 0.05

                            imu_msg.linear_acceleration_covariance[0] = 0.1
                            imu_msg.linear_acceleration_covariance[4] = 0.1
                            imu_msg.linear_acceleration_covariance[8] = 0.1

                            self.pub_imu.publish(imu_msg)

                    if latest_odom_line:
                        parts = latest_odom_line.split(',')
                        if len(parts) == 5:
                            vel_l_rads = float(parts[2])
                            vel_r_rads = float(parts[4])
                            self.process_odometry(vel_l_rads, vel_r_rads)

        except Exception as e:
            self.get_logger().warn(f'Parsing Error: {e}')
            try:
                self.ser.close()
            except:
                pass

    def process_odometry(self, vel_l_rads, vel_r_rads):
        now = self.get_clock().now()

        if self.last_odom_time is None:
            self.last_odom_time = now
            return

        dt = (now.nanoseconds - self.last_odom_time.nanoseconds) / 1e9
        self.last_odom_time = now

        # rad/s -> m/s
        v_left = vel_l_rads * self.wheel_radius
        v_right = vel_r_rads * self.wheel_radius

        # robot velocity
        v_linear = (v_right + v_left) / 2.0
        v_angular = (v_right - v_left) / self.wheel_base

        self.odom_yaw += v_angular * dt
        self.odom_x += v_linear * math.cos(self.odom_yaw) * dt
        self.odom_y += v_linear * math.sin(self.odom_yaw) * dt

        odom_msg = Odometry()
        odom_msg.header.stamp = now.to_msg()
        odom_msg.header.frame_id = "odom"
        odom_msg.child_frame_id = "base_link"

        odom_msg.pose.pose.position.x = self.odom_x
        odom_msg.pose.pose.position.y = self.odom_y

        odom_msg.pose.pose.orientation.z = math.sin(self.odom_yaw / 2.0)
        odom_msg.pose.pose.orientation.w = math.cos(self.odom_yaw / 2.0)

        odom_msg.twist.twist.linear.x = v_linear
        odom_msg.twist.twist.angular.z = v_angular


        odom_msg.pose.covariance[0] = 0.05
        odom_msg.pose.covariance[7] = 0.05
        odom_msg.pose.covariance[35] = 0.05
        odom_msg.twist.covariance[0] = 0.05
        odom_msg.twist.covariance[35] = 0.05

        self.pub_odom.publish(odom_msg)


def main(args=None):
    rclpy.init(args=args)
    node = RobotHardwareBridge()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()