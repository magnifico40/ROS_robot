#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import math
import serial
import threading
import time

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

        self.ser = None
        self.serial_lock = threading.Lock()

        self.connect_serial()

        self.sub_cmd_vel = self.create_subscription(Twist, 'cmd_vel', self.cmd_vel_callback, 10)
        self.sub_rtcm = self.create_subscription(RtcmMessage, '/rtcm', self.rtcm_callback, 10)
        self.sub_mower = self.create_subscription(Int8, '/mower_status', self.mower_callback, 10)

        self.pub_imu = self.create_publisher(Imu, 'imu/data_raw', 10)
        self.pub_nmea = self.create_publisher(Sentence, '/nmea', 10)
        self.pub_pos = self.create_publisher(Point, '/robot_position', 10)
        self.pub_odom = self.create_publisher(Odometry, '/odometry/wheel', 10)

        self.read_thread = threading.Thread(target=self.serial_read_loop, daemon=True)
        self.read_thread.start()

    def connect_serial(self):
        with self.serial_lock:
            try:
                if self.ser is not None:
                    try:
                        self.ser.close()
                    except:
                        pass
                
                self.ser = serial.Serial(self.serial_port, self.baud_rate, timeout=1.0)
                self.ser.reset_input_buffer()
                self.ser.reset_output_buffer()
                self.get_logger().info(f'ESP32 connected on {self.serial_port}')
            except Exception as e:
                self.get_logger().error(f'Port error: {e}')
                self.ser = None

    def send_to_serial(self, data_str):
        with self.serial_lock:
            if self.ser and self.ser.is_open:
                try:
                    self.ser.write(data_str.encode('utf-8'))
                    self.ser.flush()
                except Exception:
                    pass

    def cmd_vel_callback(self, msg):
        v = msg.linear.x
        omega = msg.angular.z

        v_left = v - (omega * self.wheel_base / 2.0)
        v_right = v + (omega * self.wheel_base / 2.0)

        cmd_str = f"M,{v_left:.3f},{v_right:.3f}\n"
        self.send_to_serial(cmd_str)

    def rtcm_callback(self, msg):
        hex_string = ''.join(f'{b:02x}' for b in msg.message)
        if hex_string:
            rtcm_str = f"RTCM,{hex_string}\n"
            self.send_to_serial(rtcm_str)

    def mower_callback(self, msg):
        status = msg.data
        cmd_str = f"BLDC,{status}\n"
        self.send_to_serial(cmd_str)

    def serial_read_loop(self):
        while rclpy.ok():
            if self.ser is None or not self.ser.is_open:
                time.sleep(1.0)
                self.connect_serial()
                continue

            try:
                line_bytes = self.ser.readline()
                
                if not line_bytes:
                    continue

                line = line_bytes.decode('utf-8', errors='ignore').strip()
                if line:
                    self.process_line(line)
                    
            except (serial.SerialException, OSError, AttributeError):
                self.get_logger().warn('Hardware Disconnect. Reconnecting...')
                with self.serial_lock:
                    if self.ser:
                        try:
                            self.ser.close()
                        except:
                            pass
                        self.ser = None
            except Exception:
                pass

    def is_valid_float(self, val, limit=100.0):
        # Sprawdza czy wartość nie jest NaN, nie jest Infinity i mieści się w fizycznym limicie
        return not math.isnan(val) and not math.isinf(val) and abs(val) < limit
    
    def process_line(self, line):
        try:
            if line.startswith("IMU,"):
                parts = line.split(',')
                if len(parts) == 11:

                    ax = float(parts[1])
                    ay = float(parts[2])
                    az = float(parts[3])
                    gx = float(parts[4])
                    gy = float(parts[5])
                    gz = float(parts[6])
                    w = float(parts[7])
                    x = float(parts[8])
                    y = float(parts[9])
                    z = float(parts[10])

                    # OBRONA: Jeśli jakakolwiek wartość IMU jest matematycznym śmieciem, odrzuć ramkę
                    if not all(self.is_valid_float(val, 1000.0) for val in [ax, ay, az, gx, gy, gz, w, x, y, z]):
                        self.get_logger().warn("Odrzucono ramkę IMU: Wartości inf/nan lub poza limitem!")
                        return

                    imu_msg = Imu()
                    imu_msg.header.stamp = self.get_clock().now().to_msg()
                    imu_msg.header.frame_id = "imu_link"

                    imu_msg.linear_acceleration.x = ax
                    imu_msg.linear_acceleration.y = ay
                    imu_msg.linear_acceleration.z = az

                    imu_msg.angular_velocity.x = gx
                    imu_msg.angular_velocity.y = gy
                    imu_msg.angular_velocity.z = gz

                    # Zabezpieczenie przed kwaternionem z samych zer
                    if w == 0.0 and x == 0.0 and y == 0.0 and z == 0.0:
                        w = 1.0

                    length = math.sqrt(w*w + x*x + y*y + z*z)
                    if length == 0.0:
                        length = 1.0
                    
                    # Poprawione przypisanie znormalizowanego kwaternionu!
                    imu_msg.orientation.w = w / length
                    imu_msg.orientation.x = x / length
                    imu_msg.orientation.y = y / length
                    imu_msg.orientation.z = z / length

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

            elif line.startswith("ODOM,"):
                parts = line.split(',')
                if len(parts) == 5:
                    vel_l_rads = float(parts[2])
                    vel_r_rads = float(parts[4])
                    
                    # OBRONA: Odrzucenie kosmicznych/nieprawidłowych prędkości z kół
                    if not (self.is_valid_float(vel_l_rads, 50.0) and self.is_valid_float(vel_r_rads, 50.0)):
                        self.get_logger().warn(f"Odrzucono ODOM: Kosmiczne prędkości!")
                        return
                        
                    self.process_odometry(vel_l_rads, vel_r_rads)

            elif line.startswith("$GN") or line.startswith("$GP"):
                nmea_msg = Sentence()
                nmea_msg.header.stamp = self.get_clock().now().to_msg()
                nmea_msg.header.frame_id = "gps_link"
                nmea_msg.sentence = line
                self.pub_nmea.publish(nmea_msg)

                if "GGA" in line:
                    parts = line.split(',')
                    if len(parts) >= 6 and parts[2] and parts[4]:
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

        except (ValueError, IndexError):
            pass

    def process_odometry(self, vel_l_rads, vel_r_rads):
        now = self.get_clock().now()

        if self.last_odom_time is None:
            self.last_odom_time = now
            return

        dt = (now.nanoseconds - self.last_odom_time.nanoseconds) / 1e9
        
        # OBRONA: Zapobieganie dzieleniu przez zero w EKF (zbyt szybkie ramki)
        if dt <= 0.001:
            return
            
        self.last_odom_time = now

        v_left = vel_l_rads * self.wheel_radius
        v_right = vel_r_rads * self.wheel_radius

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

        # Zabezpieczona kowariancja (Pozycja)
        odom_msg.pose.covariance[0] = 0.05
        odom_msg.pose.covariance[7] = 0.05
        odom_msg.pose.covariance[14] = 1e-9  # Z
        odom_msg.pose.covariance[21] = 1e-9  # Roll
        odom_msg.pose.covariance[28] = 1e-9  # Pitch
        odom_msg.pose.covariance[35] = 0.05
        
        # Zabezpieczona kowariancja (Prędkość)
        odom_msg.twist.covariance[0] = 0.05
        odom_msg.twist.covariance[7] = 1e-9  # Vy
        odom_msg.twist.covariance[14] = 1e-9 # Vz
        odom_msg.twist.covariance[21] = 1e-9 # Vroll
        odom_msg.twist.covariance[28] = 1e-9 # Vpitch
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