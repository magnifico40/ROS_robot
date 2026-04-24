#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Imu
from nmea_msgs.msg import Sentence
from rtcm_msgs.msg import Message as RtcmMessage
import serial

class RobotHardwareBridge(Node):
    def __init__(self):
        super().__init__('robot_hardware_bridge')

        self.declare_parameter('serial_port', '/dev/ttyACM0')
        self.declare_parameter('baud_rate', 115200)
        # Rozstaw kół
        self.declare_parameter('wheel_base_meters', 0.368)

        self.serial_port = self.get_parameter('serial_port').value
        self.baud_rate = self.get_parameter('baud_rate').value
        self.wheel_base = self.get_parameter('wheel_base_meters').value

        # Serial
        try:
            self.ser = serial.Serial(self.serial_port, self.baud_rate, timeout=0.05)
            self.get_logger().info(f'Mostek ESP32 uruchomiony na {self.serial_port}')
        except Exception as e:
            self.get_logger().error(f'Błąd portu: {e}')
            exit(1)

        # Subskrypcje
        self.sub_cmd_vel = self.create_subscription(Twist, 'cmd_vel', self.cmd_vel_callback, 10)
        self.sub_rtcm = self.create_subscription(RtcmMessage, '/rtcm', self.rtcm_callback, 10)

        # Publikacje
        self.pub_imu = self.create_publisher(Imu, 'imu/data_raw', 10)
        self.pub_nmea = self.create_publisher(Sentence, 'nmea_sentence', 10)
        
        # Timer odczytu
        self.create_timer(0.01, self.read_serial_data)

    def cmd_vel_callback(self, msg):
        v = msg.linear.x
        omega = msg.angular.z
        
        # Kinematyka
        v_left = v - (omega * self.wheel_base / 2.0)
        v_right = v + (omega * self.wheel_base / 2.0)

        cmd_str = f"M,{v_left:.3f},{v_right:.3f}\n"
        self.ser.write(cmd_str.encode('utf-8'))

    def rtcm_callback(self, msg):
        hex_string = ''.join(f'{b:02x}' for b in msg.message)
        if hex_string:
            rtcm_str = f"RTCM,{hex_string}\n"
            self.ser.write(rtcm_str.encode('utf-8'))

    def read_serial_data(self):
        while self.ser.in_waiting > 0:
            try:
                line = self.ser.readline().decode('utf-8', errors='ignore').strip()
                
                # Odbiór danych IMU
                if line.startswith("IMU,"):
                    parts = line.split(',')
                    if len(parts) == 7:
                        imu_msg = Imu()
                        imu_msg.header.stamp = self.get_clock().now().to_msg()
                        imu_msg.header.frame_id = "imu_link"
                        
                        imu_msg.linear_acceleration.x = float(parts[1])
                        imu_msg.linear_acceleration.y = float(parts[2])
                        imu_msg.linear_acceleration.z = float(parts[3])
                        
                        imu_msg.angular_velocity.x = float(parts[4])
                        imu_msg.angular_velocity.y = float(parts[5])
                        imu_msg.angular_velocity.z = float(parts[6])
                        
                        self.pub_imu.publish(imu_msg)

                # Odbiór danych GPS
                elif line.startswith("$GN") or line.startswith("$GP"):
                    nmea_msg = Sentence()
                    nmea_msg.header.stamp = self.get_clock().now().to_msg()
                    nmea_msg.header.frame_id = "gps_link"
                    nmea_msg.sentence = line
                    self.pub_nmea.publish(nmea_msg)
                    
            except Exception as e:
                self.get_logger().warn(f'Błąd parsowania: {e}')

def main(args=None):
    rclpy.init(args=args)
    node = RobotHardwareBridge()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()