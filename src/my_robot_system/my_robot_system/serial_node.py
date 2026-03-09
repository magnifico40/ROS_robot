#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Imu
import serial
import time

class RcCarDriver(Node):
    def __init__(self):
        super().__init__('rc_car_driver')

        #declare_parameter / get-parameter - Pozwlaja na zmiane przy uruchamianiu z poziomu terminala
        self.declare_parameter('serial_port', '/dev/ttyUSB0')
        self.declare_parameter('baud_rate', 115200)
        
        # Konfiguracja serwa (czas trwania impulsu w us)
        self.declare_parameter('servo_min_us', 1200)    # Maks w prawo (lub lewo)
        self.declare_parameter('servo_center_us', 1500) # Środek
        self.declare_parameter('servo_max_us', 1800)    # Maks w lewo (lub prawo)
        
        # Konfiguracja silnika (PWM)
        self.declare_parameter('motor_max_pwm', 255)    # Maks moc

        # Pobranie parametrów
        self.serial_port = self.get_parameter('serial_port').value
        self.baud_rate = self.get_parameter('baud_rate').value
        self.servo_min = self.get_parameter('servo_min_us').value
        self.servo_center = self.get_parameter('servo_center_us').value
        self.servo_max = self.get_parameter('servo_max_us').value
        self.motor_max = self.get_parameter('motor_max_pwm').value

        # Połączenie Serial
        try:
            self.ser = serial.Serial(self.serial_port, self.baud_rate, timeout=0.1)
            self.get_logger().info(f'Połączono z ESP32 na {self.serial_port}')
        except Exception as e:
            self.get_logger().error(f'Błąd portu szeregowego: {e}')
            exit(1)

        # Subskrypcja cmd_vel (sterowanie z klawiatury/nav2)
        self.subscription = self.create_subscription(
            Twist,
            'cmd_vel',
            self.cmd_vel_callback,
            10)

        # Publikacja IMU
        self.imu_publisher = self.create_publisher(Imu, 'imu/data_raw', 10)
        
        # Timer do odczytu z Seriala (50Hz)
        self.create_timer(0.02, self.read_serial_data)

    def map_value(self, x, in_min, in_max, out_min, out_max):
        return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

    def cmd_vel_callback(self, msg):
        # --- Sterowanie (Ackermann) ---
        linear_x = msg.linear.x   # Prędkość przód/tył
        angular_z = msg.angular.z # Skręt

        # 1. Obliczanie PWM Silnika
        # Zakładamy, że linear_x=1.0 to pełna moc. Możesz to przeskalować.
        pwm_val = int(linear_x * self.motor_max)
        pwm_val = max(min(pwm_val, 255), -255)

        # 2. Obliczanie Impulsu Serwa
        # angular_z > 0 to zazwyczaj w lewo. 
        # Zakładamy zakres wejściowy od -1.0 (prawo) do 1.0 (lewo) rad/s (uproszczenie)
        if angular_z == 0:
            servo_us = self.servo_center
        elif angular_z > 0: # Skręt w jedną stronę
            # Mapuj 0 do 1.0 na center do max
            servo_us = self.map_value(angular_z, 0.0, 1.0, self.servo_center, self.servo_max)
        else: # Skręt w drugą stronę
            # Mapuj -1.0 do 0 na min do center
            servo_us = self.map_value(angular_z, -1.0, 0.0, self.servo_min, self.servo_center)

        servo_us = int(max(min(servo_us, self.servo_max), self.servo_min))

        # Wyślij do ESP32
        cmd_str = f"{servo_us},{pwm_val}\n"
        self.ser.write(cmd_str.encode('utf-8'))

    def read_serial_data(self):
        if self.ser.in_waiting > 0:
            try:
                line = self.ser.readline().decode('utf-8').strip()
                if line.startswith("IMU,"):
                    parts = line.split(',')
                    if len(parts) == 7:
                        imu_msg = Imu()
                        imu_msg.header.stamp = self.get_clock().now().to_msg()
                        imu_msg.header.frame_id = "imu_link"
                        
                        # MPU6050 wysyła m/s^2 i rad/s
                        imu_msg.linear_acceleration.x = float(parts[1])
                        imu_msg.linear_acceleration.y = float(parts[2])
                        imu_msg.linear_acceleration.z = float(parts[3])
                        
                        imu_msg.angular_velocity.x = float(parts[4])
                        imu_msg.angular_velocity.y = float(parts[5])
                        imu_msg.angular_velocity.z = float(parts[6])
                        
                        # Nie obliczamy orientacji (Quaternion) tutaj, 
                        # robi to zazwyczaj imu_filter_madgwick w ROS2
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
