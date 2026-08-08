#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from vision_msgs.msg import Detection2DArray
from sensor_msgs.msg import LaserScan
import time
import math

class SafetyGuardNode(Node):
    def __init__(self):
        super().__init__('safety_guard_node')

        self.SAFE_DISTANCE = 0.9
        self.DANGEROUS_CLASSES = ['obstacle', 'person'] 

        self.TURN_SPEED = 0.5
        self.FORWARD_SPEED = 0.4
        self.state = "NORMAL" 
        
        self.latest_scan = None
        self.path_is_clear = True
        self.current_turn_direction = 1.0
        self.state_start_time = 0.0
        self.exit_start_time = 0.0

        self.lidar_clear = True
        self.camera_clear = True

        # Parametry do filtra przeciwzakłóceniowego (duchy lasera)
        self.MIN_POINTS_TO_TRIGGER = 3

        target_angle_deg = 30.0 
        target_angle_rad = math.radians(target_angle_deg)
        self.MIN_TURN_TIME = target_angle_rad / self.TURN_SPEED 

        self.create_subscription(Twist, 'rt/cmd_vel_raw', self.cmd_vel_callback, 10)
        self.create_subscription(Detection2DArray, '/fusion/detections_meters', self.vision_callback, 10)
        self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        
        self.pub_cmd_vel = self.create_publisher(Twist, 'rt/cmd_vel', 10)
        self.timer = self.create_timer(0.1, self.state_machine)
        self.get_logger().info("Avoider uruchomiony.")

    def get_scan_slice(self, msg, start_angle_deg, end_angle_deg):
        slice_ranges = []
        min_angle = min(start_angle_deg, end_angle_deg)
        max_angle = max(start_angle_deg, end_angle_deg)

        for i, r in enumerate(msg.ranges):
            if 0.05 < r < 10.0:
                # Obliczanie kąta dla bieżącego indeksu
                angle = msg.angle_min + (i * msg.angle_increment)
                # Korekta sprzętowa: obrót o 180 stopni (ponieważ przód lasera jest fizycznie z tyłu)
                angle += math.pi
                
                # Normalizacja kąta do przedziału [-180, 180] stopni
                angle_deg = math.degrees(math.atan2(math.sin(angle), math.cos(angle)))
                
                if min_angle <= angle_deg <= max_angle:
                    slice_ranges.append(r)
                    
        return slice_ranges

    def scan_callback(self, msg):
        self.latest_scan = msg
        
        # Wycinamy sektor bezpośrednio przed robotem (-30 do 30 stopni od naszej "wyprostowanej" logiki)
        front_slice = self.get_scan_slice(msg, -30, 30)

        # Filtr zakłóceń (ignorowanie pojedynczych błędnych punktów)
        danger_points = [r for r in front_slice if r < self.SAFE_DISTANCE]
        
        if len(danger_points) >= self.MIN_POINTS_TO_TRIGGER:
            self.lidar_clear = False
            min_front_dist = min(danger_points)
        else:
            self.lidar_clear = True
            min_front_dist = float('inf')

        if self.state == "NORMAL" and not self.lidar_clear:
            self.get_logger().warn(f"LIDAR WYKRYŁ PRZESZKODĘ! ({min_front_dist:.2f}m na {len(danger_points)} punktach). Omijam!")
            self.trigger_avoidance()
    
    def vision_callback(self, msg):
        danger_vision = False
        for det in msg.detections:
            if not det.results:
                continue
                
            name = det.results[0].id
            distance = det.bbox.center.y

            if name in self.DANGEROUS_CLASSES and distance < self.SAFE_DISTANCE:
                danger_vision = True               
                break
        
        self.camera_clear = not danger_vision
        if self.state == "NORMAL" and not self.camera_clear:
            self.get_logger().warn("KAMERA WYKRYŁA ZAGROŻENIE")
            self.trigger_avoidance()                

    def trigger_avoidance(self):
        if self.latest_scan:
            # Intuicyjne wycinanie sektorów bocznych dzięki normalizacji kątów
            left_slice  = self.get_scan_slice(self.latest_scan, 30, 80)
            right_slice = self.get_scan_slice(self.latest_scan, -80, -30)

            avg_left = sum(left_slice)/len(left_slice) if left_slice else 0.0
            avg_right = sum(right_slice)/len(right_slice) if right_slice else 0.0

            # Skręcamy tam, gdzie średni dystans jest większy (gdzie jest więcej wolnego miejsca)
            self.current_turn_direction = 1.0 if avg_left >= avg_right else -1.0
            
        self.state = "AVOID_TURN"
        self.state_start_time = time.time()
        self.publish_stop()

    @property
    def clear_path_ahead(self):
        return self.camera_clear and self.lidar_clear

    def state_machine(self):
        now = time.time()
        elapsed = now - self.state_start_time

        if self.state == "AVOID_TURN":
            if not self.clear_path_ahead:
                t = Twist()
                t.angular.z = self.TURN_SPEED * self.current_turn_direction
                self.pub_cmd_vel.publish(t)
            else:
                self.get_logger().info("Skręt skończony. Omijam z boku.")
                self.state = "AVOID_DRIVE"
                self.state_start_time = now

        elif self.state == "AVOID_DRIVE":
            if not self.clear_path_ahead:
                self.get_logger().warn("Nowa przeszkoda w trakcie omijania! Ponawiam skręt.")
                self.state = "AVOID_TURN"
                self.state_start_time = now
                return
            
            if elapsed < 1.2:
                t = Twist()
                t.linear.x = self.FORWARD_SPEED
                self.pub_cmd_vel.publish(t)
            else:
                if self.is_side_clear():
                    self.state = "AVOID_EXIT"
                    self.exit_start_time = now
                else:
                    self.get_logger().info("Bok wciąż zajęty, przedłużam jazdę wzdłuż przeszkody.")
                    t = Twist()
                    t.linear.x = self.FORWARD_SPEED
                    self.pub_cmd_vel.publish(t)

        elif self.state == 'AVOID_EXIT':
            exit_time = now - self.exit_start_time
            if exit_time < 1.0:
                t = Twist()
                t.linear.x = self.FORWARD_SPEED
                self.pub_cmd_vel.publish(t)
            else:
                self.get_logger().info("Wracam na trasę.")
                self.state = "NORMAL"
    
    def cmd_vel_callback(self, msg):
        if self.state == "NORMAL":
            if msg.linear.x > 0 and not self.clear_path_ahead:
                # Opcjonalne awaryjne zatrzymanie komend prędkości prosto, jeśli droga jest zablokowana
                self.pub_cmd_vel.publish(Twist())
            else:
                self.pub_cmd_vel.publish(msg)

    def publish_stop(self):
        t = Twist()
        t.linear.x = 0.0
        self.pub_cmd_vel.publish(t)

    def is_side_clear(self):
        if not self.latest_scan:
            return False
            
        # Sprawdzanie wybranego boku (po normalizacji kątów)
        if self.current_turn_direction > 0:
            side_slice = self.get_scan_slice(self.latest_scan, -100, -30) # Prawa strona
        else:
            side_slice = self.get_scan_slice(self.latest_scan, 30, 100)   # Lewa strona

        min_dist = min(side_slice) if side_slice else float('inf')
        return min_dist > 0.5

def main(args=None):
    rclpy.init(args=args)
    node = SafetyGuardNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.publish_stop()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()