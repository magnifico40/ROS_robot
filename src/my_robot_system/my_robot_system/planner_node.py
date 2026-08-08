import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point, Twist
from sensor_msgs.msg import Imu
from std_msgs.msg import String
import json
import math
from pyproj import Proj
from shapely.geometry import Polygon, LineString

class IntegratedNavigator(Node):
    def __init__(self):
        super().__init__('integrated_navigator')

        self.sub_pos = self.create_subscription(Point, '/robot_position', self.cb_pos, 10)
        self.sub_imu = self.create_subscription(Imu, 'imu/data_raw', self.cb_imu, 10)
        self.sub_mode = self.create_subscription(String, 'steering_mode', self.cb_mode, 10)
        self.sub_waypoints = self.create_subscription(String, 'waypoints', self.cb_waypoints, 10)
        self.pub_cmd = self.create_publisher(Twist, 'cmd_vel_raw', 10)

        self.proj = Proj(proj='utm', zone='34', ellps='WGS84', preserve_units=False)
        self.timer = self.create_timer(0.1, self.control_loop)

        self.robot_x = None
        self.robot_y = None
        self.robot_yaw = None
        self.mode = "manual"
        self.path = []
        self.curr_target = 0
        self.is_fence_mode = False

        self.linear_speed = 0.5
        self.angular_speed = 0.4
        self.robot_width = 0.4274
        self.lookahead_dist = 0.5
        self.step_size = 0.05
        self.row_spacing = self.robot_width * 0.8

    def cb_pos(self, msg):
        self.robot_x, self.robot_y = self.proj(msg.y, msg.x)

    def cb_imu(self, msg):
        q = msg.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.robot_yaw = math.atan2(siny_cosp, cosy_cosp)

    def cb_mode(self, msg):
        new_mode = msg.data.lower().strip()
        if new_mode != self.mode:
            self.get_logger().info(f"ZMIANA TRYBU: {self.mode} -> {new_mode}")
            self.mode = new_mode

    def cb_waypoints(self, msg):
        try:
            data = json.loads(msg.data)
            wpts = data.get("waypoints", [])
            
            if not wpts:
                return

            raw_path = []
            for wp in wpts:
                x, y = self.proj(wp["lng"], wp["lat"])
                raw_path.append((x, y))

            is_closed_loop = False
            if len(wpts) > 1:
                first_wp = wpts[0]
                last_wp = wpts[-1]
                if abs(first_wp["lat"] - last_wp["lat"]) < 1e-5 and abs(first_wp["lng"] - last_wp["lng"]) < 1e-5:
                    is_closed_loop = True


            #tryb fence
            if is_closed_loop:
                self.is_fence_mode = True
                self.get_logger().info("TRYB: FENCE")

                poly = Polygon(raw_path)
                if not poly.is_valid:
                    poly = poly.buffer(0)

                min_x, min_y, max_x, max_y = poly.bounds
                
                lines = []
                current_x = min_x + (self.row_spacing / 2.0)
                while current_x < max_x:
                    vertical_line = LineString([(current_x, min_y - 10.0), (current_x, max_y + 10.0)])
                    intersection = poly.intersection(vertical_line)
                    
                    if not intersection.is_empty:
                        if intersection.geom_type == 'LineString':
                            lines.append(list(intersection.coords))
                        elif intersection.geom_type == 'MultiLineString':
                            for line in intersection.geoms:
                                lines.append(list(line.coords))
                    current_x += self.row_spacing

                shapely_raw_path = []
                for idx, line in enumerate(lines):
                    sorted_line = sorted(line, key=lambda p: p[1])
                    if idx % 2 == 1:
                        sorted_line.reverse()
                    shapely_raw_path.extend(sorted_line)

                dense_path = []
                for i in range(len(shapely_raw_path) - 1):
                    x1, y1 = shapely_raw_path[i]
                    x2, y2 = shapely_raw_path[i + 1]

                    dense_path.append((x1, y1))
                    dist = math.hypot(x2 - x1, y2 - y1)

                    if dist > self.step_size:
                        num_steps = int(dist / self.step_size)
                        for j in range(1, num_steps):
                            fraction = j / float(num_steps)
                            interp_x = x1 + fraction * (x2 - x1)
                            interp_y = y1 + fraction * (y2 - y1)
                            dense_path.append((interp_x, interp_y))

                if shapely_raw_path:
                    dense_path.append(shapely_raw_path[-1])

                self.path = dense_path
                self.get_logger().info(f"ZAŁADOWANO {len(self.path)} PUNKTÓW ZYGZAKA PO ZAGĘSZCZENIU.")

            else:
                self.is_fence_mode = False
                self.path = raw_path
                self.get_logger().info(f"TRYB: PATH. ZAŁADOWANO {len(self.path)} PUNKTÓW BAZOWYCH.")

            self.curr_target = 0

        except Exception as e:
            self.get_logger().error(f"Błąd parsowania: {e}")

    def control_loop(self):
        twist = Twist()

        if self.robot_x is None or self.robot_yaw is None:
            return

        if self.mode != "auto":
            self.pub_cmd.publish(twist)
            return

        if not self.path:
            self.pub_cmd.publish(twist)
            return

        min_dist = float('inf')
        max_search = min(len(self.path), self.curr_target + 50)

        for i in range(self.curr_target, max_search):
            pt_x, pt_y = self.path[i]
            dx = pt_x - self.robot_x
            dy = pt_y - self.robot_y
            angle_to_pt = math.atan2(dy, dx)
            angle_diff = angle_to_pt - self.robot_yaw
            angle_diff = math.atan2(math.sin(angle_diff), math.cos(angle_diff))
            
            if abs(angle_diff) < math.pi * 0.4:
                dist = math.hypot(dx, dy)
                if dist < min_dist:
                    min_dist = dist
                    self.curr_target = i

        if self.curr_target >= len(self.path) - 2:
            self.get_logger().info("Koniec trasy", once=True)
            self.pub_cmd.publish(twist)
            self.path = []
            self.curr_target = 0
            return

        target_point = self.path[-1]
        
        for i in range(self.curr_target, len(self.path)):
            dist = math.hypot(self.robot_x - self.path[i][0], self.robot_y - self.path[i][1])
            if dist >= self.lookahead_dist:
                target_point = self.path[i]
                break

        target_x, target_y = target_point
        dx = target_x - self.robot_x
        dy = target_y - self.robot_y
        angle_to_target = math.atan2(dy, dx)
        angle_error = angle_to_target - self.robot_yaw
        angle_error = math.atan2(math.sin(angle_error), math.cos(angle_error))

        if abs(angle_error) > (math.pi / 8.0):
            twist.linear.x = 0.0
            twist.angular.z = math.copysign(self.angular_speed, angle_error)
        else:
            twist.linear.x = self.linear_speed
            twist.angular.z = (2.0 * self.linear_speed * math.sin(angle_error)) / self.lookahead_dist

        self.pub_cmd.publish(twist)

def main(args=None):
    rclpy.init(args=args)
    node = IntegratedNavigator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()