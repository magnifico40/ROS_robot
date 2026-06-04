import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan, Imu
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import Twist
import math


class Pursuit(Node):
    def __init__(self):
        super().__init__('area_coverage')
        self.subscription = self.create_subscription(
            Odometry, '/odometry/global', self.odom_callback, 10)

        self.path_sub = self.create_subscription(Path, '/planned_path', self.path_callback, 10)
        self.curr_x = 0.0
        self.curr_y = 0.0
        self.curr_yaw = 0.0

        self.path = []
        self.linear_speed = 0.5
        self.angular_speed = 0.4
        self.robot_width = 0.4274

        self.lookahead_dist = 0.5
        self.curr_target = 0


        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel_raw', 10)
        self.get_logger().info('Węzeł zygzaka uruchomiony')
        self.timer = self.create_timer(0.1, self.control_loop)

    def path_callback(self, msg):
        if len(self.path) == 0:
            self.get_logger().info(f"Otrzymano nową ścieżkę")
        raw_path = [(pose.pose.position.x, pose.pose.position.y) for pose in msg.poses]
        dense_path = []
        step_size = 0.05


        #zageszczenie sciezki
        for i in range(len(raw_path) - 1):
            x1, y1 = raw_path[i]
            x2, y2 = raw_path[i + 1]

            dense_path.append((x1, y1))

            dist = math.hypot(x2 - x1, y2 - y1)

            if dist > step_size:
                num_steps = int(dist / step_size)
                for j in range(1, num_steps):
                    fraction = j / float(num_steps)
                    interp_x = x1 + fraction * (x2 - x1)
                    interp_y = y1 + fraction * (y2 - y1)
                    dense_path.append((interp_x, interp_y))

        if raw_path:
            dense_path.append(raw_path[-1])

        self.path = dense_path

    def odom_callback(self, msg):
        self.curr_x = msg.pose.pose.position.x
        self.curr_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        self.curr_yaw = math.atan2(siny_cosp, cosy_cosp)


    def control_loop(self):
        twist = Twist()
        if not self.path:
            return

        robot_x,robot_y = self.curr_x, self.curr_y
        min_dist = float('inf')
        max_search = min(len(self.path), self.curr_target+50)

        #znajdujemy najblizszy punkt
        for i in range(self.curr_target,max_search):  #przechodzimy przez najblisze punkty
            pt_x,pt_y = self.path[i]
            dx = pt_x - robot_x
            dy = pt_y - robot_y
            angle_to_pt = math.atan2(dy, dx)
            angle_diff = angle_to_pt - self.curr_yaw
            angle_diff = math.atan2(math.sin(angle_diff), math.cos(angle_diff)) #zapewnia ze dostajemy najmniejsza wersje kata (350 w lewo = 10 prawo)
            if abs(angle_diff) < math.pi*0.4:	#ignorujemy te ktore odstaja o ponad 45 stopni od przodu
                dist = math.hypot(dx,dy)
                if dist < min_dist:
                    min_dist = dist
                    self.curr_target = i
             
        curr_path = self.path[self.curr_target]
        self.get_logger().info(f"cel: {self.curr_target}, {curr_path}")

        if self.curr_target >= len(self.path)-2: #koniec trasy
            self.get_logger().info("Koniec")
            self.cmd_vel_pub.publish(twist)
            self.path = []
            self.curr_target = 0
            return

        target_point = self.path[-1]
        #znajdujemy nasz nastepny punkt naprzod / marchewka
        for i in range(self.curr_target, len(self.path)):
            dist = math.hypot(robot_x - self.path[i][0], robot_y - self.path[i][1])
            if dist >= self.lookahead_dist:
                target_point = self.path[i]
                break

        self.get_logger().info(f"{target_point}")

        target_x,target_y = target_point

        dx = target_x - robot_x
        dy = target_y - robot_y
        angle_to_target = math.atan2(dy, dx)
        angle_error = angle_to_target - self.curr_yaw
        angle_error = math.atan2(math.sin(angle_error), math.cos(angle_error))


        if abs(angle_error)>(math.pi/8.0): #około 22.5 stopnia
            twist.linear.x = 0.0
            twist.angular.z = math.copysign(self.angular_speed,angle_error) #daje wartosc pierwszej ale znak drugiej liczby
            self.get_logger().info("Skrecamy")
        else:
            twist.linear.x = self.linear_speed
            #gotowy wzor na pure pursuit
            twist.angular.z = (2.0*self.linear_speed*math.sin(angle_error))/self.lookahead_dist

        self.cmd_vel_pub.publish(twist)


def main(args=None):
    rclpy.init(args=args)
    node = Pursuit()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
