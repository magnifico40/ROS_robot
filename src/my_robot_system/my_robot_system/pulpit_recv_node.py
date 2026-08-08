import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
from geometry_msgs.msg import Twist


class PulpitRecv(Node):
    def __init__(self):
        super().__init__('pulpit_recv_node')

        self.create_subscription(String, 'rt/steering_mode',  self.cb_steering,    10)
        self.create_subscription(Bool,   'rt/robot_status',   self.cb_robot_status, 10)
        self.create_subscription(Bool,   'rt/mowing_status',   self.cb_mowing_status, 10)
        self.create_subscription(Twist,  'rt/cmd_vel', self.cb_cmd_vel,      10) #'rt/manual_cmd_vel'
        self.create_subscription(String, 'rt/waypoints',      self.cb_waypoints,    10)

    def cb_steering(self, msg):
        print(f"[steering_mode] {msg.data}")

    def cb_robot_status(self, msg):
        print(f"[robot_status]  {msg.data}")
    
    def cb_mowing_status(self, msg):
        print(f"[mowing_status]  {msg.data}")

    def cb_cmd_vel(self, msg):
        print(f"[cmd_vel]       linear.x={msg.linear.x:.2f}  linear.y={msg.linear.y:.2f} linear.z={msg.linear.z:.2f} ")
        print(f"[cmd_vel]       angular.x={msg.angular.x:.2f} angular.y={msg.angular.y:.2f} angular.z={msg.angular.z:.2f}")

    def cb_waypoints(self, msg):
        print(f"[waypoints]     {msg.data}")


def main(args=None):
    rclpy.init(args=args)
    node = PulpitRecv()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()