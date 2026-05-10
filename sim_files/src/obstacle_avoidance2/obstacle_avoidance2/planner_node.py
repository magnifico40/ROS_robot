import rclpy
from fields2cover import Robot
from rclpy.node import Node
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
import fields2cover as f2c
from pyproj import Proj
import math
import os

class Planner(Node):
    def __init__(self):
        super().__init__('f2c_planner_node')

        self.path_publisher = self.create_publisher(Path, '/planned_path', 10)
        self.timer = self.create_timer(2.0, self.publish_path)

        self.generated_path_msg = self.generate_f2c_path()
        self.get_logger().info("Ścieżka gotowa do publikacji!")

    def publish_path(self):
        self.generated_path_msg.header.stamp = self.get_clock().now().to_msg()
        self.path_publisher.publish(self.generated_path_msg)


    def generate_f2c_path(self):
        my_proj = Proj(proj='utm', zone='34', ellps='WGS84', preserve_units=False)  # standardowe wartosci

        base_lon, base_lat = 21.012200, 52.229700
        base_x, base_y = my_proj(base_lon, base_lat)  # przeliczenie na utm w metrach

        gps_borders = [  # wspolrzedne z gps w stopniach
            (21.012200, 52.229700),
            (21.012300, 52.229700),
            (21.012500, 52.229900),
            (21.012300, 52.229800),
            (21.012300, 52.229800)

        ]

        ring = f2c.LinearRing()
        for lon, lat in gps_borders:
            global_x, global_y = my_proj(lon, lat)  # przeliczenie na utm w metrach
            relative_x = global_x - base_x
            relative_y = global_y - base_y
            ring.addPoint(relative_x, relative_y)

        ring.addPoint(0.0, 0.0)

        field = f2c.Cells(f2c.Cell(ring))
        robot_width = 0.4274
        turning_radius = 0.01
        robot = f2c.Robot(robot_width, robot_width)
        robot.setMinTurningRadius(turning_radius)

        decomp = f2c.DECOMP_TrapezoidalDecomp()  # dekompozycja
        path_planner = f2c.PP_PathPlanning()
        reeds_shepp = f2c.PP_ReedsSheppCurves()

        const_hl = f2c.HG_Const_gen()  # generator uwroci
        bf = f2c.SG_BruteForce()  # sprawdzamy kazdy kat
        n_swath = f2c.OBJ_NSwathModified()  # wybieramy kat dajacy najmniejsza liczbie rzedow
        boustrophedon_sorter = f2c.RP_Boustrophedon()  # ustawiamy kolejnosc boustrophedon

        mid_hl = const_hl.generateHeadlands(field, 0.4 * robot_width)  # tworzymy pierwszy headland
        decomp_mid_hl = decomp.decompose(mid_hl)  # rozbijamy na mniejsze komorki
        no_hl = const_hl.generateHeadlands(decomp_mid_hl, 0.4 * robot_width)  # drugi headland dla mniejszych komorek

        best_swaths = bf.generateBestSwaths(n_swath, 0.8 * robot_width,
                                            no_hl)  # tworzymy sciezki trasy, oddzielone o 0.8 szerokosci robota
        sorted_swaths_by_cells = f2c.SwathsByCells()

        for i in range(best_swaths.size()):
            # Wyciągamy ścieżki tylko dla i-tej komórki, sortujemy i wrzucamy do kontenera
            sorted_single_cell = boustrophedon_sorter.genSortedSwaths(best_swaths[i])
            sorted_swaths_by_cells.push_back(sorted_single_cell)

        route_planner = f2c.RP_RoutePlannerBase()
        route = route_planner.genRoute(mid_hl, sorted_swaths_by_cells)

        path = path_planner.planPath(robot, route, reeds_shepp)

        #tlumaczenie na ros2 path
        ros_path = Path()
        ros_path.header.frame_id = 'map'
        ros_path.header.stamp = self.get_clock().now().to_msg()

        for i in range(path.size()):
            state = path.getState(i)
            pose = PoseStamped()
            pose.header.frame_id = 'map'
            pose.pose.position.x = state.point.getX()
            pose.pose.position.y = state.point.getY()

            pose.pose.orientation.z = math.sin(state.angle / 2.0)
            pose.pose.orientation.w = math.cos(state.angle / 2.0)
            ros_path.poses.append(pose)

        return ros_path

def main(args=None):
    rclpy.init(args=args)
    node = Planner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        os._exit(0)
if __name__ == "__main__":
    main()
