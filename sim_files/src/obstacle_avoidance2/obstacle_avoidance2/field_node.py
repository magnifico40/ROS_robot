import fields2cover as f2c
import numpy as np
from fields2cover import Robot
from pyproj import Proj
from matplotlib import pyplot as plt

my_proj = Proj(proj='utm',zone='34',ellps='WGS84',preserve_units=False) #standardowe wartosci

base_lon,base_lat = 21.012200, 52.229700
base_x,base_y = my_proj(base_lon,base_lat)  #przeliczenie na utm w metrach

gps_borders = [ #wspolrzedne z gps w stopniach
    (21.012200, 52.229700),
    (21.012300, 52.229700),
    (21.012300, 52.229800),
    (21.012200, 52.229800)
]

plot_fence_x = []
plot_fence_y = []

ring = f2c.LinearRing()
for lon,lat in gps_borders:
    global_x,global_y = my_proj(lon,lat) #przeliczenie na utm w metrach
    relative_x = global_x - base_x
    relative_y = global_y - base_y
    ring.addPoint(relative_x,relative_y)

    plot_fence_x.append(relative_x)
    plot_fence_y.append(relative_y)
ring.addPoint(0.0,0.0)
plot_fence_x.append(0.0)
plot_fence_y.append(0.0)

field = f2c.Cells(f2c.Cell(ring))

robot_width = 0.4274
robot_length = 0.37
turning_radius = 0.2

robot = Robot(robot_width,robot_width)
robot.setMinRadius(turning_radius)

hl_gen = f2c.HG_Const_gen()
no_hl = hl_gen.generateHeadlands(field,robot.getRobotWidth()) #odstep dla zawracania

n_swath = f2c.OBJ_NSwath()
bf_sw_gen = f2c.SG_BruteForce()
best_swaths = bf_sw_gen.generateBestSwaths(n_swath, robot.getRobotWidth(), no_hl.getGeometry(0)) #sciezki trasy

boustrophedon_sorter = f2c.RP_Boustrophedon()
swaths = boustrophedon_sorter.genSortedSwaths(best_swaths) #ustalamy kolejnosc sciezek

path_planner = f2c.PP_PathPlanning()
dubins = f2c.PP_DubinsCurves()
path = path_planner.searchBestPath(robot, swaths, dubins)


x_path = []
y_path = []
for state in path.states:
    x_path.append(state.point.getX())
    y_path.append(state.point.getY())




# Konfiguracja wykresu
plt.figure(figsize=(6, 4))
plt.plot(plot_fence_x, plot_fence_y, 'g-', linewidth=3, label='Granice Trawnika') # Płot
plt.plot(x_path, y_path, 'b--', linewidth=2, label='Ścieżka F2C (Zygzak + Dubins)') # Trasa robota

plt.title("Plan Koszenia: Fields2Cover")
plt.xlabel("X [metry]")
plt.ylabel("Y [metry]")
plt.legend()
plt.grid(True)
plt.axis('equal') # Ważne, żeby zachować proporcje!
plt.show()