import fields2cover as f2c
from pyproj import Proj


my_proj = Proj(proj='utm', zone='34', ellps='WGS84', preserve_units=False)  # standardowe wartosci

base_lon, base_lat = 18.60814, 54.36998
base_x, base_y = my_proj(base_lon, base_lat)  # przeliczenie na utm w metrach

gps_borders = [  # wspolrzedne z gps w stopniach
    (18.60814, 54.36998),
    (18.60780900455834, 54.36996631488758),
    (18.60786440691856, 54.37001024415773),
    #(18.60752204874387, 54.37011210197716),
    (18.608134315852883,54.37003458796302),
    (18.60858889932135, 54.37011210197716)

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

decomp = f2c.DECOMP_TrapezoidalDecomp()  #dekompozycja
path_planner = f2c.PP_PathPlanning()
reeds_shepp = f2c.PP_ReedsSheppCurves()

const_hl = f2c.HG_Const_gen()   #generator uwroci
bf = f2c.SG_BruteForce()    #sprawdzamy kazdy kat
n_swath = f2c.OBJ_NSwathModified()  #wybieramy kat dajacy najmniejsza liczbie rzedow
boustrophedon_sorter = f2c.RP_Boustrophedon() #ustawiamy kolejnosc boustrophedon

mid_hl = const_hl.generateHeadlands(field, 0.4*robot_width) #tworzymy pierwszy headland
decomp_mid_hl = decomp.decompose(mid_hl)    #rozbijamy na mniejsze komorki
no_hl = const_hl.generateHeadlands(decomp_mid_hl, 0.4*robot_width)  #drugi headland dla mniejszych komorek

best_swaths = bf.generateBestSwaths(n_swath, 0.8*robot_width,no_hl)  # tworzymy sciezki trasy, oddzielone o 0.8 szerokosci robota
sorted_swaths_by_cells = f2c.SwathsByCells()

for i in range(best_swaths.size()):
    # Wyciągamy ścieżki tylko dla i-tej komórki, sortujemy i wrzucamy do kontenera
    sorted_single_cell = boustrophedon_sorter.genSortedSwaths(best_swaths[i])
    sorted_swaths_by_cells.push_back(sorted_single_cell)


route_planner = f2c.RP_RoutePlannerBase()
route = route_planner.genRoute(mid_hl, sorted_swaths_by_cells)

path = path_planner.planPath(robot, route, reeds_shepp)

f2c.Visualizer.figure()
f2c.Visualizer.plot(field)
f2c.Visualizer.plot(mid_hl)
f2c.Visualizer.plot(no_hl)
f2c.Visualizer.plot(best_swaths)
f2c.Visualizer.plot(path)
f2c.Visualizer.show()

