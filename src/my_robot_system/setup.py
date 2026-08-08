from setuptools import setup
import os
from glob import glob

package_name = 'my_robot_system'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'models'), glob('models/*.engine')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='root@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'camera_node = my_robot_system.camera_node:main',
            'ml_fusion_node_new = my_robot_system.ml_fusion_node_new:main',
            'lidar_camera_projection = my_robot_system.lidar_camera_projection:main',
            'serial_node = my_robot_system.serial_node:main',
            'obst_avoid = my_robot_system.obst_avoid:main',
            'ml_node_new = my_robot_system.ml_node_new:main',
            'pulpit_recv_node = my_robot_system.pulpit_recv_node:main',
            'simulator = my_robot_system.simulator:main',
            'planner_node = my_robot_system.planner_node:main',
            'avoider_node = my_robot_system.avoider_node:main',
            'get_intrinsic_camera_calibration = my_robot_system.get_intrinsic_camera_calibration:main',
            'ekf_deb = my_robot_system.ekf_deb:main' 
        ],
    },
)