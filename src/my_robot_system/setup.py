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
        ('share/' + package_name + '/config', ['config/general_configuration.yaml']),
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
            'ML_fusion = my_robot_system.ML_fusion:main',
            'lidar_camera_projection = ros2_camera_lidar_fusion.lidar_camera_projection:main',
            'serial_node = my_robot_system.serial_node:main',
            'obst_avoid = my_robot_system.obst_avoid:main',
            'ml_node = my_robot_system.ml_node:main',
            'ml_node_new = my_robot_system.ml_node_new:main',
        ],
    },
)
