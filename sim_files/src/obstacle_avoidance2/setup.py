from setuptools import setup
import os
from glob import glob

package_name = 'obstacle_avoidance2'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'urdf'), glob('urdf/*.xacro')),
        (os.path.join('share', package_name, 'meshes'), glob('meshes/*')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='bukaj',
    maintainer_email='bukaj@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        	'avoider = obstacle_avoidance2.obst_avoid:main',
        	'zigzag = obstacle_avoidance2.boust:main',
        	'pursuit = obstacle_avoidance2.pursuit_node:main',
        	'planner = obstacle_avoidance2.planner_node:main'
        ],
    },
)
