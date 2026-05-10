import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
import xacro

def generate_launch_description():
    pkg_name = 'obstacle_avoidance2'
    file_subpath = 'urdf/robot.xacro'

    xacro_file = os.path.join(get_package_share_directory(pkg_name), file_subpath)
    
    robot_description_raw = xacro.process_file(xacro_file).toxml()
    world_file = os.path.join(get_package_share_directory(pkg_name), 'worlds', 'my_world.world')
    node_robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': robot_description_raw,
                     'use_sim_time': True}]
    )

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([os.path.join(
            get_package_share_directory('gazebo_ros'), 'launch', 'gazebo.launch.py')]),
            launch_arguments={'world': world_file}.items()
    )

    spawn_entity = Node(package='gazebo_ros', executable='spawn_entity.py',
                        arguments=['-topic', 'robot_description',
                                   '-entity', 'my_robot',
                                   '-z', '0.1'],
                        output='screen')



    
    ekf_config_path = os.path.join(get_package_share_directory(pkg_name), 'config', 'dual_ekf.yaml')
    ekf_local = Node(
    	package='robot_localization',
    	executable='ekf_node',
    	name='ekf_local',
    	parameters=[ekf_config_path,
    		{'use_sim_time': True}],
    	remappings=[('/odometry/filtered', '/odometry/local')])
    		
    ekf_global = Node(
    	package='robot_localization',
    	executable='ekf_node',
    	name = 'ekf_global',
    	parameters = [ekf_config_path, {'use_sim_time': True}],
    	remappings=[('/odometry/filtered', '/odometry/global')])
    	
    navsat = Node(
    	package='robot_localization',
    	executable='navsat_transform_node',
    	name='navsat_transform',
    	parameters = [ekf_config_path],
    	remappings=[
    		('/imu/data','/imu'),
    		('gps/fix', '/gps/fix'),
    		('/odometry/filtered','/odometry/global')])
    	
    


    return LaunchDescription([
        gazebo,
        node_robot_state_publisher,
        spawn_entity,
        ekf_local,
        ekf_global,
        navsat
    ])
