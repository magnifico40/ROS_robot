import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    # Pobranie parametrów konfiguracyjnych (USUNIĘTY PRZECINEK NA KOŃCU!)
    config_dir = os.path.join(
        get_package_share_directory('my_robot_system'),
        'config',
        'ekf.yaml'
    )

    return LaunchDescription([
        

        Node(
            package='nmea_navsat_driver',
            executable='nmea_topic_driver',
            name='nmea_navsat_driver',
            remappings=[
                ('nmea_sentence', '/nmea'),
            ]
        ),
        # 1. Transformacje statyczne 
        Node(
            package='tf2_ros', 
            executable='static_transform_publisher', 
            name='imu_tf', 
            arguments=['0', '0', '0.1', '1.5708', '0', '0', 'base_link', 'imu_link']
        ),
             
        Node(
            package='tf2_ros', 
            executable='static_transform_publisher', 
            name='gps_tf', 
            arguments=['0', '0.1', '0.133', '0', '0', '0', 'base_link', 'gps_link']
        ),
 
        # 2. EKF Lokalny (Odometria z kół + IMU)
        Node(
            package='robot_localization', 
            executable='ekf_node', 
            name='ekf_local',
            output='screen', 
            parameters=[config_dir],
            remappings=[
                ('/odometry/filtered', '/odometry/local'),
                ('imu', '/imu/data_raw')
            ]
        ),

        # 3. EKF Globalny (Koła + IMU + GPS)
        Node(
            package='robot_localization', 
            executable='ekf_node', 
            name='ekf_global',
            output='screen', 
            parameters=[config_dir],
            remappings=[
                ('/odometry/filtered', '/odometry/global')
            ]
        ),

        # 4. Przelicznik GPS (Konwersja na metry)
        Node(
            package='robot_localization', 
            executable='navsat_transform_node', 
            name='navsat_transform',
            output='screen', 
            parameters=[config_dir],
            remappings=[
                ('/imu/data', '/imu/data_raw'),
                ('/gps/fix', '/fix'), 
                ('gps/filtered', '/gps/filtered'),
                ('odometry/gps', '/odometry/gps'),
                ('odometry/filtered', '/odometry/global')
            ]
        )
    ])