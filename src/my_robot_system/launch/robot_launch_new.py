from launch import LaunchDescription
import launch_ros.actions
import os
from ament_index_python.packages import get_package_share_directory
import launch.actions

def generate_launch_description():
    # Pobranie parametrów konfiguracyjnych dla architektury Dual EKF
    config_dir = os.path.join(
            get_package_share_directory('my_robot_system'),
            'config',
            'ekf.yaml'
        )

    return LaunchDescription([
        launch.actions.DeclareLaunchArgument(
            'output_final_position',
            default_value='false'),
        launch.actions.DeclareLaunchArgument(
            'output_location',
            default_value='~/dual_ekf_navsat_example_debug.txt'),
            
        # ==============================================================
        # 1. WĘZŁY SPRZĘTOWE I TRANSFORMACJE (Przywrócone z 1. pliku)
        # ==============================================================

        # Tłumacz ramek NMEA na topic /fix (Naprawia pusty /fix)
        launch_ros.actions.Node(
            package='nmea_navsat_driver',
            executable='nmea_topic_driver',
            name='nmea_navsat_driver',
            remappings=[
                ('nmea_sentence', '/nmea'),
            ]
        ),

        # Położenie czujnika IMU (Naprawia błąd "Invalid frame ID imu_link")
        launch_ros.actions.Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='imu_static_tf',
            arguments=['0', '0', '0.10', '1.5708', '0', '0', 'base_link', 'imu_link']
        ),

        # Położenie anteny GPS względem robota
        launch_ros.actions.Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='gps_static_tf',
            arguments=['0', '0.10', '0.133', '0', '0', '0', 'base_link', 'gps_link']
        ),

        # ==============================================================
        # 2. SYSTEM LOKALIZACJI (Z 2. pliku - Dual EKF)
        # ==============================================================

        # Lokalny EKF (publikuje odom -> base_link)
        launch_ros.actions.Node(
            package='robot_localization', 
            executable='ekf_node', 
            name='ekf_filter_node_odom',
            output='screen',
            parameters=[config_dir],
            remappings=[
                ('odometry/filtered', '/odometry/local'),
                ('imu', '/imu/data_raw')
            ]           
        ),

        # Globalny EKF (publikuje map -> odom)
        launch_ros.actions.Node(
            package='robot_localization', 
            executable='ekf_node', 
            name='ekf_filter_node_map',
            output='screen',
            parameters=[config_dir],
            remappings=[
                ('odometry/filtered', '/odometry/global')
            ]
        ),           

        # Integracja GPS do globalnego EKF
        launch_ros.actions.Node(
            package='robot_localization', 
            executable='navsat_transform_node', 
            name='navsat_transform',
            output='screen',
            parameters=[config_dir],
            remappings=[
                ('imu', '/imu/data_raw'),
                ('gps/fix', '/fix'), 
                ('gps/filtered', '/gps/filtered'),
                ('odometry/gps', '/odometry/gps'),
                ('odometry/filtered', '/odometry/global')
            ]           
        )           
    ])