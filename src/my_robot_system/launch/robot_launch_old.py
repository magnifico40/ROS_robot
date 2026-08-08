"""
    import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
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
            ('nmea_sentence', '/nmea'),  ]
        ),

        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='imu_static_tf',
            arguments=['0', '0', '0.10', '1.5708', '0', '0', 'base_link', 'imu_link']
        ),

        # 2. Transformacja dla GPS: Y = 0.10 m (lewo), Z = 0.133 m (góra)
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='gps_static_tf',
            arguments=['0', '0.10', '0.133', '0', '0', '0', 'base_link', 'gps_link']
        ),
        # 3. Węzeł przeliczający GPS na metry
        Node(
            package='robot_localization',
            executable='navsat_transform_node',
            name='navsat_transform_node',
            output='screen',
            parameters=[{
                'magnetic_declination_radians': 0.0,
                'yaw_offset': 0.0,
                'zero_altitude': True,
                'use_odometry_yaw': True, #True earlier
                'publish_filtered_gps': True
            }],
            remappings=[
               ('imu/data', '/imu/data_raw'),
                ('gps/fix', '/fix'),#change
                ('odometry/filtered', '/odometry/filtered')
            ]
        ),

        # 4. Główny filtr Kalmana EKF
        Node(
            package='robot_localization',
            executable='ekf_node',
            name='ekf_filter_node',
            output='screen',
            parameters=[config_dir],
        )
    ])
    """