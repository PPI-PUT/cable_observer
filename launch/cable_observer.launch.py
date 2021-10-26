from ament_index_python import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

import os


def generate_launch_description():
    cable_observer_pkg_prefix = get_package_share_directory('cable_observer')

    # cable_observer_param_file = os.path.join(
    #     cable_observer_pkg_prefix, 'config/param.yaml')

    camera_param_file = os.path.join(
        cable_observer_pkg_prefix, 'config/d435i.yaml')

    # Arguments

    # cable_observer_param = DeclareLaunchArgument(
    #     'cable_observer_param_file',
    #     default_value=cable_observer_param_file,
    #     description='Path to DLO tracking params'
    # )

    camera_param = DeclareLaunchArgument(
        'camera_param_file',
        default_value=camera_param_file,
        description='Camera params'
    )

    # Nodes

    camera = Node(
        package='realsense2_camera',
        executable='realsense2_camera_node',
        namespace='camera',
        name='camera',
        parameters=[LaunchConfiguration('camera_param_file')],
    )
    cable_observer = Node(
        namespace='camera',
        package='cable_observer',
        executable='cable_observer_node',
    )
    image_spline = Node(
        package='image_view',
        executable='image_view',
        name='image_view_spline_node',
        parameters=[{'autosize': False,
                     'width': 960,
                     'height': 540,
                     'image_transport': 'raw',
                     'window_name': 'Spline image'}],
        remappings=[
            ('image', '/points/image')
        ]
    )
    image_raw = Node(
        package='image_view',
        executable='image_view',
        name='image_view_raw_node',
        parameters=[{'autosize': False,
                     'width': 960,
                     'height': 540,
                     'image_transport': 'raw',
                     'window_name': 'Raw image'}],
        remappings=[
            ('image', '/camera/color/image_raw')
        ]
    )

    return LaunchDescription([
        camera_param,
        camera,
        cable_observer,
        image_spline,
        image_raw
    ])
