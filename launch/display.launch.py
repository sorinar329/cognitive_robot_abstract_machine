"""RViz view of the nacelle with joint sliders.

ros2 launch windturbine_model display.launch.py [urdf:=/abs/path/to/scenario.urdf]
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def nodes(context):
    share = get_package_share_directory("windturbine_model")
    urdf = LaunchConfiguration("urdf").perform(context) or os.path.join(share, "urdf", "windturbine.urdf")
    with open(urdf) as f:
        description = f.read()
    return [
        Node(package="robot_state_publisher", executable="robot_state_publisher",
             parameters=[{"robot_description": description}]),
        Node(package="joint_state_publisher_gui", executable="joint_state_publisher_gui"),
        Node(package="rviz2", executable="rviz2", arguments=["-d", os.path.join(share, "rviz", "nacelle.rviz")]),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("urdf", default_value="", description="URDF to show (default: healthy)"),
        OpaqueFunction(function=nodes),
    ])
