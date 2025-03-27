from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.parameter_descriptions import ParameterFile
import os
from launch.actions import TimerAction


def launch_setup(context):
    cloud2 = Node(
        package="zed_body_publisher",
        executable="zed_body_publisher",
        output="both",
        parameters=[
            {
                "point_cloud_topic_name": "zed_point_cloud_2",
                "image_topic_name": "camera_raw_2",
            }
        ],
    )
    cloud1 = TimerAction(period=5.0,
            actions=[Node(
        package="zed_body_publisher",
        executable="zed_body_publisher",
        output="both",
        parameters=[
            {
                "point_cloud_topic_name": "zed_point_cloud_1",
                "image_topic_name": "camera_raw_1",
            }
        ],
    )
    ])
        
    return [cloud2, cloud1]


def generate_launch_description():
    ld = LaunchDescription()
    opfunc = OpaqueFunction(function=launch_setup)
    ld.add_action(opfunc)
    return ld
