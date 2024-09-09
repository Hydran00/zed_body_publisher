from launch import LaunchDescription
from launch_ros.actions import Node
from os.path import expanduser

import os


def generate_launch_description():

    ld = LaunchDescription()

    yolo_seg_srv = Node(
        package="yolo_seg",
        executable="yolo_seg_service.py",
    )
    zed_tracker = Node(
        package="zed_body_publisher",
        executable="zed_body_publisher",
    )
    
    ld.add_action(yolo_seg_srv)
    ld.add_action(zed_tracker)
    return ld