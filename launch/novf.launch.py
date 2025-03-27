from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.parameter_descriptions import ParameterFile
import os
from launch.actions import TimerAction


def launch_setup(context):
    cloud_node = Node(
        package="zed_body_publisher",
        executable="zed_body_publisher",
        output="both",
        parameters=[
            {
                "point_cloud_topic_name": "zed_point_cloud_1",
                "image_topic_name": "camera_raw_1",
                "full_point_cloud_topic_name": "full_cloud",
                "resolution": "FHD", # FHD or 2K
            }
        ],
    )
    cloud2 = TimerAction(period=5.0,
            actions=[Node(
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
    ])
    pc_filter = Node(
                package="pc2_filter",
                executable="pc2_filter",
                parameters=[{
                    "frame_id" : "cam2",
                    "input_topic_name": "full_cloud",
                    "output_topic_name": "filtered_cloud",
                    "voxel_grid_size": 0.005,
                    "x_segment_distance_min": -100.0,
                    "x_segment_distance_max": 2.4,

                    "y_segment_distance_min": -0.4,
                    "y_segment_distance_max": 0.4,
                    "z_segment_distance_min": -2.0,
                    "z_segment_distance_max": 300.0,
                    "compression_method": "none"
                    }]
            )
    static_broadcaster = Node(
        package="smpl_tracking_node",
        executable="cam_tf_broadcaster.py",
        parameters=[{
        "aruco_to_cam2_path" : "/home/nardi/experiments/calibration_params_1/camera_2_transform.txt",
        "aruco_to_cam1_path" : "/home/nardi/experiments/calibration_params_1/camera_1_transform.txt",
        "robot_to_cam1_path" : "/home/nardi/experiments/calibration_params_1/camera_1_wrt_base_transform.txt",
        }]
    )
        


    return [cloud_node, pc_filter,cloud2, static_broadcaster]


def generate_launch_description():
    ld = LaunchDescription()
    opfunc = OpaqueFunction(function=launch_setup)
    ld.add_action(opfunc)
    return ld
