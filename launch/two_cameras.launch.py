from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from launch.actions import OpaqueFunction
from launch.actions import IncludeLaunchDescription
import os
from launch.actions import TimerAction
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition
import os
import pyzed.sl as sl

# Retrieve the serial number of the first ZED camera to be able to distinguish
zed = sl.Camera()
status = zed.open(sl.InitParameters())
serial_number = zed.get_camera_information().serial_number
zed.close()
print("Serial number of first camera:", serial_number)

launch_args = [
DeclareLaunchArgument(
    "publish_camera_tf",
    default_value="true",
    description="Whether to publish the camera tf or not",
)
]
def launch_setup(context):
    publish_camera_tf = LaunchConfiguration('publish_camera_tf').perform(context)
    if serial_number == 22183584:
        node_1_suffix = "_2"
        node_2_suffix = "_1"
    else:
        node_1_suffix = "_1"
        node_2_suffix = "_2"

    
    cloud1 = Node(
        package="zed_body_publisher",
        executable="zed_body_publisher",
        output="both",
        parameters=[
            {
                "suffix": node_1_suffix,
                "point_cloud_topic_name": "zed_point_cloud",
                "image_topic_name": "camera_raw",
                "frame_id": "cam",
                "full_point_cloud_topic_name": "full_cloud",
                "yolo_model_path": os.path.expanduser("~/SKEL_WS/ros2_ws/yolov8s-seg.onnx")

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
                "suffix": node_2_suffix,
                "point_cloud_topic_name": "zed_point_cloud",
                "image_topic_name": "camera_raw",
                "full_point_cloud_topic_name": "full_cloud",
                "frame_id": "cam",
                "yolo_model_path": os.path.expanduser("~/SKEL_WS/ros2_ws/yolov8s-seg.onnx")
                # "resolution": "2K", # FHD or 2K
            }
        ],
        )
    ])
    # ])
    pc_filter_1 = Node(
                package="pc2_filter",
                executable="pc2_filter",
                parameters=[{
                    "frame_id" : "cam_1",
                    "input_topic_name": "zed_point_cloud_1",
                    "output_topic_name": "filtered_cloud_1",
                    "voxel_grid_size": 0.002,
                    "x_segment_distance_min": -100.0,
                    "x_segment_distance_max": 100.0,# 2.4,

                    "y_segment_distance_min": -100.0,# -0.4,
                    "y_segment_distance_max": 100.0, #0.4,
                    "z_segment_distance_min": -100.0,#,
                    "z_segment_distance_max": 100.0,
                    "compression_method": "none"
                    }]
            )
    pc_filter_2 = Node(
            package="pc2_filter",
            executable="pc2_filter",
            parameters=[{
                "frame_id" : "cam_2",
                "input_topic_name": "zed_point_cloud_2",
                "output_topic_name": "filtered_cloud_2",
                "voxel_grid_size": 0.002,
                "x_segment_distance_min": -100.0,
                "x_segment_distance_max": 100.0, #2.4,

                "y_segment_distance_min": -100.0,# -0.4,
                "y_segment_distance_max": 100.0, #0.4,
                "z_segment_distance_min": -100.0,#,
                "z_segment_distance_max": 100.0,
                "compression_method": "none"
                }]
        )
    # static_broadcaster = Node(
    #     package="smpl_tracking_node",
    #     executable="cam_tf_broadcaster.py",
    #     parameters=[{
    #     "aruco_to_cam2_path" : "/home/nardi/experiments/calibration_params_1/camera_2_transform.txt",
    #     "aruco_to_cam1_path" : "/home/nardi/experiments/calibration_params_1/camera_1_transform.txt",
    #     "robot_to_cam1_path" : "/home/nardi/experiments/calibration_params_1/camera_1_wrt_base_transform.txt",
    #     }]
    # )

    cameras_tf_publisher = IncludeLaunchDescription(
        condition=IfCondition(publish_camera_tf),
        launch_description_source = get_package_share_directory("easy_handeye2") + "/launch/publish.launch.py"
    )
        


    return [cameras_tf_publisher, cloud1, cloud2, pc_filter_1, pc_filter_2]
    # return [cloud2, cloud1, pc_filter]


def generate_launch_description():
    ld = LaunchDescription(launch_args)
    opfunc = OpaqueFunction(function=launch_setup)
    ld.add_action(opfunc)
    return ld
