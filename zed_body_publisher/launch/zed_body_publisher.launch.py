
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription,DeclareLaunchArgument, LogInfo

def generate_launch_description():
    ld = LaunchDescription()

    zed_node = Node(
            package="zed_body_publisher",
            executable="zed_body_publisher",
            remappings=[("/zed/zed_node/point_cloud_rh_z_up", "/camera/camera/depth/color/points"),
                        ("/zed/zed_node/image_rect_color", "/camera/camera/color/image_raw")]
    )