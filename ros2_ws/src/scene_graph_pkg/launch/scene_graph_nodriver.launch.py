from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    args = [
        DeclareLaunchArgument('yolo_model',          default_value='yolov8m.pt'),
        DeclareLaunchArgument('confidence',           default_value='0.45'),
        DeclareLaunchArgument('near_threshold',       default_value='0.60'),
        DeclareLaunchArgument('voxel_length',         default_value='0.02'),
        DeclareLaunchArgument('ws_port',              default_value='8765'),
        DeclareLaunchArgument('broadcast_hz',         default_value='5.0'),
        DeclareLaunchArgument('mesh_every_n_frames',  default_value='30'),
    ]

    scene_graph_node = Node(
        package='scene_graph_pkg',
        executable='scene_graph_node',
        name='scene_graph_node',
        parameters=[{
            'yolo_model':          LaunchConfiguration('yolo_model'),
            'confidence':          LaunchConfiguration('confidence'),
            'near_threshold':      LaunchConfiguration('near_threshold'),
            'voxel_length':        LaunchConfiguration('voxel_length'),
            'ws_port':             LaunchConfiguration('ws_port'),
            'broadcast_hz':        LaunchConfiguration('broadcast_hz'),
            'mesh_every_n_frames': LaunchConfiguration('mesh_every_n_frames'),
        }],
        output='screen',
    )

    return LaunchDescription(args + [scene_graph_node])
