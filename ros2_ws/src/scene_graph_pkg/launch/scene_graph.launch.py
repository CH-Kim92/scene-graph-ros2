"""
scene_graph.launch.py
─────────────────────
Launches the full 3-D scene graph pipeline:
  1. RealSense L515 ROS2 driver
  2. Scene Graph node (detection + mesh + graph)
  3. RViz2 (optional)
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.conditions import IfCondition
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():

    # ── Arguments ──────────────────────────────────────────────────────────
    args = [
        DeclareLaunchArgument('yolo_model',         default_value='yolov8m.pt'),
        DeclareLaunchArgument('confidence',          default_value='0.45'),
        DeclareLaunchArgument('near_threshold',      default_value='0.60'),
        DeclareLaunchArgument('voxel_length',        default_value='0.02'),
        DeclareLaunchArgument('ws_port',             default_value='8765'),
        DeclareLaunchArgument('broadcast_hz',        default_value='5.0'),
        DeclareLaunchArgument('mesh_every_n_frames', default_value='30'),
        DeclareLaunchArgument('launch_rviz',         default_value='false'),
        DeclareLaunchArgument('serial_no',           default_value='',
                              description='RealSense serial number (blank = first found)'),
    ]

    # ── RealSense L515 driver ───────────────────────────────────────────────
    realsense_node = Node(
        package='realsense2_camera',
        executable='realsense2_camera_node',
        name='camera',
        namespace='camera',
        parameters=[{
            # L515 streams
            'enable_color':             True,
            'enable_depth':             True,
            'enable_infra1':            False,
            'enable_infra2':            False,
            'enable_gyro':              False,
            'enable_accel':             False,

            # Resolutions
            'color_width':              960,
            'color_height':             540,
            'color_fps':                30,
            'depth_width':              640,
            'depth_height':             480,
            'depth_fps':                30,

            # Alignment: align depth to colour frame
            'align_depth.enable':       True,
            'pointcloud.enable':        False,  # we build our own

            # L515-specific: use confidence filter
            'filters':                  'decimation,spatial,temporal',
            'decimation_filter.magnitude': 2,
            'spatial_filter.magnitude':  2,
            'temporal_filter.smooth_alpha': 0.4,
            'temporal_filter.smooth_delta': 20,

            'serial_no': LaunchConfiguration('serial_no'),
        }],
        remappings=[
            ('color/image_raw',        '/camera/color/image_raw'),
            ('color/camera_info',      '/camera/color/camera_info'),
            ('depth/image_rect_raw',   '/camera/depth/image_rect_raw'),
        ],
        output='screen',
    )

    # ── Scene Graph node ────────────────────────────────────────────────────
    scene_graph_node = Node(
        package='scene_graph_pkg',
        executable='scene_graph_node',
        name='scene_graph_node',
        parameters=[{
            'yolo_model':         LaunchConfiguration('yolo_model'),
            'confidence':         LaunchConfiguration('confidence'),
            'near_threshold':     LaunchConfiguration('near_threshold'),
            'voxel_length':       LaunchConfiguration('voxel_length'),
            'ws_port':            LaunchConfiguration('ws_port'),
            'broadcast_hz':       LaunchConfiguration('broadcast_hz'),
            'mesh_every_n_frames': LaunchConfiguration('mesh_every_n_frames'),
        }],
        output='screen',
    )

    # ── RViz2 (optional) ────────────────────────────────────────────────────
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', PathJoinSubstitution([
            FindPackageShare('scene_graph_pkg'), 'config', 'scene_graph.rviz'
        ])],
        condition=IfCondition(LaunchConfiguration('launch_rviz')),
        output='screen',
    )

    return LaunchDescription(args + [
        realsense_node,
        scene_graph_node,
        rviz_node,
    ])
