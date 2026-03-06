"""
realsense_node.py
─────────────────
Pure Python ROS2 node that captures from L515 using pyrealsense2
directly — no dependency on realsense2_camera ROS wrapper.
Publishes identical topics to what realsense2_camera_node would publish.
"""
import sys
import os
# Force correct librealsense library (2.54 from /usr/local supports L515)
os.environ['LD_LIBRARY_PATH'] = '/usr/local/lib:' + os.environ.get('LD_LIBRARY_PATH', '')
sys.path.insert(0, '/usr/local/lib/python3.12/site-packages')

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, CameraInfo
from builtin_interfaces.msg import Time
import numpy as np

import pyrealsense2 as rs


class RealSenseNode(Node):

    def __init__(self):
        super().__init__('realsense_node')

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # Publishers — same topic names as realsense2_camera
        self.pub_color = self.create_publisher(
            Image, '/camera/color/image_raw', sensor_qos)
        self.pub_depth = self.create_publisher(
            Image, '/camera/depth/image_rect_raw', sensor_qos)
        self.pub_info  = self.create_publisher(
            CameraInfo, '/camera/color/camera_info', 10)

        # RealSense pipeline
        self.pipeline = rs.pipeline()
        config = rs.config()

        # L515 streams
        config.enable_stream(rs.stream.color, 960, 540,  rs.format.rgb8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

        self.get_logger().info('Starting RealSense pipeline...')
        profile = self.pipeline.start(config)

        # Align depth to color
        self.align = rs.align(rs.stream.color)
        # Warmup — discard first 30 frames
        self.get_logger().info('Warming up camera...')
        for _ in range(30):
            self.pipeline.wait_for_frames(timeout_ms=5000)
        self.get_logger().info('Camera warmed up!')
        
        # Get camera intrinsics
        color_profile = profile.get_stream(rs.stream.color)
        intr = color_profile.as_video_stream_profile().get_intrinsics()
        self.intrinsics = intr
        self.get_logger().info(
            f'Camera ready: {intr.width}x{intr.height} '
            f'fx={intr.fx:.1f} fy={intr.fy:.1f}')

        # Timer — 30Hz
        self.create_timer(1.0 / 30.0, self._capture)
        self._frame_id = 'camera_color_optical_frame'

    def _capture(self):
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=5000)
        except RuntimeError:
            self.get_logger().warn('Frame timeout - waiting...')
            return
        aligned = self.align.process(frames)

        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()

        if not color_frame or not depth_frame:
            return

        now = self.get_clock().now().to_msg()

        # ── Publish color ──────────────────────────────────────────────
        color_np = np.asanyarray(color_frame.get_data())
        msg_color = Image()
        msg_color.header.stamp    = now
        msg_color.header.frame_id = self._frame_id
        msg_color.height          = color_np.shape[0]
        msg_color.width           = color_np.shape[1]
        msg_color.encoding        = 'rgb8'
        msg_color.step            = color_np.shape[1] * 3
        msg_color.data            = color_np.tobytes()
        self.pub_color.publish(msg_color)

        # ── Publish depth ──────────────────────────────────────────────
        depth_np = np.asanyarray(depth_frame.get_data())  # uint16 mm
        msg_depth = Image()
        msg_depth.header.stamp    = now
        msg_depth.header.frame_id = self._frame_id
        msg_depth.height          = depth_np.shape[0]
        msg_depth.width           = depth_np.shape[1]
        msg_depth.encoding        = '16UC1'
        msg_depth.step            = depth_np.shape[1] * 2
        msg_depth.data            = depth_np.tobytes()
        self.pub_depth.publish(msg_depth)

        # ── Publish camera info ────────────────────────────────────────
        intr = self.intrinsics
        info = CameraInfo()
        info.header         = msg_color.header
        info.width          = intr.width
        info.height         = intr.height
        info.distortion_model = 'plumb_bob'
        info.d = list(intr.coeffs)
        info.k = [intr.fx, 0.0, intr.ppx,
                  0.0, intr.fy, intr.ppy,
                  0.0, 0.0, 1.0]
        info.r = [1.0, 0.0, 0.0,
                  0.0, 1.0, 0.0,
                  0.0, 0.0, 1.0]
        info.p = [intr.fx, 0.0, intr.ppx, 0.0,
                  0.0, intr.fy, intr.ppy, 0.0,
                  0.0, 0.0, 1.0, 0.0]
        self.pub_info.publish(info)

    def destroy_node(self):
        self.pipeline.stop()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = RealSenseNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
