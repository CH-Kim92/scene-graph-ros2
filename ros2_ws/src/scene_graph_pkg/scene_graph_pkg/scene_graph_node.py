import sys, os
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, qos_profile_sensor_data
from sensor_msgs.msg import Image, CameraInfo
from visualization_msgs.msg import MarkerArray, Marker
from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import Point
import cv2
import numpy as np
import threading, json, asyncio, time
import websockets

try:
    from cv_bridge import CvBridge
    CV_BRIDGE_OK = True
except Exception:
    CV_BRIDGE_OK = False

from scene_graph_pkg.object_detector import ObjectDetector3D
from scene_graph_pkg.scene_graph_builder import SceneGraph3D
from scene_graph_pkg.mesh_reconstructor import MeshReconstructor


class WebSocketServer:
    def __init__(self, host="0.0.0.0", port=8765):
        self.host = host; self.port = port
        self._clients = set(); self._loop = None

    def start_in_thread(self):
        def _run():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._loop.run_until_complete(self._serve())
        threading.Thread(target=_run, daemon=True).start()
        print(f"[WebSocket] ws://{self.host}:{self.port}")

    async def _serve(self):
        async with websockets.serve(self._handler, self.host, self.port,
                            reuse_address=True):
            await asyncio.Future()

    async def _handler(self, ws):
        self._clients.add(ws)
        try: await ws.wait_closed()
        finally: self._clients.discard(ws)

    def broadcast(self, msg):
        if not self._clients or not self._loop: return
        asyncio.run_coroutine_threadsafe(
            self._bcast(json.dumps(msg)), self._loop)

    async def _bcast(self, data):
        dead = set()
        for ws in self._clients:
            try: await ws.send(data)
            except: dead.add(ws)
        self._clients -= dead


class SceneGraphNode(Node):
    def __init__(self):
        super().__init__('scene_graph_node')
        self.declare_parameter('yolo_model',         'yolov8m.pt')
        self.declare_parameter('confidence',          0.45)
        self.declare_parameter('near_threshold',      0.6)
        self.declare_parameter('voxel_length',        0.02)
        self.declare_parameter('ws_port',             8765)
        self.declare_parameter('broadcast_hz',        5.0)
        self.declare_parameter('mesh_every_n_frames', 30)

        yolo_model   = self.get_parameter('yolo_model').value
        confidence   = self.get_parameter('confidence').value
        near_thresh  = self.get_parameter('near_threshold').value
        voxel_length = self.get_parameter('voxel_length').value
        ws_port      = self.get_parameter('ws_port').value
        broadcast_hz = self.get_parameter('broadcast_hz').value
        self.mesh_every = self.get_parameter('mesh_every_n_frames').value

        self.bridge    = CvBridge() if CV_BRIDGE_OK else None
        self.detector  = ObjectDetector3D(model_path=yolo_model,
                                          confidence_threshold=confidence)
        self.sg        = SceneGraph3D(near_threshold=near_thresh)
        self.mesher    = MeshReconstructor(voxel_length=voxel_length)
        self.ws_server = WebSocketServer(port=ws_port)
        self.ws_server.start_in_thread()

        self.intrinsics = None
        self._rgb_image = None
        self._depth_image = None
        self._frame_count = 0
        self._latest_mesh_dict = {}
        self._data_lock = threading.Lock()

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST, depth=1)

        self.create_subscription(CameraInfo, '/camera/color/camera_info',
                                 self._cb_camera_info, qos_profile_sensor_data)
        self.create_subscription(Image, '/camera/color/image_raw',
                                 self._cb_rgb, sensor_qos)
        self.create_subscription(Image, '/camera/depth/image_rect_raw',
                                 self._cb_depth, sensor_qos)

        self.pub_annot   = self.create_publisher(Image, '/scene_graph/annotated_image', 1)
        self.pub_markers = self.create_publisher(MarkerArray, '/scene_graph/markers', 1)
        self.create_timer(1.0 / broadcast_hz, self._process_and_broadcast)
        self.get_logger().info("SceneGraphNode ready. Waiting for camera...")

    def _cb_camera_info(self, msg):
        if self.intrinsics is not None: return
        self.intrinsics = {
            'fx': msg.k[0], 'fy': msg.k[4],
            'cx': msg.k[2], 'cy': msg.k[5],
            'width': msg.width, 'height': msg.height,
        }
        self.get_logger().info(
            f"Camera intrinsics received: fx={self.intrinsics['fx']:.1f} "
            f"fy={self.intrinsics['fy']:.1f}")

    def _cb_rgb(self, msg):
        try:
            if self.bridge:
                rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
            else:
                rgb = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                    msg.height, msg.width, 3).copy()
            with self._data_lock:
                self._rgb_image = rgb
        except Exception as e:
            self.get_logger().error(f"RGB error: {e}")

    def _cb_depth(self, msg):
        try:
            if self.bridge:
                depth_raw = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            else:
                dtype = np.uint16 if '16' in msg.encoding else np.float32
                depth_raw = np.frombuffer(msg.data, dtype=dtype).reshape(
                    msg.height, msg.width).copy()
            depth_m = (depth_raw.astype(np.float32) / 1000.0
                       if depth_raw.dtype == np.uint16 else depth_raw.astype(np.float32))
            with self._data_lock:
                self._depth_image = depth_m
        except Exception as e:
            self.get_logger().error(f"Depth error: {e}")

    def _extract_segmented_pointcloud(self, rgb, depth, detections):
        """Extract per-object colored point clouds using detection masks."""
        fx = self.intrinsics['fx']
        fy = self.intrinsics['fy']
        cx = self.intrinsics['cx']
        cy = self.intrinsics['cy']

        h, w = depth.shape
        # Subsample for speed - every 4th pixel
        step = 4
        u = np.arange(0, w, step)
        v = np.arange(0, h, step)
        uu, vv = np.meshgrid(u, v)

        d = depth[vv, uu]
        valid = (d > 0.1) & (d < 5.0)

        x = (uu[valid] - cx) * d[valid] / fx
        y = (vv[valid] - cy) * d[valid] / fy
        z = d[valid]

        # colors = rgb[vv[valid], uu[valid]] / 255.0
        colors = rgb[vv[valid], uu[valid]].astype(np.float32)

        # Label each point by which detection it belongs to
        labels = np.full(x.shape, -1, dtype=int)
        for i, det in enumerate(detections):
            x1, y1, x2, y2 = det.bbox_2d
            # Scale bbox to subsampled coords
            in_box = (
                (uu[valid] >= x1) & (uu[valid] <= x2) &
                (vv[valid] >= y1) & (vv[valid] <= y2)
            )
            labels[in_box] = i
            # Color points by detection color
            r, g, b = det.color
            # colors[in_box] = [r/255., g/255., b/255.]
            colors[in_box] = [r, g, b]

        points = np.stack([x, y, z], axis=1)
        colors_255 = (colors * 255).astype(np.uint8)

        return {
            'points': points.flatten().tolist(),
            'colors': colors_255.flatten().tolist(),
            'count': len(points),
        }
    
    def _extract_segmented_mesh(self, rgb, depth, detections):
        """Build a simple mesh per detected object using its depth points."""
        if not detections:
            return {'vertices': [], 'triangles': [], 'colors': [],
                    'normals': [], 'vertex_count': 0, 'triangle_count': 0}

        fx = self.intrinsics['fx']; fy = self.intrinsics['fy']
        cx = self.intrinsics['cx']; cy = self.intrinsics['cy']
        h, w = depth.shape

        all_verts = []; all_tris = []; all_cols = []
        vert_offset = 0

        for det in detections:
            x1, y1, x2, y2 = det.bbox_2d
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(w-1, x2); y2 = min(h-1, y2)

            # Sample points inside bbox
            step = 3
            u = np.arange(x1, x2, step)
            v = np.arange(y1, y2, step)
            if len(u) < 2 or len(v) < 2:
                continue
            uu, vv = np.meshgrid(u, v)
            d = depth[vv.astype(int), uu.astype(int)]
            valid = (d > 0.1) & (d < 5.0)
            if valid.sum() < 4:
                continue

            # Unproject to 3D
            xs = (uu[valid] - cx) * d[valid] / fx
            ys = (vv[valid] - cy) * d[valid] / fy
            zs = d[valid]

            # Build grid mesh from valid points
            rows, cols = np.where(valid)
            nv = len(xs)
            verts = np.stack([xs, ys, zs], axis=1)

            # Color all vertices with detection color
            r, g, b = det.color
            cols_arr = np.tile([r, g, b], (nv, 1))

            # Simple triangle mesh from grid
            tris = []
            valid_idx = np.full(valid.shape, -1, dtype=int)
            valid_idx[valid] = np.arange(nv)

            grid_h, grid_w = valid.shape
            for row in range(grid_h - 1):
                for col in range(grid_w - 1):
                    i00 = valid_idx[row, col]
                    i10 = valid_idx[row+1, col]
                    i01 = valid_idx[row, col+1]
                    i11 = valid_idx[row+1, col+1]
                    if i00 >= 0 and i10 >= 0 and i01 >= 0:
                        tris.append([i00 + vert_offset,
                                     i10 + vert_offset,
                                     i01 + vert_offset])
                    if i11 >= 0 and i10 >= 0 and i01 >= 0:
                        tris.append([i11 + vert_offset,
                                     i10 + vert_offset,
                                     i01 + vert_offset])

            all_verts.append(verts)
            all_cols.append(cols_arr)
            all_tris.extend(tris)
            vert_offset += nv

        if not all_verts:
            return {'vertices': [], 'triangles': [], 'colors': [],
                    'normals': [], 'vertex_count': 0, 'triangle_count': 0}

        verts_np = np.concatenate(all_verts, axis=0)
        cols_np  = np.concatenate(all_cols,  axis=0)
        tris_np  = np.array(all_tris, dtype=np.int32)

        return {
            'vertices':       verts_np.flatten().tolist(),
            'triangles':      tris_np.flatten().tolist(),
            'colors':         cols_np.flatten().tolist(),
            'normals':        [],
            'vertex_count':   len(verts_np),
            'triangle_count': len(tris_np),
        }
    
    def _process_and_broadcast(self):
        if self.intrinsics is None: return
        with self._data_lock:
            rgb = self._rgb_image; depth = self._depth_image
        if rgb is None or depth is None: return

        self._frame_count += 1
        t0 = time.perf_counter()

        detections = self.detector.detect(rgb, depth, self.intrinsics)
        graph      = self.sg.update(detections)
        graph_dict = self.sg.to_dict()

        # self.mesher.integrate_frame(rgb, depth, self.intrinsics)
        # mesh_dict = {}
        # if self._frame_count % self.mesh_every == 0:
        #     mesh = self.mesher.extract_mesh()
        #     if mesh: mesh_dict = self.mesher.mesh_to_dict(mesh)

        # pcd      = self.mesher.extract_pointcloud()
        # pcd_dict = self.mesher.pointcloud_to_dict(pcd)
        # Per-object segmented point clouds (fast, no TSDF)
        
        pcd_dict  = self._extract_segmented_pointcloud(rgb, depth, detections)
        mesh_dict = self._extract_segmented_mesh(rgb, depth, detections)
        elapsed  = time.perf_counter() - t0

        self._publish_image(self._draw_annotations(rgb.copy(), detections))
        self._publish_markers(graph)

        if mesh_dict: self._latest_mesh_dict = mesh_dict
        self.ws_server.broadcast({
            "type": "scene_update",
            "graph": graph_dict, "pointcloud": pcd_dict,
            "mesh": mesh_dict or self._latest_mesh_dict,
            "stats": {"frame": self._frame_count, "detections": len(detections),
                      "nodes": graph.number_of_nodes(),
                      "edges": graph.number_of_edges(),
                      "process_ms": round(elapsed*1000, 1)}
        })
        self.get_logger().info(
            f"frame={self._frame_count} dets={len(detections)} "
            f"nodes={graph.number_of_nodes()} edges={graph.number_of_edges()} "
            f"{elapsed*1000:.0f}ms")

    @staticmethod
    def _draw_annotations(rgb, detections):
        img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        for det in detections:
            x1,y1,x2,y2 = det.bbox_2d
            b,g,r = det.color[2], det.color[1], det.color[0]
            cv2.rectangle(img, (x1,y1),(x2,y2),(b,g,r),2)
            cv2.rectangle(img,(x1,y1-42),(x1+220,y1),(b,g,r),-1)
            cv2.putText(img,f"{det.label} {det.confidence:.2f}",
                (x1+4,y1-26),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),1)
            p=det.position_3d
            cv2.putText(img,f"({p[0]:.2f},{p[1]:.2f},{p[2]:.2f})m",
                (x1+4,y1-8),cv2.FONT_HERSHEY_SIMPLEX,0.45,(220,220,220),1)
        return img

    def _publish_image(self, bgr):
        if self.bridge:
            self.pub_annot.publish(self.bridge.cv2_to_imgmsg(bgr, encoding='bgr8'))
        else:
            from sensor_msgs.msg import Image as Img
            m=Img(); m.height=bgr.shape[0]; m.width=bgr.shape[1]
            m.encoding='bgr8'; m.step=bgr.shape[1]*3; m.data=bgr.tobytes()
            self.pub_annot.publish(m)

    def _publish_markers(self, graph):
        arr=MarkerArray(); now=self.get_clock().now().to_msg(); mid=0
        for nid,data in graph.nodes(data=True):
            pos=data.get('position',[0,0,0]); col=data.get('color',[160,160,160])
            m=Marker(); m.header.frame_id='camera_color_optical_frame'
            m.header.stamp=now; m.ns='nodes'; m.id=mid; mid+=1
            m.type=Marker.SPHERE; m.action=Marker.ADD
            m.pose.position.x=float(pos[0]); m.pose.position.y=float(pos[1])
            m.pose.position.z=float(pos[2]); m.pose.orientation.w=1.0
            m.scale.x=m.scale.y=m.scale.z=0.08
            m.color=ColorRGBA(r=col[0]/255.,g=col[1]/255.,b=col[2]/255.,a=0.9)
            m.lifetime.sec=1; arr.markers.append(m)
            t=Marker(); t.header=m.header; t.ns='labels'; t.id=mid; mid+=1
            t.type=Marker.TEXT_VIEW_FACING; t.action=Marker.ADD
            t.pose=m.pose; t.pose.position.y-=0.07; t.scale.z=0.06
            t.color=ColorRGBA(r=1.,g=1.,b=1.,a=1.); t.text=data.get('label','')
            t.lifetime.sec=1; arr.markers.append(t)
        for src,dst,data in graph.edges(data=True):
            if not(graph.has_node(src) and graph.has_node(dst)): continue
            ps=graph.nodes[src]['position']; pd=graph.nodes[dst]['position']
            l=Marker(); l.header.frame_id='camera_color_optical_frame'
            l.header.stamp=now; l.ns='edges'; l.id=mid; mid+=1
            l.type=Marker.LINE_STRIP; l.action=Marker.ADD; l.scale.x=0.01
            l.color=ColorRGBA(r=0.4,g=1.,b=0.9,a=0.7)
            l.points=[Point(x=float(ps[0]),y=float(ps[1]),z=float(ps[2])),
                      Point(x=float(pd[0]),y=float(pd[1]),z=float(pd[2]))]
            l.lifetime.sec=1; arr.markers.append(l)
        self.pub_markers.publish(arr)


def main(args=None):
    rclpy.init(args=args)
    node = SceneGraphNode()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally: node.destroy_node(); rclpy.shutdown()

if __name__ == '__main__':
    main()