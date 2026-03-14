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
import base64

try:
    from cv_bridge import CvBridge
    CV_BRIDGE_OK = True
except Exception:
    CV_BRIDGE_OK = False

from scene_graph_pkg.object_detector import ObjectDetector3D
from scene_graph_pkg.scene_graph_builder import SceneGraph3D
from scene_graph_pkg.map_manager import MapManager

# Open3D disabled — using incremental map instead
class MeshReconstructor:
    """Disabled - Open3D not available"""
    def __init__(self, *args, **kwargs): pass
    def integrate(self, *args, **kwargs): pass
    def get_mesh(self, *args, **kwargs): return None
    def reset(self, *args, **kwargs): pass
    def save_mesh(self, *args, **kwargs): pass


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
                    reuse_address=True,
                    max_size=10_000_000):
            await asyncio.Future()

    async def _handler(self, ws):
        self._clients.add(ws)
        try: await ws.wait_closed()
        finally: self._clients.discard(ws)

    def broadcast(self, msg):
        if not self._clients or not self._loop: return
        def default(obj):
            if isinstance(obj, set): return list(obj)
            if isinstance(obj, np.ndarray): return obj.tolist()
            if isinstance(obj, (np.integer,)): return int(obj)
            if isinstance(obj, (np.floating,)): return float(obj)
            if obj is ...: return None
            if isinstance(obj, type(...)): return None
            return str(obj)
        asyncio.run_coroutine_threadsafe(
            self._bcast(json.dumps(msg, default=default)), self._loop)

    async def _bcast(self, data):
        dead = set()
        for ws in self._clients:
            try: await ws.send(data)
            except: dead.add(ws)
        self._clients -= dead


class SceneGraphNode(Node):
    def __init__(self):
        super().__init__('scene_graph_node')

        # ── Parameters ────────────────────────────────────────────────────────
        self.declare_parameter('yolo_model',           'yolov8n.engine')
        self.declare_parameter('confidence',            0.45)
        self.declare_parameter('near_threshold',        0.6)
        self.declare_parameter('voxel_length',          0.02)
        self.declare_parameter('ws_port',               8765)
        self.declare_parameter('broadcast_hz',          20.0)
        self.declare_parameter('mesh_every_n_frames',   30)
        # Persistent map parameters
        self.declare_parameter('confirmation_frames',   5)
        self.declare_parameter('disappear_timeout',     5.0)
        self.declare_parameter('map_save_dir',          os.path.expanduser('~/scene_graph/maps'))

        yolo_model          = self.get_parameter('yolo_model').value
        confidence          = self.get_parameter('confidence').value
        near_thresh         = self.get_parameter('near_threshold').value
        voxel_length        = self.get_parameter('voxel_length').value
        ws_port             = self.get_parameter('ws_port').value
        broadcast_hz        = self.get_parameter('broadcast_hz').value
        self.mesh_every     = self.get_parameter('mesh_every_n_frames').value
        confirmation_frames = self.get_parameter('confirmation_frames').value
        disappear_timeout   = self.get_parameter('disappear_timeout').value
        self.map_save_dir   = self.get_parameter('map_save_dir').value

        os.makedirs(self.map_save_dir, exist_ok=True)

        # ── Components ────────────────────────────────────────────────────────
        self.bridge    = CvBridge() if CV_BRIDGE_OK else None
        self.detector  = ObjectDetector3D(model_path=yolo_model,
                                          confidence_threshold=confidence)

        # Per-frame scene graph (original — for fast edge computation)
        self.sg = SceneGraph3D(near_threshold=near_thresh)

        # Persistent incremental map (new)
        self.map = MapManager(
            confirmation_frames=confirmation_frames,
            disappear_timeout=disappear_timeout,
            near_threshold=near_thresh,
            event_log_path=os.path.join(self.map_save_dir, 'events.json'),
        )

        self.mesher    = MeshReconstructor(voxel_length=voxel_length)
        self.ws_server = WebSocketServer(port=ws_port)
        self.ws_server.start_in_thread()

        # ── State ─────────────────────────────────────────────────────────────
        self.intrinsics        = None
        self._rgb_image        = None
        self._depth_image      = None
        self._frame_count      = 0
        self._latest_mesh_dict = {}
        self._data_lock        = threading.Lock()

        # ── ROS subscriptions / publishers ────────────────────────────────────
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

        # Auto-save map every 60 seconds
        self.create_timer(60.0, self._auto_save_map)

        self.get_logger().info(
            f"SceneGraphNode ready (persistent map enabled). "
            f"Confirmation={confirmation_frames} frames, "
            f"Timeout={disappear_timeout}s. "
            f"Waiting for camera...")

    # ── Camera callbacks ──────────────────────────────────────────────────────

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

    # ── Main processing loop ──────────────────────────────────────────────────

    def _process_and_broadcast(self):
        if self.intrinsics is None: return
        with self._data_lock:
            rgb   = self._rgb_image
            depth = self._depth_image
        if rgb is None or depth is None: return

        self._frame_count += 1
        t0        = time.perf_counter()
        timestamp = time.time()

        # Detect objects this frame
        detections = self.detector.detect(rgb, depth, self.intrinsics)

        # Update persistent map (incremental — objects confirmed over time)
        self.map.update(detections, timestamp)
        map_dict = self.map.to_dict()

        # Also update per-frame graph for comparison / fast edges
        self.sg.update(detections)

        # Point cloud and mesh
        pcd_dict  = self._extract_segmented_pointcloud(rgb, depth, detections)
        mesh_dict = self._extract_segmented_mesh(rgb, depth, detections)

        # Annotated image
        annotated = self._draw_annotations(rgb.copy(), detections)
        self._publish_image(annotated)
        self._publish_markers(self.map.graph)  # publish persistent map markers

        # Encode image to JPEG
        try:
            # ok, jpeg = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
            small = cv2.resize(annotated, (640, 360))
            ok, jpeg = cv2.imencode('.jpg', small, [cv2.IMWRITE_JPEG_QUALITY, 60])
            img_b64  = base64.b64encode(jpeg.tobytes()).decode('utf-8') if ok else ''
        except Exception:
            img_b64 = ''

        if mesh_dict:
            self._latest_mesh_dict = mesh_dict

        elapsed = time.perf_counter() - t0

        # Broadcast to browser — includes BOTH per-frame graph and persistent map
        self.ws_server.broadcast({
            "type":       "scene_update",
            "graph":      map_dict,          # persistent map (main graph)
            "frame_graph": self.sg.to_dict(), # per-frame graph (optional)
            "pointcloud": pcd_dict,
            "mesh":       mesh_dict or self._latest_mesh_dict,
            "image":      img_b64,
            "stats": {
                "frame":        self._frame_count,
                "detections":   len(detections),
                "nodes":        self.map.graph.number_of_nodes(),
                "edges":        self.map.graph.number_of_edges(),
                "candidates":   len(self.map._candidates),
                "events":       len(self.map._events),
                "process_ms":   round(elapsed * 1000, 1),
            },
        })

        self.get_logger().info(
            f"frame={self._frame_count} "
            f"dets={len(detections)} "
            f"map_nodes={self.map.graph.number_of_nodes()} "
            f"map_edges={self.map.graph.number_of_edges()} "
            f"candidates={len(self.map._candidates)} "
            f"events={len(self.map._events)} "
            f"{elapsed*1000:.0f}ms")

    # ── Agent API (called externally or via ROS service) ──────────────────────

    def add_agent(self, agent_id: str, agent_type: str, position: list) -> None:
        """Add a human/robot agent to the persistent map."""
        self.map.add_agent(agent_id, agent_type, position)
        self.get_logger().info(f"Agent added: {agent_id} ({agent_type})")

    def remove_agent(self, agent_id: str) -> None:
        """Remove an agent from the persistent map."""
        self.map.remove_agent(agent_id)

    # ── Auto-save ─────────────────────────────────────────────────────────────

    def _auto_save_map(self):
        """Save map and events every 60 seconds."""
        ts  = time.strftime('%Y%m%d_%H%M%S')
        self.map.save_map(os.path.join(self.map_save_dir, f'map_{ts}.json'))
        self.map.save_events()
        self.get_logger().info(
            f"Auto-saved map: {self.map.stats()}")

    # ── Point cloud / mesh extraction (unchanged from original) ──────────────

    def _extract_segmented_pointcloud(self, rgb, depth, detections):
        fx = self.intrinsics['fx']; fy = self.intrinsics['fy']
        cx = self.intrinsics['cx']; cy = self.intrinsics['cy']
        h, w = depth.shape
        step = 4
        u = np.arange(0, w, step); v = np.arange(0, h, step)
        uu, vv = np.meshgrid(u, v)
        d = depth[vv, uu]
        valid = (d > 0.1) & (d < 5.0)
        x = (uu[valid] - cx) * d[valid] / fx
        y = (vv[valid] - cy) * d[valid] / fy
        z = d[valid]
        colors = rgb[vv[valid], uu[valid]].astype(np.float32)
        labels = np.full(x.shape, -1, dtype=int)
        for i, det in enumerate(detections):
            x1, y1, x2, y2 = det.bbox_2d
            in_box = ((uu[valid] >= x1) & (uu[valid] <= x2) &
                      (vv[valid] >= y1) & (vv[valid] <= y2))
            labels[in_box] = i
            r, g, b = det.color
            colors[in_box] = [r, g, b]
        points = np.stack([x, y, z], axis=1)
        colors_255 = (colors * 255).astype(np.uint8)
        return {'points': points.flatten().tolist(),
                'colors': colors_255.flatten().tolist(),
                'count':  len(points)}

    def _extract_segmented_mesh(self, rgb, depth, detections):
        if not detections:
            return {'vertices': [], 'triangles': [], 'colors': [],
                    'normals': [], 'vertex_count': 0, 'triangle_count': 0}
        fx = self.intrinsics['fx']; fy = self.intrinsics['fy']
        cx = self.intrinsics['cx']; cy = self.intrinsics['cy']
        h, w = depth.shape
        all_verts=[]; all_tris=[]; all_cols=[]; vert_offset=0
        for det in detections:
            x1,y1,x2,y2 = det.bbox_2d
            x1=max(0,x1); y1=max(0,y1); x2=min(w-1,x2); y2=min(h-1,y2)
            step=3
            u=np.arange(x1,x2,step); v=np.arange(y1,y2,step)
            if len(u)<2 or len(v)<2: continue
            uu,vv=np.meshgrid(u,v)
            d=depth[vv.astype(int),uu.astype(int)]
            valid=(d>0.1)&(d<5.0)
            if valid.sum()<4: continue
            xs=(uu[valid]-cx)*d[valid]/fx
            ys=(vv[valid]-cy)*d[valid]/fy
            zs=d[valid]
            nv=len(xs)
            verts=np.stack([xs,ys,zs],axis=1)
            r,g,b=det.color
            cols_arr=np.tile([r,g,b],(nv,1))
            tris=[]; valid_idx=np.full(valid.shape,-1,dtype=int)
            valid_idx[valid]=np.arange(nv)
            grid_h,grid_w=valid.shape
            for row in range(grid_h-1):
                for col in range(grid_w-1):
                    i00=valid_idx[row,col]; i10=valid_idx[row+1,col]
                    i01=valid_idx[row,col+1]; i11=valid_idx[row+1,col+1]
                    if i00>=0 and i10>=0 and i01>=0:
                        tris.append([i00+vert_offset,i10+vert_offset,i01+vert_offset])
                    if i11>=0 and i10>=0 and i01>=0:
                        tris.append([i11+vert_offset,i10+vert_offset,i01+vert_offset])
            all_verts.append(verts); all_cols.append(cols_arr)
            all_tris.extend(tris); vert_offset+=nv
        if not all_verts:
            return {'vertices':[],'triangles':[],'colors':[],
                    'normals':[],'vertex_count':0,'triangle_count':0}
        verts_np=np.concatenate(all_verts,axis=0)
        cols_np=np.concatenate(all_cols,axis=0)
        tris_np=np.array(all_tris,dtype=np.int32)
        return {'vertices':verts_np.flatten().tolist(),
                'triangles':tris_np.flatten().tolist(),
                'colors':cols_np.flatten().tolist(),
                'normals':[],'vertex_count':len(verts_np),
                'triangle_count':len(tris_np)}

    @staticmethod
    def _draw_annotations(rgb, detections):
        img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        for det in detections:
            x1,y1,x2,y2 = det.bbox_2d
            b,g,r = det.color[2], det.color[1], det.color[0]
            cv2.rectangle(img,(x1,y1),(x2,y2),(b,g,r),2)
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
            is_agent=data.get('is_agent',False)
            m=Marker(); m.header.frame_id='camera_color_optical_frame'
            m.header.stamp=now; m.ns='nodes'; m.id=mid; mid+=1
            m.type=Marker.SPHERE; m.action=Marker.ADD
            m.pose.position.x=float(pos[0]); m.pose.position.y=float(pos[1])
            m.pose.position.z=float(pos[2]); m.pose.orientation.w=1.0
            # Agents are larger spheres
            size = 0.12 if is_agent else 0.08
            m.scale.x=m.scale.y=m.scale.z=size
            m.color=ColorRGBA(r=col[0]/255.,g=col[1]/255.,b=col[2]/255.,a=0.9)
            m.lifetime.sec=2; arr.markers.append(m)
            t=Marker(); t.header=m.header; t.ns='labels'; t.id=mid; mid+=1
            t.type=Marker.TEXT_VIEW_FACING; t.action=Marker.ADD
            t.pose=m.pose; t.pose.position.y-=0.07; t.scale.z=0.06
            t.color=ColorRGBA(r=1.,g=1.,b=1.,a=1.)
            t.text=data.get('label',''); t.lifetime.sec=2
            arr.markers.append(t)
        for src,dst,data in graph.edges(data=True):
            if not(graph.has_node(src) and graph.has_node(dst)): continue
            ps=graph.nodes[src]['position']; pd=graph.nodes[dst]['position']
            l=Marker(); l.header.frame_id='camera_color_optical_frame'
            l.header.stamp=now; l.ns='edges'; l.id=mid; mid+=1
            l.type=Marker.LINE_STRIP; l.action=Marker.ADD; l.scale.x=0.01
            # Agent edges are brighter
            is_agent_edge=data.get('is_agent_edge',False)
            l.color=ColorRGBA(r=1.0,g=1.0,b=0.2,a=0.9) if is_agent_edge \
                    else ColorRGBA(r=0.4,g=1.,b=0.9,a=0.7)
            l.points=[Point(x=float(ps[0]),y=float(ps[1]),z=float(ps[2])),
                      Point(x=float(pd[0]),y=float(pd[1]),z=float(pd[2]))]
            l.lifetime.sec=2; arr.markers.append(l)
        self.pub_markers.publish(arr)


def main(args=None):
    rclpy.init(args=args)
    node = SceneGraphNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        # Save map on exit
        node.map.save_map(
            os.path.join(node.map_save_dir, 'map_final.json'))
        node.map.save_events()
        node.get_logger().info("Map saved on shutdown.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
