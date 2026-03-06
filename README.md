# 3D Scene Graph — RealSense L515 + ROS2 Jazzy + YOLOv8

A real-time 3D scene understanding system that builds a semantic scene graph from live RGB-D camera data. Objects are detected, localized in 3D space, and connected by spatial relationships — all visualized in a browser-based Three.js interface.

---

## System Overview

```
RealSense L515 (HOST)
        │
        ▼
 realsense_node.py          ← Python ROS2 node (runs on HOST)
  /camera/color/image_raw
  /camera/depth/image_rect_raw
  /camera/color/camera_info
        │
        ▼ ROS2 topics (network_mode: host)
        │
 Docker Container
  ┌─────────────────────────────────┐
  │  scene_graph_node               │
  │  ├── YOLOv8 Object Detection    │
  │  ├── 3D Point Unprojection      │
  │  ├── Scene Graph (NetworkX)     │
  │  └── WebSocket Broadcaster      │
  └─────────────────────────────────┘
        │
        ▼ ws://localhost:8766
  Browser (Three.js Visualizer)
  ├── RGB Point Cloud view
  ├── Segmented Mesh view
  └── Scene Graph view
```

---

## Hardware Requirements

| Component | Spec |
|-----------|------|
| Camera | Intel RealSense L515 |
| GPU | NVIDIA GPU (tested on RTX 5070 Ti) |
| OS | Ubuntu 24.04 |
| CPU | Intel Z890 or equivalent |

---

## Software Requirements

- **ROS2 Jazzy** (installed on HOST)
- **Docker** + **Docker Compose**
- **NVIDIA Container Toolkit**
- **pyrealsense2** (built from source — see Setup)

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/scene-graph-ros2.git ~/scene_graph
cd ~/scene_graph
```

### 2. Build pyrealsense2 from source (required for L515)

The pip version of pyrealsense2 does not support the L515. Build from the Intel source:

```bash
cd ~
git clone https://github.com/IntelRealSense/librealsense.git \
  -b v2.56.4 --depth 1 librealsense2_564
cd librealsense2_564
mkdir build && cd build

cmake .. \
  -DBUILD_PYTHON_BINDINGS=ON \
  -DPYTHON_EXECUTABLE=/usr/bin/python3 \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_EXAMPLES=OFF \
  -DBUILD_GRAPHICAL_EXAMPLES=OFF \
  -DFORCE_RSUSB_BACKEND=ON

make -j$(nproc) pyrealsense2
```

Copy the working Python bindings:

```bash
sudo mkdir -p /usr/local/lib/python3.12/site-packages
sudo cp /usr/local/OFF/pyrealsense2*.so /usr/local/lib/python3.12/site-packages/
```

Add library path to `~/.bashrc`:

```bash
echo 'export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

Verify L515 is detected:

```bash
LD_LIBRARY_PATH=/usr/local/lib python3 -c "
import pyrealsense2 as rs
ctx = rs.context()
print('Devices:', len(ctx.devices))
for d in ctx.devices:
    print(' -', d.get_info(rs.camera_info.name))
"
# Expected: Devices: 1 / Intel RealSense L515
```

### 3. Configure FastDDS for localhost (required for HOST↔Docker ROS2 communication)

```bash
cat > ~/fastdds_localhost.xml << 'EOF'
<?xml version="1.0" encoding="UTF-8" ?>
<profiles xmlns="http://www.eprosima.com/XMLSchemas/fastRTPS_Profiles">
    <transport_descriptors>
        <transport_descriptor>
            <transport_id>udp_localhost</transport_id>
            <type>UDPv4</type>
        </transport_descriptor>
    </transport_descriptors>
    <participant profile_name="localhost_participant" is_default_profile="true">
        <rtps>
            <userTransports>
                <transport_id>udp_localhost</transport_id>
            </userTransports>
            <useBuiltinTransports>false</useBuiltinTransports>
            <defaultUnicastLocatorList>
                <locator>
                    <udpv4>
                        <address>127.0.0.1</address>
                    </udpv4>
                </locator>
            </defaultUnicastLocatorList>
        </rtps>
    </participant>
</profiles>
EOF

echo 'export FASTRTPS_DEFAULT_PROFILES_FILE=~/fastdds_localhost.xml' >> ~/.bashrc
echo 'export ROS_DOMAIN_ID=0' >> ~/.bashrc
source ~/.bashrc
```

### 4. Build Docker image

```bash
cd ~/scene_graph
docker compose -f docker/docker-compose.yml build
```

---

## Running the System

You need **3 terminals** running simultaneously.

### Terminal 1 — Camera Node (HOST)

```bash
source /opt/ros/jazzy/setup.bash
cd ~/scene_graph/ros2_ws/src/scene_graph_pkg
LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH \
PYTHONPATH=$(pwd):$PYTHONPATH \
python3 scene_graph_pkg/realsense_node.py
```

Wait for:
```
[realsense_node]: Camera warmed up!
[realsense_node]: Camera ready: 960x540 fx=683.0 fy=683.5
```

### Terminal 2 — Scene Graph Processing (Docker)

```bash
cd ~/scene_graph
docker compose -f docker/docker-compose.yml up
```

Wait for:
```
[scene_graph_node]: Camera intrinsics received: fx=683.0
[scene_graph_node]: frame=1 dets=2 nodes=2 edges=1 85ms
```

### Terminal 3 — Web Visualizer

```bash
cd ~/scene_graph/viz
python3 -m http.server 8080
```

Open browser: **http://localhost:8080**

In the WebSocket field, enter: `ws://localhost:8766`

Click **Connect**.

---

## Visualizer Controls

| Key | Mode |
|-----|------|
| `1` | RGB Point Cloud |
| `2` | Segmented Mesh (per object) |
| `3` | Scene Graph (nodes + edges) |
| Mouse drag | Rotate view |
| Scroll | Zoom |

---

## Project Structure

```
scene_graph/
├── docker/
│   ├── Dockerfile              # CUDA + ROS2 Jazzy + ML deps
│   └── docker-compose.yml      # Container config
├── ros2_ws/
│   └── src/scene_graph_pkg/
│       ├── scene_graph_pkg/
│       │   ├── realsense_node.py       # Camera driver (HOST)
│       │   ├── scene_graph_node.py     # Main processing node
│       │   ├── object_detector.py      # YOLOv8 3D detection
│       │   ├── scene_graph_builder.py  # NetworkX graph
│       │   └── mesh_reconstructor.py  # TSDF/Open3D
│       ├── launch/
│       │   └── scene_graph_nodriver.launch.py
│       ├── setup.py
│       └── package.xml
└── viz/
    └── index.html              # Three.js visualizer
```

---

## Architecture Details

### Camera Node (realsense_node.py)
Runs directly on the HOST using pyrealsense2 built against librealsense 2.54.2 (the version that supports L515). Publishes standard ROS2 sensor topics at 30Hz.

### Scene Graph Node (scene_graph_node.py)
Runs inside Docker with GPU access. Pipeline per frame:
1. **Detection**: YOLOv8m detects objects in RGB image
2. **3D Localization**: Depth values unprojected using camera intrinsics
3. **Graph Update**: NetworkX graph updated with object nodes and spatial edges
4. **Point Cloud**: Per-object colored point clouds extracted from depth
5. **WebSocket Broadcast**: JSON payload sent to browser at 5Hz

### Spatial Relations
Objects are connected by edges when their 3D distance is below a configurable threshold. Relations include: `near`, `left_of`, `right_of`, `above`, `below`, `in_front_of`, `behind`.

---

## Troubleshooting

### Camera not detected
```bash
lsusb | grep Intel   # Check USB connection
/usr/local/bin/rs-enumerate-devices  # Should list L515
```

### ROS2 topics not flowing between HOST and Docker
Ensure FastDDS config is set on both HOST and in docker-compose.yml:
```bash
echo $FASTRTPS_DEFAULT_PROFILES_FILE   # Should point to fastdds_localhost.xml
echo $ROS_DOMAIN_ID                    # Should be 0
```

### WebSocket not connecting
Check port 8766 is open:
```bash
sudo ss -tlnp | grep 8766
```

### NumPy version conflict in Docker
```bash
docker exec -it scene_graph bash
/opt/venv/bin/pip install "numpy<2" "opencv-python<4.10"
```

---

## Known Limitations

- **RTX 5070 Ti (Blackwell)**: PyTorch stable does not support sm_120. Install nightly build for full GPU acceleration
- **L515 + ROS2**: `ros-jazzy-realsense2-camera` does not support L515 — custom Python node used instead
- **TSDF mesh**: Disabled in favor of fast per-frame segmented mesh for better performance

---

## License

MIT
