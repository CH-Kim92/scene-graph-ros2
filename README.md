# 3D Scene Graph — GroundingDINO + SAM2 + CLIP Pipeline

> **Branch:** `feature/gdino-sam2-clip`  
> **Base branch:** `main` (YOLOv8 version)

An upgrade to the original 3D scene graph system replacing fixed-class YOLOv8 detection with an open-vocabulary pipeline. You can now detect **any object by describing it in natural language** — no retraining required.

---

## What Changed From `main`

| Component | `main` (YOLOv8) | `feature/gdino-sam2-clip` |
|-----------|----------------|--------------------------|
| Detection | YOLOv8m — fixed 80 COCO classes | GroundingDINO — any object by text |
| Segmentation | Bounding box only | SAM2 — pixel-precise mask |
| Classification | YOLO confidence score | CLIP semantic re-scoring |
| 3D Position | Box center + depth | Mask pixels + depth (more accurate) |
| CUDA base | 12.6 | 12.8 (Blackwell RTX 5000 support) |
| Config | `yolo_model`, `confidence` | `text_prompt`, `gdino_box_threshold`, `clip_threshold` |

---

## Model Pipeline

```
RGB + Depth (RealSense L515 @ 30Hz)
         │
         ▼
┌─────────────────────────────────────────┐
│  GroundingDINO (SwinT + BERT)           │
│  Input:  RGB image + text prompt        │
│  Output: bounding boxes + labels        │
│  "person, cup, laptop, chair..."        │
└─────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│  SAM2 (Segment Anything Model 2)        │
│  Input:  RGB image + bounding boxes     │
│  Output: pixel-precise mask per object  │
└─────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│  CLIP (ViT-B/32)                        │
│  Input:  cropped image patch per object │
│  Output: verified label + semantic score│
└─────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│  Depth Unprojection                     │
│  Input:  SAM2 mask + depth image        │
│  Output: 3D centroid + 3D bounding box  │
└─────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│  Scene Graph (NetworkX)                 │
│  Nodes: detected objects with 3D pos    │
│  Edges: spatial relations (near, above) │
└─────────────────────────────────────────┘
         │
         ▼
  WebSocket → Three.js Browser Visualizer
```

---

## Hardware Requirements

| Component | Requirement |
|-----------|-------------|
| Camera | Intel RealSense L515 |
| GPU | NVIDIA RTX 5000 series (Blackwell sm_120) or older |
| CUDA | 12.8+ |
| RAM | 16GB+ recommended |
| OS | Ubuntu 24.04 |

---

## Software Requirements

- ROS2 Jazzy (installed on HOST)
- Docker + Docker Compose
- NVIDIA Container Toolkit
- pyrealsense2 built from source (see main branch README)

---

## Setup

### 1. Clone and switch to this branch

```bash
git clone https://github.com/CH-Kim92/scene-graph-ros2.git ~/scene_graph
cd ~/scene_graph
git checkout feature/gdino-sam2-clip
```

### 2. Set up pyrealsense2 on HOST (same as main branch)

Follow the pyrealsense2 build instructions in the main branch README. This only needs to be done once regardless of branch.

### 3. Configure FastDDS for HOST↔Docker communication

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

This installs all Python dependencies including SAM2, CLIP, and transformers. Model weights (~2GB) are downloaded automatically on first run.

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

**First run only** — GroundingDINO CUDA ops are compiled at startup (~5 minutes):
```
Cloning GroundingDINO...
Patching CUDA source for PyTorch 2.x...
Compiling GroundingDINO...
CUDA ops compiled OK
```

Wait for all three models to load:
```
[ObjectDetector3D] GroundingDINO ready.
[ObjectDetector3D] SAM2 ready.
[ObjectDetector3D] CLIP ready.
[ObjectDetector3D] Ready. Prompt: 'person, cup, bottle, laptop...'
[scene_graph_node]: Camera intrinsics received: fx=683.0
[scene_graph_node]: frame=1 dets=2 nodes=2 edges=1 340ms
```

### Terminal 3 — Web Visualizer

```bash
cd ~/scene_graph/viz
python3 -m http.server 8080 --bind 0.0.0.0
```

Open browser: **http://localhost:8080**

Set WebSocket URL to `ws://localhost:8766` → click **Connect**

**Share with others on same WiFi:** open `http://<your-ip>:8080` on any device.

---

## Changing What Objects to Detect

No retraining needed — just edit the `text_prompt` in `docker/docker-compose.yml`:

```yaml
'text_prompt:=person, cup, bottle, laptop, chair, table, phone, keyboard'
```

Change to anything you want:
```yaml
'text_prompt:=screwdriver, robot arm, safety helmet, conveyor belt, fire extinguisher'
```

Restart Docker and it detects your new objects immediately.

---

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `text_prompt` | `person, cup, bottle...` | Objects to detect (natural language) |
| `gdino_box_threshold` | `0.35` | GroundingDINO minimum box confidence |
| `gdino_text_threshold` | `0.25` | GroundingDINO minimum text alignment score |
| `clip_verify` | `true` | Enable CLIP re-classification |
| `clip_threshold` | `0.20` | Minimum CLIP similarity score to keep detection |
| `near_threshold` | `0.60` | Max distance (metres) to create a scene graph edge |
| `ws_port` | `8766` | WebSocket port for browser visualizer |
| `broadcast_hz` | `5.0` | Visualizer update rate (Hz) |

---

## Visualizer Controls

| Key | Mode |
|-----|------|
| `1` | RGB Point Cloud |
| `2` | Segmented Mesh (per object) |
| `3` | Scene Graph (nodes + edges) |
| `4` | Live Camera (2D annotated RGB) |
| Mouse drag | Rotate |
| Scroll | Zoom |

---

## Project Structure Changes vs `main`

```
scene_graph/
├── docker/
│   ├── Dockerfile              # CUDA 12.8 base + GroundingDINO/SAM2/CLIP deps
│   ├── docker-compose.yml      # Runtime CUDA ops compilation
│   └── patch_gdino.py          # PyTorch 2.x compatibility patch for GroundingDINO
├── ros2_ws/src/scene_graph_pkg/
│   ├── scene_graph_pkg/
│   │   ├── object_detector.py  # NEW: GroundingDINO + SAM2 + CLIP pipeline
│   │   ├── scene_graph_node.py # Updated parameters
│   │   └── ...
│   └── launch/
│       └── scene_graph_nodriver.launch.py  # Updated launch args
```

---

## Known Issues

- **First startup is slow (~5 min)** due to GroundingDINO CUDA ops compilation. Subsequent restarts reuse the compiled ops if the container is not removed.
- **Processing is slower than YOLOv8** (~300-500ms per frame vs ~100ms) due to SAM2 segmentation. Reduce `broadcast_hz` if needed.
- **RTX 5070 Ti (Blackwell)** requires the CUDA 12.8 base image. Older GPUs should still work.

---

## Switching Back to YOLOv8 Version

```bash
git checkout main
docker compose -f docker/docker-compose.yml build
docker compose -f docker/docker-compose.yml up
```

---

## License

MIT
