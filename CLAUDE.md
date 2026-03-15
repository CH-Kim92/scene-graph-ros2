# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a ROS2-based 3D scene graph system using an Intel RealSense L515 camera. The **main branch** uses YOLOv8 for detection. The **feature/gdino-sam2-clip branch** replaces YOLOv8 with GroundingDINO + SAM2 + CLIP for open-vocabulary detection.

The system runs split across HOST (camera node) and Docker (ML processing), communicating via ROS2 DDS over localhost.

## Running the System

**Terminal 1 — Camera node (HOST)**:
```bash
cd ~/scene_graph/ros2_ws/src/scene_graph_pkg
LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH PYTHONPATH=$(pwd):$PYTHONPATH \
python3 scene_graph_pkg/realsense_node.py
```

**Terminal 2 — Scene graph processing (Docker)**:
```bash
cd ~/scene_graph
docker compose -f docker/docker-compose.yml up
```
First run compiles GroundingDINO CUDA ops (~5 min). Subsequent starts are fast.

**Terminal 3 — Web visualizer (HOST)**:
```bash
cd ~/scene_graph/viz
python3 -m http.server 8080 --bind 0.0.0.0
```
Then open `http://localhost:8080` in a browser.

## Build Commands

**Build Docker image**:
```bash
docker compose -f docker/docker-compose.yml build
```

**Build ROS2 package** (inside container or host with ROS2 Jazzy):
```bash
cd ros2_ws
colcon build --symlink-install --packages-select scene_graph_pkg
source install/setup.bash
```

**ROS2 linting** (ament tools):
```bash
cd ros2_ws
colcon test --packages-select scene_graph_pkg
colcon test-result --verbose
```

## Architecture

### Data Flow
```
HOST: realsense_node.py (30Hz)
  → ROS2 topics (DDS localhost via fastdds_localhost.xml)
DOCKER: scene_graph_node.py (5Hz)
  → WebSocket (port 8766)
BROWSER: viz/index.html (Three.js)
```

### Key Source Files (`ros2_ws/src/scene_graph_pkg/scene_graph_pkg/`)

| File | Role |
|------|------|
| `scene_graph_node.py` | Main ROS2 node; orchestrates pipeline, handles WebSocket broadcasting |
| `object_detector.py` | `ObjectDetector3D`: GroundingDINO → SAM2 → CLIP → 3D unprojection |
| `scene_graph_builder.py` | `SceneGraph3D`: persistent node/edge graph with EMA smoothing |
| `mesh_reconstructor.py` | `MeshReconstructor`: TSDF volume integration via Open3D |
| `realsense_node.py` | ROS2 publisher for RealSense L515 RGB+depth @ 30Hz |

### Processing Pipeline (inside Docker)
1. **GroundingDINO** — text-prompted open-vocabulary detection → bounding boxes
2. **SAM2** — instance segmentation masks from detected boxes
3. **CLIP** — semantic re-classification/verification of detected labels
4. **Depth unprojection** — masks + aligned depth → 3D centroids + bounding boxes
5. **SceneGraph3D** — maintains persistent objects with spatial edges (near, above, on_top_of, left_of, etc.), EMA smoothing (α=0.7), decay over 10 missed frames

### ROS2 Topics
- **Subscribed**: `/camera/color/image_raw`, `/camera/depth/image_rect_raw`, `/camera/color/camera_info`
- **Published**: `/scene_graph/annotated_image`, `/scene_graph/markers`

### Configurable Launch Parameters
Defined in `launch/scene_graph_nodriver.launch.py`:
- `text_prompt` — comma-separated object descriptions (default: `"person, cup, bottle, laptop, chair, table, phone, keyboard"`)
- `gdino_box_threshold` (0.35), `gdino_text_threshold` (0.25) — GroundingDINO confidence thresholds
- `clip_threshold` (0.20), `clip_verify` (True) — CLIP re-scoring
- `near_threshold` (0.6m) — spatial edge proximity
- `voxel_length` (0.02m) — TSDF mesh resolution
- `ws_port` (8766), `broadcast_hz` (5), `mesh_every_n_frames` — streaming

### Visualizer Controls (viz/index.html)
- **1–4 keys**: switch view modes (RGB pointcloud / segmented mesh / scene graph / live camera)
- **Mouse drag**: rotate; **scroll**: zoom

## Environment Notes

- Docker base: `nvidia/cuda:12.8.0-devel-ubuntu24.04` + ROS2 Jazzy
- Python dependencies run inside `/opt/venv` in the container
- `fastdds_localhost.xml` must be set via `FASTRTPS_DEFAULT_PROFILES_FILE` for HOST↔Docker DDS communication (handled by docker-compose.yml)
- Model weights are pre-downloaded into the image (~2GB): GroundingDINO SwinT, SAM2.1 Hiera Large
- Camera: Intel RealSense L515 (960×540 RGB, 640×480 depth aligned)
