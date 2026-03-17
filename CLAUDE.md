# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

ROS2-based 3D scene graph system using an Intel RealSense L515 camera. Split across two git worktrees on the same repository:

| Directory | Branch | Detection | `ROS_DOMAIN_ID` | WebSocket |
|---|---|---|---|---|
| `~/scene_graph` | `main` | YOLOv8 (fixed 80 COCO classes) | 0 | 8766 |
| `~/scene_graph_gdino` | `feature/gdino-sam2-clip` | GroundingDINO + SAM2 + CLIP (open-vocabulary) | 1 | 8767 |

**Never run `git checkout` inside a worktree directory.** Each directory is permanently tied to its branch.

## Running the System (main branch)

**Terminal 1 — Camera (HOST)**:
```bash
cd ~/scene_graph/ros2_ws/src/scene_graph_pkg
LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH PYTHONPATH=$(pwd):$PYTHONPATH \
python3 scene_graph_pkg/realsense_node.py
```

**Terminal 2 — Scene graph (Docker)**:
```bash
cd ~/scene_graph
docker compose -f docker/docker-compose.yml up
```

**Terminal 3 — Visualizer (HOST)**:
```bash
cd ~/scene_graph/viz && python3 -m http.server 8080
# Open http://localhost:8080 → connects to ws://localhost:8766
```

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

**ROS2 linting**:
```bash
cd ros2_ws && colcon test --packages-select scene_graph_pkg && colcon test-result --verbose
```

## Git Workflow

```bash
cd ~/scene_graph   # always work from this directory for main branch
git pull
# edit → commit → push
git push

git worktree list  # show both worktrees
git fetch --all    # fetch all remotes
```

### Docker management
```bash
docker ps
docker logs scene_graph
docker compose -f docker/docker-compose.yml down
```

## Architecture

### Data Flow
```
HOST: realsense_node.py (30Hz)
  → ROS2 topics (DDS localhost, fastdds_localhost.xml)
DOCKER: scene_graph_node.py (5Hz)
  → WebSocket (port 8766)
BROWSER: viz/index.html (Three.js)
```

### Key Source Files (`ros2_ws/src/scene_graph_pkg/scene_graph_pkg/`)

| File | Role |
|------|------|
| `scene_graph_node.py` | Main ROS2 node; orchestrates pipeline, WebSocket broadcasting |
| `scene_graph_builder.py` | `SceneGraph3D`: persistent node/edge graph with EMA smoothing (α=0.7) |
| `mesh_reconstructor.py` | `MeshReconstructor`: TSDF volume integration via Open3D |
| `realsense_node.py` | ROS2 publisher for RealSense L515 RGB+depth @ 30Hz |

### ROS2 Topics
- **Subscribed**: `/camera/color/image_raw`, `/camera/depth/image_rect_raw`, `/camera/color/camera_info`
- **Published**: `/scene_graph/annotated_image`, `/scene_graph/markers`

### Visualizer Controls (`viz/index.html`)
- **1–4 keys**: switch view modes (RGB pointcloud / segmented mesh / scene graph / live camera)
- **Mouse drag**: rotate; **scroll**: zoom

## Environment Notes

- Docker base: `nvidia/cuda:12.8.0-devel-ubuntu24.04` + ROS2 Jazzy
- Python deps inside `/opt/venv` in container
- `fastdds_localhost.xml` required for HOST↔Docker DDS (set automatically by docker-compose.yml)
- Camera: Intel RealSense L515 (960×540 RGB, 640×480 depth aligned)
