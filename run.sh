#!/bin/bash
# ═══════════════════════════════════════════════════════
#  scene-graph-ros2 — Native Jetson launcher
#  Usage: ./run.sh [--reset-map]
# ═══════════════════════════════════════════════════════

source ~/.bashrc

MAP_DIR="$HOME/scene_graph/maps"
LOG_DIR="$HOME/logs"
mkdir -p "$MAP_DIR" "$LOG_DIR"

# ── Optional: reset map on launch ───────────────────────
if [[ "$1" == "--reset-map" ]]; then
    echo "⚠  Resetting map — deleting all saved maps and events"
    rm -f "$MAP_DIR"/*.json
fi

# ── Kill any existing processes ──────────────────────────
echo "Stopping any existing processes..."
pkill -f realsense_node   2>/dev/null || true
pkill -f scene_graph_node 2>/dev/null || true
pkill -f "http.server 8080" 2>/dev/null || true
sleep 2

# ── Terminal 1: Camera node ──────────────────────────────
echo "Starting camera node..."
cd "$HOME/scene_graph/ros2_ws/src/scene_graph_pkg"
python3 scene_graph_pkg/realsense_node.py \
    2>&1 | tee "$LOG_DIR/camera_node.log" &
CAMERA_PID=$!

# Wait for camera to warm up
echo "Waiting for camera warmup (5s)..."
sleep 5

# Check camera actually started
if ! kill -0 $CAMERA_PID 2>/dev/null; then
    echo "ERROR: Camera node failed to start. Check $LOG_DIR/camera_node.log"
    exit 1
fi
echo "✓ Camera node running (PID $CAMERA_PID)"

# ── Terminal 2: Scene graph node ─────────────────────────
echo "Starting scene graph node (persistent map)..."
cd "$HOME/scene_graph/ros2_ws"
ros2 launch scene_graph_pkg scene_graph_nodriver.launch.py \
    2>&1 | tee "$LOG_DIR/scene_graph.log" &
GRAPH_PID=$!
sleep 3

if ! kill -0 $GRAPH_PID 2>/dev/null; then
    echo "ERROR: Scene graph node failed. Check $LOG_DIR/scene_graph.log"
    exit 1
fi
echo "✓ Scene graph node running (PID $GRAPH_PID)"

# ── Terminal 3: Web visualizer ───────────────────────────
echo "Starting web visualizer..."
cd "$HOME/scene_graph/viz"
python3 -m http.server 8080 --bind 0.0.0.0\
    2>&1 | tee "$LOG_DIR/web_server.log" &
WEB_PID=$!
sleep 1
echo "✓ Web visualizer running (PID $WEB_PID)"

# ── Summary ──────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════"
echo "  All systems running!"
echo "  Browser:    http://localhost:8080"
echo "  Remote:     http://$(hostname -I | awk '{print $1}'):8080"
echo "  Maps saved: $MAP_DIR/"
echo "  Logs:       $LOG_DIR/"
echo ""
echo "  Commands:"
echo "  tail -f $LOG_DIR/scene_graph.log   ← watch graph"
echo "  ls $MAP_DIR/                        ← see saved maps"
echo "  ./run.sh --reset-map               ← start fresh"
echo "═══════════════════════════════════════════"

# Keep script alive — Ctrl+C to stop everything
trap "echo 'Shutting down...'; kill $CAMERA_PID $GRAPH_PID $WEB_PID 2>/dev/null; exit" INT
wait
