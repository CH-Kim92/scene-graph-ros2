"""
map_manager.py
──────────────
Persistent incremental map for 3D scene graph.

Instead of rebuilding the graph every frame, this module:
  1. Confirms objects before adding them (reduces false positives)
  2. Keeps objects in the map even when temporarily not detected
  3. Tracks when each object was first/last seen
  4. Supports adding human/robot agents with spatio-temporal logging
  5. Records all agent-object interaction events over time

Usage:
    from scene_graph_pkg.map_manager import MapManager
    manager = MapManager()
    manager.update(detections, timestamp)
    graph_dict = manager.to_dict()
"""

from __future__ import annotations

import time
import json
import os
import numpy as np
import networkx as nx
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple


# ── Spatial relationship helpers (mirrors scene_graph_builder.py) ─────────────

RELATION_COLORS = {
    "near":        "#64ffda",
    "above":       "#ff6d00",
    "below":       "#ff6d00",
    "left_of":     "#40c4ff",
    "right_of":    "#40c4ff",
    "in_front_of": "#ea80fc",
    "behind":      "#ea80fc",
    "on_top_of":   "#ffff00",
    "interacting": "#ffffff",
}


# ── Spatio-temporal event ─────────────────────────────────────────────────────

@dataclass
class SpatioTemporalEvent:
    """A single recorded interaction between an agent and an object."""
    agent_id:   str
    relation:   str         # e.g. 'near', 'picked_up', 'moved_to'
    object_id:  str
    position:   List[float] # where it happened [x, y, z]
    timestamp:  float       # Unix time


# ── Main MapManager ───────────────────────────────────────────────────────────

class MapManager:
    """
    Persistent incremental scene graph map.

    Parameters
    ----------
    confirmation_frames : int   – frames an object must appear before added to map
    disappear_timeout   : float – seconds without detection before object removed
    position_grid       : float – quantisation grid (m) for node identity
    near_threshold      : float – max distance (m) for 'near' edge
    above_min_dy        : float – min vertical gap (m) for above/below
    overlap_xy_thresh   : float – max XZ distance (m) for on_top_of
    event_log_path      : str   – path to save event log JSON (None = no save)
    """

    def __init__(
        self,
        confirmation_frames: int   = 5,
        disappear_timeout:   float = 5.0,
        position_grid:       float = 0.30,
        near_threshold:      float = 0.60,
        above_min_dy:        float = 0.10,
        overlap_xy_thresh:   float = 0.20,
        event_log_path:      Optional[str] = None,
    ):
        self.confirmation_frames = confirmation_frames
        self.disappear_timeout   = disappear_timeout
        self.position_grid       = position_grid
        self.near_threshold      = near_threshold
        self.above_min_dy        = above_min_dy
        self.overlap_xy_thresh   = overlap_xy_thresh
        self.event_log_path      = event_log_path

        # Persistent graph — nodes stay until timeout
        self.graph: nx.DiGraph = nx.DiGraph()

        # Candidate objects seen but not yet confirmed
        self._candidates: Dict[str, Dict] = {}

        # Timestamp when each node was last detected
        self._last_seen: Dict[str, float] = {}

        # Timestamp when each node was first confirmed
        self._first_seen: Dict[str, float] = {}

        # How many times each confirmed node has been detected
        self._detection_count: Dict[str, int] = {}

        # Spatio-temporal event log
        self._events: List[SpatioTemporalEvent] = []

        # Frame counter
        self._frame_id = 0

        print("[MapManager] Initialized — persistent incremental map ready.")

    # ── Public API ────────────────────────────────────────────────────────────

    def update(self, detections, timestamp: Optional[float] = None) -> nx.DiGraph:
        """
        Called every frame with new detections.
        Updates the persistent map incrementally.

        Parameters
        ----------
        detections : List[Detection3D]  from object_detector.py
        timestamp  : float              Unix time (defaults to now)

        Returns
        -------
        nx.DiGraph  the current persistent map graph
        """
        if timestamp is None:
            timestamp = time.time()

        self._frame_id += 1
        detected_node_ids = set()

        # ── 1. Process each detection ─────────────────────────────────────────
        for det in detections:
            node_id = self._make_node_id(det)
            detected_node_ids.add(node_id)

            if self.graph.has_node(node_id):
                # Already confirmed — just update position (EMA smoothing)
                self._update_node_position(node_id, det, timestamp)
                self._detection_count[node_id] = self._detection_count.get(node_id, 0) + 1
                self._last_seen[node_id] = timestamp
            else:
                # Not yet confirmed — add to candidates
                if node_id not in self._candidates:
                    self._candidates[node_id] = {
                        'det': det,
                        'count': 0,
                        'first_candidate': timestamp,
                    }
                self._candidates[node_id]['count'] += 1
                self._candidates[node_id]['det'] = det  # update with latest

                # Confirm object after N frames
                if self._candidates[node_id]['count'] >= self.confirmation_frames:
                    self._confirm_node(node_id, det, timestamp)
                    del self._candidates[node_id]

        # ── 2. Age out objects not seen for timeout period ────────────────────
        nodes_to_remove = []
        for node_id, last_t in list(self._last_seen.items()):
            # Skip agents — they are manually managed
            if self.graph.has_node(node_id) and \
               self.graph.nodes[node_id].get('is_agent', False):
                continue
            if timestamp - last_t > self.disappear_timeout:
                nodes_to_remove.append(node_id)

        for node_id in nodes_to_remove:
            self._remove_node(node_id)

        # ── 3. Rebuild spatial edges ──────────────────────────────────────────
        self._rebuild_edges()

        # ── 4. Update agent-object edges ──────────────────────────────────────
        self._update_agent_edges(timestamp)

        return self.graph

    def add_agent(
        self,
        agent_id:   str,
        agent_type: str,        # 'human' or 'robot'
        position:   List[float],
        timestamp:  Optional[float] = None,
        color:      List[int] = None,
    ) -> None:
        """
        Add or update a human/robot agent in the persistent map.

        Parameters
        ----------
        agent_id   : unique string ID, e.g. 'human_0', 'robot_arm'
        agent_type : 'human' or 'robot'
        position   : [x, y, z] in camera frame (metres)
        timestamp  : Unix time (defaults to now)
        color      : [R,G,B] 0-255
        """
        if timestamp is None:
            timestamp = time.time()

        if color is None:
            color = [255, 80, 80] if agent_type == 'human' else [80, 80, 255]

        if self.graph.has_node(agent_id):
            # Update existing agent position
            old_pos = np.array(self.graph.nodes[agent_id]['position'])
            new_pos = (0.5 * old_pos + 0.5 * np.array(position)).tolist()
            self.graph.nodes[agent_id]['position']  = new_pos
            self.graph.nodes[agent_id]['last_seen'] = timestamp
        else:
            # Add new agent
            self.graph.add_node(
                agent_id,
                label=agent_type,
                position=position,
                color=color,
                is_agent=True,
                agent_type=agent_type,
                first_seen=timestamp,
                last_seen=timestamp,
                confidence=1.0,
                bbox_min=[0, 0, 0],
                bbox_max=[0, 0, 0],
                frame=self._frame_id,
            )
            self._last_seen[agent_id] = timestamp
            self._first_seen[agent_id] = timestamp
            print(f"[MapManager] Agent added: {agent_id} ({agent_type}) at {position}")

        # Record spatial events with nearby objects
        self._record_agent_events(agent_id, position, timestamp)
        self._update_agent_edges(timestamp)

    def remove_agent(self, agent_id: str) -> None:
        """Remove an agent from the map."""
        self._remove_node(agent_id)
        print(f"[MapManager] Agent removed: {agent_id}")

    def get_events(
        self,
        agent_id:   Optional[str]   = None,
        object_id:  Optional[str]   = None,
        time_range: Optional[Tuple] = None,
        relation:   Optional[str]   = None,
    ) -> List[Dict]:
        """
        Query the spatio-temporal event log.

        Parameters
        ----------
        agent_id   : filter by agent
        object_id  : filter by object node ID
        time_range : (start_time, end_time) Unix timestamps
        relation   : filter by relation type

        Returns
        -------
        List of event dicts
        """
        results = self._events

        if agent_id:
            results = [e for e in results if e.agent_id == agent_id]
        if object_id:
            results = [e for e in results if e.object_id == object_id]
        if relation:
            results = [e for e in results if e.relation == relation]
        if time_range:
            results = [e for e in results
                       if time_range[0] <= e.timestamp <= time_range[1]]

        return [
            {
                'agent':     e.agent_id,
                'relation':  e.relation,
                'object':    e.object_id,
                'position':  e.position,
                'timestamp': e.timestamp,
                'time_str':  time.strftime('%H:%M:%S', time.localtime(e.timestamp)),
            }
            for e in results
        ]

    def save_events(self, path: Optional[str] = None) -> None:
        """Save event log to JSON file."""
        save_path = path or self.event_log_path
        if not save_path:
            print("[MapManager] No event log path set.")
            return
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(self.get_events(), f, indent=2)
        print(f"[MapManager] Events saved to {save_path}")

    def save_map(self, path: str) -> None:
        """Save current map graph to JSON file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"[MapManager] Map saved to {path}")

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the persistent map to JSON-friendly dict for visualiser."""
        nodes = []
        for nid, data in self.graph.nodes(data=True):
            nodes.append({
                "id":              nid,
                "label":           data.get("label", ""),
                "position":        data.get("position", [0, 0, 0]),
                "confidence":      round(data.get("confidence", 0.0), 3),
                "color":           data.get("color", [160, 160, 160]),
                "bbox_min":        data.get("bbox_min", [0, 0, 0]),
                "bbox_max":        data.get("bbox_max", [0, 0, 0]),
                "frame":           data.get("frame", 0),
                "is_agent":        data.get("is_agent", False),
                "agent_type":      data.get("agent_type", ""),
                "first_seen":      data.get("first_seen", 0),
                "last_seen":       data.get("last_seen", 0),
                "detection_count": self._detection_count.get(nid, 0),
            })

        edges = []
        for src, dst, data in self.graph.edges(data=True):
            rel = data.get("relation", "near")
            edges.append({
                "source":   src,
                "target":   dst,
                "relation": rel,
                "distance": round(data.get("distance", 0.0), 3),
                "color":    RELATION_COLORS.get(rel, "#ffffff"),
            })

        return {
            "frame":       self._frame_id,
            "nodes":       nodes,
            "edges":       edges,
            "timestamp":   time.time(),
            "node_count":  self.graph.number_of_nodes(),
            "edge_count":  self.graph.number_of_edges(),
            "event_count": len(self._events),
            "candidates":  len(self._candidates),
        }

    def stats(self) -> str:
        agents = sum(
            1 for _, d in self.graph.nodes(data=True) if d.get('is_agent', False)
        )
        objects = self.graph.number_of_nodes() - agents
        return (
            f"PersistentMap | frame={self._frame_id} "
            f"objects={objects} agents={agents} "
            f"edges={self.graph.number_of_edges()} "
            f"events={len(self._events)} "
            f"candidates={len(self._candidates)}"
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _make_node_id(self, det) -> str:
        """
        Stable ID based on class label + quantised XZ position.
        Matches the logic in scene_graph_builder.py for compatibility.
        """
        qx = round(det.position_3d[0] / self.position_grid)
        qz = round(det.position_3d[2] / self.position_grid)
        return f"{det.label}_{qx}_{qz}"

    def _confirm_node(self, node_id: str, det, timestamp: float) -> None:
        """Add a confirmed object to the persistent graph."""
        self.graph.add_node(
            node_id,
            label=det.label,
            position=det.position_3d,
            confidence=det.confidence,
            color=det.color,
            bbox_min=det.bbox_3d_min,
            bbox_max=det.bbox_3d_max,
            frame=self._frame_id,
            is_agent=False,
            first_seen=timestamp,
            last_seen=timestamp,
        )
        self._last_seen[node_id]      = timestamp
        self._first_seen[node_id]     = timestamp
        self._detection_count[node_id] = self.confirmation_frames
        print(f"[MapManager] ✓ Confirmed: {node_id} at {det.position_3d}")

    def _update_node_position(self, node_id: str, det, timestamp: float) -> None:
        """Smooth update of an existing confirmed node."""
        alpha   = 0.7  # EMA weight for old position
        old_pos = np.array(self.graph.nodes[node_id]['position'])
        new_pos = (alpha * old_pos + (1 - alpha) * np.array(det.position_3d)).tolist()
        self.graph.nodes[node_id]['position']   = new_pos
        self.graph.nodes[node_id]['confidence'] = det.confidence
        self.graph.nodes[node_id]['frame']      = self._frame_id
        self.graph.nodes[node_id]['last_seen']  = timestamp

    def _remove_node(self, node_id: str) -> None:
        """Remove a node and clean up tracking dicts."""
        if self.graph.has_node(node_id):
            self.graph.remove_node(node_id)
        self._last_seen.pop(node_id, None)
        self._first_seen.pop(node_id, None)
        self._detection_count.pop(node_id, None)
        print(f"[MapManager] ✗ Removed: {node_id}")

    def _rebuild_edges(self) -> None:
        """Recompute all spatial edges between confirmed nodes."""
        # Remove old spatial edges (keep agent edges rebuilt separately)
        non_agent_edges = [
            (u, v) for u, v, d in self.graph.edges(data=True)
            if not d.get('is_agent_edge', False)
        ]
        self.graph.remove_edges_from(non_agent_edges)

        nodes = [(n, d) for n, d in self.graph.nodes(data=True)
                 if not d.get('is_agent', False)]

        for i, (ni, di) in enumerate(nodes):
            for j, (nj, dj) in enumerate(nodes):
                if i >= j:
                    continue
                self._add_spatial_edges(ni, di, nj, dj)

    def _add_spatial_edges(
        self, ni: str, di: Dict, nj: str, dj: Dict
    ) -> None:
        """Compute and add spatial relation edges between two object nodes."""
        pi = np.array(di['position'])
        pj = np.array(dj['position'])

        dx = pi[0] - pj[0]
        dy = pi[1] - pj[1]
        dz = pi[2] - pj[2]

        dist_3d = float(np.linalg.norm(pi - pj))
        dist_xz = float(np.sqrt(dx**2 + dz**2))

        if dist_3d < self.near_threshold:
            self.graph.add_edge(ni, nj, relation="near", distance=dist_3d,
                                is_agent_edge=False)

        if abs(dy) > self.above_min_dy:
            if dy > 0:
                self.graph.add_edge(ni, nj, relation="below",  distance=abs(dy),
                                    is_agent_edge=False)
                self.graph.add_edge(nj, ni, relation="above",  distance=abs(dy),
                                    is_agent_edge=False)
            else:
                self.graph.add_edge(ni, nj, relation="above",  distance=abs(dy),
                                    is_agent_edge=False)
                self.graph.add_edge(nj, ni, relation="below",  distance=abs(dy),
                                    is_agent_edge=False)

        if abs(dy) > self.above_min_dy and dist_xz < self.overlap_xy_thresh:
            if dy < 0:
                self.graph.add_edge(ni, nj, relation="on_top_of", distance=dist_xz,
                                    is_agent_edge=False)
            else:
                self.graph.add_edge(nj, ni, relation="on_top_of", distance=dist_xz,
                                    is_agent_edge=False)

        if abs(dx) > 0.20:
            if dx > 0:
                self.graph.add_edge(ni, nj, relation="right_of", distance=abs(dx),
                                    is_agent_edge=False)
            else:
                self.graph.add_edge(ni, nj, relation="left_of",  distance=abs(dx),
                                    is_agent_edge=False)

        if abs(dz) > 0.25:
            if dz < 0:
                self.graph.add_edge(ni, nj, relation="in_front_of", distance=abs(dz),
                                    is_agent_edge=False)
            else:
                self.graph.add_edge(nj, ni, relation="in_front_of", distance=abs(dz),
                                    is_agent_edge=False)

    def _update_agent_edges(self, timestamp: float) -> None:
        """Recompute edges between agents and nearby objects."""
        # Remove old agent edges
        agent_edges = [
            (u, v) for u, v, d in self.graph.edges(data=True)
            if d.get('is_agent_edge', False)
        ]
        self.graph.remove_edges_from(agent_edges)

        agents = [
            (nid, data) for nid, data in self.graph.nodes(data=True)
            if data.get('is_agent', False)
        ]
        objects = [
            (nid, data) for nid, data in self.graph.nodes(data=True)
            if not data.get('is_agent', False)
        ]

        for agent_id, agent_data in agents:
            pa = np.array(agent_data['position'])
            for obj_id, obj_data in objects:
                po  = np.array(obj_data['position'])
                dist = float(np.linalg.norm(pa - po))
                if dist < self.near_threshold * 2.0:
                    self.graph.add_edge(
                        agent_id, obj_id,
                        relation="near",
                        distance=round(dist, 3),
                        is_agent_edge=True,
                    )

    def _record_agent_events(
        self, agent_id: str, position: List[float], timestamp: float
    ) -> None:
        """Record spatio-temporal events for an agent near objects."""
        pa = np.array(position)

        for obj_id, obj_data in self.graph.nodes(data=True):
            if obj_data.get('is_agent', False):
                continue
            po   = np.array(obj_data['position'])
            dist = float(np.linalg.norm(pa - po))

            if dist < self.near_threshold * 2.0:
                # Only log if this is a new proximity event
                # (avoid flooding log with same event every frame)
                recent = [
                    e for e in self._events[-20:]
                    if e.agent_id == agent_id
                    and e.object_id == obj_id
                    and timestamp - e.timestamp < 2.0
                ]
                if not recent:
                    event = SpatioTemporalEvent(
                        agent_id=agent_id,
                        relation="near",
                        object_id=obj_id,
                        position=position,
                        timestamp=timestamp,
                    )
                    self._events.append(event)

                    # Auto-save if path configured
                    if self.event_log_path and len(self._events) % 10 == 0:
                        self.save_events()
