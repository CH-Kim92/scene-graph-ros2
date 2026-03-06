"""
scene_graph_builder.py
──────────────────────
Builds and maintains a 3-D scene graph from lists of Detection3D objects.

  Nodes: objects detected in the scene
    - id, label, position_3d, confidence, color, bbox_3d

  Edges: directed spatial relationships
    - near / above / below / left_of / right_of / in_front_of / behind
    - distance (metres)

The graph is persisted across frames with temporal smoothing so that
objects do not flicker in/out of the visualisation.
"""

from __future__ import annotations
import time
import numpy as np
import networkx as nx
from typing import List, Dict, Any, Tuple
from scene_graph_pkg.object_detector import Detection3D


# ── Spatial relationship catalogue ────────────────────────────────────────────

RELATIONS = [
    "near", "above", "below",
    "left_of", "right_of",
    "in_front_of", "behind",
    "on_top_of",        # above + overlapping XZ footprint
]

RELATION_COLORS = {
    "near":        "#64ffda",
    "above":       "#ff6d00",
    "below":       "#ff6d00",
    "left_of":     "#40c4ff",
    "right_of":    "#40c4ff",
    "in_front_of": "#ea80fc",
    "behind":      "#ea80fc",
    "on_top_of":   "#ffff00",
}


# ── Graph ─────────────────────────────────────────────────────────────────────

class SceneGraph3D:
    """
    Maintains a 3-D directed scene graph that is updated every perception cycle.

    Parameters
    ----------
    near_threshold      : float  – max distance (m) to create a 'near' edge
    above_min_dy        : float  – min vertical separation (m) for above/below
    overlap_xy_thresh   : float  – min XZ overlap fraction for 'on_top_of'
    decay_frames        : int    – how many missed frames before a node is removed
    """

    def __init__(
        self,
        near_threshold: float = 0.6,
        above_min_dy: float = 0.10,
        overlap_xy_thresh: float = 0.20,
        decay_frames: int = 10,
    ):
        self.near_threshold    = near_threshold
        self.above_min_dy      = above_min_dy
        self.overlap_xy_thresh = overlap_xy_thresh
        self.decay_frames      = decay_frames

        self.G: nx.DiGraph = nx.DiGraph()
        self._node_miss: Dict[str, int] = {}   # frames since last seen
        self._frame_id = 0

    # ── Public ────────────────────────────────────────────────────────────────

    def update(self, detections: List[Detection3D]) -> nx.DiGraph:
        """Update graph with current-frame detections. Returns updated graph."""
        self._frame_id += 1
        detected_ids = set()

        # ── 1. Upsert nodes ───────────────────────────────────────────────────
        for det in detections:
            nid = self._make_node_id(det)
            detected_ids.add(nid)
            self._upsert_node(nid, det)
            self._node_miss[nid] = 0

        # ── 2. Decay / remove unseen nodes ────────────────────────────────────
        for nid in list(self._node_miss.keys()):
            if nid not in detected_ids:
                self._node_miss[nid] = self._node_miss.get(nid, 0) + 1
                if self._node_miss[nid] > self.decay_frames:
                    if self.G.has_node(nid):
                        self.G.remove_node(nid)
                    del self._node_miss[nid]

        # ── 3. Rebuild edges ──────────────────────────────────────────────────
        self.G.remove_edges_from(list(self.G.edges()))
        nodes = [(n, d) for n, d in self.G.nodes(data=True)]

        for i, (ni, di) in enumerate(nodes):
            for j, (nj, dj) in enumerate(nodes):
                if i >= j:
                    continue
                self._compute_and_add_edges(ni, di, nj, dj)

        return self.G

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the graph to a JSON-friendly dict for the web visualiser."""
        nodes = []
        for nid, data in self.G.nodes(data=True):
            nodes.append({
                "id":         nid,
                "label":      data.get("label", ""),
                "position":   data.get("position", [0, 0, 0]),
                "confidence": round(data.get("confidence", 0.0), 3),
                "color":      data.get("color", [160, 160, 160]),
                "bbox_min":   data.get("bbox_min", [0, 0, 0]),
                "bbox_max":   data.get("bbox_max", [0, 0, 0]),
                "frame":      data.get("frame", 0),
            })

        edges = []
        for src, dst, data in self.G.edges(data=True):
            rel = data.get("relation", "near")
            edges.append({
                "source":   src,
                "target":   dst,
                "relation": rel,
                "distance": round(data.get("distance", 0.0), 3),
                "color":    RELATION_COLORS.get(rel, "#ffffff"),
            })

        return {
            "frame":    self._frame_id,
            "nodes":    nodes,
            "edges":    edges,
            "timestamp": time.time(),
        }

    def stats(self) -> str:
        return (f"SceneGraph | frame={self._frame_id} "
                f"nodes={self.G.number_of_nodes()} "
                f"edges={self.G.number_of_edges()}")

    # ── Internal ──────────────────────────────────────────────────────────────

    @staticmethod
    def _make_node_id(det: Detection3D) -> str:
        """
        Stable ID that groups the same object type at roughly the same location.
        Quantise position to 0.3 m grid to reduce jitter.
        """
        qx = round(det.position_3d[0] / 0.30)
        qz = round(det.position_3d[2] / 0.30)
        return f"{det.label}_{qx}_{qz}"

    def _upsert_node(self, nid: str, det: Detection3D) -> None:
        alpha = 0.7  # EMA smoothing for position
        if self.G.has_node(nid):
            old_pos = np.array(self.G.nodes[nid]["position"])
            new_pos = (alpha * old_pos +
                       (1 - alpha) * np.array(det.position_3d)).tolist()
        else:
            new_pos = det.position_3d

        self.G.add_node(
            nid,
            label=det.label,
            position=new_pos,
            confidence=det.confidence,
            color=det.color,
            bbox_min=det.bbox_3d_min,
            bbox_max=det.bbox_3d_max,
            frame=self._frame_id,
        )

    def _compute_and_add_edges(
        self,
        ni: str, di: Dict,
        nj: str, dj: Dict,
    ) -> None:
        pi = np.array(di["position"])
        pj = np.array(dj["position"])

        dx = pi[0] - pj[0]   # camera X  (left/right)
        dy = pi[1] - pj[1]   # camera Y  (up/down in image, +Y = down)
        dz = pi[2] - pj[2]   # depth Z

        dist_3d  = float(np.linalg.norm(pi - pj))
        dist_xz  = float(np.sqrt(dx**2 + dz**2))

        # near ─────────────────────────────────────────────────────────────────
        if dist_3d < self.near_threshold:
            self.G.add_edge(ni, nj, relation="near", distance=dist_3d)

        # above / below ────────────────────────────────────────────────────────
        if abs(dy) > self.above_min_dy:
            # In camera frame +Y points downward in image.
            # Physically, a lower Y in image → higher in real world.
            if dy > 0:   # i has larger Y → i is lower in image → i is physically below j
                self.G.add_edge(ni, nj, relation="below",  distance=abs(dy))
                self.G.add_edge(nj, ni, relation="above",  distance=abs(dy))
            else:
                self.G.add_edge(ni, nj, relation="above",  distance=abs(dy))
                self.G.add_edge(nj, ni, relation="below",  distance=abs(dy))

        # on_top_of  (above + close XZ footprint) ─────────────────────────────
        if abs(dy) > self.above_min_dy and dist_xz < self.overlap_xy_thresh:
            if dy < 0:
                self.G.add_edge(ni, nj, relation="on_top_of", distance=dist_xz)
            else:
                self.G.add_edge(nj, ni, relation="on_top_of", distance=dist_xz)

        # left_of / right_of ───────────────────────────────────────────────────
        if abs(dx) > 0.20:
            if dx > 0:
                self.G.add_edge(ni, nj, relation="right_of", distance=abs(dx))
            else:
                self.G.add_edge(ni, nj, relation="left_of",  distance=abs(dx))

        # in_front_of / behind ─────────────────────────────────────────────────
        if abs(dz) > 0.25:
            if dz < 0:   # i is closer to camera
                self.G.add_edge(ni, nj, relation="in_front_of", distance=abs(dz))
            else:
                self.G.add_edge(nj, ni, relation="in_front_of", distance=abs(dz))
