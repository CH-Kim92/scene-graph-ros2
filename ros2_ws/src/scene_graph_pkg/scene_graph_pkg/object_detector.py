"""
object_detector.py
──────────────────
YOLOv8-based 2D object detection fused with L515 depth data
to produce 3D-localized detections with semantic labels.

Outputs per-detection:
  - label          : COCO class name
  - confidence     : float [0,1]
  - bbox_2d        : [x1, y1, x2, y2] pixels
  - position_3d    : [x, y, z] meters in camera frame
  - bbox_3d        : axis-aligned 3D bounding box corners
  - color          : mean RGB inside bbox (for node colouring)
  - mask_points    : sampled point cloud inside bbox (Nx3)
"""

from __future__ import annotations
import numpy as np
import cv2
import torch
from ultralytics import YOLO
from dataclasses import dataclass, field
from typing import List, Optional


# ── Detection result container ────────────────────────────────────────────────

@dataclass
class Detection3D:
    label: str
    confidence: float
    class_id: int
    bbox_2d: List[int]             # [x1,y1,x2,y2]
    position_3d: List[float]       # [x,y,z] metres, camera frame
    bbox_3d_min: List[float]       # [xmin, ymin, zmin]
    bbox_3d_max: List[float]       # [xmax, ymax, zmax]
    color: List[int]               # [R,G,B] 0-255
    mask_points: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))

    @property
    def node_id(self) -> str:
        return f"{self.label}_{self.class_id}_{int(self.position_3d[2]*100)}"


# ── Detector ──────────────────────────────────────────────────────────────────

class ObjectDetector3D:
    """
    Wraps YOLOv8 and fuses detections with aligned depth from RealSense L515.
    """

    # Per-class colour palette for graph nodes
    CLASS_COLORS = {
        "person":       (255, 80,  80),
        "chair":        (80,  200, 80),
        "laptop":       (80,  80,  255),
        "cup":          (255, 200, 80),
        "bottle":       (200, 80,  255),
        "book":         (80,  200, 255),
        "keyboard":     (255, 160, 80),
        "mouse":        (160, 255, 80),
        "monitor":      (80,  160, 255),
        "desk":         (200, 200, 80),
        "phone":        (255, 80,  200),
    }
    DEFAULT_COLOR = (160, 160, 160)

    def __init__(
        self,
        model_path: str = "yolov8m.pt",
        confidence_threshold: float = 0.45,
        depth_sample_stride: int = 4,
        depth_min_m: float = 0.1,
        depth_max_m: float = 6.0,
    ):
        self.conf_thresh = confidence_threshold
        self.depth_stride = depth_sample_stride
        self.depth_min = depth_min_m
        self.depth_max = depth_max_m

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[ObjectDetector3D] Loading YOLOv8 on {self.device} ...")
        self.model = YOLO(model_path)
        self.model.to(self.device)
        print(f"[ObjectDetector3D] Ready. {len(self.model.names)} classes.")

    # ── Public API ────────────────────────────────────────────────────────────

    def detect(
        self,
        rgb_image: np.ndarray,         # H×W×3  uint8
        depth_image: np.ndarray,       # H×W    float32 [metres]
        intrinsics: dict,              # {fx, fy, cx, cy}
    ) -> List[Detection3D]:
        """Run detection + 3-D fusion. Returns list of Detection3D."""

        h, w = rgb_image.shape[:2]
        results = self.model(
             rgb_image,
             conf=self.conf_thresh,
             verbose=False,
             device=self.device,
         )

        detections: List[Detection3D] = []
        for r in results:
            for box in r.boxes:
                det = self._process_box(box, rgb_image, depth_image, intrinsics, h, w)
                if det is not None:
                    detections.append(det)

        # Non-maximum suppression in 3-D (remove duplicate objects)
        detections = self._nms_3d(detections, iou_thresh_3d=0.5)
        return detections

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _process_box(self, box, rgb, depth, intr, H, W) -> Optional[Detection3D]:
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W - 1, x2), min(H - 1, y2)

        if x2 <= x1 or y2 <= y1:
            return None

        conf  = float(box.conf[0])
        cls   = int(box.cls[0])
        label = self.model.names[cls]

        roi_depth = depth[y1:y2, x1:x2]
        valid_mask = (roi_depth > self.depth_min) & (roi_depth < self.depth_max)
        valid_depths = roi_depth[valid_mask]

        if len(valid_depths) < 10:
            return None

        # Robust depth estimate – lower 30th percentile (closest surface)
        z = float(np.percentile(valid_depths, 30))

        fx, fy = intr['fx'], intr['fy']
        cx, cy = intr['cx'], intr['cy']

        cx_2d = (x1 + x2) / 2.0
        cy_2d = (y1 + y2) / 2.0
        x3d = (cx_2d - cx) * z / fx
        y3d = (cy_2d - cy) * z / fy

        # Build per-ROI 3-D point cloud (sampled)
        pts_3d = self._roi_to_3d(roi_depth, x1, y1, intr, valid_mask)

        # 3-D bounding box
        if len(pts_3d):
            bb_min = pts_3d.min(axis=0).tolist()
            bb_max = pts_3d.max(axis=0).tolist()
        else:
            bb_min = [x3d - 0.1, y3d - 0.1, z - 0.05]
            bb_max = [x3d + 0.1, y3d + 0.1, z + 0.05]

        # Mean colour inside bbox
        roi_rgb = rgb[y1:y2, x1:x2]
        color = tuple(int(c) for c in roi_rgb.mean(axis=(0, 1)))

        # Override with class colour if available
        class_color = self.CLASS_COLORS.get(label, self.DEFAULT_COLOR)

        return Detection3D(
            label=label,
            confidence=conf,
            class_id=cls,
            bbox_2d=[x1, y1, x2, y2],
            position_3d=[round(x3d, 4), round(y3d, 4), round(z, 4)],
            bbox_3d_min=bb_min,
            bbox_3d_max=bb_max,
            color=list(class_color),
            mask_points=pts_3d,
        )

    def _roi_to_3d(
        self,
        roi_depth: np.ndarray,
        x_off: int,
        y_off: int,
        intr: dict,
        valid_mask: np.ndarray,
    ) -> np.ndarray:
        fx, fy = intr['fx'], intr['fy']
        cx, cy = intr['cx'], intr['cy']
        s = self.depth_stride

        # Vectorised projection (fast)
        h, w = roi_depth.shape
        v_idx, u_idx = np.mgrid[0:h:s, 0:w:s]
        z_vals = roi_depth[v_idx, u_idx]
        mask   = valid_mask[v_idx, u_idx]

        v_idx, u_idx, z_vals = v_idx[mask], u_idx[mask], z_vals[mask]
        if len(z_vals) == 0:
            return np.empty((0, 3), dtype=np.float32)

        x_vals = (u_idx + x_off - cx) * z_vals / fx
        y_vals = (v_idx + y_off - cy) * z_vals / fy
        return np.stack([x_vals, y_vals, z_vals], axis=1).astype(np.float32)

    @staticmethod
    def _nms_3d(dets: List[Detection3D], iou_thresh_3d: float) -> List[Detection3D]:
        """Remove detections whose 3-D centroids are too close (same object)."""
        if len(dets) <= 1:
            return dets
        kept = []
        for d in sorted(dets, key=lambda x: -x.confidence):
            pos = np.array(d.position_3d)
            duplicate = False
            for k in kept:
                if k.label == d.label:
                    dist = np.linalg.norm(pos - np.array(k.position_3d))
                    if dist < 0.25:  # 25 cm – same object
                        duplicate = True
                        break
            if not duplicate:
                kept.append(d)
        return kept
