"""
object_detector.py
──────────────────
Open-vocabulary 3D object detection pipeline:
  1. Grounding DINO  – detect any object described in natural language
  2. SAM2            – precise segmentation mask per detection
  3. CLIP            – verify / re-classify each detection semantically
  4. Depth fusion    – unproject mask pixels → 3D centroid + bbox
"""

import os
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import cv2
from dataclasses import dataclass, field
from typing import List, Optional

# ── GroundingDINO ─────────────────────────────────────────────────────────────
from groundingdino.util.inference import load_model as gdino_load_model
from groundingdino.util.inference import predict as gdino_predict
from groundingdino.util import box_ops
import torchvision.transforms as T

# ── SAM2 ──────────────────────────────────────────────────────────────────────
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# ── CLIP ──────────────────────────────────────────────────────────────────────
import open_clip


# ── Data classes ─────────────────────────────────────────────────────────────
@dataclass
class Detection3D:
    label:       str
    confidence:  float
    bbox_2d:     List[int]          # [x1, y1, x2, y2] pixels
    position_3d: List[float]        # [x, y, z] metres in camera frame
    bbox_min:    List[float]        # 3D bbox min corner
    bbox_max:    List[float]        # 3D bbox max corner
    mask:        Optional[np.ndarray] = field(default=None, repr=False)
    color:       List[int] = field(default_factory=lambda: [100, 200, 255])
    clip_label:  str = ''
    clip_score:  float = 0.0


# ── Colour palette ────────────────────────────────────────────────────────────
_PALETTE = [
    [255,  80,  80], [ 80, 255,  80], [ 80,  80, 255],
    [255, 255,  80], [255,  80, 255], [ 80, 255, 255],
    [255, 160,  80], [160, 255,  80], [ 80, 160, 255],
    [200, 100, 200], [100, 200, 100], [200, 200, 100],
]


# ── Main detector class ───────────────────────────────────────────────────────
class ObjectDetector3D:
    """
    Parameters
    ----------
    text_prompt : str
        Comma-separated object names, e.g.
        "person, cup, laptop, chair, bottle"
    gdino_box_threshold   : float  (default 0.35)
    gdino_text_threshold  : float  (default 0.25)
    clip_verify           : bool   (default True)  run CLIP re-scoring
    clip_threshold        : float  (default 0.20)  min CLIP score to keep
    """

    GDINO_CFG     = "/weights/GroundingDINO_SwinT_OGC.py"
    GDINO_WEIGHTS = "/weights/groundingdino_swint_ogc.pth"
    SAM2_CFG      = "configs/sam2.1/sam2.1_hiera_l"
    SAM2_WEIGHTS  = "/weights/sam2.1_hiera_large.pt"
    CLIP_MODEL    = "ViT-B-32"
    CLIP_PRETRAIN = "openai"

    def __init__(
        self,
        text_prompt:          str   = "cup, laptop, chair, monitor",
        gdino_box_threshold:  float = 0.35,
        gdino_text_threshold: float = 0.25,
        clip_verify:          bool  = True,
        clip_threshold:       float = 0.20,
    ):
        self.text_prompt          = text_prompt
        self.gdino_box_threshold  = gdino_box_threshold
        self.gdino_text_threshold = gdino_text_threshold
        self.clip_verify          = clip_verify
        self.clip_threshold       = clip_threshold

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[ObjectDetector3D] Device: {self.device}")

        self._load_gdino()
        self._load_sam2()
        if clip_verify:
            self._load_clip()

        # Build CLIP text embeddings once
        self._clip_labels   = [l.strip() for l in text_prompt.split(",") if l.strip()]
        self._clip_text_emb = None
        if clip_verify:
            self._encode_clip_labels()

        print(f"[ObjectDetector3D] Ready. Prompt: '{text_prompt}'")

    # ── Model loaders ─────────────────────────────────────────────────────────

    def _load_gdino(self):
        print("[ObjectDetector3D] Loading GroundingDINO …")
        self.gdino = gdino_load_model(
            self.GDINO_CFG, self.GDINO_WEIGHTS, device=self.device
        )
        self._gdino_transform = T.Compose([
            T.ToPILImage(),
            T.Resize(800),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.gdino.eval()
        print("[ObjectDetector3D] GroundingDINO ready.")

    def _load_sam2(self):
        print("[ObjectDetector3D] Loading SAM2 …")
        sam2_model = build_sam2(
            self.SAM2_CFG, self.SAM2_WEIGHTS,
            device=self.device,
            apply_postprocessing=False,
        )
        self.sam2 = SAM2ImagePredictor(sam2_model)
        print("[ObjectDetector3D] SAM2 ready.")

    def _load_clip(self):
        print("[ObjectDetector3D] Loading CLIP …")
        self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
            self.CLIP_MODEL, pretrained=self.CLIP_PRETRAIN
        )
        self.clip_model = self.clip_model.to(self.device).eval()
        self.clip_tokenizer = open_clip.get_tokenizer(self.CLIP_MODEL)
        print("[ObjectDetector3D] CLIP ready.")

    def _encode_clip_labels(self):
        texts = self.clip_tokenizer(self._clip_labels).to(self.device)
        with torch.no_grad():
            self._clip_text_emb = self.clip_model.encode_text(texts)
            self._clip_text_emb /= self._clip_text_emb.norm(dim=-1, keepdim=True)

    # ── Update prompt at runtime ───────────────────────────────────────────────

    def update_prompt(self, new_prompt: str):
        self.text_prompt  = new_prompt
        self._clip_labels = [l.strip() for l in new_prompt.split(",") if l.strip()]
        if self.clip_verify:
            self._encode_clip_labels()
        print(f"[ObjectDetector3D] Prompt updated: '{new_prompt}'")

    # ── Main detect method ────────────────────────────────────────────────────

    def detect(
        self,
        rgb:        np.ndarray,   # H×W×3 uint8 RGB
        depth:      np.ndarray,   # H×W float32 metres
        intrinsics: dict,
    ) -> List[Detection3D]:

        h, w = rgb.shape[:2]

        # ── 1. GroundingDINO: open-vocab detection ────────────────────────────
        img_tensor = self._gdino_transform(rgb).to(self.device)

        with torch.no_grad():
            boxes, logits, phrases = gdino_predict(
                model=self.gdino,
                image=img_tensor,
                caption=self.text_prompt,
                box_threshold=self.gdino_box_threshold,
                text_threshold=self.gdino_text_threshold,
                device=self.device,
            )

        if len(boxes) == 0:
            return []

        # Convert normalised cxcywh → pixel xyxy
        boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.tensor(
            [w, h, w, h], dtype=torch.float32
        )
        boxes_np = boxes_xyxy.cpu().numpy().astype(int)
        scores   = logits.cpu().numpy()

        # ── 2. SAM2: segment each box ─────────────────────────────────────────
        self.sam2.set_image(rgb)
        masks_all = []
        for box in boxes_np:
            x1, y1, x2, y2 = box
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(w-1, x2); y2 = min(h-1, y2)
            try:
                m, _, _ = self.sam2.predict(
                    point_coords=None,
                    point_labels=None,
                    box=np.array([x1, y1, x2, y2])[None, :],
                    multimask_output=False,
                )
                masks_all.append(m[0].astype(bool))   # H×W bool
            except Exception:
                # Fallback: rectangular mask
                fallback = np.zeros((h, w), dtype=bool)
                fallback[y1:y2, x1:x2] = True
                masks_all.append(fallback)

        # ── 3. CLIP: verify / re-classify ─────────────────────────────────────
        clip_labels = phrases
        clip_scores = scores.tolist()

        if self.clip_verify and self._clip_text_emb is not None:
            clip_labels, clip_scores = self._clip_reclassify(
                rgb, boxes_np, phrases, scores
            )

        # ── 4. 3D fusion ──────────────────────────────────────────────────────
        detections = []
        for i, (box, mask, label, score) in enumerate(
            zip(boxes_np, masks_all, clip_labels, clip_scores)
        ):
            if score < self.clip_threshold and self.clip_verify:
                continue

            pos3d, bmin, bmax = self._unproject_mask(mask, depth, intrinsics)
            if pos3d is None:
                continue

            color = _PALETTE[i % len(_PALETTE)]
            x1, y1, x2, y2 = box

            detections.append(Detection3D(
                label=label,
                confidence=float(score),
                bbox_2d=[int(x1), int(y1), int(x2), int(y2)],
                position_3d=pos3d,
                bbox_min=bmin,
                bbox_max=bmax,
                mask=mask,
                color=color,
                clip_label=label,
                clip_score=float(score),
            ))

        return detections

    # ── CLIP re-classification ────────────────────────────────────────────────

    def _clip_reclassify(self, rgb, boxes_np, phrases, scores):
        out_labels = list(phrases)
        out_scores = list(scores)

        crops = []
        for box in boxes_np:
            x1, y1, x2, y2 = box
            x1 = max(0, x1); y1 = max(0, y1)
            crop = rgb[y1:y2, x1:x2]
            if crop.size == 0:
                crops.append(None)
                continue
            pil = self.clip_preprocess(
                __import__('PIL').Image.fromarray(crop)
            ).unsqueeze(0).to(self.device)
            crops.append(pil)

        for i, pil in enumerate(crops):
            if pil is None:
                continue
            try:
                with torch.no_grad():
                    img_emb = self.clip_model.encode_image(pil)
                    img_emb /= img_emb.norm(dim=-1, keepdim=True)
                    sims = (img_emb @ self._clip_text_emb.T).squeeze(0)
                    best_idx  = sims.argmax().item()
                    best_score = sims[best_idx].item()
                out_labels[i] = self._clip_labels[best_idx]
                out_scores[i] = best_score
            except Exception:
                pass

        return out_labels, out_scores

    # ── 3D unprojection ───────────────────────────────────────────────────────

    def _unproject_mask(self, mask, depth, intrinsics):
        fx = intrinsics['fx']; fy = intrinsics['fy']
        cx = intrinsics['cx']; cy = intrinsics['cy']

        ys, xs = np.where(mask)
        if len(xs) == 0:
            return None, None, None

        d = depth[ys, xs]
        valid = (d > 0.1) & (d < 8.0)
        if valid.sum() < 5:
            return None, None, None

        d  = d[valid]
        xs = xs[valid]; ys = ys[valid]

        X = (xs - cx) * d / fx
        Y = (ys - cy) * d / fy
        Z = d

        # Median centroid (robust to noise)
        pos3d = [float(np.median(X)), float(np.median(Y)), float(np.median(Z))]
        bmin  = [float(X.min()), float(Y.min()), float(Z.min())]
        bmax  = [float(X.max()), float(Y.max()), float(Z.max())]

        return pos3d, bmin, bmax
