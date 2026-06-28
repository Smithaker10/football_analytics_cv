"""
Pose Estimation — YOLO11x-pose (COCO-17 skeleton)
═════════════════════════════════════════════════════
Replaces MediaPipe Pose with YOLO11x-pose for:
  • Faster GPU inference (single pass)
  • Native COCO-17 keypoints (no remapping needed)
  • Better lower-body detection
  • Built-in confidence scoring

Provides action classification:
  • Kicking (ankle above knee threshold)
  • Ball control (foot near ball)
  • Running (speed-based)
"""

import logging
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

from config import CFG

logger = logging.getLogger(__name__)


# COCO-17 skeleton connections
SKELETON_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),       # face
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # arms
    (5, 11), (6, 12), (11, 12),             # torso
    (11, 13), (13, 15), (12, 14), (14, 16),  # legs
]


class Keypoints:
    """COCO-17 keypoint container."""

    def __init__(self, data: np.ndarray):
        self.data = data  # (17, 3) = (x, y, conf)

    @property
    def xy(self) -> np.ndarray:
        return self.data[:, :2]

    @property
    def conf(self) -> np.ndarray:
        return self.data[:, 2]

    def __getitem__(self, idx) -> dict:
        return {"x": int(self.data[idx, 0]), "y": int(self.data[idx, 1]), "vis": float(self.data[idx, 2])}

    def __len__(self) -> int:
        return 17

    def __bool__(self) -> bool:
        return self.data is not None and len(self.data) == 17


class PoseDetector:
    """YOLO11x-pose based pose estimator."""

    def __init__(self, config=CFG):
        self.cfg = config.pose
        self.perf = config.perf
        from ultralytics import YOLO
        self.device = config.perf.device if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading pose model: {config.paths.yolo_pose_model}")
        self._model = YOLO(config.paths.yolo_pose_model, verbose=False)
        self._model.to(self.device)
        self._smooth: Dict[int, List[np.ndarray]] = {}

    @torch.inference_mode()
    def detect(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[Keypoints]:
        """
        Detect COCO-17 keypoints for a single player bounding box.
        Uses YOLO11x-pose on the full frame (single-pass).
        """
        x1, y1, x2, y2 = bbox
        fh, fw = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(fw, x2), min(fh, y2)

        if (x2 - x1) < 20 or (y2 - y1) < 20:
            return None

        # Run pose on cropped region for speed
        crop = frame[y1:y2, x1:x2]
        results = self._model(
            crop,
            conf=self.cfg.conf_threshold,
            half=self.perf.half_precision,
            verbose=False,
        )[0]

        if results.keypoints is None or len(results.keypoints.data) == 0:
            return None

        kp_data = results.keypoints.data[0].cpu().numpy()  # (17, 3)
        # Offset keypoints to original frame coordinates
        kp_data[:, 0] += x1
        kp_data[:, 1] += y1
        return Keypoints(kp_data)

    def detect_batch(
        self, frame: np.ndarray, bboxes: List[Tuple[int, int, int, int]]
    ) -> Dict[int, Optional[Keypoints]]:
        """Detect pose for multiple bboxes. Uses single full-frame YOLO pose pass."""
        results = self._model(
            frame,
            conf=self.cfg.conf_threshold,
            half=self.perf.half_precision,
            verbose=False,
        )[0]

        if results.keypoints is None:
            return {i: None for i in range(len(bboxes))}

        all_kps = results.keypoints.data.cpu().numpy()  # (N, 17, 3)

        # Associate detected poses to requested bboxes via IoU
        det_boxes = results.boxes.xyxy.cpu().numpy() if results.boxes is not None else None
        if det_boxes is None or len(det_boxes) == 0:
            return {i: None for i in range(len(bboxes))}

        assigned = {}
        for i, bbox in enumerate(bboxes):
            best_iou, best_kp = -1, None
            for j, dbox in enumerate(det_boxes):
                iou = self._iou(bbox, tuple(map(int, dbox)))
                if iou > best_iou:
                    best_iou = iou
                    if j < len(all_kps):
                        best_kp = Keypoints(all_kps[j])
            assigned[i] = best_kp if best_iou > 0.3 else None
        return assigned

    def smooth(self, tid: int, kps: Keypoints, alpha: float = 0.6) -> Keypoints:
        """Temporal exponential smoothing."""
        if tid not in self._smooth:
            self._smooth[tid] = []
        buf = self._smooth[tid]
        buf.append(kps.data.copy())
        if len(buf) > 5:
            buf.pop(0)
        smoothed = buf[0].copy()
        for arr in buf[1:]:
            smoothed = smoothed * alpha + arr * (1 - alpha)
        return Keypoints(smoothed)

    @staticmethod
    def confidence(kps: Optional[Keypoints]) -> float:
        if kps is None:
            return 0.0
        vis = kps.conf[kps.conf > 0.05]
        return float(np.mean(vis)) if len(vis) > 0 else 0.0

    @staticmethod
    def has_lower_body(kps: Optional[Keypoints]) -> bool:
        """
        Require ≥4 of 6 lower-body keypoints (hips, knees, ankles).
        Prevents face-only skeleton on close-up crops.
        """
        if kps is None or len(kps) != 17:
            return False
        lower_idx = [11, 12, 13, 14, 15, 16]
        visible = sum(1 for i in lower_idx if kps[i]["vis"] > 0.35)
        return visible >= 4

    @staticmethod
    def classify_action(
        kps: Optional[Keypoints],
        ball_pos: Optional[Tuple[int, int]],
        speed_kmh: float,
        cfg=CFG.pose,
    ) -> str:
        """Classify player action from pose keypoints."""
        if kps is None or len(kps) != 17:
            return "idle"

        lkne, rkne = kps[cfg.idx_lkne], kps[cfg.idx_rkne]
        lank, rank = kps[cfg.idx_lank], kps[cfg.idx_rank]

        lkick = (
            lank["vis"] > cfg.kp_visibility
            and lkne["vis"] > cfg.kp_visibility
            and lank["y"] < lkne["y"] - cfg.kick_px
        )
        rkick = (
            rank["vis"] > cfg.kp_visibility
            and rkne["vis"] > cfg.kp_visibility
            and rank["y"] < rkne["y"] - cfg.kick_px
        )
        if lkick or rkick:
            return "kicking"

        if ball_pos:
            bx, by = ball_pos
            ld = np.hypot(lank["x"] - bx, lank["y"] - by)
            rd = np.hypot(rank["x"] - bx, rank["y"] - by)
            if min(ld, rd) < cfg.ball_px and (lank["vis"] > 0.25 or rank["vis"] > 0.25):
                return "ball_control"

        return "running" if speed_kmh > cfg.run_kmh else "idle"

    @staticmethod
    def _iou(box1: Tuple, box2: Tuple) -> float:
        xa = max(box1[0], box2[0])
        ya = max(box1[1], box2[1])
        xb = min(box1[2], box2[2])
        yb = min(box1[3], box2[3])
        inter = max(0, xb - xa) * max(0, yb - ya)
        if inter == 0:
            return 0.0
        a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        return inter / (a1 + a2 - inter + 1e-9)
