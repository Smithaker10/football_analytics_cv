"""
Detection Module — YOLO11x-seg (single model for detection + segmentation)
════════════════════════════════════════════════════════════════════════════
Detects players, goalkeepers, referees, and football.
Applies crowd / false-positive filtering for broadcast football.
Uses a single YOLO11x-seg model for both detection and segmentation masks.
"""

import logging
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from config import CFG

logger = logging.getLogger(__name__)


class DetectionResult:
    """Single detection output."""

    __slots__ = ("x1", "y1", "x2", "y2", "conf", "cls_id", "cls_name", "mask")

    def __init__(
        self,
        x1: int, y1: int, x2: int, y2: int,
        conf: float,
        cls_id: int,
        cls_name: str,
        mask: Optional[np.ndarray] = None,
    ):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.conf = conf
        self.cls_id = cls_id
        self.cls_name = cls_name
        self.mask = mask

    @property
    def box(self) -> Tuple[int, int, int, int]:
        return (self.x1, self.y1, self.x2, self.y2)

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2.0

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    def __repr__(self) -> str:
        return f"Det({self.cls_name} conf={self.conf:.2f} box=({self.x1},{self.y1},{self.x2},{self.y2}))"

    def __hash__(self) -> int:
        return id(self)

    def __eq__(self, other) -> bool:
        return self is other


class Detector:
    """YOLO11x-seg based detector — single model for boxes + masks."""

    PERSON_CLASS = 0
    BALL_CLASS = 32

    def __init__(self, config=CFG):
        self.cfg = config.detection
        self.perf = config.perf
        self._device = config.perf.device if torch.cuda.is_available() else "cpu"

        logger.info(f"Loading model: {config.paths.yolo_seg_model}")
        self._model = YOLO(config.paths.yolo_seg_model, verbose=False)
        self._model.to(self._device)

    @torch.inference_mode()
    def infer(self, frame: np.ndarray) -> Tuple[List[DetectionResult], List[DetectionResult]]:
        """
        Run single model inference.
        Returns (player_detections, ball_detections).
        """
        fh, fw = frame.shape[:2]
        min_h_px = int(fh * self.cfg.min_h_frac)
        field_y = fh * self.cfg.field_top

        results = self._model(
            frame,
            conf=self.cfg.conf_threshold,
            iou=self.cfg.iou_threshold,
            half=self.perf.half_precision,
            verbose=False,
        )[0]

        boxes_data = results.boxes
        masks_data = results.masks

        players: List[DetectionResult] = []
        balls: List[DetectionResult] = []

        if boxes_data is None:
            return players, balls

        n_boxes = len(boxes_data)

        for i in range(n_boxes):
            cls_id = int(boxes_data.cls[i])
            cls_name = results.names[cls_id]
            x1, y1, x2, y2 = map(int, boxes_data.xyxy[i].tolist())
            conf = float(boxes_data.conf[i])
            bw, bh = x2 - x1, y2 - y1
            cy = (y1 + y2) / 2.0

            # Person detections
            if cls_id == self.PERSON_CLASS:
                if conf < self.cfg.min_player_conf:
                    continue
                if bh < min_h_px:
                    continue
                if bh / max(bw, 1) < self.cfg.min_aspect_ratio:
                    continue
                if cy < field_y:
                    continue

                mask: Optional[np.ndarray] = None
                if masks_data is not None and i < len(masks_data.data):
                    raw = masks_data.data[i].cpu().numpy()
                    mask = cv2.resize(raw, (fw, fh), interpolation=cv2.INTER_NEAREST)

                det = DetectionResult(x1, y1, x2, y2, conf, cls_id, cls_name, mask)
                players.append(det)
                continue

            # Ball detections
            if cls_id == self.BALL_CLASS:
                det = DetectionResult(x1, y1, x2, y2, conf, cls_id, cls_name)
                balls.append(det)
                continue

        # Cap max detections (avoid phantom tracks)
        if len(players) > self.cfg.max_detections:
            players.sort(key=lambda d: -d.conf)
            players = players[: self.cfg.max_detections]

        return players, balls

    def cleanup(self):
        """Release GPU memory."""
        del self._model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
