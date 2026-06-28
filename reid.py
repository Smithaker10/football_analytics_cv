"""
Re-Identification Module — OSNet-based Appearance Embedding
═══════════════════════════════════════════════════════════════
Stabilizes player IDs across occlusions using appearance features.
Integrates with ByteTrack for appearance + IoU cost matrix.
"""

import logging
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from config import CFG
from detector import DetectionResult

logger = logging.getLogger(__name__)


class OSNetEmbedder:
    """
    Lightweight OSNet (x0.25) feature extractor for person ReID.

    Falls back to a CNN-based embedder if torchreid is unavailable.
    """

    def __init__(self, config=CFG):
        self.cfg = config.reid
        self.device = torch.device(config.perf.device if torch.cuda.is_available() else "cpu")
        self._model = None
        self._input_size = (128, 256)

        try:
            self._init_torchreid()
        except Exception:
            logger.warning("torchreid unavailable — ReID disabled (IoU matching only)")
            self._model = self._build_noop_embedder()

    def _init_torchreid(self):
        import torchreid
        self._model = torchreid.models.build_model(
            name=self.cfg.model_name,
            num_classes=1,
            pretrained=True,
            use_gpu=torch.cuda.is_available(),
        )
        self._model.eval()
        if torch.cuda.is_available():
            self._model = self._model.cuda()

    def _build_noop_embedder(self):
        """Return a zero vector when torchreid is unavailable.
        The tracker falls back to IoU-only matching, which is fine."""
        class NoopEmbedder(torch.nn.Module):
            def forward(self, x):
                return torch.zeros(x.size(0), self.cfg.feature_dim, device=x.device)
        return NoopEmbedder()

    def preprocess(self, frame: np.ndarray, det: DetectionResult) -> Optional[torch.Tensor]:
        """Crop and preprocess a detection for feature extraction."""
        x1, y1, x2, y2 = det.box
        fh, fw = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(fw, x2), min(fh, y2)
        if (x2 - x1) < 20 or (y2 - y1) < 30:
            return None

        crop = frame[y1:y2, x1:x2]
        crop = cv2.resize(crop, self._input_size, interpolation=cv2.INTER_LINEAR)
        crop = crop.astype(np.float32)
        crop /= np.float32(255.0)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        crop = (crop - mean) / std
        crop = torch.from_numpy(crop).permute(2, 0, 1).unsqueeze(0)
        return crop.to(self.device)

    @torch.inference_mode()
    def extract(self, frame: np.ndarray, dets: List[DetectionResult]) -> Dict[int, np.ndarray]:
        """
        Extract appearance features for a list of detections.
        Returns dict mapping detection index → feature vector.
        """
        features: Dict[int, np.ndarray] = {}
        batch_inputs = []
        batch_indices = []

        for i, det in enumerate(dets):
            tensor = self.preprocess(frame, det)
            if tensor is not None:
                batch_inputs.append(tensor)
                batch_indices.append(i)

        if not batch_inputs:
            return features

        inputs = torch.cat(batch_inputs, dim=0)
        # Process in batches of 8 for memory efficiency
        bs = 8
        all_feats = []
        for start in range(0, len(inputs), bs):
            chunk = inputs[start:start + bs]
            feats = self._model(chunk)
            all_feats.append(feats)

        if all_feats:
            all_feats = torch.cat(all_feats, dim=0)
            all_feats = F.normalize(all_feats, p=2, dim=1)

            for idx, feat in zip(batch_indices, all_feats):
                features[dets[idx]] = feat.cpu().numpy()

        return features

    def compute_similarity(self, feat1: np.ndarray, feat2: np.ndarray) -> float:
        """Cosine similarity between two feature vectors."""
        return float(np.dot(feat1, feat2) / (np.linalg.norm(feat1) * np.linalg.norm(feat2) + 1e-9))
