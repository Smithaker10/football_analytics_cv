"""
Team Classifier — Automatic Team Assignment
════════════════════════════════════════════
Assigns each detected player to:
  • Team A (home)
  • Team B (away)
  • Goalkeeper (detected via jersey color dissimilarity)
  • Referee (detected via dark/black uniform)

Uses HSV + LAB color features with KMeans clustering.
Temporal smoothing with confidence voting prevents flickering.
"""

import logging
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from config import CFG

logger = logging.getLogger(__name__)


class TeamClassifier:
    """
    Automatic team assignment using HSV/LAB color features.

    Calibration phase: first N frames collect features,
    then KMeans establishes team centroids.
    After calibration, assignments use nearest-centroid + temporal voting.
    """

    GOALKEEPER = "goalkeeper"
    REFEREE = "referee"
    TEAM_A = "team_a"
    TEAM_B = "team_b"
    UNKNOWN = "unknown"

    def __init__(self, config=CFG):
        self.cfg = config.team
        self._calibrated = False
        self._feature_buffer: List[Tuple[float, float, float]] = []
        self._centroid_a: Optional[np.ndarray] = None
        self._centroid_b: Optional[np.ndarray] = None
        self._centroid_gk: Optional[np.ndarray] = None
        self._centroid_ref: Optional[np.ndarray] = None
        self._history: Dict[int, List[str]] = defaultdict(list)

    @property
    def is_calibrated(self) -> bool:
        return self._calibrated

    def extract_features(self, frame: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> Optional[np.ndarray]:
        """
        Extract color features from player jersey region.
        Uses upper torso area (15%-60% height, 20%-80% width).
        Returns [white_score, blue_score, dark_score, sat_mean, hue_mean].
        """
        ty = int(y1 + (y2 - y1) * 0.15)
        by = int(y1 + (y2 - y1) * 0.60)
        tx = int(x1 + (x2 - x1) * 0.20)
        bx = int(x1 + (x2 - x1) * 0.80)

        if by <= ty or bx <= tx:
            return None

        crop = frame[ty:by, tx:bx]
        if crop.size < 120:
            return None

        # HSV features
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).astype(np.float32)
        H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]

        # LAB features
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).astype(np.float32)
        L, A, B = lab[..., 0], lab[..., 1], lab[..., 2]

        white_score = float(np.mean((V > 180) & (S < 40)))
        blue_score = float(np.mean((H > 90) & (H < 140) & (S > 60) & (V > 60)))
        red_score = float(np.mean(((H < 10) | (H > 170)) & (S > 80) & (V > 60)))
        dark_score = float(np.mean(V < 60))
        sat_mean = float(np.mean(S))
        val_mean = float(np.mean(V))
        hue_mean = float(np.mean(H))

        return np.array([
            white_score, blue_score, red_score, dark_score,
            sat_mean / 255.0, val_mean / 255.0, hue_mean / 180.0,
        ], dtype=np.float32)

    def _calibrate(self):
        """Run KMeans on accumulated features to establish team centroids."""
        if len(self._feature_buffer) < self.cfg.calibrate_samples:
            return

        data = np.array(self._feature_buffer, dtype=np.float32)

        # KMeans++ initialization
        n_clusters = min(4, len(data))
        centroids = data[np.random.choice(len(data), n_clusters, replace=False)]

        for _ in range(self.cfg.kmeans_iters):
            dists = np.array([
                np.linalg.norm(data - c, axis=1) for c in centroids
            ])
            labels = np.argmin(dists, axis=0)
            new_centroids = []
            for k in range(n_clusters):
                mask = labels == k
                if mask.sum() > 0:
                    new_centroids.append(data[mask].mean(axis=0))
                else:
                    new_centroids.append(centroids[k])
            centroids = np.array(new_centroids)
            n_clusters = centroids.shape[0]

        # Sort centroids by white_score descending to find teams
        white_scores = centroids[:, 0]
        blue_scores = centroids[:, 1]
        dark_scores = centroids[:, 3]

        # Higher white_score → team A (white jerseys)
        # Higher blue_score → team B (blue jerseys)
        # Higher dark_score → referees
        # Higher red_score → goalkeepers (or assign by exclusion)

        sorted_by_white = np.argsort(-white_scores)
        sorted_by_dark = np.argsort(-dark_scores)
        sorted_by_blue = np.argsort(-blue_scores)

        # Ref is darkest
        ref_idx = sorted_by_dark[0]

        # Among remaining, highest white = team A, highest blue = team B
        remaining = [i for i in range(n_clusters) if i != ref_idx]
        if len(remaining) >= 2:
            remaining_by_white = sorted(remaining, key=lambda i: -white_scores[i])
            team_a_idx = remaining_by_white[0]
            team_b_candidates = [i for i in remaining_by_white[1:] if i != team_a_idx]
            if team_b_candidates:
                team_b_idx = max(team_b_candidates, key=lambda i: blue_scores[i])
            else:
                team_b_idx = team_a_idx
        elif len(remaining) == 1:
            team_a_idx = remaining[0]
            team_b_idx = remaining[0]
        else:
            return

        if white_scores[team_a_idx] >= white_scores[team_b_idx]:
            self._centroid_a = centroids[team_a_idx]
            self._centroid_b = centroids[team_b_idx]
        else:
            self._centroid_a = centroids[team_b_idx]
            self._centroid_b = centroids[team_a_idx]

        self._centroid_ref = centroids[ref_idx]

        # Goalkeeper: if red_score > 0.3, assign separately
        gk_candidates = [i for i in range(n_clusters)
                         if i not in (team_a_idx, team_b_idx, ref_idx)]
        if gk_candidates:
            gk_idx = gk_candidates[0]
            self._centroid_gk = centroids[gk_idx]

        self._calibrated = True
        logger.info(
            f"Team calibrated: A={self._centroid_a[:3]} "
            f"B={self._centroid_b[:3]} "
            f"Ref={self._centroid_ref[:3]}"
        )

    def assign(
        self, frame: np.ndarray, x1: int, y1: int, x2: int, y2: int,
        tid: Optional[int] = None,
    ) -> str:
        """
        Assign a player detection to a team.

        Returns one of: 'team_a', 'team_b', 'goalkeeper', 'referee', 'unknown'.
        """
        feat = self.extract_features(frame, x1, y1, x2, y2)
        if feat is None:
            return self.TEAM_A

        if not self._calibrated:
            self._feature_buffer.append(feat)
            if len(self._feature_buffer) >= self.cfg.calibrate_samples:
                self._calibrate()
            # Pre-calibration: heuristic guess
            white_s, blue_s, red_s, dark_s, *_ = feat
            if dark_s > 0.4:
                return self.REFEREE
            if red_s > 0.25:
                return self.GOALKEEPER
            return self.TEAM_A if white_s >= blue_s else self.TEAM_B

        # Nearest-centroid assignment
        centroids = {
            self.TEAM_A: self._centroid_a,
            self.TEAM_B: self._centroid_b,
            self.REFEREE: self._centroid_ref,
        }
        if self._centroid_gk is not None:
            centroids[self.GOALKEEPER] = self._centroid_gk

        best_label = self.TEAM_A
        best_dist = float("inf")

        for label, centroid in centroids.items():
            if centroid is None:
                continue
            dist = float(np.linalg.norm(feat - centroid))
            if dist < best_dist:
                best_dist = dist
                best_label = label

        # Temporal smoothing
        if tid is not None:
            self._history[tid].append(best_label)
            if len(self._history[tid]) > self.cfg.temporal_smooth:
                self._history[tid].pop(0)
            if self.cfg.confidence_vote:
                counter = Counter(self._history[tid])
                most_common = counter.most_common(1)[0]
                if most_common[1] >= self.cfg.temporal_smooth // 2:
                    best_label = most_common[0]

        return best_label

    def get_team_colors(self) -> Dict[str, Tuple[int, int, int]]:
        """Return BGR colors for each team (for visualization)."""
        return {
            self.TEAM_A: (220, 220, 220),  # white
            self.TEAM_B: (255, 100, 50),   # blue
            self.REFEREE: (50, 50, 50),    # dark
            self.GOALKEEPER: (50, 255, 100),  # green
        }
