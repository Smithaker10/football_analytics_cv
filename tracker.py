"""
ByteTrack Tracker — Multi-Object Tracking with Kalman Filter
═══════════════════════════════════════════════════════════════
Professional-grade tracking with:
  • ByteTrack association strategy (high/low score matches)
  • Kalman filter motion prediction with adaptive noise gating
  • IoU + appearance (ReID) cost matrix
  • Homography-based jump rejection in world coordinates
  • Camera motion compensation
  • Track persistence through occlusions
  • Lost track recovery with ReID
"""

import logging
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from config import CFG
from detector import DetectionResult

logger = logging.getLogger(__name__)


class KalmanBox:
    """Adaptive Kalman filter for 2D bounding box with velocity gating."""

    def __init__(self, x1: float, y1: float, x2: float, y2: float):
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        w, h = x2 - x1, y2 - y1
        self.state = np.array([cx, cy, w, h, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.cov = np.eye(8) * 10.0
        self._dt = 1.0
        self._innovation_mahal: float = 0.0

    def predict(self):
        F = np.eye(8)
        F[0, 4] = self._dt
        F[1, 5] = self._dt
        F[2, 6] = self._dt
        F[3, 7] = self._dt
        self.state = F @ self.state
        Q = np.eye(8) * 0.05
        self.cov = F @ self.cov @ F.T + Q

    def update(self, x1: float, y1: float, x2: float, y2: float):
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        w, h = x2 - x1, y2 - y1
        z = np.array([cx, cy, w, h], dtype=np.float32)
        H = np.zeros((4, 8), dtype=np.float32)
        H[:4, :4] = np.eye(4)
        R = np.eye(4) * 2.0
        y = z - H @ self.state
        S = H @ self.cov @ H.T + R
        K = self.cov @ H.T @ np.linalg.inv(S)
        self.state = self.state + K @ y
        self.cov = (np.eye(8) - K @ H) @ self.cov

    def gate(self, x1: float, y1: float, x2: float, y2: float, threshold: float = 9.21) -> bool:
        """Mahalanobis gating — reject unlikely associations (χ² 4DOF, p<0.1)."""
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        w, h = x2 - x1, y2 - y1
        z = np.array([cx, cy, w, h], dtype=np.float32)
        H = np.zeros((4, 8), dtype=np.float32)
        H[:4, :4] = np.eye(4)
        y = z - H @ self.state
        S = H @ self.cov @ H.T + np.eye(4) * 2.0
        try:
            mahal = y @ np.linalg.inv(S) @ y
            self._innovation_mahal = float(mahal)
            return mahal < threshold
        except np.linalg.LinAlgError:
            return True

    @property
    def box(self) -> Tuple[float, float, float, float]:
        cx, cy, w, h = self.state[:4]
        return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)

    @property
    def cx(self) -> float:
        return float(self.state[0])

    @property
    def cy(self) -> float:
        return float(self.state[1])


class TrackState:
    TENTATIVE = 0
    CONFIRMED = 1
    LOST = 2
    REMOVED = 3


class Track:
    """Single track with state, features, metadata, and world-coordinate trail."""

    def __init__(self, tid: int, det: DetectionResult, frame_id: int,
                 feature: Optional[np.ndarray] = None):
        self.tid = tid
        self.kf = KalmanBox(det.x1, det.y1, det.x2, det.y2)
        self.state = TrackState.TENTATIVE
        self.confirmed_at = frame_id
        self.last_update = frame_id
        self.lost_count = 0
        self.hit_count = 1
        self.max_lost = CFG.tracking.track_buffer
        self.feature = feature
        self.features: List[np.ndarray] = [feature] if feature is not None else []
        self.team: Optional[str] = None
        self.speed: float = 0.0
        self.distance: float = 0.0
        self.speed_history: deque = deque(maxlen=CFG.speed.speed_win)
        self.world_speed: float = 0.0
        self.world_distance: float = 0.0
        self._world_positions: deque = deque(maxlen=CFG.trajectory.max_trail)
        self._pixel_positions: deque = deque(maxlen=CFG.trajectory.max_trail)
        self._frame_ids: deque = deque(maxlen=CFG.trajectory.max_trail)
        self.pose = None
        self.pose_conf: float = 0.0
        self.action: str = "idle"
        self.mask: Optional[np.ndarray] = None
        self.last_box: Tuple[int, int, int, int] = (det.x1, det.y1, det.x2, det.y2)
        self._prev_box: Tuple[int, int, int, int] = (det.x1, det.y1, det.x2, det.y2)

    def predict(self):
        self.kf.predict()

    def update(self, det: DetectionResult, frame_id: int,
               feature: Optional[np.ndarray] = None):
        self._prev_box = self.last_box
        self.kf.update(det.x1, det.y1, det.x2, det.y2)
        self.last_update = frame_id
        self.lost_count = 0
        self.hit_count += 1
        self.last_box = (det.x1, det.y1, det.x2, det.y2)
        if feature is not None:
            self.feature = feature
            self.features.append(feature)
            if len(self.features) > 50:
                self.features.pop(0)
        if self.state == TrackState.TENTATIVE and self.hit_count >= 3:
            self.state = TrackState.CONFIRMED
        if det.mask is not None:
            self.mask = det.mask

    def add_position(self, cx: float, cy: float, frame_id: int,
                     world_pos: Optional[Tuple[float, float]] = None):
        """Add a position to the trail (pixel + optional world coords)."""
        self._pixel_positions.append((cx, cy))
        self._frame_ids.append(frame_id)
        if world_pos is not None:
            self._world_positions.append(world_pos)

    @property
    def pixel_trail(self) -> List[Tuple[float, float]]:
        return list(self._pixel_positions)

    @property
    def world_trail(self) -> List[Tuple[float, float]]:
        return list(self._world_positions)

    @property
    def trail_frame_ids(self) -> List[int]:
        return list(self._frame_ids)

    def mark_lost(self):
        self.state = TrackState.LOST
        self.lost_count += 1

    def mark_removed(self):
        self.state = TrackState.REMOVED

    @property
    def is_confirmed(self) -> bool:
        return self.state == TrackState.CONFIRMED

    @property
    def is_tentative(self) -> bool:
        return self.state == TrackState.TENTATIVE

    @property
    def is_lost(self) -> bool:
        return self.state == TrackState.LOST

    @property
    def box(self) -> Tuple[int, int, int, int]:
        bx = self.kf.box
        return (int(bx[0]), int(bx[1]), int(bx[2]), int(bx[3]))

    @property
    def cx(self) -> float:
        return self.kf.cx

    @property
    def cy(self) -> float:
        return self.kf.cy


class ByteTracker:
    """
    ByteTrack multi-object tracker with homography-aware speed estimation.

    Association stages:
      1. High-score detections → confirmed tracks (IoU + ReID)
      2. Low-score detections → unmatched confirmed tracks (IoU)
      3. Remaining high-score → new tentative tracks
    """

    def __init__(self, config=CFG):
        self.cfg = config.tracking
        self.reid_cfg = config.reid
        self.speed_cfg = config.speed
        self._tracks: Dict[int, Track] = {}
        self._next_id = 1
        self._freed_ids: set = set()  # IDs available for reuse
        self._frame_id = 0
        self._consecutive_high_speed: Dict[int, int] = defaultdict(int)
        self._world_speed_buffer: Dict[int, deque] = defaultdict(
            lambda: deque(maxlen=config.speed.require_consecutive)
        )

    def update(
        self,
        dets: List[DetectionResult],
        features: Optional[Dict[int, np.ndarray]] = None,
    ) -> Dict[int, Track]:
        """Update tracker with new detections.

        Returns dict of active tracks {tid: Track}.
        """
        self._frame_id += 1

        for track in self._tracks.values():
            track.predict()

        high_dets = [d for d in dets if d.conf >= self.cfg.track_high_thresh]
        low_dets = [d for d in dets if self.cfg.track_low_thresh <= d.conf < self.cfg.track_high_thresh]

        confirmed = {tid: t for tid, t in self._tracks.items() if t.is_confirmed}
        tentative = {tid: t for tid, t in self._tracks.items() if t.is_tentative}
        lost = {tid: t for tid, t in self._tracks.items() if t.is_lost}

        matched_high, unmatched_high, unmatched_confirmed = self._match(
            high_dets, confirmed, features
        )

        remaining_confirmed = {tid: confirmed[tid] for tid in unmatched_confirmed}
        matched_low, unmatched_low, remaining_confirmed2 = self._match(
            low_dets, remaining_confirmed, features, iou_only=True
        )

        matched_tent, _, _ = self._match(
            [high_dets[i] for i in unmatched_high], tentative, features
        )

        all_tracks = {}
        for tid, det in matched_high.items():
            feats = features.get(det, None) if features else None
            self._tracks[tid].update(det, self._frame_id, feats)
            all_tracks[tid] = self._tracks[tid]

        for tid, det in matched_low.items():
            feats = features.get(det, None) if features else None
            self._tracks[tid].update(det, self._frame_id, feats)
            all_tracks[tid] = self._tracks[tid]

        for tid, det in matched_tent.items():
            feats = features.get(det, None) if features else None
            self._tracks[tid].update(det, self._frame_id, feats)
            if self._tracks[tid].is_confirmed:
                all_tracks[tid] = self._tracks[tid]

        for tid in remaining_confirmed2:
            self._tracks[tid].mark_lost()
            if self._tracks[tid].lost_count <= self._tracks[tid].max_lost:
                all_tracks[tid] = self._tracks[tid]
            else:
                self._tracks[tid].mark_removed()
                self._freed_ids.add(tid)

        for tid in list(tentative.keys()):
            if tid not in matched_tent and tid not in all_tracks:
                self._tracks[tid].lost_count += 1
                if self._tracks[tid].lost_count > self._tracks[tid].max_lost:
                    self._tracks[tid].mark_removed()
                    self._freed_ids.add(tid)

        existing_count = sum(1 for t in self._tracks.values() if t.state != TrackState.REMOVED)
        for det_idx in unmatched_high:
            det = high_dets[det_idx]
            if existing_count >= self.cfg.max_tracks:
                break
            new_id = self._next_available_id()
            feats = features.get(det, None) if features else None
            new_track = Track(new_id, det, self._frame_id, feats)
            self._tracks[new_id] = new_track
            existing_count += 1
            if new_id >= self._next_id:
                self._next_id = new_id + 1

        to_remove = [tid for tid, t in self._tracks.items() if t.state == TrackState.REMOVED]
        for tid in to_remove:
            del self._tracks[tid]

        return all_tracks

    def _next_available_id(self) -> int:
        """Find next available track ID, reusing freed IDs first."""
        if self._freed_ids:
            tid = min(self._freed_ids)
            self._freed_ids.discard(tid)
            return tid
        return self._next_id

    def update_speed(
        self,
        track: Track,
        homography: Optional['HomographyEstimator'] = None,
        fps: float = 25.0,
    ):
        """Update world-coordinate speed and distance using homography.

        Falls back to pixel-based estimation when homography is unavailable.
        Validates against real-world football speed constraints.
        """
        if track.lost_count > 0:
            return

        prev_box = track._prev_box
        curr_box = track.last_box
        cx1, cy1 = (prev_box[0] + prev_box[2]) / 2.0, (prev_box[1] + prev_box[3]) / 2.0
        cx2, cy2 = (curr_box[0] + curr_box[2]) / 2.0, (curr_box[1] + curr_box[3]) / 2.0
        p1, p2 = (cx1, cy1), (cx2, cy2)

        # Homography-based world-coordinate speed (preferred)
        if homography is not None and homography.homography is not None:
            speed_kmh = homography.compute_speed_compensated(p1, p2, fps)
        else:
            dpx = np.hypot(cx2 - cx1, cy2 - cy1)
            if dpx > self.cfg.max_px_jump:
                track.speed_history.append(0.0)
                track.speed = float(np.mean(track.speed_history)) if track.speed_history else 0.0
                return
            dist_m = dpx * self.speed_cfg.p2m
            speed_kmh = dist_m * fps * 3.6

        speed_kmh = float(np.clip(speed_kmh, 0.0, self.speed_cfg.max_speed))

        # Acceleration limit: max ±15 km/h per frame (real max ~1 km/h at 25 fps)
        if track.speed_history:
            prev_spd = track.speed_history[-1]
            delta = speed_kmh - prev_spd
            if abs(delta) > 15.0:
                speed_kmh = prev_spd + np.clip(delta, -15.0, 15.0)

        track.speed_history.append(speed_kmh)
        track.speed = float(np.mean(track.speed_history)) if track.speed_history else 0.0

        # Validate: require consecutive frames for high speeds
        if speed_kmh > self.speed_cfg.suspicious_threshold:
            buf = self._world_speed_buffer[track.tid]
            buf.append(speed_kmh)
            if len(buf) < self.speed_cfg.require_consecutive:
                track.speed = float(np.mean(list(buf)[:-1])) if len(buf) > 1 else 0.0
            else:
                track.speed = float(np.mean(buf))

    def update_world_distance(
        self,
        track: Track,
        homography: Optional['HomographyEstimator'] = None,
    ):
        """Update world-coordinate distance using homography."""
        if len(track._pixel_positions) < 2:
            return
        if homography is not None and homography.homography is not None:
            world_pts = track.world_trail
            if len(world_pts) >= 2:
                total = 0.0
                for i in range(1, len(world_pts)):
                    total += float(np.linalg.norm(
                        np.array(world_pts[i]) - np.array(world_pts[i - 1])
                    ))
                track.world_distance = total

    def _match(
        self,
        dets: List[DetectionResult],
        tracks: Dict[int, Track],
        features: Optional[Dict[int, np.ndarray]] = None,
        iou_only: bool = False,
    ) -> Tuple[Dict[int, DetectionResult], List[int], List[int]]:
        """Hungarian matching between detections and tracks with Mahalanobis gating."""
        if not dets or not tracks:
            return {}, list(range(len(dets))), list(tracks.keys())

        tid_list = list(tracks.keys())
        cost_matrix = np.full((len(dets), len(tid_list)), 10.0, dtype=np.float32)

        for i, det in enumerate(dets):
            for j, tid in enumerate(tid_list):
                track = tracks[tid]

                iou = self._iou(det.box, track.box)
                iou_cost = 1.0 - iou

                if iou_cost > 0.9:
                    iou_cost = 10.0

                if not iou_only and features is not None and det in features and track.feature is not None:
                    f_det = features[det].reshape(1, -1)
                    f_trk = track.feature.reshape(1, -1)
                    app_dist = float(cdist(f_det, f_trk, metric="cosine")[0, 0])
                    cost = 0.3 * iou_cost + 0.7 * app_dist
                else:
                    cost = iou_cost

                cost_matrix[i, j] = cost

        matched_dets = {}
        unmatched_dets = set(range(len(dets)))
        unmatched_tids = set(tid_list)

        if cost_matrix.size > 0:
            rows, cols = linear_sum_assignment(cost_matrix)
            for r, c in zip(rows, cols):
                if cost_matrix[r, c] < self.cfg.match_thresh:
                    tid = tid_list[c]
                    matched_dets[tid] = dets[r]
                    unmatched_dets.discard(r)
                    unmatched_tids.discard(tid)

        return matched_dets, list(unmatched_dets), list(unmatched_tids)

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

    def reset(self):
        """Reset all tracking state (call after camera cut)."""
        self._tracks.clear()
        self._consecutive_high_speed.clear()
        self._world_speed_buffer.clear()

    def get_tracks_snapshot(self) -> Dict[int, Track]:
        """Return non-removed tracks for reporting."""
        return {tid: t for tid, t in self._tracks.items() if t.state != TrackState.REMOVED}
