"""
Ball Tracker — Dedicated Football Tracking
════════════════════════════════════════════
Handles:
  • Fast ball movement
  • Motion blur
  • Temporary disappearance (occlusions / out-of-frame)
  • Velocity-based position prediction
  • Trajectory smoothing
  • Ball possession assignment
"""

import logging
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np

from config import CFG
from detector import DetectionResult

logger = logging.getLogger(__name__)


class BallTracker:
    """
    Dedicated ball tracker with Kalman-like velocity filtering.
    Maintains trajectory history for analytics and visualization.
    """

    def __init__(self, config=CFG):
        self.cfg = config.ball
        self.speed_cfg = config.speed

        self._pos: Optional[Tuple[int, int]] = None
        self._prev_pos: Optional[Tuple[int, int]] = None
        self._velocity: Tuple[float, float] = (0.0, 0.0)
        self._lost_count: int = 0
        self._trajectory: deque = deque(maxlen=self.cfg.trajectory_length)
        self._velocity_history: deque = deque(maxlen=self.cfg.velocity_smooth)
        self._conf_history: deque = deque(maxlen=5)

        # Possession state machine
        self._possession_state: Optional[str] = None
        self._possession_sustain: int = 0
        self._possession_sustain_frames: int = 10  # frames of same team before counting
        self._possession_uncertainty_px: int = 60  # if nearest player > 60px, treat as contested

    @property
    def position(self) -> Optional[Tuple[int, int]]:
        return self._pos

    @property
    def prev_position(self) -> Optional[Tuple[int, int]]:
        return self._prev_pos

    @property
    def trajectory(self) -> List[Tuple[int, int]]:
        return list(self._trajectory)

    @property
    def velocity(self) -> Tuple[float, float]:
        return self._velocity

    @property
    def speed(self) -> float:
        vx, vy = self._velocity
        return np.hypot(vx, vy)

    @property
    def is_lost(self) -> bool:
        return self._lost_count > 0

    @property
    def lost_frames(self) -> int:
        return self._lost_count

    def update(self, ball_dets: List[DetectionResult]) -> Optional[Tuple[int, int]]:
        """
        Update ball position from detections with velocity smoothing.

        Strategy:
          1. Best confidence detection → primary position
          2. If no detection, predict from velocity
          3. Apply velocity-based smoothing
          4. Clamp to reasonable movement bounds
        """
        best_pos: Optional[Tuple[int, int]] = None
        best_conf = 0.0
        best_size = 0

        for det in ball_dets:
            cx, cy = det.cx, det.cy
            size = max(det.width, det.height)
            if size > self.cfg.max_radius:
                continue
            if det.conf > best_conf:
                best_pos = (int(cx), int(cy))
                best_conf = det.conf
                best_size = size

        if best_pos is not None:
            self._prev_pos = self._pos
            self._pos = best_pos
            self._lost_count = 0
            self._conf_history.append(best_conf)

            # Update velocity
            if self._prev_pos is not None:
                dx = self._pos[0] - self._prev_pos[0]
                dy = self._pos[1] - self._prev_pos[1]
                self._velocity_history.append((dx, dy))
                # Smoothed velocity
                if len(self._velocity_history) >= 2:
                    vx = float(np.mean([v[0] for v in self._velocity_history]))
                    vy = float(np.mean([v[1] for v in self._velocity_history]))
                    self._velocity = (vx, vy)

            self._trajectory.append(self._pos)
        else:
            self._lost_count += 1
            # Predict position from velocity
            if self._pos is not None and self._lost_count <= self.cfg.max_disappear:
                vx, vy = self._velocity
                predicted = (int(self._pos[0] + vx), int(self._pos[1] + vy))
            elif self._pos is not None:
                predicted = self._pos
            else:
                return None

            # Clamp to frame (caller should update bounds)
            self._prev_pos = self._pos
            self._pos = predicted
            self._trajectory.append(self._pos)

        return self._pos

    def reset(self):
        """Reset ball tracker state (e.g., for new half)."""
        self._pos = None
        self._prev_pos = None
        self._velocity = (0.0, 0.0)
        self._lost_count = 0
        self._trajectory.clear()
        self._velocity_history.clear()
        self._conf_history.clear()
        self._possession_state = None
        self._possession_sustain = 0

    def compute_possession(
        self,
        ball_pos: Optional[Tuple[int, int]],
        tracks: Dict,
        near_radius: int = CFG.ball.near_player_radius,
    ) -> Optional[str]:
        """
        Determine which team has ball possession with temporal smoothing.
        Returns "team_a", "team_b", or None.

        State machine:
          NEUTRAL → same team nearest for N consecutive frames → counting possession
          COUNTING → team loses ball for N frames or contested → back to NEUTRAL
        """
        if ball_pos is None:
            self._possession_state = None
            self._possession_sustain = 0
            return None

        from tracker import Track

        best_d = near_radius + 1
        best_team = None
        bx, by = ball_pos

        for tid, track in tracks.items():
            if not isinstance(track, Track):
                continue
            d = np.hypot(track.cx - bx, track.cy - by)
            if d < best_d:
                best_d = d
                best_team = track.team

        # Contested or too far from ball — reset sustain
        if best_d > self._possession_uncertainty_px:
            self._possession_sustain = 0
            self._possession_state = None
            return None

        # Same team as current state
        if best_team == self._possession_state:
            self._possession_sustain += 1
        else:
            # Different team — start counting toward new possession
            self._possession_sustain = 1
            self._possession_state = best_team

        # Only return possession after sustain threshold
        if self._possession_sustain >= self._possession_sustain_frames:
            return best_team
        return None

    @staticmethod
    def compute_trajectory_stats(
        trajectory: List[Tuple[int, int]],
        homography: Optional['HomographyEstimator'] = None,
        fps: float = 25.0,
    ) -> dict:
        """Compute ball statistics from trajectory.

        Uses homography-based world-coordinate speeds when available.
        """
        if len(trajectory) < 2:
            return {}

        speeds = []
        distances = []

        for i in range(1, len(trajectory)):
            p1, p2 = trajectory[i - 1], trajectory[i]

            if homography is not None and homography.homography is not None:
                speed_kmh = homography.compute_speed(p1, p2, fps)
                distances.append(speed_kmh / (fps * 3.6))  # back to meters
            else:
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                d = np.hypot(dx, dy)
                distances.append(d)
                speed_kmh = d * fps * 3.6 * CFG.speed.p2m

            speeds.append(speed_kmh)

        return {
            "total_distance_px": float(np.sum(distances)),
            "avg_speed_kmh": float(np.mean(speeds)) if speeds else 0.0,
            "max_speed_kmh": float(np.max(speeds)) if speeds else 0.0,
            "trajectory_length": len(trajectory),
        }
