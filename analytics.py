"""
Analytics Engine — Match Statistics and Metrics
═════════════════════════════════════════════════
Computes all match analytics including:
  • Distance covered (per player, per team) in world meters
  • Speed statistics via homography-based world coordinates
  • Ball possession (team, individual)
  • Sprint detection
  • Heatmap accumulation
  • Kick zones
  • Player action distribution
"""

import logging
from collections import Counter, defaultdict, deque
from typing import Dict, List, Optional, Tuple

import numpy as np

from config import CFG

logger = logging.getLogger(__name__)


class AnalyticsEngine:
    """
    Core analytics engine accumulating match statistics frame by frame.
    All speed/distance calculations use homography-based world coordinates
    where available, falling back to pixel-based estimation.
    """

    def __init__(self, config=CFG):
        self.cfg = config

        # Per-player stats
        self._distances: Dict[int, float] = defaultdict(float)
        self._world_distances: Dict[int, float] = defaultdict(float)
        self._speeds: Dict[int, float] = defaultdict(float)
        self._world_speeds: Dict[int, float] = defaultdict(float)
        self._speed_history: Dict[int, deque] = defaultdict(
            lambda: deque(maxlen=config.speed.speed_win)
        )
        self._world_speed_history: Dict[int, deque] = defaultdict(
            lambda: deque(maxlen=config.speed.speed_win)
        )
        self._sprint_count: Dict[int, int] = defaultdict(int)
        self._action_counts: Dict[int, Counter] = defaultdict(Counter)
        self._pos_history: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
        self._world_pos_history: Dict[int, List[Tuple[float, float]]] = defaultdict(list)

        # Team stats
        self._team_distances: Dict[str, float] = defaultdict(float)
        self._team_sprints: Dict[str, int] = defaultdict(int)
        self._possession_frames: Counter = Counter()
        self._total_kicks: int = 0
        self._kick_positions: List[Tuple[int, int]] = []

        # Historical data
        self._avg_speed_history: List[float] = []
        self._activity_history: List[float] = []
        self._possession_history: List[Tuple[float, float]] = []

        # Per-frame tracking
        self._prev_pos: Dict[int, Tuple[float, float]] = {}
        self._is_sprinting: Dict[int, bool] = defaultdict(bool)
        self._sprint_frames: Dict[int, int] = defaultdict(int)

    def update(
        self,
        tid: int,
        team: str,
        cx: float,
        cy: float,
        action: str,
        ball_pos: Optional[Tuple[int, int]] = None,
        fps: float = 25.0,
        world_pos: Optional[Tuple[float, float]] = None,
        track_speed: Optional[float] = None,
    ):
        """Update all analytics for a single player in one frame.

        Args:
            tid: Track ID
            team: Team name
            cx, cy: Center pixel coordinates
            action: Current action string
            ball_pos: Ball position (optional)
            fps: Current frames per second
            world_pos: World-coordinate position in meters (optional, from homography)
            track_speed: Pre-computed speed from tracker (includes camera compensation)
        """
        max_speed = self.cfg.speed.max_speed

        if tid in self._prev_pos:
            px, py = self._prev_pos[tid]
            dpx = np.hypot(cx - px, cy - py)

            if track_speed is not None:
                # Use tracker's camera-compensated speed
                spd = track_speed
                if spd <= max_speed:
                    self._world_speed_history[tid].append(spd)
                    self._world_speeds[tid] = float(np.mean(self._world_speed_history[tid]))
                    # Distance: estimate from speed
                    world_dist = spd / (fps * 3.6)
                    self._world_distances[tid] += world_dist
                    self._team_distances[team] += world_dist
            elif world_pos is not None and self._world_pos_history.get(tid):
                # Homography-based world-coordinate distance
                prev_world = self._world_pos_history[tid][-1]
                world_dist = float(np.linalg.norm(
                    np.array(world_pos) - np.array(prev_world)
                ))
                spd = world_dist * fps * 3.6
                if spd <= max_speed:
                    self._world_distances[tid] += world_dist
                    self._team_distances[team] += world_dist
                    self._world_speed_history[tid].append(spd)
                    self._world_speeds[tid] = float(np.mean(self._world_speed_history[tid]))
            elif dpx <= 110:
                # Fallback: pixel-based
                p2m = self.cfg.speed.p2m
                dist_m = dpx * p2m
                spd = dist_m * fps * 3.6
                if spd <= max_speed:
                    self._distances[tid] += dist_m
                    self._team_distances[team] += dist_m
                else:
                    spd = 0.0
                self._speed_history[tid].append(spd)
                self._speeds[tid] = float(np.mean(self._speed_history[tid]))

                # Sprint detection
                if spd > 20.0:
                    self._sprint_frames[tid] += 1
                    if not self._is_sprinting[tid]:
                        self._is_sprinting[tid] = True
                else:
                    if self._is_sprinting[tid] and self._sprint_frames[tid] > 5:
                        self._sprint_count[tid] += 1
                        self._team_sprints[team] += 1
                    self._is_sprinting[tid] = False
                    self._sprint_frames[tid] = 0

        self._prev_pos[tid] = (cx, cy)
        self._pos_history[tid].append((int(cx), int(cy)))
        if world_pos is not None:
            self._world_pos_history[tid].append(world_pos)

        # Action tracking
        self._action_counts[tid][action] += 1

        # Kick tracking
        if action == "kicking":
            self._total_kicks += 1
            if ball_pos:
                self._kick_positions.append(ball_pos)

    def update_possession(self, team: str):
        """Increment possession counter for a team."""
        self._possession_frames[team] += 1

    def finalize_frame(self, avg_speed: float, activity: float):
        """Record end-of-frame statistics."""
        self._avg_speed_history.append(avg_speed)
        self._activity_history.append(activity)
        pa = self.possession_pct("team_a")
        pb = self.possession_pct("team_b")
        self._possession_history.append((pa, pb))

    # ── Player Queries ─────────────────────────────────────────────────────────

    def speed(self, tid: int) -> float:
        return self._world_speeds.get(tid, self._speeds.get(tid, 0.0))

    def distance(self, tid: int) -> float:
        return self._world_distances.get(tid, self._distances.get(tid, 0.0))

    def sprint_count(self, tid: int) -> int:
        return self._sprint_count.get(tid, 0)

    def top_speeds(self, n: int = 5) -> List[Tuple[int, float]]:
        combined = {}
        for tid in set(list(self._world_speeds.keys()) + list(self._speeds.keys())):
            combined[tid] = self.speed(tid)
        return sorted(combined.items(), key=lambda x: -x[1])[:n]

    def top_distances(self, n: int = 5) -> List[Tuple[int, float]]:
        combined = {}
        for tid in set(list(self._world_distances.keys()) + list(self._distances.keys())):
            combined[tid] = self.distance(tid)
        return sorted(combined.items(), key=lambda x: -x[1])[:n]

    # ── Team Queries ────────────────────────────────────────────────────────────

    def team_distance(self, team: str) -> float:
        return self._team_distances.get(team, 0.0)

    def possession_pct(self, team: str) -> float:
        total = sum(self._possession_frames.values())
        if total == 0:
            return 50.0
        return round(self._possession_frames.get(team, 0) / total * 100, 1)

    def possession(self) -> Tuple[float, float]:
        return (self.possession_pct("team_a"), self.possession_pct("team_b"))

    @property
    def total_kicks(self) -> int:
        return self._total_kicks

    @property
    def kick_positions(self) -> List[Tuple[int, int]]:
        return self._kick_positions

    # ── Match Queries ──────────────────────────────────────────────────────────

    @property
    def avg_speed_history(self) -> List[float]:
        return self._avg_speed_history

    @property
    def activity_history(self) -> List[float]:
        return self._activity_history

    @property
    def possession_history(self) -> List[Tuple[float, float]]:
        return self._possession_history

    def peak_speed(self) -> float:
        combined = {}
        for tid in set(list(self._world_speeds.keys()) + list(self._speeds.keys())):
            combined[tid] = self.speed(tid)
        return max(combined.values()) if combined else 0.0

    def average_speed(self) -> float:
        return float(np.mean(self._avg_speed_history)) if self._avg_speed_history else 0.0

    def action_distribution(self) -> Dict[str, int]:
        total = Counter()
        for tid_counter in self._action_counts.values():
            total += tid_counter
        return dict(total)

    def player_action_distribution(self, tid: int) -> Dict[str, int]:
        return dict(self._action_counts.get(tid, Counter()))

    # ── Heatmap / Position Data ───────────────────────────────────────────────

    def all_positions_by_team(self, teams: Dict[int, str]) -> Dict[str, List[Tuple[int, int]]]:
        """Group pixel position history by team."""
        result = {"team_a": [], "team_b": [], "referee": [], "goalkeeper": []}
        for tid, pos_list in self._pos_history.items():
            team = teams.get(tid, "unknown")
            if team in result:
                result[team].extend(pos_list)
        return result

    def all_world_positions_by_team(
        self, teams: Dict[int, str]
    ) -> Dict[str, List[Tuple[float, float]]]:
        """Group world-coordinate position history by team."""
        result = {"team_a": [], "team_b": [], "referee": [], "goalkeeper": []}
        for tid, pos_list in self._world_pos_history.items():
            team = teams.get(tid, "unknown")
            if team in result:
                result[team].extend(pos_list)
        return result

    # ── Sprint Stats ──────────────────────────────────────────────────────────

    def sprint_stats(self) -> Dict[str, int]:
        """Total sprints per team."""
        return dict(self._team_sprints)

    # ── Reset ──────────────────────────────────────────────────────────────────

    def reset(self):
        """Reset all analytics (e.g., after camera cut)."""
        self._distances.clear()
        self._world_distances.clear()
        self._speeds.clear()
        self._world_speeds.clear()
        self._speed_history.clear()
        self._world_speed_history.clear()
        self._sprint_count.clear()
        self._action_counts.clear()
        self._pos_history.clear()
        self._world_pos_history.clear()
        self._team_distances.clear()
        self._team_sprints.clear()
        self._possession_frames.clear()
        self._total_kicks = 0
        self._kick_positions.clear()
        self._avg_speed_history.clear()
        self._activity_history.clear()
        self._possession_history.clear()
        self._prev_pos.clear()
        self._is_sprinting.clear()
        self._sprint_frames.clear()
