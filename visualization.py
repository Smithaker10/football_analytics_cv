"""
Visualization Module — Premium Broadcast Overlays
═══════════════════════════════════════════════════
All drawing functions for player overlays, HUD, and on-screen graphics.
Maintains the broadcast-style visual appearance from v6.1.
"""

import logging
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from config import CFG
from pose import SKELETON_CONNECTIONS

logger = logging.getLogger(__name__)


class Visualizer:
    """
    Professional broadcast-style overlay renderer.

    Features:
      • Minimal player bracket tags (no giant stat panels)
      • Segmentation silhouettes clamped to bounding box
      • COCO-17 skeleton with lower-body validation
      • Action pills (KICK / BALL only — no noise)
      • Velocity-based ball glow
      • Speed text below feet (only when moving)
    """

    def __init__(self, config=CFG):
        self.cfg = config.viz
        self.colors = config.colors
        self.pose_cfg = config.pose
        self.speed_cfg = config.speed
        self.font = cv2.FONT_HERSHEY_DUPLEX

    def draw_silhouette(
        self, canvas: np.ndarray, mask: Optional[np.ndarray],
        bbox: Tuple[int, int, int, int], team: str,
    ):
        """
        Semi-transparent silhouette clamped to bounding box.
        Morphological close fills holes, crisp contour edge.
        """
        if mask is None:
            return
        x1, y1, x2, y2 = bbox
        pad = 8
        clamped = np.zeros_like(mask)
        r1 = max(0, y1 - pad)
        r2 = min(mask.shape[0], y2 + pad)
        c1 = max(0, x1 - pad)
        c2 = min(mask.shape[1], x2 + pad)
        clamped[r1:r2, c1:c2] = mask[r1:r2, c1:c2]

        m_u8 = (clamped > 0.4).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        m_u8 = cv2.morphologyEx(m_u8, cv2.MORPH_CLOSE, kernel)
        if m_u8.max() == 0:
            return

        color = self.colors.team_a if team == "team_a" else self.colors.team_b
        ov = canvas.copy()
        ov[m_u8 > 0] = color
        cv2.addWeighted(ov, self.cfg.silhouette_alpha, canvas, 1 - self.cfg.silhouette_alpha, 0, canvas)

        contours, _ = cv2.findContours(m_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, contours, -1, color, 1, cv2.LINE_AA)

    def draw_skeleton(
        self, canvas: np.ndarray,
        kps: Optional[object],
        team: str, confidence: float,
    ):
        """
        COCO-17 skeleton with lower-body validation.
        Only draws when hips + knees + ankles are visible.
        """
        if kps is None or confidence < self.pose_cfg.skel_conf:
            return
        # Lower-body guard
        lower_idx = [11, 12, 13, 14, 15, 16]
        visible_lower = sum(1 for i in lower_idx if len(kps) > i and kps[i]["vis"] > self.pose_cfg.kp_visibility)
        if visible_lower < 4:
            return

        skel_col = self.colors.skeleton_a if team == "team_a" else self.colors.skeleton_b
        for s, e in SKELETON_CONNECTIONS:
            if s >= len(kps) or e >= len(kps):
                continue
            ks, ke = kps[s], kps[e]
            if ks["vis"] > self.pose_cfg.kp_visibility and ke["vis"] > self.pose_cfg.kp_visibility:
                cv2.line(canvas,
                         (int(ks["x"]), int(ks["y"])),
                         (int(ke["x"]), int(ke["y"])),
                         skel_col, 2, cv2.LINE_AA)
        for k in kps:
            if k["vis"] > self.pose_cfg.kp_visibility:
                pt = (int(k["x"]), int(k["y"]))
                cv2.circle(canvas, pt, 3, skel_col, -1, cv2.LINE_AA)
                cv2.circle(canvas, pt, 3, self.colors.white, 1, cv2.LINE_AA)

    def draw_player_tag(
        self, canvas: np.ndarray,
        x1: int, y1: int, x2: int, y2: int,
        tid: int, team: str, speed: float, has_pose: bool,
    ):
        """
        Minimal bracket corners + tiny ID tag + speed below feet.
        No dark rectangles — field stays visible.
        """
        color = self.colors.team_a if team == "team_a" else self.colors.team_b
        L = max(7, min(13, (x2 - x1) // 6))

        for px, py, dx, dy in [
            (x1, y1, 1, 1), (x2, y1, -1, 1),
            (x1, y2, 1, -1), (x2, y2, -1, -1),
        ]:
            cv2.line(canvas, (px, py), (px + dx * L, py), color, 2, cv2.LINE_AA)
            cv2.line(canvas, (px, py), (px, py + dy * L), color, 2, cv2.LINE_AA)

        self._txt(canvas, f"P{tid}", x1 + 2, y1 - 3, 0.38, color, 1)

        if has_pose:
            cv2.circle(canvas, (x2 + 4, y1 - 4), 4, self.colors.green, -1, cv2.LINE_AA)

        if speed >= self.speed_cfg.show_speed_at:
            cx_mid = (x1 + x2) // 2
            spd_col = self.colors.green if speed > 18 else (150, 255, 150)
            self._txt(canvas, f"{speed:.0f}", cx_mid - 9, y2 + 13, 0.40, spd_col, 1)

    def draw_action_pill(
        self, canvas: np.ndarray, x1: int, y1: int, x2: int, action: str,
    ):
        """Compact pill for KICK / BALL control only."""
        if action not in ("kicking", "ball_control"):
            return
        labels = {"kicking": "KICK", "ball_control": "BALL"}
        text = labels[action]
        colors_map = {"kicking": self.colors.action_kick, "ball_control": self.colors.action_ball}
        color = colors_map[action]
        cx = (x1 + x2) // 2
        tw = len(text) * 8 + 6
        tx, ty = cx - tw // 2, y1 - 9
        self._fill(canvas, tx - 3, ty - 13, tx + tw + 3, ty + 3, self.colors.bg, 0.82)
        self._txt(canvas, text, tx, ty, 0.50, color, 1)

    def draw_ball(self, canvas: np.ndarray, bx: int, by: int):
        """
        Multi-ring ball glow with white core.
        Decreasing opacity rings for motion feel.
        """
        for r, a in [(26, 28), (18, 60), (11, 130), (5, 255)]:
            ov = canvas.copy()
            cv2.circle(ov, (bx, by), r, self.colors.ball, -1, cv2.LINE_AA)
            cv2.addWeighted(ov, a / 255.0, canvas, 1 - a / 255.0, 0, canvas)
        cv2.circle(canvas, (bx, by), 7, self.colors.white, 2, cv2.LINE_AA)

    def draw_ball_trajectory(
        self, canvas: np.ndarray, trajectory: List[Tuple[int, int]],
    ):
        """Draw ball trajectory trail."""
        if len(trajectory) < 2:
            return
        for i in range(1, len(trajectory)):
            alpha = i / len(trajectory)
            thickness = max(1, int(4 * alpha))
            color = tuple(int(c * (0.3 + 0.7 * alpha)) for c in self.colors.ball)
            cv2.line(canvas, trajectory[i - 1], trajectory[i], color, thickness, cv2.LINE_AA)

    def draw_player_trail(
        self, canvas: np.ndarray,
        trail: List[Tuple[int, int]], team: str,
    ):
        """Draw player movement trail — only consecutive positions of confirmed tracks."""
        if len(trail) < 2:
            return
        color = self.colors.team_a if team == "team_a" else self.colors.team_b
        alpha_step = 1.0 / len(trail)
        for i in range(1, len(trail)):
            alpha = i * alpha_step
            thickness = max(1, int(3 * alpha))
            fade = tuple(int(c * (0.3 + 0.7 * alpha)) for c in color)
            pts = np.array([trail[i - 1], trail[i]], dtype=np.int32)
            cv2.line(canvas, tuple(pts[0]), tuple(pts[1]), fade, thickness, cv2.LINE_AA)

    def draw_field_borders(self, canvas: np.ndarray, W: int, H: int):
        """Draw subtle field boundary indicators."""
        top_line_y = int(H * CFG.detection.field_top)
        cv2.line(canvas, (0, top_line_y), (W, top_line_y),
                 (40, 60, 80), 1, cv2.LINE_AA)

    def _txt(self, img: np.ndarray, text: str, x: int, y: int,
             scale: float = 0.55, color: Tuple[int, int, int] = (255, 255, 255),
             thick: int = 1):
        """Text with drop shadow — safe coordinate clipping."""
        h, w = img.shape[:2]
        x = int(np.clip(x, 0, w - 1))
        y = int(np.clip(y, 0, h - 1))
        pos = (x, y)
        cv2.putText(img, text, pos, self.font, scale, self.colors.black, thick + 3, cv2.LINE_AA)
        cv2.putText(img, text, pos, self.font, scale, color, thick, cv2.LINE_AA)

    def _fill(self, canvas: np.ndarray, x1: int, y1: int, x2: int, y2: int,
              color: Tuple[int, int, int], alpha: float):
        """Semi-transparent filled rectangle — safe coordinate clipping."""
        h, w = canvas.shape[:2]
        x1 = int(np.clip(x1, 0, w - 1))
        y1 = int(np.clip(y1, 0, h - 1))
        x2 = int(np.clip(x2, x1 + 1, w))
        y2 = int(np.clip(y2, y1 + 1, h))
        ov = canvas.copy()
        cv2.rectangle(ov, (x1, y1), (x2, y2), color, -1)
        cv2.addWeighted(ov, alpha, canvas, 1 - alpha, 0, canvas)

    def _bar(self, canvas: np.ndarray, x: int, y: int, w: int, h: int,
             pct: float, col_fill: Tuple[int, int, int],
             col_bg: Tuple[int, int, int] = (30, 30, 50)):
        """Compact horizontal fill bar — safe coordinate clipping."""
        canvas_h, canvas_w = canvas.shape[:2]
        x = int(np.clip(x, 0, canvas_w - 1))
        y = int(np.clip(y, 0, canvas_h - 1))
        w = int(np.clip(w, 1, canvas_w - x))
        h = int(np.clip(h, 1, canvas_h - y))
        cv2.rectangle(canvas, (x, y), (x + w, y + h), col_bg, -1)
        fw = int(w * max(0.0, min(1.0, pct)))
        if fw > 0:
            cv2.rectangle(canvas, (x, y), (x + fw, y + h), col_fill, -1)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (55, 55, 75), 1)

    def draw_analytics_card(
        self, canvas: np.ndarray, pa: float, pb: float,
        avg_spd: float, peak_spd: float, n_tracks: int,
        kicks: int, top_speeds: List[Tuple[int, float]],
    ):
        """
        Compact analytics card in top-left corner.
        Single card replaces per-player stat panels from v6.0.
        """
        cw, ch = self.cfg.card_width, self.cfg.card_height
        cx, cy = self.cfg.card_x, self.cfg.card_y
        self._fill(canvas, cx, cy, cx + cw, cy + ch, self.colors.bg, 0.88)
        cv2.rectangle(canvas, (cx, cy), (cx + cw, cy + ch), self.colors.cyan, 1)

        self._fill(canvas, cx, cy, cx + cw, cy + 20, self.colors.cyan, 0.30)
        self._txt(canvas, "MATCH ANALYTICS", cx + 6, cy + 15, 0.45, self.colors.cyan, 1)
        cv2.line(canvas, (cx, cy + 20), (cx + cw, cy + 20), self.colors.cyan, 1)

        bw = 100
        bx = cx + 28
        self._txt(canvas, "A", cx + 6, cy + 35, 0.44, self.colors.team_a, 1)
        self._bar(canvas, bx, cy + 25, bw, 8, pa / 100.0, self.colors.team_a)
        self._txt(canvas, f"{pa:.0f}%", bx + bw + 4, cy + 35, 0.40, self.colors.team_a, 1)

        self._txt(canvas, "B", cx + 6, cy + 52, 0.44, self.colors.team_b, 1)
        self._bar(canvas, bx, cy + 42, bw, 8, pb / 100.0, self.colors.team_b)
        self._txt(canvas, f"{pb:.0f}%", bx + bw + 4, cy + 52, 0.40, self.colors.team_b, 1)

        cv2.line(canvas, (cx, cy + 60), (cx + cw, cy + 60), (40, 40, 60), 1)

        self._txt(canvas, f"AVG  {avg_spd:4.1f} km/h", cx + 6, cy + 76, 0.44, (180, 255, 180), 1)
        self._txt(canvas, f"PEAK {peak_spd:4.1f} km/h", cx + 6, cy + 92, 0.44, self.colors.green, 1)
        self._txt(canvas, f"ON FIELD {n_tracks:2d}  KICKS {kicks:2d}",
                  cx + 6, cy + 108, 0.40, (170, 170, 205), 1)

        cv2.line(canvas, (cx, cy + 115), (cx + cw, cy + 115), (40, 40, 60), 1)

        self._txt(canvas, "FASTEST", cx + 6, cy + 130, 0.43, self.colors.cyan, 1)
        for idx, (pid, spd) in enumerate(top_speeds[:3]):
            fy = cy + 142 + idx * 20
            bar = int(70 * min(spd, 40) / 40)
            cv2.rectangle(canvas, (cx + 50, fy), (cx + 50 + bar, fy + 7), self.colors.green, -1)
            cv2.rectangle(canvas, (cx + 50, fy), (cx + 50 + 70, fy + 7), (40, 40, 60), 1)
            self._txt(canvas, f"P{pid}", cx + 6, fy + 8, 0.40, (200, 200, 200), 1)
            self._txt(canvas, f"{spd:.0f}", cx + 126, fy + 8, 0.40, self.colors.green, 1)

    def draw_hud(
        self, canvas: np.ndarray, pa: float, pb: float,
        avg_spd: float, peak_spd: float, n_players: int,
        kicks: int, frame_idx: int, fps: float, W: int, H: int,
    ):
        """Bottom HUD bar with possession, stats, and clock."""
        hh = self.cfg.hud_height
        ys = H - hh
        self._fill(canvas, 0, ys, W, H, self.colors.bg, 0.90)
        cv2.line(canvas, (0, ys), (W, ys), self.colors.cyan, 2)

        bw = min(400, W // 3)
        bx = W // 2 - bw // 2
        by = ys + 10
        bht = 10
        self._bar(canvas, bx, by, bw, bht, pa / 100.0, self.colors.team_a)

        aw = int(bw * pa / 100.0)
        if aw < bw:
            cv2.rectangle(canvas, (bx + aw, by), (bx + bw, by + bht), self.colors.team_b, -1)
        cv2.rectangle(canvas, (bx, by), (bx + bw, by + bht), self.colors.cyan, 1)

        self._txt(canvas,
                  f"{pa:.0f}%  TEAM A  |  POSSESSION  |  TEAM B  {pb:.0f}%",
                  W // 2 - 185, ys + 29, 0.65, self.colors.cyan)

        secs = int(frame_idx / max(fps, 1))
        clock = f"{secs // 60:02d}:{secs % 60:02d}"
        self._txt(canvas,
                  f"AVG {avg_spd:.1f}   PEAK {peak_spd:.1f} km/h  |  {n_players} PLAYERS  |  KICKS {kicks}",
                  18, ys + 52, 0.58, self.colors.green)
        self._txt(canvas, f"CLOCK {clock}", W - 170, ys + 52, 0.56, (180, 180, 220))

        brand = f"Football Analytics v{CFG.pid}  •  YOLO11 + ByteTrack + ReID  |  {CFG.author}"
        self._txt(canvas, brand, W // 2 - 360, H - 4, 0.43, (80, 80, 110))
