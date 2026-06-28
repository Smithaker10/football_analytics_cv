"""
Minimap — Bird's-Eye View Radar
═════════════════════════════════
Real-time minimap in top-right corner with:
  • Realistic field markings (penalty areas, center circle, halfway line)
  • Player positions color-coded by team
  • Player movement trails (world-coordinate when homography available)
  • Ball position indicator
  • Semi-transparent background for readability
"""

import logging
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from config import CFG

logger = logging.getLogger(__name__)


class Minimap:
    """
    Broadcast-style minimap radar overlay.

    Positioned in the top-right corner of the frame.
    Converts pixel coordinates to minimap space using aspect-ratio-preserving scaling,
    or uses homography-based world-to-minimap projection when available.
    """

    def __init__(self, config=CFG):
        self.cfg = config.viz
        self.colors = config.colors
        self._size = config.viz.minimap_size
        self._trail_cfg = config.trajectory

    def draw(
        self,
        canvas: np.ndarray,
        tracks: Dict,
        teams: Dict[int, str],
        W: int, H: int,
        ball_pos: Optional[Tuple[int, int]] = None,
        homography=None,
    ):
        """
        Draw minimap overlay on the canvas.

        Args:
            canvas: Frame to draw on (modified in-place)
            tracks: Dict of {tid: track_object} with .cx, .cy attributes
            teams: Dict of {tid: team_name}
            W, H: Frame dimensions
            ball_pos: Optional (x, y) of ball
            homography: Optional HomographyEstimator for pitch-based projection
        """
        sz = self._size
        mw, mh = sz, int(sz * 0.65)
        mx, my = W - mw - 12, 12

        # Background
        ov = canvas.copy()
        cv2.rectangle(ov, (mx, my), (mx + mw, my + mh), self.colors.bg, -1)
        cv2.addWeighted(ov, 0.92, canvas, 0.08, 0, canvas)
        cv2.rectangle(canvas, (mx, my), (mx + mw, my + mh), self.colors.cyan, 2)

        # Field markings
        self._draw_field_markings(canvas, mx, my, mw, mh)

        # Label
        self._txt(canvas, "FIELD", mx + 6, my + 13, 0.44, self.colors.cyan, 1)

        # World-coordinate minimap or pixel-based
        use_world = homography is not None and hasattr(homography, 'homography') and homography.homography is not None

        # Player trails (world-coordinate if available)
        if use_world and self._trail_cfg.minimap_trail:
            for tid, track in tracks.items():
                if not track.is_confirmed:
                    continue
                team = teams.get(tid, "unknown")
                color = self.colors.team_a if team == "team_a" else self.colors.team_b
                trail = track.world_trail if use_world else track.pixel_trail
                if len(trail) < 2:
                    continue
                trail_pts = []
                for wpt in trail:
                    if use_world:
                        # World coords: map 105x68 pitch to minimap
                        wx, wy = wpt
                        px = int(mx + (wx / 105.0) * mw)
                        py = int(my + (wy / 68.0) * mh)
                    else:
                        px = int(mx + (wpt[0] / W) * mw)
                        py = int(my + (wpt[1] / H) * mh)
                    px = np.clip(px, mx + 3, mx + mw - 3)
                    py = np.clip(py, my + 3, my + mh - 3)
                    trail_pts.append((px, py))

                for i in range(1, len(trail_pts)):
                    alpha = i / len(trail_pts)
                    thickness = max(1, int(2 * alpha))
                    fade = tuple(int(c * (0.2 + 0.6 * alpha)) for c in color)
                    cv2.line(canvas, trail_pts[i - 1], trail_pts[i],
                             fade, thickness, cv2.LINE_AA)

        # Player positions
        for tid, track in tracks.items():
            if not track.is_confirmed:
                continue
            if use_world and track.world_trail:
                last_w = track.world_trail[-1]
                wx, wy = last_w
                pdx = int(mx + (wx / 105.0) * mw)
                pdy = int(my + (wy / 68.0) * mh)
            else:
                pdx = int(mx + (track.cx / W) * mw)
                pdy = int(my + (track.cy / H) * mh)
            pdx = np.clip(pdx, mx + 4, mx + mw - 4)
            pdy = np.clip(pdy, my + 4, my + mh - 4)
            col = self.colors.team_a if teams.get(tid) == "team_a" else self.colors.team_b
            cv2.circle(canvas, (pdx, pdy), 5, col, -1, cv2.LINE_AA)
            cv2.circle(canvas, (pdx, pdy), 5, self.colors.white, 1, cv2.LINE_AA)

        # Ball
        if ball_pos is not None:
            if use_world:
                ball_w = homography.project_to_pitch(
                    np.array([[ball_pos[0], ball_pos[1]]], dtype=np.float32)
                )
                if ball_w is not None and len(ball_w) > 0:
                    bpx = int(mx + (ball_w[0, 0] / 105.0) * mw)
                    bpy = int(my + (ball_w[0, 1] / 68.0) * mh)
                else:
                    bpx = int(mx + (ball_pos[0] / W) * mw)
                    bpy = int(my + (ball_pos[1] / H) * mh)
            else:
                bpx = int(mx + (ball_pos[0] / W) * mw)
                bpy = int(my + (ball_pos[1] / H) * mh)
            bpx = np.clip(bpx, mx + 3, mx + mw - 3)
            bpy = np.clip(bpy, my + 3, my + mh - 3)
            cv2.circle(canvas, (bpx, bpy), 4, self.colors.ball, -1, cv2.LINE_AA)
            cv2.circle(canvas, (bpx, bpy), 4, self.colors.white, 1, cv2.LINE_AA)

    def _draw_field_markings(self, canvas: np.ndarray, mx: int, my: int, mw: int, mh: int):
        """Draw realistic football field markings on the minimap."""
        cdark = self.colors.cyan_dark

        cv2.rectangle(canvas, (mx + 3, my + 3), (mx + mw - 3, my + mh - 3), cdark, 1)

        cv2.line(canvas, (mx + mw // 2, my + 3), (mx + mw // 2, my + mh - 3), cdark, 1)

        cr = int(mw * 9.15 / 105)
        cv2.circle(canvas, (mx + mw // 2, my + mh // 2), cr, cdark, 1)
        cv2.circle(canvas, (mx + mw // 2, my + mh // 2), 2, cdark, -1)

        pw = int(mw * 20 / 105)
        ph = int(mh * 40 / 68)
        cx_center = mx + mw // 2
        cy_center = my + mh // 2

        cv2.rectangle(canvas,
                      (mx + 3, cy_center - ph // 2),
                      (mx + 3 + pw, cy_center + ph // 2),
                      cdark, 1)
        cv2.rectangle(canvas,
                      (mx + mw - 3 - pw, cy_center - ph // 2),
                      (mx + mw - 3, cy_center + ph // 2),
                      cdark, 1)

        gw = int(mw * 8 / 105)
        gh = int(mh * 18 / 68)
        cv2.rectangle(canvas,
                      (mx + 3, cy_center - gh // 2),
                      (mx + 3 + gw, cy_center + gh // 2),
                      cdark, 1)
        cv2.rectangle(canvas,
                      (mx + mw - 3 - gw, cy_center - gh // 2),
                      (mx + mw - 3, cy_center + gh // 2),
                      cdark, 1)

    def _txt(self, img: np.ndarray, text: str, x: int, y: int,
             scale: float = 0.55, color: Tuple[int, int, int] = (255, 255, 255),
             thick: int = 1):
        """Text with drop shadow — safe coordinate clipping."""
        h, w = img.shape[:2]
        x = int(np.clip(x, 0, w - 1))
        y = int(np.clip(y, 0, h - 1))
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_DUPLEX, scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_DUPLEX, scale, color, thick, cv2.LINE_AA)
