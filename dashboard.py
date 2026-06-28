"""
Dashboard — Post-Match Visualization Reports
══════════════════════════════════════════════
Generates matplotlib-based:
  • Team position heatmaps
  • Kick zone heatmaps
  • Match summary dashboard
  • Speed / activity charts
"""

import logging
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import CFG

logger = logging.getLogger(__name__)


class DashboardGenerator:
    """
    Generates post-match visual reports using matplotlib.
    All outputs are saved as high-resolution PNG images.
    """

    def __init__(self, config=CFG):
        self.cfg = config
        self.ver = CFG.pid
        self.author = CFG.author

    def save_heatmap(
        self,
        positions_a: List[Tuple[int, int]],
        positions_b: List[Tuple[int, int]],
        shape: Tuple[int, int],
        path: str,
        kick_pts: Optional[List[Tuple[int, int]]] = None,
    ):
        """
        Generate team position density heatmaps with optional kick zones.
        """
        H, W = shape[:2]
        ncols = 3 if kick_pts else 2
        fig, axes = plt.subplots(1, ncols, figsize=(5.5 * ncols, 7), facecolor="#0a1528")

        pairs = [
            (positions_a, "TEAM A — POSITION DENSITY", "Greys"),
            (positions_b, "TEAM B — POSITION DENSITY", "Blues"),
        ]

        for ax, (pts, title, cmap) in zip(axes[:2], pairs):
            ax.set_facecolor("#08101e")
            hm = np.zeros((H // 4, W // 4), dtype=np.float32)
            for px, py in pts:
                xi, yi = int(px / 4), int(py / 4)
                if 0 <= xi < hm.shape[1] and 0 <= yi < hm.shape[0]:
                    hm[yi, xi] += 1
            hm = gaussian_filter(hm, sigma=9)
            if hm.max() > 0:
                hm /= hm.max()
            ax.imshow(hm, cmap=cmap, aspect="auto", origin="upper", interpolation="bilinear")
            ax.set_title(title, color="#00d4ff", fontsize=12, fontweight="bold", pad=10)
            ax.axis("off")

        if kick_pts and len(axes) > 2:
            ax = axes[2]
            ax.set_facecolor("#08101e")
            hm = np.zeros((H // 4, W // 4), dtype=np.float32)
            for px, py in kick_pts:
                xi, yi = int(px / 4), int(py / 4)
                if 0 <= xi < hm.shape[1] and 0 <= yi < hm.shape[0]:
                    hm[yi, xi] += 1
            hm = gaussian_filter(hm, sigma=11)
            if hm.max() > 0:
                hm /= hm.max()
            ax.imshow(hm, cmap="hot", aspect="auto", origin="upper", interpolation="bilinear")
            ax.set_title("KICK ZONES", color="#ff6644", fontsize=12, fontweight="bold", pad=10)
            ax.axis("off")

        fig.suptitle(
            f"Football Analytics v{self.ver} — Heatmaps  |  {self.author}",
            color="#00d4ff", fontsize=13, fontweight="bold",
        )
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close()
        logger.info(f"Heatmap saved: {path}")

    def save_dashboard(
        self,
        hist_spd: List[float],
        hist_act: List[float],
        analytics,
        tracker,
        n_frames: int,
        action_counts: Dict[str, int],
        path: str,
    ):
        """
        Generate comprehensive match dashboard with charts and summary table.
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 8), facecolor="#0a1528")
        for ax in axes.flat:
            ax.set_facecolor("#0a1220")
            for sp in ax.spines.values():
                sp.set_color("#1a2840")

        fig.suptitle(
            f"Football Analytics v{self.ver} — Dashboard  |  {self.author}",
            color="#00d4ff", fontsize=14, fontweight="bold", y=0.98,
        )

        # Speed chart
        fx = np.arange(len(hist_spd))
        ax = axes[0, 0]
        ax.fill_between(fx, hist_spd, alpha=0.30, color="#00d464")
        ax.plot(fx, hist_spd, color="#00d464", lw=1.5)
        ax.set_title("Avg Player Speed (km/h)", color="#00d4ff", fontsize=11)
        ax.set_ylabel("km/h", color="#6a8aaa")
        ax.tick_params(colors="#6a8aaa")

        # Activity chart
        ax = axes[0, 1]
        ax.fill_between(fx, hist_act, alpha=0.30, color="#ffa000")
        ax.plot(fx, hist_act, color="#ffa000", lw=1.5)
        ax.set_title("Match Activity Score", color="#00d4ff", fontsize=11)
        ax.set_ylabel("Score", color="#6a8aaa")
        ax.tick_params(colors="#6a8aaa")

        # Action distribution
        ax = axes[1, 0]
        if action_counts:
            lbls = list(action_counts.keys())
            vals = list(action_counts.values())
            bcolors = ["#FF4444", "#00EEFF", "#44FF88", "#888888"][:len(lbls)]
            bars = ax.bar(lbls, vals, color=bcolors, edgecolor="#0a1528", width=0.5)
            ax.set_title("Action Distribution (frames)", color="#00d4ff", fontsize=11)
            ax.tick_params(colors="#6a8aaa")
            for bar, v in zip(bars, vals):
                ax.text(
                    bar.get_x() + bar.get_width() / 2, v + max(vals) * 0.01,
                    str(v), ha="center", va="bottom", color="#deeeff", fontsize=9,
                )

        # Summary table
        ax = axes[1, 1]
        ax.axis("off")
        pa, pb = analytics.possession()
        peak_s = max(hist_spd) if hist_spd else 0
        avg_s = float(np.mean(hist_spd)) if hist_spd else 0
        td = sum(analytics._distances.values()) if hasattr(analytics, '_distances') else 0
        unique_ids = len(tracker._tracks) if hasattr(tracker, '_tracks') else 0

        rows = [
            ["Frames Processed", str(n_frames)],
            ["Avg Speed", f"{avg_s:.1f} km/h"],
            ["Peak Speed", f"{peak_s:.1f} km/h"],
            ["Team A Possession", f"{pa}%"],
            ["Team B Possession", f"{pb}%"],
            ["Total Distance", f"{td:.0f} m"],
            ["Total Kicks", str(analytics.total_kicks)],
        ]

        tbl = ax.table(
            cellText=rows, colLabels=["Metric", "Value"],
            loc="center", cellLoc="left",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        tbl.scale(1, 2.1)
        for r in range(len(rows) + 1):
            tbl[(r, 0)].set_facecolor("#0a2050")
            tbl[(r, 0)].set_text_props(color="#deeeff")
            tbl[(r, 1)].set_facecolor("#0a1535")
            tbl[(r, 1)].set_text_props(color="#deeeff")
        for c in range(2):
            tbl[(0, c)].set_facecolor("#00d4ff")
            tbl[(0, c)].set_text_props(weight="bold", color="#000")

        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close()
        logger.info(f"Dashboard saved: {path}")
