"""
Reports — Data Export and Interactive HTML Reports
═════════════════════════════════════════════════════
Generates:
  • JSON match report (machine-readable)
  • CSV player statistics table
  • Interactive HTML report with embedded visualizations
"""

import csv
import json
import logging
import os
from typing import Dict, List, Optional

from config import CFG

logger = logging.getLogger(__name__)


class ReportGenerator:
    """
    Generates post-match reports in multiple formats.
    """

    def __init__(self, config=CFG):
        self.cfg = config
        self.ver = CFG.pid
        self.author = CFG.author

    def save_json(
        self,
        path: str,
        video_path: str,
        fps: float,
        W: int, H: int,
        n_frames: int,
        analytics,
        teams: Dict[int, str],
        config_info: Optional[dict] = None,
    ):
        """
        Save machine-readable match report as JSON.
        """
        pa, pb = analytics.possession()
        data = {
            "version": self.ver,
            "video": video_path,
            "resolution": [W, H],
            "fps": fps,
            "frames_processed": n_frames,
            "duration_sec": round(n_frames / max(fps, 1), 2),
            "actions_detected": analytics.action_distribution(),
            "kicks_total": analytics.total_kicks,
            "possession": {"team_a": pa, "team_b": pb},
            "speed": {
                "average_kmh": round(analytics.average_speed(), 2),
                "peak_kmh": round(analytics.peak_speed(), 2),
            },
            "distance": {
                "team_a_m": round(analytics.team_distance("team_a"), 2),
                "team_b_m": round(analytics.team_distance("team_b"), 2),
            },
            "players": self._build_player_stats(analytics, teams),
        }
        if config_info:
            data["config"] = config_info

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.info(f"JSON report saved: {path}")

    def save_csv(
        self,
        path: str,
        analytics,
        teams: Dict[int, str],
    ):
        """
        Save per-player statistics as CSV.
        """
        rows = []
        # Collect all player IDs
        all_tids = set()
        for key in ('_distances', '_world_distances', '_speeds', '_world_speeds'):
            if hasattr(analytics, key):
                all_tids.update(getattr(analytics, key).keys())

        for tid in sorted(all_tids):
            peak = 0.0
            for hist_key in ('_speed_history', '_world_speed_history'):
                if hasattr(analytics, hist_key):
                    hist = getattr(analytics, hist_key).get(tid, [0])
                    if hist:
                        peak = max(peak, max(hist))
            rows.append({
                "player_id": f"P{tid}",
                "team": teams.get(tid, "unknown"),
                "distance_m": round(analytics.distance(tid), 2),
                "avg_speed_kmh": round(analytics.speed(tid), 2),
                "peak_speed_kmh": round(peak, 2),
                "sprints": analytics.sprint_count(tid),
            })

        if not rows:
            logger.warning("No player data for CSV export")
            return

        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        logger.info(f"CSV report saved: {path}")

    def save_html(
        self,
        path: str,
        json_path: str,
        heatmap_path: str,
        dashboard_path: str,
        config_info: Optional[dict] = None,
    ):
        """
        Generate interactive HTML report with embedded visuals and data.
        """
        # Load JSON data if available
        report_data = {}
        if os.path.exists(json_path):
            with open(json_path) as f:
                report_data = json.load(f)

        pa = report_data.get("possession", {}).get("team_a", 50)
        pb = report_data.get("possession", {}).get("team_b", 50)
        avg_speed = report_data.get("speed", {}).get("average_kmh", 0)
        peak_speed = report_data.get("speed", {}).get("peak_kmh", 0)
        total_kicks = report_data.get("kicks_total", 0)
        frames = report_data.get("frames_processed", 0)

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Football Analytics Match Report v{self.ver}</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ background: #0a1528; color: #deeeff; font-family: 'Segoe UI', Arial, sans-serif; }}
  .container {{ max-width: 1200px; margin: 0 auto; padding: 20px; }}
  h1 {{ color: #00d4ff; border-bottom: 2px solid #00d4ff; padding-bottom: 10px; margin-bottom: 20px; }}
  h2 {{ color: #00d4ff; margin: 20px 0 10px; }}
  .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }}
  .stat-card {{ background: #0a2050; border: 1px solid #1a3050; border-radius: 8px; padding: 15px; text-align: center; }}
  .stat-card .value {{ font-size: 28px; font-weight: bold; color: #00d4ff; }}
  .stat-card .label {{ font-size: 12px; color: #6a8aaa; text-transform: uppercase; margin-top: 5px; }}
  .visuals {{ display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin: 20px 0; }}
  .visuals img {{ width: 100%; border: 1px solid #1a3050; border-radius: 8px; }}
  .possession-bar {{ background: #0a2050; border-radius: 8px; overflow: hidden; height: 30px; display: flex; margin: 10px 0; border: 1px solid #00d4ff; }}
  .possession-bar .team-a {{ background: #dcdcdc; display: flex; align-items: center; justify-content: center; color: #000; font-weight: bold; }}
  .possession-bar .team-b {{ background: #3264ff; display: flex; align-items: center; justify-content: center; color: #fff; font-weight: bold; }}
  table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
  th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #1a3050; }}
  th {{ background: #0a2050; color: #00d4ff; }}
  tr:hover {{ background: #0a1830; }}
  .footer {{ text-align: center; color: #4a6a8a; padding: 20px 0; font-size: 12px; }}
</style>
</head>
<body>
<div class="container">
  <h1>⚽ Football Analytics — Match Report</h1>
  <p>Version {self.ver} | Generated by {self.author}</p>

  <div class="stats-grid">
    <div class="stat-card"><div class="value">{frames}</div><div class="label">Frames</div></div>
    <div class="stat-card"><div class="value">{avg_speed:.1f}</div><div class="label">Avg Speed (km/h)</div></div>
    <div class="stat-card"><div class="value">{peak_speed:.1f}</div><div class="label">Peak Speed (km/h)</div></div>
    <div class="stat-card"><div class="value">{total_kicks}</div><div class="label">Total Kicks</div></div>
  </div>

  <h2>Possession</h2>
  <div class="possession-bar">
    <div class="team-a" style="width: {pa}%;">{pa:.0f}%</div>
    <div class="team-b" style="width: {pb}%;">{pb:.0f}%</div>
  </div>

  <h2>Visual Reports</h2>
  <div class="visuals">
    <img src="{os.path.basename(heatmap_path)}" alt="Heatmap" onerror="this.style.display='none'">
    <img src="{os.path.basename(dashboard_path)}" alt="Dashboard" onerror="this.style.display='none'">
  </div>

  <h2>Summary Statistics</h2>
  <table>
    <tr><th>Metric</th><th>Value</th></tr>
    <tr><td>Team A Possession</td><td>{pa:.1f}%</td></tr>
    <tr><td>Team B Possession</td><td>{pb:.1f}%</td></tr>
    <tr><td>Average Speed</td><td>{avg_speed:.2f} km/h</td></tr>
    <tr><td>Peak Speed</td><td>{peak_speed:.2f} km/h</td></tr>
    <tr><td>Total Kicks</td><td>{total_kicks}</td></tr>
    <tr><td>Frames Processed</td><td>{frames}</td></tr>
  </table>

  <div class="footer">
    Football Analytics v{self.ver} — Built with YOLO11 + ByteTrack + ReID + PyTorch
  </div>
</div>
</body>
</html>"""

        with open(path, "w", encoding="utf-8") as f:
            f.write(html)
        logger.info(f"HTML report saved: {path}")

    def _build_player_stats(self, analytics, teams: Dict[int, str]) -> Dict:
        """Build per-player statistics dict for JSON export."""
        players = {}
        all_tids = set()
        for key in ('_distances', '_world_distances', '_speeds', '_world_speeds'):
            if hasattr(analytics, key):
                all_tids.update(getattr(analytics, key).keys())

        for tid in sorted(all_tids):
            players[f"P{tid}"] = {
                "team": teams.get(tid, "unknown"),
                "distance_m": round(analytics.distance(tid), 2),
                "avg_speed_kmh": round(analytics.speed(tid), 2),
                "sprints": analytics.sprint_count(tid),
                "actions": analytics.player_action_distribution(tid) if hasattr(analytics, 'player_action_distribution') else {},
            }
        return players
