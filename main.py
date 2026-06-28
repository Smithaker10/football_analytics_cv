"""
Football Match Analytics — Main Pipeline
══════════════════════════════════════════
Professional broadcast football analysis engine.

Pipeline:
  1. Frame capture
  2. Detection (YOLO11x-seg)
  3. ReID feature extraction (optional)
  4. ByteTrack tracking with Kalman + Mahalanobis gating
  5. Pose estimation (YOLO11x-pose) — batched per frame
  6. Team classification
  7. Ball tracking
  8. Homography estimation + camera motion compensation
  9. World-coordinate speed/distance (via homography)
  10. Analytics computation
  11. Visualization (overlays, minimap, HUD, analytics card)
  12. Frame writing
  13. Post-match reports (heatmap, dashboard, JSON, CSV, HTML)
"""

import logging
import os
import sys
import time
from collections import defaultdict
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import torch
from tqdm import tqdm

from config import CFG
from detector import Detector
from tracker import ByteTracker
from reid import OSNetEmbedder
from pose import PoseDetector
from ball_tracker import BallTracker
from homography import HomographyEstimator
from team_classifier import TeamClassifier
from analytics import AnalyticsEngine
from visualization import Visualizer
from minimap import Minimap
from dashboard import DashboardGenerator
from reports import ReportGenerator
from utils import (
    setup_logging,
    find_video,
    get_video_info,
    create_video_writer,
    check_gpu,
)

logger = logging.getLogger(__name__)

IDX_LANK, IDX_RANK = CFG.pose.idx_lank, CFG.pose.idx_rank


class FootballAnalyticsPipeline:
    """
    End-to-end football match analytics pipeline.

    Manages module lifecycle, frame processing, and report generation.
    """

    def __init__(self, config=CFG):
        self.cfg = config
        self._init_modules()

    def _init_modules(self):
        """Initialize all processing modules."""
        logger.info("Initializing Football Analytics Pipeline v%s", CFG.pid)
        self.detector = Detector(self.cfg)
        self.tracker = ByteTracker(self.cfg)
        self.reid = OSNetEmbedder(self.cfg) if self.cfg.reid.enabled else None
        self.pose = PoseDetector(self.cfg)
        self.ball_tracker = BallTracker(self.cfg)
        self.homography = HomographyEstimator(self.cfg)
        self.team_classifier = TeamClassifier(self.cfg)
        self.analytics = AnalyticsEngine(self.cfg)
        self.visualizer = Visualizer(self.cfg)
        self.minimap = Minimap(self.cfg)
        self.dashboard = DashboardGenerator(self.cfg)
        self.reports = ReportGenerator(self.cfg)

    def process(self, video_path: str) -> str:
        """
        Process a broadcast football video through the full pipeline.

        Args:
            video_path: Path to input MP4 video

        Returns:
            Path to output video
        """
        W, H, fps, total = get_video_info(video_path)
        cap = cv2.VideoCapture(video_path)

        output_dir = "outputs"
        os.makedirs(output_dir, exist_ok=True)

        out_video = os.path.join(output_dir, "football_output.mp4")
        out_heatmap = os.path.join(output_dir, "football_heatmap.png")
        out_dashboard = os.path.join(output_dir, "football_dashboard.png")
        out_json = os.path.join(output_dir, "pose_analysis.json")
        out_csv = os.path.join(output_dir, "player_stats.csv")
        out_html = os.path.join(output_dir, "football_report.html")

        writer = create_video_writer(out_video, fps, (W, H))

        self._print_header(video_path, W, H, fps, total)

        peak_speed = 0.0
        frame_idx = 0
        pos_a, pos_b = [], []
        kick_pts = []
        teams_dict: Dict[int, str] = {}
        _prev_frame_gray: Optional[np.ndarray] = None

        # CUDA warmup
        if torch.cuda.is_available():
            logger.info("Warming up GPU...")
            _warmup = np.zeros((640, 640, 3), dtype=np.uint8)
            try:
                self.detector.infer(_warmup)
                self.pose.detect(_warmup, (100, 100, 200, 400))
                torch.cuda.synchronize()
                logger.info("GPU warmup complete")
            except Exception:
                logger.warning("GPU warmup failed (non-fatal), continuing...")

        logger.info("Processing video...")
        pbar = tqdm(total=total, unit="fr", ncols=70, colour="cyan")

        max_frames = int(os.environ.get("MAX_FRAMES", "0"))
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if max_frames > 0 and frame_idx >= max_frames:
                break

            try:
                canvas = frame.copy()
                inference_start = time.perf_counter()

                # ── 0. Camera cut detection ──────────────────────────────────
                if _detect_camera_cut(frame, _prev_frame_gray,
                                      threshold=0.55):
                    logger.info("Camera cut detected at frame %d, resetting tracking", frame_idx)
                    self.tracker.reset()
                    self.homography.reset()
                    self.ball_tracker.reset()
                    # Clear position trails so old positions aren't drawn at wrong locations
                    pos_a.clear()
                    pos_b.clear()
                    kick_pts.clear()
                    self.analytics.reset()
                _prev_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # ── 1. Detection ─────────────────────────────────────────────
                players, ball_dets = self.detector.infer(frame)

                # ── 2. ReID feature extraction ────────────────────────────────
                reid_features = None
                if self.reid is not None and players:
                    reid_features = self.reid.extract(frame, players)

                # ── 3. Tracking ──────────────────────────────────────────────
                tracks = self.tracker.update(players, reid_features)

                # ── 4. Team classification ────────────────────────────────────
                for tid, track in tracks.items():
                    if track.team is None:
                        x1, y1, x2, y2 = track.last_box
                        team = self.team_classifier.assign(
                            frame, x1, y1, x2, y2, tid=tid
                        )
                        track.team = team

                # ── 5. Ball tracking ─────────────────────────────────────────
                ball_pos = self.ball_tracker.update(ball_dets)
                if ball_pos is None and ball_dets:
                    best = max(ball_dets, key=lambda d: d.conf)
                    ball_pos = (int(best.cx), int(best.cy))
                    self.ball_tracker.update([best])

                # ── 6. Homography ────────────────────────────────────────────
                self.homography.estimate(frame)

                # ── 7. Pose — batched per frame ──────────────────────────────
                pose_inputs = [(tid, track.last_box) for tid, track in tracks.items()
                               if track.hit_count > 2 and track.is_confirmed]
                if pose_inputs:
                    pose_tids, pose_boxes = zip(*pose_inputs)
                    batch_kps = self.pose.detect_batch(frame, list(pose_boxes))
                else:
                    pose_tids, pose_boxes, batch_kps = (), (), {}

                # ── 8. Player analytics + world-coordinate trails ────────────
                spds = []
                for tid, track in tracks.items():
                    x1, y1, x2, y2 = track.last_box
                    team = track.team or "team_a"
                    cx, cy = track.cx, track.cy

                    # World-coordinate position (via homography)
                    world_pos: Optional[Tuple[float, float]] = None
                    if self.homography.is_calibrated:
                        wp = self.homography.project_to_pitch(
                            np.array([[cx, cy]], dtype=np.float32)
                        )
                        if wp is not None and len(wp) > 0:
                            world_pos = (float(wp[0, 0]), float(wp[0, 1]))

                    # Trail recording (filtered by confidence)
                    if track.hit_count > 2 and track.is_confirmed:
                        if len(track._frame_ids) == 0 or \
                           frame_idx - track._frame_ids[-1] >= CFG.trajectory.trail_min_gap:
                            track.add_position(cx, cy, frame_idx, world_pos)

                    # Speed estimation via homography
                    self.tracker.update_speed(track, self.homography, fps)
                    self.tracker.update_world_distance(track, self.homography)

                    spd = track.speed
                    spds.append(spd)
                    peak_speed = max(peak_speed, spd)

                    # Pose
                    kps = None
                    p_conf = 0.0
                    action = "idle"

                    if track.hit_count > 2 and track.is_confirmed:
                        try:
                            idx = pose_tids.index(tid)
                            kps = batch_kps.get(idx)
                        except (ValueError, AttributeError):
                            pass

                        if kps:
                            kps = self.pose.smooth(tid, kps)
                            p_conf = PoseDetector.confidence(kps)
                            action = PoseDetector.classify_action(kps, ball_pos, spd)
                            track.pose = kps
                            track.pose_conf = p_conf

                    track.action = action
                    self.analytics.update(tid, team, track.cx, track.cy,
                                          action, ball_pos, fps, world_pos,
                                          track_speed=track.speed)

                    if action == "kicking":
                        if kps and len(kps) == 17:
                            for kidx in (IDX_LANK, IDX_RANK):
                                if kps[kidx]["vis"] > 0.28:
                                    kick_pts.append((kps[kidx]["x"], kps[kidx]["y"]))

                # ── 9. Possession ────────────────────────────────────────────
                possession_team = self.ball_tracker.compute_possession(ball_pos, tracks)
                if possession_team:
                    self.analytics.update_possession(possession_team)

                # ── 10. Visualization ────────────────────────────────────────
                for tid, track in tracks.items():
                    if track.is_confirmed:
                        self.visualizer.draw_silhouette(
                            canvas, track.mask, track.last_box, track.team or "team_a"
                        )

                # Ball
                if ball_pos:
                    self.visualizer.draw_ball(canvas, *ball_pos)
                    if len(self.ball_tracker.trajectory) > 1:
                        self.visualizer.draw_ball_trajectory(
                            canvas, self.ball_tracker.trajectory
                        )

                # Player overlays + trail
                for tid, track in tracks.items():
                    if not track.is_confirmed and track.hit_count < 3:
                        continue

                    x1, y1, x2, y2 = track.last_box
                    team = track.team or "team_a"
                    spd = track.speed

                    if track.is_confirmed:
                        trail_pts = [(int(p[0]), int(p[1]))
                                     for p in track.pixel_trail]
                        if len(trail_pts) > 1:
                            self.visualizer.draw_player_trail(
                                canvas, trail_pts, team
                            )

                    if track.pose and track.pose_conf > 0:
                        self.visualizer.draw_skeleton(
                            canvas, track.pose, team, track.pose_conf
                        )

                    self.visualizer.draw_player_tag(
                        canvas, x1, y1, x2, y2, tid, team, spd,
                        track.pose is not None
                    )
                    self.visualizer.draw_action_pill(
                        canvas, x1, y1, x2, track.action
                    )

                    cx, cy = int(track.cx), int(track.cy)
                    if team == "team_a":
                        pos_a.append((cx, cy))
                    elif team == "team_b":
                        pos_b.append((cx, cy))

                # Minimap with trails
                teams_dict = {tid: t.team or "team_a" for tid, t in tracks.items()}
                self.minimap.draw(
                    canvas, tracks, teams_dict, W, H, ball_pos,
                    homography=self.homography if self.homography.is_calibrated else None,
                )

                # HUD
                pa, pb = self.analytics.possession()
                expected_players = 22
                filled = spds + [0.0] * (expected_players - len(spds))
                avg_s = float(np.mean(filled)) if filled else 0.0
                top_speeds = self.analytics.top_speeds(3)
                self.visualizer.draw_analytics_card(
                    canvas, pa, pb, avg_s, peak_speed,
                    len(tracks), self.analytics.total_kicks, top_speeds,
                )

                self.analytics.finalize_frame(avg_s, min(len(tracks) * avg_s / 10.0, 100.0))
                self.visualizer.draw_hud(
                    canvas, pa, pb, avg_s, peak_speed,
                    len(tracks), self.analytics.total_kicks,
                    frame_idx, fps, W, H,
                )

                inference_time = (time.perf_counter() - inference_start) * 1000
                cv2.putText(canvas, f"{inference_time:.0f}ms",
                            (8, H - self.cfg.viz.hud_height - 10),
                            cv2.FONT_HERSHEY_DUPLEX, 0.5, (100, 200, 100), 1)

                writer.write(canvas)
                frame_idx += 1
                pbar.update(1)

            except Exception:
                logger.exception("Frame %d failed, skipping", frame_idx)
                pbar.update(1)
                frame_idx += 1
                continue

        cap.release()
        writer.release()
        pbar.close()

        self._print_summary(frame_idx, peak_speed)

        logger.info("Generating reports...")
        self.dashboard.save_heatmap(
            pos_a, pos_b, (H, W), out_heatmap, kick_pts or None
        )
        action_dist = self.analytics.action_distribution()
        self.dashboard.save_dashboard(
            self.analytics.avg_speed_history,
            self.analytics.activity_history,
            self.analytics, self.tracker,
            frame_idx, action_dist, out_dashboard,
        )
        self.reports.save_json(
            out_json, video_path, fps, W, H, frame_idx,
            self.analytics, teams_dict,
            config_info={"max_tracks": CFG.tracking.max_tracks, "model": "YOLO11x"},
        )
        self.reports.save_csv(out_csv, self.analytics, teams_dict)
        self.reports.save_html(out_html, out_json, out_heatmap, out_dashboard)

        logger.info("All outputs saved to %s/", output_dir)
        return out_video

    def _print_header(self, video_path: str, W: int, H: int, fps: float, total: int):
        print(f"\n{'═' * 62}")
        print(f"  Football Analytics v{CFG.pid}  |  {CFG.author}")
        print(f"  Video  : {video_path}")
        print(f"  Size   : {W}×{H} @ {fps:.1f} fps  |  {total} frames")
        print(f"  Model  : YOLO11x-seg + YOLO11x-pose")
        print(f"  Tracker: ByteTrack + ReID  max={CFG.tracking.max_tracks}")
        print(f"  Speed  : Homography + camera compensation  max={CFG.speed.max_speed} km/h")
        if torch.cuda.is_available():
            print(f"  GPU    : {torch.cuda.get_device_name(0)}")
        print(f"{'═' * 62}\n")

    def _print_summary(self, frame_idx: int, peak_speed: float):
        print(f"\n  {'─' * 44}")
        print(f"  Frames processed : {frame_idx}")
        print(f"  Peak speed       : {peak_speed:.1f} km/h")
        print(f"  Total kicks      : {self.analytics.total_kicks}")
        pa, pb = self.analytics.possession()
        print(f"  Possession       : A={pa}%  B={pb}%")
        print(f"  Actions:")
        for act, cnt in sorted(
            self.analytics.action_distribution().items(),
            key=lambda x: -x[1],
        ):
            print(f"    {act:<14}: {cnt}")
        print(f"  {'─' * 44}\n")


def _detect_camera_cut(
    frame: np.ndarray,
    prev_gray: Optional[np.ndarray],
    threshold: float = 0.45,
) -> bool:
    """Fast histogram-based camera cut detection.

    Compares grayscale histogram between consecutive frames.
    Returns True when a cut is likely.
    """
    if prev_gray is None:
        return False
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [64], [0, 256])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    prev_hist = cv2.calcHist([prev_gray], [0], None, [64], [0, 256])
    cv2.normalize(prev_hist, prev_hist, 0, 1, cv2.NORM_MINMAX)
    diff = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CHISQR)
    return diff > threshold


def main():
    """Entry point."""
    setup_logging("INFO")

    video_path = None
    if len(sys.argv) >= 2 and os.path.isfile(sys.argv[1]):
        video_path = sys.argv[1]
    else:
        video_path = find_video()
        if not video_path:
            print("[ERROR] No input video found.")
            print("  Usage: python main.py <video.mp4>")
            sys.exit(1)
        print(f"  [AUTO] Using  →  {video_path}")

    check_gpu()

    pipeline = FootballAnalyticsPipeline()
    output = pipeline.process(video_path)
    print(f"\n  [DONE] Output: {output}")
    print("  All outputs saved. ✓\n")


if __name__ == "__main__":
    main()
