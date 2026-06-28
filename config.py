"""
Football Match Analytics — Configuration
═══════════════════════════════════════════
All tunable parameters in one place.
RTX 4060 optimized defaults.
"""

from dataclasses import dataclass, field
from typing import Tuple, List


# ── Version ─────────────────────────────────────────────────────────────────────
VER: str = "7.0"
AUTHOR: str = "Smit Thaker"

# ── Paths ───────────────────────────────────────────────────────────────────────
@dataclass
class Paths:
    yolo_model: str = "yolo11x.pt"
    yolo_seg_model: str = "yolo11x-seg.pt"
    yolo_pose_model: str = "yolo11x-pose.pt"
    output_video: str = "outputs/football_output.mp4"
    output_heatmap: str = "outputs/football_heatmap.png"
    output_dashboard: str = "outputs/football_dashboard.png"
    output_json: str = "outputs/pose_analysis.json"
    output_html: str = "outputs/football_report.html"


# ── Detection ───────────────────────────────────────────────────────────────────
@dataclass
class DetectionConfig:
    conf_threshold: float = 0.45
    min_player_conf: float = 0.55
    min_aspect_ratio: float = 1.15
    max_ball_size: int = 55
    iou_threshold: float = 0.5
    # Crowd / false positive filters
    field_top: float = 0.18
    min_h_frac: float = 0.055
    max_detections: int = 14


# ── ByteTrack ───────────────────────────────────────────────────────────────────
@dataclass
class ByteTrackConfig:
    track_high_thresh: float = 0.55
    track_low_thresh: float = 0.25
    new_track_thresh: float = 0.55  # lower = fewer spurious new IDs
    match_thresh: float = 0.85
    track_buffer: int = 90  # 3 seconds at 30 fps
    max_tracks: int = 30  # 22 players + subs + refs
    # Motion
    max_px_jump: int = 110
    cent_fallback: int = 115


# ── ReID ────────────────────────────────────────────────────────────────────────
@dataclass
class ReIDConfig:
    enabled: bool = False  # requires torchreid; off by default to avoid CUDA warmup hangs
    model_name: str = "osnet_x0_25"  # light reid model
    feature_dim: int = 512
    match_threshold: float = 0.70
    gallery_size: int = 100
    max_feature_age: int = 90
    use_cuda: bool = True


# ── Pose (YOLO11x-pose) ─────────────────────────────────────────────────────────
@dataclass
class PoseConfig:
    conf_threshold: float = 0.60
    kp_visibility: float = 0.35
    skel_conf: float = 0.62
    kick_px: int = 28
    run_kmh: float = 3.5
    ball_px: int = 88
    # COCO keypoint indices
    idx_lhip: int = 11
    idx_rhip: int = 12
    idx_lkne: int = 13
    idx_rkne: int = 14
    idx_lank: int = 15
    idx_rank: int = 16


# ── Homography ──────────────────────────────────────────────────────────────────
@dataclass
class HomographyConfig:
    enabled: bool = True
    pitch_width_m: float = 105.0
    pitch_length_m: float = 68.0
    refine_every: int = 50
    min_matches: int = 10


# ── Camera Motion Compensation ─────────────────────────────────────────────────
@dataclass
class CameraConfig:
    enabled: bool = True
    orb_features: int = 2000
    orb_scale: float = 1.2
    orb_levels: int = 8
    min_matches: int = 10
    match_ratio: float = 0.75
    max_warp_error: float = 5.0
    flow_clamp: float = 100.0


# ── Speed / Distance ────────────────────────────────────────────────────────────
@dataclass
class SpeedConfig:
    p2m: float = 0.075  # fallback when homography unavailable
    max_speed: float = 38.0  # real-world football max ~38 km/h (Mbappé ~36)
    speed_win: int = 10
    show_speed_at: float = 5.0
    # Validation
    require_consecutive: int = 3  # frames above threshold before reporting
    suspicious_threshold: float = 40.0  # km/h — treat as tracking error
    min_world_delta: float = 0.15  # meters — ignore sub-pixel noise


# ── Trajectory / Trail ──────────────────────────────────────────────────────────
@dataclass
class TrajectoryConfig:
    max_trail: int = 30  # frames
    trail_min_gap: int = 2  # skip frames for performance
    minimap_trail: bool = True
    world_coords: bool = True  # store world (meters) trails when homography available


# ── Ball Tracking ───────────────────────────────────────────────────────────────
@dataclass
class BallConfig:
    conf_threshold: float = 0.35
    max_radius: int = 40
    trajectory_length: int = 30
    velocity_smooth: int = 5
    max_disappear: int = 15
    near_player_radius: int = 130


# ── Team Classification ─────────────────────────────────────────────────────────
@dataclass
class TeamConfig:
    calibrate_samples: int = 50
    kmeans_iters: int = 25
    temporal_smooth: int = 10
    confidence_vote: bool = True


# ── Visualization ───────────────────────────────────────────────────────────────
@dataclass
class VizConfig:
    font: int = 0  # cv2.FONT_HERSHEY_DUPLEX
    silhouette_alpha: float = 0.22
    minimap_size: int = 280
    card_width: int = 188
    card_height: int = 205
    card_x: int = 14
    card_y: int = 14
    hud_height: int = 80


# ── BGR Colors ──────────────────────────────────────────────────────────────────
@dataclass
class Colors:
    bg: Tuple[int, int, int] = (10, 21, 40)
    cyan: Tuple[int, int, int] = (255, 212, 0)
    cyan_dark: Tuple[int, int, int] = (160, 140, 0)
    team_a: Tuple[int, int, int] = (220, 220, 220)
    team_b: Tuple[int, int, int] = (255, 100, 50)
    ball: Tuple[int, int, int] = (255, 220, 0)
    white: Tuple[int, int, int] = (255, 255, 255)
    black: Tuple[int, int, int] = (0, 0, 0)
    green: Tuple[int, int, int] = (100, 255, 50)
    red: Tuple[int, int, int] = (50, 50, 255)
    grey: Tuple[int, int, int] = (120, 120, 120)
    action_kick: Tuple[int, int, int] = (50, 50, 255)
    action_ball: Tuple[int, int, int] = (255, 255, 0)
    action_run: Tuple[int, int, int] = (100, 255, 50)
    action_idle: Tuple[int, int, int] = (120, 120, 120)

    # COCO skeleton colors per limb
    skeleton_a: Tuple[int, int, int] = (200, 230, 255)
    skeleton_b: Tuple[int, int, int] = (255, 210, 100)


# ── GPU / Performance ───────────────────────────────────────────────────────────
@dataclass
class PerformanceConfig:
    device: str = "cuda:0"
    half_precision: bool = False  # disabled by default — RTX 4060 FP16 + YOLO11x-seg can hang CUDA compiler
    batch_size: int = 1
    frame_skip: int = 0
    async_pipeline: bool = True
    inference_mode: bool = True
    max_video_width: int = 1920
    max_video_height: int = 1080


# ── Singleton config instance ───────────────────────────────────────────────────
@dataclass
class Config:
    pid: str = VER
    author: str = AUTHOR
    paths: Paths = field(default_factory=Paths)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    tracking: ByteTrackConfig = field(default_factory=ByteTrackConfig)
    reid: ReIDConfig = field(default_factory=ReIDConfig)
    pose: PoseConfig = field(default_factory=PoseConfig)
    homography: HomographyConfig = field(default_factory=HomographyConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    speed: SpeedConfig = field(default_factory=SpeedConfig)
    trajectory: TrajectoryConfig = field(default_factory=TrajectoryConfig)
    ball: BallConfig = field(default_factory=BallConfig)
    team: TeamConfig = field(default_factory=TeamConfig)
    viz: VizConfig = field(default_factory=VizConfig)
    colors: Colors = field(default_factory=Colors)
    perf: PerformanceConfig = field(default_factory=PerformanceConfig)


CFG = Config()
