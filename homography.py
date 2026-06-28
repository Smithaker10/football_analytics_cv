"""
Homography Estimation — Broadcast → Bird's-Eye View
════════════════════════════════════════════════════
Automatically estimates the field homography from broadcast video.

Features:
  • Line detection (field lines, penalty areas, center circle)
  • Keypoint matching to canonical pitch model
  • RANSAC-based homography estimation
  • Temporal smoothing of homography matrix
  • Point projection (pixel ↔ world coordinates)
  • Camera motion compensation via feature tracking
  • Broadcast cut detection
  • Calibrated speed/distance computation
"""

import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np

from config import CFG

logger = logging.getLogger(__name__)


# Canonical pitch model (FIFA standard): 105m × 68m
PITCH_WIDTH = 105.0
PITCH_LENGTH = 68.0

# Key pitch landmarks in world coordinates (meters)
# Origin at top-left of pitch
PITCH_LANDMARKS = np.array([
    [0.0, 0.0],              # top-left corner
    [PITCH_WIDTH, 0.0],      # top-right corner
    [0.0, PITCH_LENGTH],     # bottom-left corner
    [PITCH_WIDTH, PITCH_LENGTH],  # bottom-right corner
    [PITCH_WIDTH / 2, 0.0],               # top center
    [PITCH_WIDTH / 2, PITCH_LENGTH],      # bottom center
    [0.0, PITCH_LENGTH / 2],             # left center
    [PITCH_WIDTH, PITCH_LENGTH / 2],     # right center
    [PITCH_WIDTH / 2, PITCH_LENGTH / 2], # center spot
    # Penalty areas
    [0.0, PITCH_LENGTH / 2 - 20.15],     # left penalty top
    [0.0, PITCH_LENGTH / 2 + 20.15],     # left penalty bottom
    [PITCH_WIDTH, PITCH_LENGTH / 2 - 20.15],  # right penalty top
    [PITCH_WIDTH, PITCH_LENGTH / 2 + 20.15],  # right penalty bottom
], dtype=np.float32)


class CameraMotionCompensator:
    """Estimates camera motion between consecutive frames using ORB feature matching."""

    def __init__(self, config=CFG):
        self.cfg = config.camera
        self._prev_gray: Optional[np.ndarray] = None
        self._prev_kps: Optional[List[cv2.KeyPoint]] = None
        self._prev_des: Optional[np.ndarray] = None
        self._orb = cv2.ORB_create(
            nfeatures=self.cfg.orb_features,
            scaleFactor=self.cfg.orb_scale,
            nlevels=self.cfg.orb_levels,
        )
        self._bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self._warp: Optional[np.ndarray] = None

    def estimate(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Estimate camera motion between previous and current frame.

        Returns 2x3 affine transform [[dx, dy]] expressing camera pixel movement,
        or None if estimation fails (first frame, cut, low texture).
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self._prev_gray is None:
            kps, des = self._orb.detectAndCompute(gray, None)
            self._prev_gray = gray
            self._prev_kps = kps
            self._prev_des = des
            return None

        kps, des = self._orb.detectAndCompute(gray, None)
        if des is None or self._prev_des is None or len(kps) < self.cfg.min_matches:
            self._prev_gray = gray
            self._prev_kps = kps
            self._prev_des = des
            return None

        matches = self._bf.knnMatch(self._prev_des, des, k=2)
        good = []
        for m, n in matches:
            if m.distance < self.cfg.match_ratio * n.distance:
                good.append(m)

        if len(good) < self.cfg.min_matches:
            self._prev_gray = gray
            self._prev_kps = kps
            self._prev_des = des
            return None

        src_pts = np.float32([self._prev_kps[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kps[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        warp, inliers = cv2.estimateAffinePartial2D(
            src_pts, dst_pts, method=cv2.RANSAC,
            ransacReprojThreshold=self.cfg.max_warp_error,
        )

        self._prev_gray = gray
        self._prev_kps = kps
        self._prev_des = des

        if warp is not None:
            self._warp = warp
        return self._warp

    def compensate(self, pt: Tuple[float, float]) -> Tuple[float, float]:
        """Subtract camera motion from a pixel position (returns corrected position)."""
        if self._warp is None:
            return pt
        x, y = pt
        dx = self._warp[0, 2]
        dy = self._warp[1, 2]
        return (x - dx, y - dy)

    def compensate_displacement(
        self, p1: Tuple[float, float], p2: Tuple[float, float]
    ) -> Tuple[float, float]:
        """Compensated displacement vector between two points.

        Returns (dx, dy) of player motion after subtracting camera motion.
        p1 is from frame N-1 (already in reference coordinates), p2 is from
        frame N (needs camera motion subtracted to convert to reference frame).
        """
        if self._warp is None:
            return (p2[0] - p1[0], p2[1] - p1[1])
        dx, dy = self._warp[0, 2], self._warp[1, 2]
        return (p2[0] - p1[0] - dx, p2[1] - p1[1] - dy)

    def reset(self):
        """Reset for new scene (e.g., after camera cut)."""
        self._prev_gray = None
        self._prev_kps = None
        self._prev_des = None
        self._warp = None


class HomographyEstimator:
    """
    Estimates and maintains the broadcast-to-bird's-eye homography.

    Strategy:
      1. Detect field lines using color segmentation + Hough
      2. Intersect lines to find pitch corners
      3. Match to canonical pitch model via RANSAC
      4. Temporal smoothing to avoid jitter
    """

    def __init__(self, config=CFG):
        self.cfg = config.homography
        self.camera_cfg = config.camera
        self._H: Optional[np.ndarray] = None  # 3×3 homography matrix
        self._H_inv: Optional[np.ndarray] = None
        self._frame_count = 0
        self._last_valid_H = None
        self._consecutive_failures = 0
        self.camera = CameraMotionCompensator(config)

    @property
    def is_calibrated(self) -> bool:
        return self._H is not None

    @property
    def homography(self) -> Optional[np.ndarray]:
        return self._H

    def estimate(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Estimate homography from a broadcast frame.

        Returns the 3×3 homography matrix or None if estimation fails.
        Also updates camera motion estimate.
        """
        self._frame_count += 1

        if self.camera_cfg.enabled:
            self.camera.estimate(frame)

        if self._H is not None and self._frame_count % self.cfg.refine_every != 0:
            return self._H

        try:
            lines = self._detect_field_lines(frame)
            if lines is None or len(lines) < 4:
                raise ValueError(f"Not enough lines: {len(lines) if lines is not None else 0}")

            img_pts = self._find_intersections(lines, frame.shape[:2])
            if img_pts is None or len(img_pts) < 4:
                raise ValueError(f"Not enough intersections: {len(img_pts) if img_pts is not None else 0}")

            H = self._compute_homography_ransac(img_pts, self._get_pitch_landmarks(len(img_pts)))
            if H is None:
                raise ValueError("RANSAC failed to find homography")

            if self._last_valid_H is not None:
                H = H * 0.7 + self._last_valid_H * 0.3

            self._H = H
            self._H_inv = np.linalg.inv(H)
            self._last_valid_H = H
            self._consecutive_failures = 0
            return H

        except Exception as e:
            self._consecutive_failures += 1
            if self._consecutive_failures > 10:
                self._H = None
                self._consecutive_failures = 0
            return self._H if self._H is not None else None

    def project_to_pitch(
        self, img_points: np.ndarray
    ) -> Optional[np.ndarray]:
        """Project image points to pitch coordinates (meters)."""
        if self._H is None:
            return None
        pts = cv2.perspectiveTransform(
            img_points.reshape(-1, 1, 2).astype(np.float32),
            self._H_inv,
        )
        return pts.reshape(-1, 2)

    def project_to_image(
        self, pitch_points: np.ndarray
    ) -> Optional[np.ndarray]:
        """Project pitch coordinates (meters) to image pixels."""
        if self._H is None:
            return None
        pts = cv2.perspectiveTransform(
            pitch_points.reshape(-1, 1, 2).astype(np.float32),
            self._H,
        )
        return pts.reshape(-1, 2)

    def compute_speed(
        self, img_p1: Tuple[float, float], img_p2: Tuple[float, float], fps: float
    ) -> float:
        """Compute ground-truth speed in km/h between two image points.

        Uses homography to get real-world distance, clips to physically plausible range.
        """
        if self._H is None:
            return 0.0
        pts = np.array([img_p1, img_p2], dtype=np.float32)
        world_pts = self.project_to_pitch(pts)
        if world_pts is None:
            return 0.0
        dist_m = float(np.linalg.norm(world_pts[1] - world_pts[0]))
        speed_kmh = dist_m * fps * 3.6
        return float(np.clip(speed_kmh, 0.0, CFG.speed.max_speed))

    def compute_speed_compensated(
        self,
        p1: Tuple[float, float], p2: Tuple[float, float],
        fps: float,
    ) -> float:
        """Compute world-coordinate speed with camera motion compensation."""
        dx, dy = self.camera.compensate_displacement(p1, p2)
        compensated_p2 = (p1[0] + dx, p1[1] + dy)
        return self.compute_speed(p1, compensated_p2, fps)

    def compute_distance(self, points: List[Tuple[float, float]]) -> float:
        """Compute total distance in meters from a list of image points."""
        if self._H is None or len(points) < 2:
            return 0.0
        world_pts = self.project_to_pitch(np.array(points, dtype=np.float32))
        if world_pts is None:
            return 0.0
        total = 0.0
        for i in range(1, len(world_pts)):
            total += float(np.linalg.norm(world_pts[i] - world_pts[i - 1]))
        return total

    def detect_camera_cut(self, frame: np.ndarray, threshold: float = 0.45) -> bool:
        """Detect broadcast camera cut using histogram difference.

        Returns True if a cut is detected.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [64], [0, 256])
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)

        if not hasattr(self, '_prev_hist'):
            self._prev_hist = hist
            return False

        diff = cv2.compareHist(self._prev_hist, hist, cv2.HISTCMP_CHISQR)
        self._prev_hist = hist

        # Low camera motion + high histogram change = cut
        warp = self.camera._warp
        is_static = warp is None or (abs(warp[0, 2]) < 5 and abs(warp[1, 2]) < 5)
        return diff > threshold and is_static

    def reset(self):
        """Reset homography state."""
        self._H = None
        self._H_inv = None
        self._last_valid_H = None
        self._consecutive_failures = 0
        self.camera.reset()

    def _detect_field_lines(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Detect field lines using green-color segmentation + edge detection."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])
        field_mask = cv2.inRange(hsv, lower_green, upper_green)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        field_mask = cv2.morphologyEx(field_mask, cv2.MORPH_CLOSE, kernel)
        field_mask = cv2.morphologyEx(field_mask, cv2.MORPH_OPEN, kernel)

        white_mask = cv2.inRange(
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 200, 255
        )
        line_mask = cv2.bitwise_and(white_mask, cv2.bitwise_not(field_mask))

        edges = cv2.Canny(line_mask, 50, 150, apertureSize=3)

        lines = cv2.HoughLinesP(
            edges, rho=1, theta=np.pi / 180,
            threshold=80, minLineLength=60, maxLineGap=30,
        )
        if lines is None or len(lines) < 4:
            return None

        filtered = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            if angle < 30 or angle > 60:
                filtered.append(line[0])
        return np.array(filtered) if len(filtered) >= 4 else None

    def _find_intersections(
        self, lines: np.ndarray, shape: Tuple[int, int]
    ) -> Optional[np.ndarray]:
        """Find intersection points of detected field lines."""
        h, w = shape
        points = []

        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                x1, y1, x2, y2 = lines[i]
                x3, y3, x4, y4 = lines[j]

                denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
                if abs(denom) < 1e-6:
                    continue

                ix = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
                iy = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom

                if 0 <= ix <= w and 0 <= iy <= h:
                    points.append([ix, iy])

        if len(points) < 4:
            return None

        points = np.array(points, dtype=np.float32)

        if len(points) > 20:
            from scipy.cluster.vq import kmeans2
            n_clusters = min(12, len(points))
            centroid, _ = kmeans2(points, n_clusters, iter=10, minit='points')
            return centroid.astype(np.float32)

        return points

    def _compute_homography_ransac(
        self, src_pts: np.ndarray, dst_pts: np.ndarray
    ) -> Optional[np.ndarray]:
        """Compute homography with RANSAC."""
        n = min(len(src_pts), len(dst_pts))
        if n < 4:
            return None

        H, mask = cv2.findHomography(
            src_pts[:n], dst_pts[:n],
            cv2.RANSAC, ransacReprojThreshold=5.0,
        )
        if H is None or mask is None:
            return None
        inlier_count = np.sum(mask)
        if inlier_count < self.cfg.min_matches:
            return None
        return H.astype(np.float32)

    def _get_pitch_landmarks(self, n: int) -> np.ndarray:
        """Get first n pitch landmarks from the canonical model."""
        return PITCH_LANDMARKS[:min(n, len(PITCH_LANDMARKS))]
