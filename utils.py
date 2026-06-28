"""
Utilities — Logging, Video I/O, Helpers
═════════════════════════════════════════
Shared utility functions used across the project.
"""

import logging
import os
import sys
from typing import Optional, Tuple

import cv2
import numpy as np
import torch


def setup_logging(level: str = "INFO"):
    """
    Configure project-wide logging with consistent format.

    Args:
        level: One of DEBUG, INFO, WARNING, ERROR, CRITICAL
    """
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    datefmt = "%H:%M:%S"
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=fmt,
        datefmt=datefmt,
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def find_video(directory: str = ".") -> Optional[str]:
    """
    Auto-discover video file in directory.
    Prefers highlight/football/match/game/soccer keywords.
    Falls back to largest MP4.
    """
    if not os.path.isdir(directory):
        directory = os.path.dirname(directory) or "."

    mp4s = [
        f for f in os.listdir(directory)
        if f.lower().endswith(".mp4")
        and "output" not in f.lower()
    ]
    if not mp4s:
        return None
    if len(mp4s) == 1:
        return os.path.join(directory, mp4s[0])

    # Prefer keyword matches
    keywords = ("highlight", "football", "match", "game", "soccer")
    for f in mp4s:
        if any(k in f.lower() for k in keywords):
            return os.path.join(directory, f)

    # Fallback to largest
    largest = max(mp4s, key=lambda f: os.path.getsize(os.path.join(directory, f)))
    return os.path.join(directory, largest)


def get_video_info(path: str) -> Tuple[int, int, float, int]:
    """
    Read video metadata.

    Returns:
        Tuple of (width, height, fps, total_frames)
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {path}")
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return W, H, fps, total


def create_video_writer(output_path: str, fps: float, size: Tuple[int, int]) -> cv2.VideoWriter:
    """
    Create video writer with MP4V codec.

    Args:
        output_path: Path for output video
        fps: Frames per second
        size: (width, height) tuple

    Returns:
        cv2.VideoWriter instance
    """
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(output_path, fourcc, fps, size)


def check_gpu() -> bool:
    """Check if CUDA GPU is available and print diagnostics."""
    cuda_avail = torch.cuda.is_available()
    if cuda_avail:
        device_count = torch.cuda.device_count()
        device_name = torch.cuda.get_device_name(0)
        print(f"  [GPU] {device_name} ({device_count} device(s)) — CUDA ready")
    else:
        print("  [GPU] CUDA not available — falling back to CPU")
    return cuda_avail
