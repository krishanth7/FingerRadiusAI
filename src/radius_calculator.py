"""
radius_calculator.py - Finger radius computation & hand status detection.
Supports both 2D and 3D (depth-aware) distance calculation.
Professional visualization with clean metrics overlay.
"""

from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np

from src.hand_tracker import FINGER_TIPS, LandmarkIndex
from src.utils import (
    euclidean_distance, euclidean_distance_3d,
    ExponentialMovingAverage,
    COLORS, FINGER_NAMES, FINGER_KEYS,
    draw_clean_line, draw_dot, draw_label, draw_status_badge,
    draw_filled_rect,
)


RADIUS_PAIRS = [
    ("Thumb-Index",  LandmarkIndex.THUMB_TIP,  LandmarkIndex.INDEX_TIP),
    ("Index-Middle", LandmarkIndex.INDEX_TIP,   LandmarkIndex.MIDDLE_TIP),
    ("Middle-Ring",  LandmarkIndex.MIDDLE_TIP,  LandmarkIndex.RING_TIP),
    ("Ring-Pinky",   LandmarkIndex.RING_TIP,    LandmarkIndex.PINKY_TIP),
]

WRIST_TIP_PAIRS = [
    ("Wrist-Thumb",  LandmarkIndex.WRIST, LandmarkIndex.THUMB_TIP),
    ("Wrist-Index",  LandmarkIndex.WRIST, LandmarkIndex.INDEX_TIP),
    ("Wrist-Middle", LandmarkIndex.WRIST, LandmarkIndex.MIDDLE_TIP),
    ("Wrist-Ring",   LandmarkIndex.WRIST, LandmarkIndex.RING_TIP),
    ("Wrist-Pinky",  LandmarkIndex.WRIST, LandmarkIndex.PINKY_TIP),
]


class RadiusCalculator:
    """
    Computes smoothed finger-pair and wrist-to-tip distances (2D or 3D),
    classifies hand gestures, and renders professional metric overlays.
    """

    OPEN_THRESHOLD = 100
    CLOSED_THRESHOLD = 50
    PINCH_THRESHOLD = 40

    def __init__(self, smoothing_alpha: float = 0.35):
        # 2D smoothers
        self._pair_smoothers: Dict[str, ExponentialMovingAverage] = {
            name: ExponentialMovingAverage(smoothing_alpha) for name, _, _ in RADIUS_PAIRS
        }
        self._wrist_smoothers: Dict[str, ExponentialMovingAverage] = {
            name: ExponentialMovingAverage(smoothing_alpha) for name, _, _ in WRIST_TIP_PAIRS
        }
        # 3D smoothers (separate set for clean switching)
        self._pair_smoothers_3d: Dict[str, ExponentialMovingAverage] = {
            name: ExponentialMovingAverage(smoothing_alpha) for name, _, _ in RADIUS_PAIRS
        }
        self._wrist_smoothers_3d: Dict[str, ExponentialMovingAverage] = {
            name: ExponentialMovingAverage(smoothing_alpha) for name, _, _ in WRIST_TIP_PAIRS
        }

    def compute(
        self,
        landmarks: List[Tuple[int, int]],
        landmarks_3d: Optional[List[Tuple[int, int, float]]] = None,
        use_3d: bool = False,
    ) -> tuple:
        """
        Compute all radius values.
        Returns (pair_radii, wrist_radii, hand_status, depth_deltas).
        depth_deltas maps pair name -> z-difference (only when 3D data available).
        """
        pair_radii: Dict[str, float] = {}
        depth_deltas: Dict[str, float] = {}

        for name, a, b in RADIUS_PAIRS:
            if use_3d and landmarks_3d is not None:
                raw = euclidean_distance_3d(landmarks_3d[a], landmarks_3d[b])
                pair_radii[name] = self._pair_smoothers_3d[name].update(raw)
                depth_deltas[name] = abs(landmarks_3d[a][2] - landmarks_3d[b][2])
            else:
                raw = euclidean_distance(landmarks[a], landmarks[b])
                pair_radii[name] = self._pair_smoothers[name].update(raw)
                depth_deltas[name] = 0.0

        wrist_radii: Dict[str, float] = {}
        for name, a, b in WRIST_TIP_PAIRS:
            if use_3d and landmarks_3d is not None:
                raw = euclidean_distance_3d(landmarks_3d[a], landmarks_3d[b])
                wrist_radii[name] = self._wrist_smoothers_3d[name].update(raw)
            else:
                raw = euclidean_distance(landmarks[a], landmarks[b])
                wrist_radii[name] = self._wrist_smoothers[name].update(raw)

        hand_status = self._classify_hand(pair_radii)
        return pair_radii, wrist_radii, hand_status, depth_deltas

    def _classify_hand(self, pair_radii: Dict[str, float]) -> str:
        values = list(pair_radii.values())
        if pair_radii.get("Thumb-Index", 999) < self.PINCH_THRESHOLD:
            return "Pinch"
        if all(v < self.CLOSED_THRESHOLD for v in values):
            return "Closed"
        if all(v > self.OPEN_THRESHOLD for v in values):
            return "Open"
        return "Partial"

    def draw_radii(
        self,
        frame: np.ndarray,
        landmarks: List[Tuple[int, int]],
        pair_radii: Dict[str, float],
        wrist_radii: Dict[str, float],
        use_3d: bool = False,
        depth_deltas: Optional[Dict[str, float]] = None,
    ):
        """Draw professional radius visualization on the video frame."""
        h, w = frame.shape[:2]

        for i, (name, a, b) in enumerate(RADIUS_PAIRS):
            pa, pb = landmarks[a], landmarks[b]
            cx = (pa[0] + pb[0]) // 2
            cy = (pa[1] + pb[1]) // 2
            radius_px = int(pair_radii[name] / 2)
            color = COLORS[FINGER_KEYS[i]]

            # Connecting line
            cv2.line(frame, pa, pb, color, 1, cv2.LINE_AA)

            # Radius circle
            if radius_px > 3:
                overlay = frame.copy()
                cv2.circle(overlay, (cx, cy), radius_px, color, 1, cv2.LINE_AA)
                cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

            cv2.circle(frame, (cx, cy), 2, color, -1, cv2.LINE_AA)

            # Build label text
            val_str = f"{pair_radii[name]:.0f}"
            if use_3d and depth_deltas and name in depth_deltas:
                dz = depth_deltas[name]
                val_str = f"{pair_radii[name]:.0f} z{dz:.0f}"

            font = cv2.FONT_HERSHEY_SIMPLEX
            fs = 0.35
            (tw, th), bl = cv2.getTextSize(val_str, font, fs, 1)
            lx, ly = cx + 12, cy - 4

            # Label background
            draw_filled_rect(frame,
                             (lx - 3, ly - th - 2),
                             (lx + tw + 3, ly + bl + 2),
                             COLORS["bg_primary"], alpha=0.7)
            cv2.putText(frame, val_str, (lx, ly), font, fs, color, 1, cv2.LINE_AA)

        # Wrist-to-tip dotted lines
        wrist = landmarks[LandmarkIndex.WRIST]
        for i, (name, _, tip_idx) in enumerate(WRIST_TIP_PAIRS):
            tip = landmarks[tip_idx]
            self._draw_dotted_line(frame, wrist, tip, COLORS["bone"], gap=8)

    def _draw_dotted_line(self, img, pt1, pt2, color, gap=8):
        dx = pt2[0] - pt1[0]
        dy = pt2[1] - pt1[1]
        dist = max(1, int((dx**2 + dy**2) ** 0.5))
        num_segments = dist // gap
        for j in range(0, num_segments, 2):
            t1 = j / max(num_segments, 1)
            t2 = min((j + 1) / max(num_segments, 1), 1.0)
            x1 = int(pt1[0] + dx * t1)
            y1 = int(pt1[1] + dy * t1)
            x2 = int(pt1[0] + dx * t2)
            y2 = int(pt1[1] + dy * t2)
            cv2.line(img, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)

    def draw_status(self, frame: np.ndarray, status: str, position: Tuple[int, int] = (15, 60)):
        color_map = {
            "Open":    COLORS["success"],
            "Closed":  COLORS["warning"],
            "Pinch":   COLORS["thumb"],
            "Partial": COLORS["text_secondary"],
        }
        color = color_map.get(status, COLORS["text_secondary"])
        draw_status_badge(frame, f"HAND: {status.upper()}", position, color)
