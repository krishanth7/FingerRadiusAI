"""
radius_calculator.py - Finger radius computation & hand status detection.
Professional visualization with clean metrics overlay.
"""

from typing import Dict, List, Tuple

import cv2
import numpy as np

from src.hand_tracker import FINGER_TIPS, LandmarkIndex
from src.utils import (
    euclidean_distance,
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
    Computes smoothed finger-pair and wrist-to-tip distances,
    classifies hand gestures, and renders professional metric overlays.
    """

    OPEN_THRESHOLD = 100
    CLOSED_THRESHOLD = 50
    PINCH_THRESHOLD = 40

    def __init__(self, smoothing_alpha: float = 0.35):
        self._pair_smoothers: Dict[str, ExponentialMovingAverage] = {
            name: ExponentialMovingAverage(smoothing_alpha) for name, _, _ in RADIUS_PAIRS
        }
        self._wrist_smoothers: Dict[str, ExponentialMovingAverage] = {
            name: ExponentialMovingAverage(smoothing_alpha) for name, _, _ in WRIST_TIP_PAIRS
        }

    def compute(
        self, landmarks: List[Tuple[int, int]]
    ) -> tuple:
        """Compute all radius values. Returns (pair_radii, wrist_radii, hand_status)."""
        pair_radii: Dict[str, float] = {}
        for name, a, b in RADIUS_PAIRS:
            raw = euclidean_distance(landmarks[a], landmarks[b])
            pair_radii[name] = self._pair_smoothers[name].update(raw)

        wrist_radii: Dict[str, float] = {}
        for name, a, b in WRIST_TIP_PAIRS:
            raw = euclidean_distance(landmarks[a], landmarks[b])
            wrist_radii[name] = self._wrist_smoothers[name].update(raw)

        hand_status = self._classify_hand(pair_radii)
        return pair_radii, wrist_radii, hand_status

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
    ):
        """Draw professional radius visualization on the video frame."""
        h, w = frame.shape[:2]

        # 1) Adjacent finger-pair radius lines and labels
        for i, (name, a, b) in enumerate(RADIUS_PAIRS):
            pa, pb = landmarks[a], landmarks[b]
            cx = (pa[0] + pb[0]) // 2
            cy = (pa[1] + pb[1]) // 2
            radius_px = int(pair_radii[name] / 2)
            color = COLORS[FINGER_KEYS[i]]

            # Dashed-style connecting line between tips
            cv2.line(frame, pa, pb, color, 1, cv2.LINE_AA)

            # Clean radius circle (semi-transparent)
            if radius_px > 3:
                overlay = frame.copy()
                cv2.circle(overlay, (cx, cy), radius_px, color, 1, cv2.LINE_AA)
                cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

            # Midpoint dot
            cv2.circle(frame, (cx, cy), 2, color, -1, cv2.LINE_AA)

            # Clean numerical label with background
            label = f"{pair_radii[name]:.0f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            fs = 0.38
            (tw, th), bl = cv2.getTextSize(label, font, fs, 1)
            lx, ly = cx + 12, cy - 4
            # Label background
            draw_filled_rect(frame,
                             (lx - 3, ly - th - 2),
                             (lx + tw + 3, ly + bl + 2),
                             COLORS["bg_primary"], alpha=0.65)
            cv2.putText(frame, label, (lx, ly), font, fs, color, 1, cv2.LINE_AA)

        # 2) Wrist-to-tip lines (very subtle dashed look)
        wrist = landmarks[LandmarkIndex.WRIST]
        for i, (name, _, tip_idx) in enumerate(WRIST_TIP_PAIRS):
            tip = landmarks[tip_idx]
            # Draw dotted line by drawing short segments
            self._draw_dotted_line(frame, wrist, tip, COLORS["bone"], gap=8)

    def _draw_dotted_line(self, img, pt1, pt2, color, gap=8):
        """Draw a dotted line between two points."""
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
        """Draw hand status badge in corporate style."""
        color_map = {
            "Open":    COLORS["success"],
            "Closed":  COLORS["warning"],
            "Pinch":   COLORS["thumb"],
            "Partial": COLORS["text_secondary"],
        }
        color = color_map.get(status, COLORS["text_secondary"])
        draw_status_badge(frame, f"HAND: {status.upper()}", position, color)
