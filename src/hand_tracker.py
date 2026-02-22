"""
hand_tracker.py - MediaPipe Tasks API hand landmark detection.
Uses mp.tasks.vision.HandLandmarker with professional clean visualization.
"""

import os
from typing import List, Optional, Tuple
from collections import deque

import cv2
import numpy as np
import mediapipe as mp

from src.utils import (
    COLORS, draw_clean_line, draw_dot, draw_label,
    ExponentialMovingAverage, FINGER_KEYS,
)

# ── MediaPipe Tasks imports ──
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


class LandmarkIndex:
    """Mapping of MediaPipe hand landmark names to integer IDs."""
    WRIST = 0
    THUMB_CMC = 1;  THUMB_MCP = 2;  THUMB_IP = 3;   THUMB_TIP = 4
    INDEX_MCP = 5;  INDEX_PIP = 6;  INDEX_DIP = 7;   INDEX_TIP = 8
    MIDDLE_MCP = 9; MIDDLE_PIP = 10; MIDDLE_DIP = 11; MIDDLE_TIP = 12
    RING_MCP = 13;  RING_PIP = 14;  RING_DIP = 15;   RING_TIP = 16
    PINKY_MCP = 17; PINKY_PIP = 18; PINKY_DIP = 19;  PINKY_TIP = 20


FINGER_TIPS = [
    LandmarkIndex.THUMB_TIP,
    LandmarkIndex.INDEX_TIP,
    LandmarkIndex.MIDDLE_TIP,
    LandmarkIndex.RING_TIP,
    LandmarkIndex.PINKY_TIP,
]

# Skeleton connections
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]

# Finger bone groups for coloring skeleton segments
FINGER_BONE_GROUPS = {
    "thumb":  [(0, 1), (1, 2), (2, 3), (3, 4)],
    "index":  [(0, 5), (5, 6), (6, 7), (7, 8)],
    "middle": [(0, 9), (9, 10), (10, 11), (11, 12)],
    "ring":   [(0, 13), (13, 14), (14, 15), (15, 16)],
    "pinky":  [(0, 17), (17, 18), (18, 19), (19, 20)],
}

DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models", "hand_landmarker.task"
)


class HandTracker:
    """
    Professional hand landmark tracker using the MediaPipe Tasks API.
    Clean skeleton visualization with finger-colored bones and subtle trails.
    """

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        max_num_hands: int = 1,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.6,
        trail_length: int = 15,
        smoothing_alpha: float = 0.4,
    ):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Hand landmarker model not found at: {model_path}\n"
                "Download it from: https://storage.googleapis.com/mediapipe-models/"
                "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
            )

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.VIDEO,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.landmarker = HandLandmarker.create_from_options(options)
        self._frame_timestamp_ms = 0

        # EMA smoothers for each landmark
        self._smoothers_x = [ExponentialMovingAverage(smoothing_alpha) for _ in range(21)]
        self._smoothers_y = [ExponentialMovingAverage(smoothing_alpha) for _ in range(21)]

        # Motion trails
        self._trails = {tip: deque(maxlen=trail_length) for tip in FINGER_TIPS}

        self.landmarks: Optional[List[Tuple[int, int]]] = None

    def process(self, frame: np.ndarray) -> Optional[List[Tuple[int, int]]]:
        """Run hand detection on a BGR frame. Returns 21 (x, y) tuples or None."""
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        self._frame_timestamp_ms += 33
        result = self.landmarker.detect_for_video(mp_image, self._frame_timestamp_ms)

        if result.hand_landmarks and len(result.hand_landmarks) > 0:
            hand = result.hand_landmarks[0]
            coords = []
            for i, lm in enumerate(hand):
                px = int(lm.x * w)
                py = int(lm.y * h)
                sx = int(self._smoothers_x[i].update(px))
                sy = int(self._smoothers_y[i].update(py))
                coords.append((sx, sy))

            for tip_idx in FINGER_TIPS:
                self._trails[tip_idx].append(coords[tip_idx])

            self.landmarks = coords
            return coords
        else:
            self.landmarks = None
            return None

    def draw_skeleton(self, frame: np.ndarray):
        """Draw a clean, professional skeleton with finger-colored bones."""
        if self.landmarks is None:
            return

        lm = self.landmarks

        # Draw finger bones with finger colors (subtle)
        for finger_key, connections in FINGER_BONE_GROUPS.items():
            color = COLORS[finger_key]
            # Mute the color for bones
            muted = tuple(max(0, min(255, int(c * 0.5))) for c in color)
            for (a, b) in connections:
                if a == 0:
                    # Wrist connections are more subtle
                    cv2.line(frame, lm[a], lm[b], COLORS["bone"], 1, cv2.LINE_AA)
                else:
                    cv2.line(frame, lm[a], lm[b], muted, 2, cv2.LINE_AA)

        # Palm connections
        for (a, b) in [(5, 9), (9, 13), (13, 17)]:
            cv2.line(frame, lm[a], lm[b], COLORS["bone"], 1, cv2.LINE_AA)

        # Draw landmark dots
        for i, (x, y) in enumerate(lm):
            if i in FINGER_TIPS:
                finger_idx = FINGER_TIPS.index(i)
                color = COLORS[FINGER_KEYS[finger_idx]]
                # Outer ring + inner dot
                cv2.circle(frame, (x, y), 7, color, 1, cv2.LINE_AA)
                cv2.circle(frame, (x, y), 3, color, -1, cv2.LINE_AA)
            elif i == LandmarkIndex.WRIST:
                cv2.circle(frame, (x, y), 5, COLORS["wrist"], 1, cv2.LINE_AA)
                cv2.circle(frame, (x, y), 2, COLORS["wrist"], -1, cv2.LINE_AA)
            else:
                cv2.circle(frame, (x, y), 2, COLORS["joint"], -1, cv2.LINE_AA)

    def draw_trails(self, frame: np.ndarray):
        """Draw clean, fading motion trails for fingertips."""
        if self.landmarks is None:
            return

        for idx, tip in enumerate(FINGER_TIPS):
            trail = self._trails[tip]
            if len(trail) < 2:
                continue
            color = COLORS[FINGER_KEYS[idx]]
            for i in range(1, len(trail)):
                alpha = i / len(trail)
                faded = tuple(max(0, min(255, int(c * alpha * 0.5))) for c in color)
                thickness = max(1, int(alpha * 2))
                cv2.line(frame, trail[i - 1], trail[i], faded, thickness, cv2.LINE_AA)

    def release(self):
        """Release MediaPipe resources."""
        self.landmarker.close()
