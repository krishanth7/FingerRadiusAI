"""
hand_tracker.py - MediaPipe Tasks API multi-hand landmark detection.
Supports tracking up to 2 hands simultaneously with per-hand smoothing and trails.
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

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


class LandmarkIndex:
    WRIST = 0
    THUMB_CMC = 1;  THUMB_MCP = 2;  THUMB_IP = 3;   THUMB_TIP = 4
    INDEX_MCP = 5;  INDEX_PIP = 6;  INDEX_DIP = 7;   INDEX_TIP = 8
    MIDDLE_MCP = 9; MIDDLE_PIP = 10; MIDDLE_DIP = 11; MIDDLE_TIP = 12
    RING_MCP = 13;  RING_PIP = 14;  RING_DIP = 15;   RING_TIP = 16
    PINKY_MCP = 17; PINKY_PIP = 18; PINKY_DIP = 19;  PINKY_TIP = 20


FINGER_TIPS = [
    LandmarkIndex.THUMB_TIP, LandmarkIndex.INDEX_TIP,
    LandmarkIndex.MIDDLE_TIP, LandmarkIndex.RING_TIP,
    LandmarkIndex.PINKY_TIP,
]

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]

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

# Hand label colors for distinguishing Left vs Right
HAND_LABEL_COLORS = {
    "Left":  COLORS["accent"],
    "Right": COLORS["info"],
}


class _HandState:
    """Per-hand smoothing and trail state."""

    def __init__(self, smoothing_alpha: float, trail_length: int):
        self.smoothers_x = [ExponentialMovingAverage(smoothing_alpha) for _ in range(21)]
        self.smoothers_y = [ExponentialMovingAverage(smoothing_alpha) for _ in range(21)]
        self.smoothers_z = [ExponentialMovingAverage(smoothing_alpha) for _ in range(21)]
        self.trails = {tip: deque(maxlen=trail_length) for tip in FINGER_TIPS}
        self.landmarks: Optional[List[Tuple[int, int]]] = None
        self.landmarks_3d: Optional[List[Tuple[int, int, float]]] = None
        self.label: str = "Unknown"  # "Left" or "Right"

    def reset_trails(self):
        for t in self.trails.values():
            t.clear()


class HandTracker:
    """
    Multi-hand landmark tracker using MediaPipe Tasks API.
    Tracks up to max_num_hands simultaneously with independent smoothing.
    """

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.6,
        trail_length: int = 15,
        smoothing_alpha: float = 0.4,
    ):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found: {model_path}\n"
                "Download from: https://storage.googleapis.com/mediapipe-models/"
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
        self.max_num_hands = max_num_hands

        # Per-hand state (indexed 0, 1)
        self._hand_states = [
            _HandState(smoothing_alpha, trail_length)
            for _ in range(max_num_hands)
        ]

        # Public: list of detected hand data this frame
        self.hands: List[_HandState] = []
        self.num_hands: int = 0

    def process(self, frame: np.ndarray) -> int:
        """
        Detect hands. Returns the number of hands found (0, 1, or 2).
        Access results via self.hands list.
        """
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        self._frame_timestamp_ms += 33
        result = self.landmarker.detect_for_video(mp_image, self._frame_timestamp_ms)

        num = len(result.hand_landmarks) if result.hand_landmarks else 0
        self.num_hands = num
        self.hands = []

        for hand_idx in range(num):
            hand_lm = result.hand_landmarks[hand_idx]
            state = self._hand_states[hand_idx]

            # Handedness label (MediaPipe gives mirrored label, so we flip)
            if result.handedness and hand_idx < len(result.handedness):
                raw_label = result.handedness[hand_idx][0].category_name
                # MediaPipe reports from camera's perspective; flip for mirrored view
                state.label = "Right" if raw_label == "Left" else "Left"
            else:
                state.label = f"Hand {hand_idx + 1}"

            coords = []
            coords_3d = []
            for i, lm in enumerate(hand_lm):
                px = int(lm.x * w)
                py = int(lm.y * h)
                # z is relative depth; scale to pixel-space using image width
                pz = lm.z * w
                sx = int(state.smoothers_x[i].update(px))
                sy = int(state.smoothers_y[i].update(py))
                sz = state.smoothers_z[i].update(pz)
                coords.append((sx, sy))
                coords_3d.append((sx, sy, sz))

            for tip_idx in FINGER_TIPS:
                state.trails[tip_idx].append(coords[tip_idx])

            state.landmarks = coords
            state.landmarks_3d = coords_3d
            self.hands.append(state)

        # Clear state for hands that disappeared
        for hand_idx in range(num, self.max_num_hands):
            self._hand_states[hand_idx].landmarks = None
            self._hand_states[hand_idx].landmarks_3d = None

        return num

    def draw_skeleton(self, frame: np.ndarray, hand_idx: int = 0):
        """Draw skeleton for a specific hand index."""
        if hand_idx >= len(self.hands):
            return
        state = self.hands[hand_idx]
        if state.landmarks is None:
            return

        lm = state.landmarks
        label_color = HAND_LABEL_COLORS.get(state.label, COLORS["text_secondary"])

        # Finger bones
        for finger_key, connections in FINGER_BONE_GROUPS.items():
            color = COLORS[finger_key]
            muted = tuple(max(0, min(255, int(c * 0.5))) for c in color)
            for (a, b) in connections:
                if a == 0:
                    cv2.line(frame, lm[a], lm[b], COLORS["bone"], 1, cv2.LINE_AA)
                else:
                    cv2.line(frame, lm[a], lm[b], muted, 2, cv2.LINE_AA)

        # Palm
        for (a, b) in [(5, 9), (9, 13), (13, 17)]:
            cv2.line(frame, lm[a], lm[b], COLORS["bone"], 1, cv2.LINE_AA)

        # Landmark dots
        for i, (x, y) in enumerate(lm):
            if i in FINGER_TIPS:
                fi = FINGER_TIPS.index(i)
                c = COLORS[FINGER_KEYS[fi]]
                cv2.circle(frame, (x, y), 7, c, 1, cv2.LINE_AA)
                cv2.circle(frame, (x, y), 3, c, -1, cv2.LINE_AA)
            elif i == LandmarkIndex.WRIST:
                cv2.circle(frame, (x, y), 5, COLORS["wrist"], 1, cv2.LINE_AA)
                cv2.circle(frame, (x, y), 2, COLORS["wrist"], -1, cv2.LINE_AA)
            else:
                cv2.circle(frame, (x, y), 2, COLORS["joint"], -1, cv2.LINE_AA)

        # Hand label near wrist
        wx, wy = lm[LandmarkIndex.WRIST]
        draw_label(frame, state.label, (wx - 15, wy + 20),
                   color=label_color, font_scale=0.4, bg=True)

    def draw_trails(self, frame: np.ndarray, hand_idx: int = 0):
        """Draw trails for a specific hand index."""
        if hand_idx >= len(self.hands):
            return
        state = self.hands[hand_idx]
        if state.landmarks is None:
            return

        for idx, tip in enumerate(FINGER_TIPS):
            trail = state.trails[tip]
            if len(trail) < 2:
                continue
            color = COLORS[FINGER_KEYS[idx]]
            for i in range(1, len(trail)):
                alpha = i / len(trail)
                faded = tuple(max(0, min(255, int(c * alpha * 0.5))) for c in color)
                thickness = max(1, int(alpha * 2))
                cv2.line(frame, trail[i - 1], trail[i], faded, thickness, cv2.LINE_AA)

    def draw_all(self, frame: np.ndarray, show_trails: bool = True):
        """Draw skeletons and trails for all detected hands."""
        for i in range(self.num_hands):
            self.draw_skeleton(frame, i)
            if show_trails:
                self.draw_trails(frame, i)

    def get_landmarks(self, hand_idx: int = 0):
        """Get 2D landmarks (x, y) for a specific hand, or None."""
        if hand_idx < len(self.hands) and self.hands[hand_idx].landmarks:
            return self.hands[hand_idx].landmarks
        return None

    def get_landmarks_3d(self, hand_idx: int = 0):
        """Get 3D landmarks (x, y, z) for a specific hand, or None."""
        if hand_idx < len(self.hands) and self.hands[hand_idx].landmarks_3d:
            return self.hands[hand_idx].landmarks_3d
        return None

    def get_label(self, hand_idx: int = 0) -> str:
        if hand_idx < len(self.hands):
            return self.hands[hand_idx].label
        return "N/A"

    def release(self):
        self.landmarker.close()
