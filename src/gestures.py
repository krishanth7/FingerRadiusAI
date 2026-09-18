"""
gestures.py - Static hand gesture recognition from MediaPipe landmarks.

The recogniser is deliberately geometric rather than learned: every decision
comes from distances and angles between the 21 landmarks, so it needs no
training data, no model file and no calibration, and it can be unit-tested
against synthetic hands.

Finger extension is decided by comparing how far the fingertip sits from the
wrist against how far the PIP joint sits from the wrist. That ratio is
invariant to hand size and to how close the hand is to the camera, which a
raw pixel threshold would not be. It is also rotation-invariant, so a gesture
is still recognised with the hand turned sideways -- which comparing y
coordinates (the common shortcut) cannot do.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from src.hand_tracker import LandmarkIndex
from src.utils import euclidean_distance

__all__ = [
    "Gesture",
    "GestureResult",
    "GestureRecognizer",
    "GESTURE_NAMES",
]


class Gesture:
    """Gesture identifiers. Plain strings so they survive CSV and JSON."""

    UNKNOWN = "Unknown"
    FIST = "Fist"
    OPEN_PALM = "Open Palm"
    PEACE = "Peace"
    THUMBS_UP = "Thumbs Up"
    THUMBS_DOWN = "Thumbs Down"
    OK = "OK"
    POINTING = "Pointing"
    PINCH = "Pinch"


GESTURE_NAMES: List[str] = [
    Gesture.FIST,
    Gesture.OPEN_PALM,
    Gesture.PEACE,
    Gesture.THUMBS_UP,
    Gesture.THUMBS_DOWN,
    Gesture.OK,
    Gesture.POINTING,
    Gesture.PINCH,
    Gesture.UNKNOWN,
]

#: (tip, pip, mcp) for the four fingers that bend the same way.
_FINGER_JOINTS: Dict[str, Tuple[int, int, int]] = {
    "index": (LandmarkIndex.INDEX_TIP, LandmarkIndex.INDEX_PIP, LandmarkIndex.INDEX_MCP),
    "middle": (LandmarkIndex.MIDDLE_TIP, LandmarkIndex.MIDDLE_PIP, LandmarkIndex.MIDDLE_MCP),
    "ring": (LandmarkIndex.RING_TIP, LandmarkIndex.RING_PIP, LandmarkIndex.RING_MCP),
    "pinky": (LandmarkIndex.PINKY_TIP, LandmarkIndex.PINKY_PIP, LandmarkIndex.PINKY_MCP),
}


@dataclass
class GestureResult:
    """One frame's recognition result.

    Attributes:
        gesture: The winning gesture name, or ``Gesture.UNKNOWN``.
        confidence: 0.0-1.0. How cleanly the finger pattern matched, not a
            probability from a model -- there is no model.
        fingers: Per-finger extension flags, thumb first.
        extended_count: How many fingers are extended.
        raw: The gesture before temporal stabilisation, useful for debugging
            a jittery reading.
    """

    gesture: str = Gesture.UNKNOWN
    confidence: float = 0.0
    fingers: Dict[str, bool] = field(default_factory=dict)
    extended_count: int = 0
    raw: str = Gesture.UNKNOWN

    def __str__(self) -> str:
        return f"{self.gesture} ({self.confidence:.0%})"


class GestureRecognizer:
    """Recognises static hand gestures from 21 landmarks.

    Args:
        hold_frames: How many consecutive frames a gesture must win before it
            is reported. Raw per-frame recognition flickers between
            neighbouring poses while a hand is moving; requiring a short run
            removes that without adding noticeable lag. Set to 1 to disable.
        pinch_ratio: Thumb-to-index tip distance, as a fraction of hand size,
            below which the two are considered touching. Scaled by hand size
            so it holds at any distance from the camera.
        extend_ratio: How much further than its PIP joint a fingertip must sit
            from the wrist to count as extended.
        thumb_extend_ratio: The same idea for the thumb, measured against the
            thumb MCP instead of a PIP joint.

    Example:
        >>> recognizer = GestureRecognizer(hold_frames=1)
        >>> result = recognizer.update(landmarks)   # doctest: +SKIP
        >>> result.gesture                          # doctest: +SKIP
        'Peace'
    """

    def __init__(
        self,
        hold_frames: int = 3,
        pinch_ratio: float = 0.28,
        extend_ratio: float = 1.12,
        thumb_extend_ratio: float = 1.45,
    ) -> None:
        if hold_frames < 1:
            raise ValueError(f"hold_frames must be at least 1, got {hold_frames}.")
        self.hold_frames = int(hold_frames)
        self.pinch_ratio = float(pinch_ratio)
        self.extend_ratio = float(extend_ratio)
        self.thumb_extend_ratio = float(thumb_extend_ratio)

        self._candidate: str = Gesture.UNKNOWN
        self._streak: int = 0
        self._stable: str = Gesture.UNKNOWN

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------
    @staticmethod
    def hand_scale(landmarks: Sequence[Tuple[int, int]]) -> float:
        """Return a size for this hand, in pixels.

        Wrist to middle-finger MCP: a bone that does not change length when
        the fingers move, which is exactly what a scale reference needs.
        """
        return max(
            1.0,
            euclidean_distance(
                landmarks[LandmarkIndex.WRIST], landmarks[LandmarkIndex.MIDDLE_MCP]
            ),
        )

    def _finger_extended(
        self, landmarks: Sequence[Tuple[int, int]], tip: int, pip: int
    ) -> bool:
        """True when the fingertip reaches further from the wrist than its PIP."""
        wrist = landmarks[LandmarkIndex.WRIST]
        return (
            euclidean_distance(wrist, landmarks[tip])
            > euclidean_distance(wrist, landmarks[pip]) * self.extend_ratio
        )

    def _thumb_extended(self, landmarks: Sequence[Tuple[int, int]]) -> bool:
        """True when the thumb is held out rather than folded across the palm.

        The thumb pivots sideways instead of curling, so the PIP comparison
        used for the other fingers does not apply. What does separate the two
        states is how far the tip reaches from the wrist relative to the thumb
        MCP: folded across the palm the tip comes back toward the wrist, held
        out it carries on away from it.

        Measured across the synthetic poses in the test suite this ratio is
        about 1.26 folded and about 1.77 extended, so the threshold sits
        between them with room on both sides. Comparing the tip against the
        index MCP instead -- the common shortcut -- fails on a thumbs-up,
        where the thumb points the same way as the knuckles.
        """
        wrist = landmarks[LandmarkIndex.WRIST]
        reach = euclidean_distance(wrist, landmarks[LandmarkIndex.THUMB_TIP])
        knuckle = max(1.0, euclidean_distance(wrist, landmarks[LandmarkIndex.THUMB_MCP]))
        return reach / knuckle > self.thumb_extend_ratio

    def finger_states(self, landmarks: Sequence[Tuple[int, int]]) -> Dict[str, bool]:
        """Return which fingers are extended, thumb first."""
        states = {"thumb": self._thumb_extended(landmarks)}
        for name, (tip, pip, _mcp) in _FINGER_JOINTS.items():
            states[name] = self._finger_extended(landmarks, tip, pip)
        return states

    def _pinching(self, landmarks: Sequence[Tuple[int, int]]) -> bool:
        """True when the thumb and index tips are touching."""
        scale = self.hand_scale(landmarks)
        gap = euclidean_distance(
            landmarks[LandmarkIndex.THUMB_TIP], landmarks[LandmarkIndex.INDEX_TIP]
        )
        return gap < scale * self.pinch_ratio

    @staticmethod
    def _thumb_direction(landmarks: Sequence[Tuple[int, int]]) -> float:
        """Signed vertical offset of the thumb tip from the wrist, in pixels.

        Negative is upward, because image y grows downward.
        """
        return landmarks[LandmarkIndex.THUMB_TIP][1] - landmarks[LandmarkIndex.WRIST][1]

    # ------------------------------------------------------------------
    # Recognition
    # ------------------------------------------------------------------
    def classify(self, landmarks: Sequence[Tuple[int, int]]) -> GestureResult:
        """Classify one frame with no temporal smoothing.

        Raises:
            ValueError: If fewer than 21 landmarks are supplied.
        """
        if landmarks is None or len(landmarks) < 21:
            raise ValueError(
                f"A hand has 21 landmarks; got {0 if landmarks is None else len(landmarks)}."
            )

        fingers = self.finger_states(landmarks)
        thumb = fingers["thumb"]
        index = fingers["index"]
        middle = fingers["middle"]
        ring = fingers["ring"]
        pinky = fingers["pinky"]
        count = sum(fingers.values())
        pinching = self._pinching(landmarks)
        scale = self.hand_scale(landmarks)

        gesture = Gesture.UNKNOWN
        confidence = 0.0

        # OK is checked before pinch: both have the thumb and index touching,
        # and the remaining three fingers are what separates them.
        # Both OK and pinch put the thumb and index tips together, so the
        # other three fingers are what tells them apart. Both also require an
        # extended index finger: in a closed fist the curled index tip comes
        # to rest beside the tucked thumb, which is close enough to look like
        # a pinch if only the gap is measured.
        if pinching and index and middle and ring and pinky:
            gesture = Gesture.OK
            confidence = 0.95
        elif index and middle and not ring and not pinky and not thumb:
            gesture = Gesture.PEACE
            confidence = 0.95
        elif thumb and count == 1:
            vertical = self._thumb_direction(landmarks)
            if vertical < -scale * 0.5:
                gesture = Gesture.THUMBS_UP
                confidence = 0.92
            elif vertical > scale * 0.5:
                gesture = Gesture.THUMBS_DOWN
                confidence = 0.92
            else:
                # Thumb out sideways is neither, and saying so beats guessing.
                gesture = Gesture.UNKNOWN
                confidence = 0.30
        elif index and count == 1:
            gesture = Gesture.POINTING
            confidence = 0.93
        elif pinching and index:
            gesture = Gesture.PINCH
            confidence = 0.85
        elif count == 0:
            gesture = Gesture.FIST
            confidence = 0.95
        elif count == 5:
            gesture = Gesture.OPEN_PALM
            confidence = 0.95
        else:
            gesture = Gesture.UNKNOWN
            confidence = 0.2

        return GestureResult(
            gesture=gesture,
            confidence=confidence,
            fingers=fingers,
            extended_count=count,
            raw=gesture,
        )

    def update(self, landmarks: Optional[Sequence[Tuple[int, int]]]) -> GestureResult:
        """Classify a frame and apply temporal stabilisation.

        A gesture is only reported once it has won ``hold_frames`` frames in a
        row. Passing ``None`` (no hand this frame) resets the streak.
        """
        if landmarks is None:
            self._candidate = Gesture.UNKNOWN
            self._streak = 0
            self._stable = Gesture.UNKNOWN
            return GestureResult()

        result = self.classify(landmarks)

        if result.raw == self._candidate:
            self._streak += 1
        else:
            self._candidate = result.raw
            self._streak = 1

        if self._streak >= self.hold_frames:
            self._stable = self._candidate

        result.gesture = self._stable
        if result.gesture != result.raw:
            # Reported gesture is the held one, so the confidence shown should
            # not be the incoming frame's.
            result.confidence = min(result.confidence, 0.5)
        return result

    def reset(self) -> None:
        """Forget the held gesture and the streak."""
        self._candidate = Gesture.UNKNOWN
        self._streak = 0
        self._stable = Gesture.UNKNOWN
