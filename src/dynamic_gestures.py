"""
dynamic_gestures.py - Gestures that only exist over time.

A static recogniser looks at one frame and asks what shape the hand is in. It
can never see a swipe, because a swipe is not a shape -- it is a shape that
moved. This module keeps a short history of fingertip positions and classifies
the *trajectory*.

Everything is scale-normalised by hand size, the same reason the static
recogniser uses ratios: a swipe near the camera covers far more pixels than
the same swipe at arm's length, and a pixel threshold would only work at one
distance.

Recognised motions:

* **Swipe** left, right, up, down -- fast, straight, sustained travel.
* **Tap** -- the fingertip drops toward the camera and returns, which shows up
  in the z coordinate rather than in x or y.
* **Hold** -- the hand stays still for a while. Useful as a trigger that
  cannot be produced accidentally while moving.
* **Circle** -- the trajectory closes on itself with consistent turning.

A gesture fires once and then locks out for ``cooldown_frames`` so a single
swipe is not reported thirty times as the hand decelerates.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Iterable, List, Optional, Sequence, Tuple

from src.hand_tracker import LandmarkIndex
from src.utils import euclidean_distance

__all__ = ["DynamicGesture", "MotionEvent", "DynamicGestureRecognizer"]


class DynamicGesture:
    """Identifiers for motions. Strings, so they survive CSV and JSON."""

    NONE = "None"
    SWIPE_LEFT = "Swipe Left"
    SWIPE_RIGHT = "Swipe Right"
    SWIPE_UP = "Swipe Up"
    SWIPE_DOWN = "Swipe Down"
    TAP = "Tap"
    HOLD = "Hold"
    CIRCLE = "Circle"


@dataclass
class MotionEvent:
    """One recognised motion.

    Attributes:
        gesture: Which motion, or ``DynamicGesture.NONE``.
        confidence: 0.0-1.0, from how cleanly the trajectory matched.
        speed: Travel in hand-widths per frame, averaged over the window.
        direction_degrees: Heading of the movement, 0 = right, 90 = down.
        travel: Total distance covered, in hand-widths.
    """

    gesture: str = DynamicGesture.NONE
    confidence: float = 0.0
    speed: float = 0.0
    direction_degrees: float = 0.0
    travel: float = 0.0

    @property
    def is_swipe(self) -> bool:
        return self.gesture in {
            DynamicGesture.SWIPE_LEFT, DynamicGesture.SWIPE_RIGHT,
            DynamicGesture.SWIPE_UP, DynamicGesture.SWIPE_DOWN,
        }

    def __str__(self) -> str:
        if self.gesture == DynamicGesture.NONE:
            return "-"
        return f"{self.gesture} ({self.confidence:.0%})"


class DynamicGestureRecognizer:
    """Classifies fingertip trajectories into motions.

    Args:
        history: How many frames of motion to consider. At 30 fps, 16 frames
            is a little over half a second -- long enough to contain a swipe,
            short enough not to merge two of them.
        swipe_travel: Minimum distance, in hand-widths, for a swipe. Hand
            width is the wrist-to-middle-MCP bone, so this is distance-
            invariant.
        swipe_straightness: How direct the path must be, as the ratio of
            straight-line distance to path length. 1.0 is a perfect line;
            a wandering path scores low and is rejected.
        hold_movement: Below this much travel the hand counts as still.
        circle_turn: Degrees of accumulated turn needed for a circle.
        cooldown_frames: Frames to stay silent after firing, so one swipe is
            reported once.

    Example:
        >>> recognizer = DynamicGestureRecognizer()
        >>> event = recognizer.update(landmarks)      # doctest: +SKIP
        >>> event.gesture                             # doctest: +SKIP
        'Swipe Right'
    """

    def __init__(
        self,
        history: int = 16,
        swipe_travel: float = 1.6,
        swipe_straightness: float = 0.82,
        hold_movement: float = 0.16,
        hold_frames: int = 24,
        tap_depth: float = 0.35,
        circle_turn: float = 200.0,
        cooldown_frames: int = 12,
    ) -> None:
        if history < 4:
            raise ValueError(f"history must be at least 4 frames, got {history}.")

        self.history = int(history)
        self.swipe_travel = float(swipe_travel)
        self.swipe_straightness = float(swipe_straightness)
        self.hold_movement = float(hold_movement)
        self.hold_frames = int(hold_frames)
        self.tap_depth = float(tap_depth)
        self.circle_turn = float(circle_turn)
        self.cooldown_frames = int(cooldown_frames)

        self._points: Deque[Tuple[float, float]] = deque(maxlen=self.history)
        self._depths: Deque[float] = deque(maxlen=self.history)
        self._scales: Deque[float] = deque(maxlen=self.history)
        self._still_for = 0
        self._cooldown = 0
        self.last_event = MotionEvent()

    # ------------------------------------------------------------------
    @staticmethod
    def _scale(landmarks: Sequence[Tuple[int, int]]) -> float:
        """Hand size in pixels: the wrist-to-middle-MCP bone."""
        return max(
            1.0,
            euclidean_distance(
                landmarks[LandmarkIndex.WRIST], landmarks[LandmarkIndex.MIDDLE_MCP]
            ),
        )

    def _path_metrics(self) -> Tuple[float, float, float, float]:
        """Return (travel, straightness, heading_degrees, net_distance).

        ``travel`` is the length of the path walked; ``net_distance`` is the
        straight line from first point to last. Their ratio is what separates
        a swipe from a hand waving back and forth over the same ground.
        """
        points = list(self._points)
        if len(points) < 2:
            return 0.0, 0.0, 0.0, 0.0

        scale = sum(self._scales) / len(self._scales)
        path = 0.0
        for a, b in zip(points, points[1:]):
            path += math.dist(a, b)
        path /= scale

        dx = (points[-1][0] - points[0][0]) / scale
        dy = (points[-1][1] - points[0][1]) / scale
        net = math.hypot(dx, dy)

        straightness = net / path if path > 1e-9 else 0.0
        heading = math.degrees(math.atan2(dy, dx)) % 360.0
        return path, straightness, heading, net

    def _turning(self) -> float:
        """Total signed turn along the path, in degrees.

        A circle accumulates roughly +/-360; a straight line accumulates
        almost nothing. Signed, so a figure-of-eight cancels rather than
        counting as two circles.
        """
        points = list(self._points)
        if len(points) < 4:
            return 0.0
        total = 0.0
        for a, b, c in zip(points, points[1:], points[2:]):
            v1 = math.atan2(b[1] - a[1], b[0] - a[0])
            v2 = math.atan2(c[1] - b[1], c[0] - b[0])
            delta = math.degrees(v2 - v1)
            # Wrap into (-180, 180] so crossing the +/-pi boundary does not
            # register as a 350-degree turn.
            total += (delta + 180.0) % 360.0 - 180.0
        return total

    def _turn_consistency(self) -> float:
        """Fraction of turns that share the majority direction, 0.0-1.0.

        A circle turns the same way throughout and scores near 1.0. A zigzag
        alternates and scores near 0.5, which is what keeps it from
        accumulating enough turn to be mistaken for a circle.
        """
        points = list(self._points)
        if len(points) < 4:
            return 0.0
        signs: List[int] = []
        for a, b, c in zip(points, points[1:], points[2:]):
            v1 = math.atan2(b[1] - a[1], b[0] - a[0])
            v2 = math.atan2(c[1] - b[1], c[0] - b[0])
            delta = (math.degrees(v2 - v1) + 180.0) % 360.0 - 180.0
            if abs(delta) > 1.0:
                signs.append(1 if delta > 0 else -1)
        if not signs:
            return 0.0
        return max(signs.count(1), signs.count(-1)) / len(signs)

    def _tap_detected(self) -> bool:
        """True when the fingertip dipped toward the camera and came back.

        MediaPipe's z is negative toward the camera, so a tap is a dip in z
        followed by a recovery -- a V shape in depth with little x/y travel.
        """
        depths = list(self._depths)
        if len(depths) < 6:
            return False
        scale = sum(self._scales) / len(self._scales)
        lowest = min(depths)
        index = depths.index(lowest)
        # The dip has to be in the middle: a still-falling finger is not a tap.
        if index == 0 or index == len(depths) - 1:
            return False
        approach = (depths[0] - lowest) / scale
        retreat = (depths[-1] - lowest) / scale
        return approach > self.tap_depth and retreat > self.tap_depth * 0.6

    # ------------------------------------------------------------------
    def update(
        self,
        landmarks: Optional[Sequence[Tuple[int, int]]],
        landmarks_3d: Optional[Sequence[Tuple[int, int, float]]] = None,
        tip: int = LandmarkIndex.INDEX_TIP,
    ) -> MotionEvent:
        """Feed one frame and get any motion recognised this frame.

        Args:
            landmarks: 2-D landmarks, or ``None`` when no hand is present,
                which clears the history.
            landmarks_3d: Optional 3-D landmarks. Tap detection needs these;
                without them everything else still works.
            tip: Which landmark to track. The index fingertip by default.

        Returns:
            A :class:`MotionEvent`. ``DynamicGesture.NONE`` most frames --
            a motion fires on the frame it completes.
        """
        if landmarks is None:
            self.reset()
            return MotionEvent()

        scale = self._scale(landmarks)
        self._points.append((float(landmarks[tip][0]), float(landmarks[tip][1])))
        self._scales.append(scale)
        if landmarks_3d is not None:
            self._depths.append(float(landmarks_3d[tip][2]))

        if self._cooldown > 0:
            self._cooldown -= 1
            return MotionEvent()

        if len(self._points) < self.history:
            return MotionEvent()

        travel, straightness, heading, net = self._path_metrics()
        speed = travel / max(1, len(self._points) - 1)

        # Tap is checked BEFORE hold, and deliberately so. A tap moves toward
        # the camera, not across it, so its lateral travel is near zero -- the
        # stillness test below would otherwise swallow every tap before this
        # ever ran.
        if travel < self.swipe_travel and self._tap_detected():
            return self._fire(MotionEvent(DynamicGesture.TAP, 0.85, speed, heading, travel))

        # Still: track how long, then fire a hold once.
        if travel < self.hold_movement:
            self._still_for += 1
            if self._still_for == self.hold_frames:
                return self._fire(MotionEvent(
                    DynamicGesture.HOLD, 0.9, speed, heading, travel
                ))
            return MotionEvent()
        self._still_for = 0

        # Circle: sustained turning in one direction that comes back on
        # itself. The window rarely holds a whole revolution -- a circle drawn
        # over 24 frames shows about 218 degrees inside a 16-frame window --
        # so the threshold is set for a strong arc, and consistency of
        # direction is what stops a zigzag accumulating its way over the line.
        turning = self._turning()
        if (
            abs(turning) > self.circle_turn
            and self._turn_consistency() > 0.7
            and net < travel * 0.45
            and travel > self.swipe_travel
        ):
            return self._fire(MotionEvent(
                DynamicGesture.CIRCLE, min(0.95, abs(turning) / 360.0), speed, heading, travel
            ))

        # Swipe: far enough, straight enough.
        if travel >= self.swipe_travel and straightness >= self.swipe_straightness:
            return self._fire(MotionEvent(
                self._direction(heading), min(0.99, straightness), speed, heading, travel
            ))

        return MotionEvent()

    @staticmethod
    def _direction(heading: float) -> str:
        """Map a heading in degrees to one of four swipes.

        Quadrants are centred on the axes -- 45 degrees either side -- so a
        diagonal resolves to whichever axis it is closer to rather than being
        rejected.
        """
        if heading < 45.0 or heading >= 315.0:
            return DynamicGesture.SWIPE_RIGHT
        if heading < 135.0:
            return DynamicGesture.SWIPE_DOWN      # image y grows downward
        if heading < 225.0:
            return DynamicGesture.SWIPE_LEFT
        return DynamicGesture.SWIPE_UP

    def _fire(self, event: MotionEvent) -> MotionEvent:
        """Emit an event, then go quiet so one motion is reported once."""
        self._cooldown = self.cooldown_frames
        self._points.clear()
        self._depths.clear()
        self._scales.clear()
        self._still_for = 0
        self.last_event = event
        return event

    def reset(self) -> None:
        """Forget all history. Call when the hand leaves the frame."""
        self._points.clear()
        self._depths.clear()
        self._scales.clear()
        self._still_for = 0
        self._cooldown = 0


def retire_absent(
    recognizers: Sequence["DynamicGestureRecognizer"],
    present: Iterable[int],
) -> None:
    """Reset every recogniser whose hand was not seen this frame.

    A recogniser only hears about a hand while that hand is detected, so one
    that drops out keeps its trajectory forever. If the hand comes back
    somewhere else, the jump between the old and new fingertip positions is a
    large, fast displacement -- indistinguishable from a real swipe, and quite
    capable of completing a circle. Clearing the history on the frames where
    the hand is missing is what keeps the two apart.
    """
    seen = set(present)
    for index, recognizer in enumerate(recognizers):
        if index not in seen:
            recognizer.reset()
