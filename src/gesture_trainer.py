"""
gesture_trainer.py - Record your own gestures and have them recognised.

The geometric recogniser in :mod:`src.gestures` knows eight poses and cannot
learn a ninth without someone writing new rules. This module learns from
examples instead: hold a pose, record a few samples, give it a name, and it is
recognised from then on.

How it works, and why this rather than a neural network:

A hand is turned into a **descriptor** -- a 21x2 vector that has been
translated to the wrist, scaled by hand size, and rotated so the middle
metacarpal points a fixed way. After that normalisation, the same pose made by
a large hand at the edge of frame and a small one in the centre produces
nearly the same numbers. Recognition is then nearest-neighbour by Euclidean
distance against the stored templates.

That is a deliberate choice. It trains from three samples instead of three
thousand, it runs in microseconds, the stored model is readable JSON, and when
it gets something wrong you can see exactly which template it matched and by
how far. A small network would need far more data to beat it on eight classes
and would be much harder to debug.

Models are plain JSON -- no pickle, so loading someone else's gesture file
cannot execute code.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.hand_tracker import LandmarkIndex
from src.utils import euclidean_distance

__all__ = [
    "GestureTemplate",
    "TrainedGestureRecognizer",
    "describe_landmarks",
    "describe_with_orientation",
]

#: Bumped when the descriptor changes, so an old model is refused rather than
#: silently compared against incompatible numbers.
MODEL_VERSION = 1


def describe_landmarks(landmarks: Sequence[Tuple[int, int]]) -> np.ndarray:
    """Turn 21 landmarks into a pose descriptor invariant to place and size.

    Three normalisations, in order:

    1. **Translate** so the wrist is the origin -- removes where in frame the
       hand is.
    2. **Scale** by the wrist-to-middle-MCP bone -- removes how near the
       camera it is. That bone is used because its length does not change
       when the fingers move.
    3. **Rotate** so that bone points along +y -- removes hand roll, so a
       tilted peace sign matches an upright one.

    Returns:
        A flat array of 42 floats: 21 landmarks, x and y interleaved.

    Raises:
        ValueError: If fewer than 21 landmarks are supplied.

    Example:
        >>> import numpy as np
        >>> from tests.synthetic_hands import peace
        >>> a = describe_landmarks(peace())
        >>> b = describe_landmarks(peace(scale=180.0, rotation=40.0))
        >>> bool(np.linalg.norm(a - b) < 0.35)   # same pose, very different framing
        True
    """
    if landmarks is None or len(landmarks) < 21:
        raise ValueError(
            f"A hand has 21 landmarks; got {0 if landmarks is None else len(landmarks)}."
        )

    points = np.asarray(landmarks[:21], dtype=np.float64)
    wrist = points[LandmarkIndex.WRIST].copy()
    points -= wrist

    reference = points[LandmarkIndex.MIDDLE_MCP]
    scale = float(np.hypot(*reference))
    if scale < 1e-6:
        raise ValueError("Degenerate hand: the wrist and middle MCP coincide.")
    points /= scale

    # Rotate the (now unit-length) reference bone onto +y.
    angle = math.atan2(reference[1], reference[0]) - math.pi / 2.0
    cos_a, sin_a = math.cos(-angle), math.sin(-angle)
    rotation = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float64)
    points = points @ rotation.T

    return points.reshape(-1)


def describe_with_orientation(landmarks: Sequence[Tuple[int, int]]) -> np.ndarray:
    """The 42-value descriptor plus 2 values giving the hand's orientation.

    :func:`describe_landmarks` rotation-normalises on purpose, so a tilted
    peace sign matches an upright one. For most gestures that is exactly
    right. For two of them it is fatal: a thumbs-up and a thumbs-down are the
    *same shape* and differ only in which way it points, so rotation
    normalisation deletes the only thing that tells them apart.

    Measured: a classifier trained on the 42-value descriptor alone reaches
    100% on the other six gestures and chance -- 0.34 and 0.65 -- on thumbs-up
    and thumbs-down. Appending the orientation fixes that without giving up
    invariance for the poses that need it, because the shape half of the
    vector is still normalised.

    Returns:
        44 floats: the 42 normalised coordinates, then ``cos`` and ``sin`` of
        the wrist-to-middle-MCP direction in image space. Sine and cosine
        rather than the angle itself, so that 359 degrees and 1 degree are
        adjacent numbers instead of opposite ends of a range.
    """
    shape = describe_landmarks(landmarks)
    wrist = np.asarray(landmarks[LandmarkIndex.WRIST], dtype=np.float64)
    knuckle = np.asarray(landmarks[LandmarkIndex.MIDDLE_MCP], dtype=np.float64)
    direction = knuckle - wrist
    norm = float(np.hypot(*direction))
    if norm < 1e-9:
        return np.concatenate([shape, [0.0, 0.0]])
    return np.concatenate([shape, direction / norm])


@dataclass
class GestureTemplate:
    """One learned gesture.

    Attributes:
        name: What the user called it.
        samples: Descriptors recorded for it.
        threshold: Distance beyond which a match is rejected. Derived from how
            varied the samples were, so a pose recorded consistently gets a
            tight threshold and a sloppy one gets a loose one.
    """

    name: str
    samples: List[np.ndarray] = field(default_factory=list)
    threshold: float = 0.0

    @property
    def centroid(self) -> np.ndarray:
        """Mean descriptor across the samples."""
        return np.mean(np.stack(self.samples), axis=0)

    def spread(self) -> float:
        """Mean distance from the samples to their own centroid."""
        if len(self.samples) < 2:
            return 0.0
        centre = self.centroid
        return float(np.mean([np.linalg.norm(s - centre) for s in self.samples]))

    def distance(self, descriptor: np.ndarray) -> float:
        """Distance to the nearest sample of this gesture.

        Nearest sample rather than distance to the centroid: a gesture can be
        made in genuinely different ways, and averaging them produces a
        centre that matches none of them.
        """
        return float(min(np.linalg.norm(descriptor - s) for s in self.samples))

    def to_json(self) -> dict:
        return {
            "name": self.name,
            "threshold": self.threshold,
            "samples": [s.tolist() for s in self.samples],
        }

    @classmethod
    def from_json(cls, data: dict) -> "GestureTemplate":
        return cls(
            name=data["name"],
            samples=[np.asarray(s, dtype=np.float64) for s in data["samples"]],
            threshold=float(data.get("threshold", 0.0)),
        )


class TrainedGestureRecognizer:
    """Learns gestures from examples and recognises them.

    Args:
        base_threshold: Distance below which an unseen pose matches a template
            with only one sample, where no spread can be measured.
        margin: How much slack to add over a template's observed spread when
            setting its threshold.

    Example:
        >>> from tests.synthetic_hands import peace, fist
        >>> recognizer = TrainedGestureRecognizer()
        >>> for _ in range(3):
        ...     _ = recognizer.record("my-peace", peace())
        >>> recognizer.finalise()
        >>> name, distance = recognizer.predict(peace(scale=140.0))
        >>> name
        'my-peace'
    """

    def __init__(self, base_threshold: float = 0.55, margin: float = 2.5) -> None:
        self.base_threshold = float(base_threshold)
        self.margin = float(margin)
        self.templates: Dict[str, GestureTemplate] = {}

    # ------------------------------------------------------------------
    def record(self, name: str, landmarks: Sequence[Tuple[int, int]]) -> int:
        """Add one sample for ``name``. Returns how many it now has."""
        if not name or not name.strip():
            raise ValueError("A gesture needs a name.")
        descriptor = describe_landmarks(landmarks)
        template = self.templates.setdefault(name, GestureTemplate(name=name))
        template.samples.append(descriptor)
        return len(template.samples)

    def finalise(self) -> None:
        """Set each template's threshold from how varied its samples were.

        A pose recorded consistently gets a tight threshold; one recorded
        loosely gets a wide one. That is better than a single global number,
        which would be too strict for the sloppy gestures and too generous for
        the precise ones.
        """
        for template in self.templates.values():
            spread = template.spread()
            template.threshold = (
                self.base_threshold if spread <= 1e-9
                else max(self.base_threshold * 0.6, spread * self.margin)
            )

    def forget(self, name: str) -> bool:
        """Remove a gesture. Returns False if it was not there."""
        return self.templates.pop(name, None) is not None

    # ------------------------------------------------------------------
    def predict(
        self, landmarks: Optional[Sequence[Tuple[int, int]]]
    ) -> Tuple[Optional[str], float]:
        """Return ``(name, distance)``, or ``(None, distance)`` for no match.

        ``None`` means every template was further away than its own threshold.
        Reporting that honestly matters more than always returning the closest
        label: an unknown pose is not a weak match for the nearest gesture.
        """
        if landmarks is None or not self.templates:
            return None, float("inf")

        descriptor = describe_landmarks(landmarks)
        scored = sorted(
            ((t.distance(descriptor), t) for t in self.templates.values()),
            key=lambda pair: pair[0],
        )
        distance, template = scored[0]
        if distance <= template.threshold:
            return template.name, distance
        return None, distance

    def rank(self, landmarks: Sequence[Tuple[int, int]]) -> List[Tuple[str, float]]:
        """Every gesture with its distance, nearest first. For debugging."""
        descriptor = describe_landmarks(landmarks)
        return sorted(
            ((t.name, t.distance(descriptor)) for t in self.templates.values()),
            key=lambda pair: pair[1],
        )

    # ------------------------------------------------------------------
    def save(self, path: str = "gestures.json") -> str:
        """Write the model as JSON and return the path.

        JSON rather than pickle, deliberately: a gesture file is something
        people will share, and loading a shared pickle executes whatever is
        inside it.
        """
        payload = {
            "version": MODEL_VERSION,
            "base_threshold": self.base_threshold,
            "margin": self.margin,
            "templates": [t.to_json() for t in self.templates.values()],
        }
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        print(f"[GestureTrainer] {len(self.templates)} gesture(s) -> {path}")
        return path

    @classmethod
    def open_for_recording(cls, path: str = "gestures.json") -> "TrainedGestureRecognizer":
        """Return a recogniser to record into, loaded from ``path`` if it exists.

        Recording writes the whole model back with :meth:`save`, so starting
        from an empty recogniser would delete every gesture already in the
        file the moment a new one was saved. Building a multi-gesture set
        across sessions only works if each session starts from what the last
        one left behind.

        Raises:
            ValueError: If ``path`` exists but cannot be read. Overwriting a
                file we failed to understand would destroy it, so the caller
                is told to stop rather than silently starting fresh.
        """
        if not os.path.exists(path):
            return cls()
        try:
            return cls.load(path)
        except Exception as error:
            raise ValueError(
                f"{path} exists but could not be read ({error}). Recording "
                "into it would overwrite it, so nothing was opened."
            ) from error

    @classmethod
    def load(cls, path: str = "gestures.json") -> "TrainedGestureRecognizer":
        """Read a model back.

        Raises:
            FileNotFoundError: If the file is missing.
            ValueError: If it was written by a different descriptor version,
                since the stored numbers would not be comparable.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"No gesture model at {path}")
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)

        version = payload.get("version")
        if version != MODEL_VERSION:
            raise ValueError(
                f"{path} was written with descriptor version {version}, but this "
                f"build uses version {MODEL_VERSION}. Re-record the gestures."
            )

        recognizer = cls(
            base_threshold=float(payload.get("base_threshold", 0.55)),
            margin=float(payload.get("margin", 2.5)),
        )
        for entry in payload.get("templates", []):
            template = GestureTemplate.from_json(entry)
            recognizer.templates[template.name] = template
        return recognizer

    def describe(self) -> str:
        """One line per gesture, for the console."""
        if not self.templates:
            return "no gestures recorded"
        return "; ".join(
            f"{t.name} ({len(t.samples)} samples, threshold {t.threshold:.2f})"
            for t in self.templates.values()
        )
