"""Regenerate ports/fixtures/conformance.json from the Python implementation.

Run this after changing anything the ports mirror -- gesture thresholds, the
radius pairs, the Kalman defaults -- then rerun ports/run_conformance.sh to
find out which ports need the same change.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.gestures import GestureRecognizer
from src.kalman import KalmanFilter1D
from src.radius_calculator import RADIUS_PAIRS
from src.utils import euclidean_distance
from tests import synthetic_hands as hands

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "ports", "fixtures", "conformance.json")

BUILDERS = [
    ("fist", hands.fist), ("open_palm", hands.open_palm),
    ("peace", hands.peace), ("pointing", hands.pointing),
    ("thumbs_up", hands.thumbs_up), ("thumbs_down", hands.thumbs_down),
    ("ok_sign", hands.ok_sign), ("pinch", hands.pinch),
]


def main() -> int:
    recognizer = GestureRecognizer(hold_frames=1)
    cases = []
    for name, builder in BUILDERS:
        for scale, rotation in [(90.0, 0.0), (160.0, 45.0), (55.0, 210.0)]:
            landmarks = builder(scale=scale, rotation=rotation)
            result = recognizer.classify(landmarks)
            cases.append({
                "id": f"{name}_s{scale:g}_r{rotation:g}",
                "landmarks": [[int(x), int(y)] for x, y in landmarks],
                "expected": {
                    "gesture": result.gesture,
                    "fingers": result.fingers,
                    "extended_count": result.extended_count,
                    "radii": {
                        pair: round(euclidean_distance(landmarks[a], landmarks[b]), 9)
                        for pair, a, b in RADIUS_PAIRS
                    },
                },
            })

    rng = np.random.default_rng(42)
    signal = [
        float(round(120 + 60 * np.sin(i / 9.0) + rng.normal(0, 6), 6))
        for i in range(60)
    ]
    filt = KalmanFilter1D()
    expected = [round(filt.update(v), 9) for v in signal]

    payload = {
        "description": (
            "Conformance fixture. Every port must reproduce these outputs "
            "exactly (gestures, finger flags) and to 1e-6 (radii, Kalman)."
        ),
        "kalman": {
            "process_noise": 0.05,
            "measurement_noise": 36.0,
            "input": signal,
            "expected": expected,
        },
        "gestures": cases,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=1)
    print(f"{OUT}: {len(cases)} gesture cases, {len(signal)} Kalman samples")
    return 0


if __name__ == "__main__":
    sys.exit(main())
