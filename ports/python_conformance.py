"""Check the Python implementation still matches its own fixture.

The fixture was generated from this code, so a failure here means the Python
implementation changed without the fixture being regenerated -- in which case
every other port is now being held to a stale standard.
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.gestures import GestureRecognizer
from src.kalman import KalmanFilter1D
from src.radius_calculator import RADIUS_PAIRS
from src.utils import euclidean_distance

TOLERANCE = 1e-6
FIXTURE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "fixtures", "conformance.json")


def main() -> int:
    with open(FIXTURE, encoding="utf-8") as handle:
        fixture = json.load(handle)

    failures = []
    checks = 0
    recognizer = GestureRecognizer(hold_frames=1)

    for case in fixture["gestures"]:
        landmarks = [tuple(p) for p in case["landmarks"]]
        expected = case["expected"]
        result = recognizer.classify(landmarks)
        checks += 1

        if result.gesture != expected["gesture"]:
            failures.append(f"{case['id']}: {result.gesture} != {expected['gesture']}")
        if result.fingers != expected["fingers"]:
            failures.append(f"{case['id']}: finger states differ")

        for name, a, b in RADIUS_PAIRS:
            checks += 1
            got = euclidean_distance(landmarks[a], landmarks[b])
            want = expected["radii"][name]
            if abs(got - want) > TOLERANCE:
                failures.append(f"{case['id']}: radius {name} {got} != {want}")

    kalman = fixture["kalman"]
    filt = KalmanFilter1D(
        process_noise=kalman["process_noise"],
        measurement_noise=kalman["measurement_noise"],
    )
    worst = 0.0
    for i, value in enumerate(kalman["input"]):
        checks += 1
        got = filt.update(value)
        delta = abs(got - kalman["expected"][i])
        worst = max(worst, delta)
        if delta > TOLERANCE:
            failures.append(f"kalman[{i}]: delta {delta:.3e}")

    if failures:
        print(f"python: FAIL - {len(failures)} of {checks} checks")
        for line in failures[:10]:
            print(f"  {line}")
        return 1
    print(f"python: PASS - {checks} checks against the Python fixture "
          f"(Kalman worst {worst:.3e})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
