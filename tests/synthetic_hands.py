"""
synthetic_hands.py - Build MediaPipe-shaped landmark sets for tests.

There is no camera in CI and no recorded hand in the repository, so the
gesture tests need hands they can construct. These are anatomically plausible
rather than real: a wrist, a palm arch of MCP joints, and four fingers that
either extend from their MCP or curl back toward the palm, plus a thumb that
swings away from the index MCP.

Coordinates are in pixels with y growing downward, matching OpenCV.
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

Point = Tuple[int, int]

# Landmark order is MediaPipe's: wrist, thumb(4), index(4), middle(4),
# ring(4), pinky(4).
_MCP_ANGLES = {"index": -100.0, "middle": -90.0, "ring": -80.0, "pinky": -70.0}
_FINGER_ORDER = ["index", "middle", "ring", "pinky"]


def _polar(origin: Point, angle_deg: float, length: float) -> Point:
    rad = math.radians(angle_deg)
    return (
        int(round(origin[0] + math.cos(rad) * length)),
        int(round(origin[1] + math.sin(rad) * length)),
    )


def make_hand(
    extended: Dict[str, bool],
    *,
    wrist: Point = (300, 400),
    scale: float = 90.0,
    rotation: float = 0.0,
    thumb_index_gap: float | None = None,
    thumb_angle: float | None = None,
) -> List[Point]:
    """Build 21 landmarks for a hand in the requested pose.

    Args:
        extended: Which of thumb/index/middle/ring/pinky are straight.
        wrist: Where to put landmark 0.
        scale: Wrist-to-middle-MCP length in pixels; every other bone is
            derived from it, so the whole hand scales together.
        rotation: Degrees to rotate the whole hand about the wrist, for
            testing that recognition is rotation-invariant.
        thumb_index_gap: If given, place the thumb tip exactly this many
            pixels from the index tip -- used to build pinch and OK poses.
        thumb_angle: Override the thumb's direction, in degrees.

    Returns:
        A list of 21 ``(x, y)`` points.
    """
    points: List[Point] = [(0, 0)] * 21

    def place(index: int, pt: Point) -> None:
        points[index] = pt

    place(0, wrist)

    # Palm: the four MCP joints fan out from the wrist.
    mcps: Dict[str, Point] = {}
    for name, angle in _MCP_ANGLES.items():
        length = scale if name == "middle" else scale * 0.92
        mcps[name] = _polar(wrist, angle + rotation, length)

    mcp_index = {"index": 5, "middle": 9, "ring": 13, "pinky": 17}
    for name in _FINGER_ORDER:
        place(mcp_index[name], mcps[name])

    # Fingers: extended reaches outward from the MCP; curled folds back
    # toward the wrist so the tip ends up nearer than the PIP.
    for name in _FINGER_ORDER:
        base = mcp_index[name]
        mcp = mcps[name]
        angle = _MCP_ANGLES[name] + rotation
        seg = scale * 0.42

        if extended.get(name, False):
            pip = _polar(mcp, angle, seg)
            dip = _polar(pip, angle, seg * 0.75)
            tip = _polar(dip, angle, seg * 0.6)
        else:
            # Curl: PIP still moves outward, then the finger folds back.
            pip = _polar(mcp, angle, seg * 0.85)
            dip = _polar(pip, angle + 150, seg * 0.55)
            tip = _polar(dip, angle + 170, seg * 0.5)

        place(base + 1, pip)
        place(base + 2, dip)
        place(base + 3, tip)

    # Thumb. Anatomy matters here: a tucked thumb folds ACROSS the palm and
    # ends up close to the wrist, while an extended thumb swings out to the
    # side. Getting this wrong makes a fist look like a pinch.
    index_mcp = mcps["index"]
    if extended.get("thumb", False):
        angle = thumb_angle if thumb_angle is not None else (-150.0 + rotation)
        cmc = _polar(wrist, angle + 25, scale * 0.35)
        mcp = _polar(cmc, angle, scale * 0.38)
        ip = _polar(mcp, angle, scale * 0.30)
        tip = _polar(ip, angle, scale * 0.26)
    else:
        angle = thumb_angle if thumb_angle is not None else (-140.0 + rotation)
        cmc = _polar(wrist, angle + 25, scale * 0.32)
        mcp = _polar(cmc, angle, scale * 0.30)
        # Folded across the palm: the tip comes back toward the wrist, which
        # is what makes it measurably "not extended".
        ip = _polar(mcp, angle + 85, scale * 0.24)
        tip = _polar(ip, angle + 120, scale * 0.22)

    place(1, cmc)
    place(2, mcp)
    place(3, ip)
    place(4, tip)

    # Optionally pull the thumb tip to a precise distance from the index tip,
    # which is how a pinch or an OK sign is built.
    if thumb_index_gap is not None:
        index_tip = points[8]
        direction = math.radians(200 + rotation)
        place(
            4,
            (
                int(round(index_tip[0] + math.cos(direction) * thumb_index_gap)),
                int(round(index_tip[1] + math.sin(direction) * thumb_index_gap)),
            ),
        )

    return points


# ---------------------------------------------------------------- presets
def fist(**kw) -> List[Point]:
    return make_hand(
        {"thumb": False, "index": False, "middle": False, "ring": False, "pinky": False}, **kw
    )


def open_palm(**kw) -> List[Point]:
    return make_hand(
        {"thumb": True, "index": True, "middle": True, "ring": True, "pinky": True}, **kw
    )


def peace(**kw) -> List[Point]:
    return make_hand(
        {"thumb": False, "index": True, "middle": True, "ring": False, "pinky": False}, **kw
    )


def pointing(**kw) -> List[Point]:
    return make_hand(
        {"thumb": False, "index": True, "middle": False, "ring": False, "pinky": False}, **kw
    )


def thumbs_up(**kw) -> List[Point]:
    """A fist turned on its side with the thumb pointing up.

    The rotation matters: in a real thumbs-up the curled fingers point
    sideways and the thumb stands perpendicular to them. A thumb pointing the
    same way as the fingers is not a thumbs-up, it is a hand at rest.
    """
    kw.setdefault("rotation", 90.0)
    kw.setdefault("thumb_angle", -90.0)
    return make_hand(
        {"thumb": True, "index": False, "middle": False, "ring": False, "pinky": False}, **kw
    )


def thumbs_down(**kw) -> List[Point]:
    kw.setdefault("rotation", 90.0)
    kw.setdefault("thumb_angle", 90.0)
    return make_hand(
        {"thumb": True, "index": False, "middle": False, "ring": False, "pinky": False}, **kw
    )


def ok_sign(**kw) -> List[Point]:
    kw.setdefault("thumb_index_gap", 8.0)
    return make_hand(
        {"thumb": True, "index": True, "middle": True, "ring": True, "pinky": True}, **kw
    )


def pinch(**kw) -> List[Point]:
    kw.setdefault("thumb_index_gap", 8.0)
    return make_hand(
        {"thumb": True, "index": True, "middle": False, "ring": False, "pinky": False}, **kw
    )
