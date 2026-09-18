"""
multi_camera.py - Metric 3D from two cameras.

MediaPipe's z is *relative*: roughly how far a landmark sits in front of the
wrist, in units scaled to the hand. It is enough to tell which finger is
nearer, which is what the 3D radius mode uses it for. It is not a measurement
-- you cannot ask it how many centimetres apart two fingertips are, because it
never knew the answer.

Two calibrated cameras can answer that. The same landmark seen from two known
viewpoints gives two rays through space, and where those rays meet is the
point, in whatever real units the calibration was done in.

Rays that meet exactly are a convenience of textbooks. With real detections
they pass near each other and miss, so this solves for the point that is
closest to both at once, via the linear triangulation of Hartley & Zisserman
(*Multiple View Geometry*, section 12.2): stack the four constraints the two
2-D observations impose on the homogeneous 3-D point and take the singular
vector with the smallest singular value.

The **reprojection error** that comes back is the honest quality signal. It is
the distance, in pixels, between where each camera actually saw the landmark
and where the solved point projects to. A large value means the calibration is
wrong or the two cameras are not looking at the same hand -- and it is far
better to surface that than to return a confident wrong number.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

__all__ = ["CameraCalibration", "StereoRig", "TriangulationResult"]


@dataclass
class CameraCalibration:
    """One camera's intrinsics and pose.

    Attributes:
        name: A label, for error messages.
        matrix: 3x3 intrinsics ``K`` -- focal lengths and principal point.
        rotation: 3x3 world-to-camera rotation ``R``.
        translation: 3-vector world-to-camera translation ``t``.
    """

    name: str
    matrix: np.ndarray
    rotation: np.ndarray
    translation: np.ndarray

    def __post_init__(self) -> None:
        self.matrix = np.asarray(self.matrix, dtype=np.float64).reshape(3, 3)
        self.rotation = np.asarray(self.rotation, dtype=np.float64).reshape(3, 3)
        self.translation = np.asarray(self.translation, dtype=np.float64).reshape(3)

        # A rotation matrix must be orthonormal with determinant +1. A
        # determinant of -1 is a reflection, which silently mirrors the whole
        # reconstruction, so it is worth refusing rather than debugging later.
        determinant = float(np.linalg.det(self.rotation))
        if not math.isclose(determinant, 1.0, abs_tol=1e-3):
            raise ValueError(
                f"Camera {self.name!r}: rotation has determinant {determinant:.4f}, "
                "expected 1.0. A value near -1 means the axes are mirrored."
            )

    @property
    def projection(self) -> np.ndarray:
        """The 3x4 projection matrix ``P = K [R|t]``."""
        return self.matrix @ np.hstack([self.rotation, self.translation.reshape(3, 1)])

    @property
    def centre(self) -> np.ndarray:
        """Camera centre in world coordinates."""
        return -self.rotation.T @ self.translation

    def project(self, point_world: Sequence[float]) -> Tuple[float, float]:
        """Project a world point to pixels.

        Raises:
            ValueError: If the point is behind the camera, where the
                projection would still produce plausible-looking pixels.
        """
        point = np.asarray(point_world, dtype=np.float64).reshape(3)
        camera_point = self.rotation @ point + self.translation
        if camera_point[2] <= 1e-9:
            raise ValueError(
                f"Camera {self.name!r}: point is at or behind the image plane."
            )
        pixel = self.matrix @ camera_point
        return float(pixel[0] / pixel[2]), float(pixel[1] / pixel[2])

    @staticmethod
    def simple(
        name: str,
        position: Sequence[float],
        look_at: Sequence[float] = (0.0, 0.0, 0.0),
        focal: float = 800.0,
        width: int = 640,
        height: int = 480,
    ) -> "CameraCalibration":
        """Build a calibration from a position and a point to look at.

        A convenience for setting up a rig or a test without running a full
        checkerboard calibration. Real deployments should calibrate properly;
        the focal length here is a guess, and a guessed focal length scales
        the whole reconstruction.
        """
        eye = np.asarray(position, dtype=np.float64).reshape(3)
        target = np.asarray(look_at, dtype=np.float64).reshape(3)

        forward = target - eye
        norm = np.linalg.norm(forward)
        if norm < 1e-9:
            raise ValueError(f"Camera {name!r} cannot look at its own position.")
        forward /= norm

        world_up = np.array([0.0, -1.0, 0.0])
        if abs(float(forward @ world_up)) > 0.999:
            world_up = np.array([0.0, 0.0, 1.0])

        right = np.cross(world_up, forward)
        right /= np.linalg.norm(right)
        up = np.cross(forward, right)

        rotation = np.vstack([right, up, forward])
        translation = -rotation @ eye
        matrix = np.array(
            [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]]
        )
        return CameraCalibration(name, matrix, rotation, translation)


@dataclass
class TriangulationResult:
    """A solved 3-D point and how much to trust it.

    Attributes:
        points: ``(n, 3)`` world coordinates.
        reprojection_errors: Per-landmark mean pixel error across the two
            cameras. This is the number to threshold on.
    """

    points: np.ndarray
    reprojection_errors: np.ndarray

    @property
    def mean_error(self) -> float:
        """Mean reprojection error over all landmarks, in pixels."""
        return float(np.mean(self.reprojection_errors))

    @property
    def max_error(self) -> float:
        return float(np.max(self.reprojection_errors))

    def is_reliable(self, tolerance: float = 3.0) -> bool:
        """True when every landmark reprojects within ``tolerance`` pixels."""
        return bool(self.max_error <= tolerance)

    def distance(self, a: int, b: int) -> float:
        """Distance between two landmarks, in calibration units."""
        return float(np.linalg.norm(self.points[a] - self.points[b]))


class StereoRig:
    """Two calibrated cameras, triangulating a hand between them.

    Args:
        left: One camera.
        right: The other. Order is only a naming convention.

    Raises:
        ValueError: If both cameras sit in the same place, which gives no
            baseline and therefore no depth.

    Example:
        >>> import numpy as np
        >>> left = CameraCalibration.simple("L", (-0.15, 0.0, 0.0), (0, 0, 1.0))
        >>> right = CameraCalibration.simple("R", (0.15, 0.0, 0.0), (0, 0, 1.0))
        >>> rig = StereoRig(left, right)
        >>> truth = np.array([[0.02, -0.03, 1.10]])
        >>> a = [left.project(truth[0])]
        >>> b = [right.project(truth[0])]
        >>> result = rig.triangulate(a, b)
        >>> bool(np.allclose(result.points, truth, atol=1e-6))
        True
    """

    def __init__(self, left: CameraCalibration, right: CameraCalibration) -> None:
        baseline = float(np.linalg.norm(left.centre - right.centre))
        if baseline < 1e-6:
            raise ValueError(
                "Both cameras are at the same point, so there is no baseline "
                "and no depth to recover. Move them apart."
            )
        self.left = left
        self.right = right
        self.baseline = baseline

    # ------------------------------------------------------------------
    @staticmethod
    def _triangulate_one(
        p_left: np.ndarray, p_right: np.ndarray,
        x_left: Sequence[float], x_right: Sequence[float],
    ) -> np.ndarray:
        """Linear triangulation of one correspondence (Hartley & Zisserman)."""
        design = np.vstack([
            x_left[0] * p_left[2] - p_left[0],
            x_left[1] * p_left[2] - p_left[1],
            x_right[0] * p_right[2] - p_right[0],
            x_right[1] * p_right[2] - p_right[1],
        ])
        _u, _s, vt = np.linalg.svd(design)
        homogeneous = vt[-1]
        if abs(homogeneous[3]) < 1e-12:
            # A point at infinity: parallel rays. Returning inf is honest;
            # dividing would produce an enormous arbitrary coordinate.
            return np.full(3, np.inf)
        return homogeneous[:3] / homogeneous[3]

    def triangulate(
        self,
        left_points: Sequence[Tuple[float, float]],
        right_points: Sequence[Tuple[float, float]],
    ) -> TriangulationResult:
        """Solve 3-D positions for matched 2-D observations.

        Args:
            left_points: Pixel coordinates from the left camera.
            right_points: The same landmarks, from the right camera, in the
                same order.

        Raises:
            ValueError: If the two lists differ in length -- which means the
                correspondence is broken, and triangulating mismatched
                landmarks would produce confident nonsense.
        """
        if len(left_points) != len(right_points):
            raise ValueError(
                f"Both cameras must see the same landmarks in the same order: "
                f"got {len(left_points)} and {len(right_points)}."
            )
        if not left_points:
            raise ValueError("No points to triangulate.")

        p_left = self.left.projection
        p_right = self.right.projection

        points = np.zeros((len(left_points), 3), dtype=np.float64)
        errors = np.zeros(len(left_points), dtype=np.float64)

        for i, (xl, xr) in enumerate(zip(left_points, right_points)):
            point = self._triangulate_one(p_left, p_right, xl, xr)
            points[i] = point
            if not np.all(np.isfinite(point)):
                errors[i] = float("inf")
                continue
            try:
                back_left = self.left.project(point)
                back_right = self.right.project(point)
            except ValueError:
                # Solved to a point behind a camera: the observation pair
                # cannot be right.
                errors[i] = float("inf")
                continue
            errors[i] = 0.5 * (
                math.dist(back_left, xl) + math.dist(back_right, xr)
            )

        return TriangulationResult(points, errors)

    def triangulate_hand(
        self,
        left_landmarks: Sequence[Tuple[int, int]],
        right_landmarks: Sequence[Tuple[int, int]],
    ) -> TriangulationResult:
        """Triangulate a full 21-landmark hand.

        Raises:
            ValueError: If either view does not have 21 landmarks.
        """
        for name, landmarks in (("left", left_landmarks), ("right", right_landmarks)):
            if landmarks is None or len(landmarks) < 21:
                raise ValueError(
                    f"The {name} view needs 21 landmarks; got "
                    f"{0 if landmarks is None else len(landmarks)}."
                )
        return self.triangulate(
            [(float(x), float(y)) for x, y in left_landmarks[:21]],
            [(float(x), float(y)) for x, y in right_landmarks[:21]],
        )

    # ------------------------------------------------------------------
    def save(self, path: str = "stereo_rig.json") -> str:
        """Write the calibration as JSON."""
        payload = {
            "baseline": self.baseline,
            "cameras": [
                {
                    "name": camera.name,
                    "matrix": camera.matrix.tolist(),
                    "rotation": camera.rotation.tolist(),
                    "translation": camera.translation.tolist(),
                }
                for camera in (self.left, self.right)
            ],
        }
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        return path

    @classmethod
    def load(cls, path: str = "stereo_rig.json") -> "StereoRig":
        """Read a calibration written by :meth:`save`."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"No rig calibration at {path}")
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
        cameras = [
            CameraCalibration(c["name"], np.array(c["matrix"]),
                              np.array(c["rotation"]), np.array(c["translation"]))
            for c in payload["cameras"]
        ]
        return cls(cameras[0], cameras[1])

    def describe(self) -> str:
        """One line for the console banner."""
        return (
            f"stereo rig: {self.left.name} + {self.right.name}, "
            f"baseline {self.baseline:.3f} units"
        )
