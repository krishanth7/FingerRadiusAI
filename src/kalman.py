"""
kalman.py - Kalman smoothing for landmark and radius signals.

The exponential moving average this replaces has one setting, alpha, and it
buys smoothness with lag: a high alpha tracks fast motion but keeps the
jitter, a low alpha is smooth but trails behind the hand. There is no value
that does both, because an EMA has no notion of where the hand is heading.

A constant-velocity Kalman filter does. It carries position *and* velocity,
predicts forward between measurements, and corrects by an amount that depends
on how much it currently trusts its own prediction. While the hand moves
steadily it leans on the prediction and cuts jitter hard; when the hand
changes direction the innovation grows and it snaps back to the measurement.

State is [position, velocity]. Implemented directly in NumPy -- the matrices
are 2x2, so a dependency for this would be heavier than the maths.

Measured against the EMA it replaces, one fixed setting each, on three
synthetic signals at the same noise level (see ``tests/test_kalman.py``):

===============  ==========  ==============  ========  =============
signal           raw RMSE    EMA alpha=0.35  Kalman    improvement
===============  ==========  ==============  ========  =============
smooth sine           7.599           4.831     3.694         23.5%
move-and-hold         7.599           3.885     3.228         16.9%
drift + tremor        7.599           3.710     3.675          1.0%
mean                                  4.142     3.532         14.7%
===============  ==========  ==============  ========  =============

The defaults below are the single pair that minimises the worst case across
all three. Tuned per signal the filter does better still, but a shipped
default has to work on a signal it has not seen. Note that an EMA re-tuned
for each signal individually can beat a single fixed Kalman on the smoothest
of them -- the gain here comes from not having to pick alpha at all.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

__all__ = ["KalmanFilter1D", "KalmanPoint2D", "KalmanLandmarkSet"]


class KalmanFilter1D:
    """Constant-velocity Kalman filter over a scalar signal.

    Args:
        process_noise: How much the velocity is expected to change between
            frames. Raise it to follow quick direction changes, lower it for
            heavier smoothing. This is the dial that replaces EMA's alpha.
        measurement_noise: How noisy the incoming measurement is. Raise it to
            trust the model over the sensor.
        dt: Timestep between updates. Left at 1.0 the units are per-frame,
            which is what a fixed-rate video loop wants.

    Example:
        >>> f = KalmanFilter1D()
        >>> for _ in range(50):
        ...     value = f.update(10.0)
        >>> round(value, 3)
        10.0
    """

    def __init__(
        self,
        process_noise: float = 0.05,
        measurement_noise: float = 36.0,
        dt: float = 1.0,
    ) -> None:
        if process_noise <= 0:
            raise ValueError(f"process_noise must be positive, got {process_noise}.")
        if measurement_noise <= 0:
            raise ValueError(
                f"measurement_noise must be positive, got {measurement_noise}."
            )

        self.dt = float(dt)
        self.process_noise = float(process_noise)
        self.measurement_noise = float(measurement_noise)

        # x = [position, velocity]
        self._x = np.zeros(2, dtype=np.float64)
        # State transition: position advances by velocity * dt.
        self._F = np.array([[1.0, self.dt], [0.0, 1.0]], dtype=np.float64)
        # We observe position only.
        self._H = np.array([[1.0, 0.0]], dtype=np.float64)
        # Process noise for a constant-velocity model driven by random
        # acceleration (the standard discrete white-noise-acceleration form).
        t, t2 = self.dt, self.dt ** 2
        self._Q = self.process_noise * np.array(
            [[t2 * t2 / 4.0, t2 * t / 2.0], [t2 * t / 2.0, t2]], dtype=np.float64
        )
        self._R = np.array([[self.measurement_noise]], dtype=np.float64)
        self._P = np.eye(2, dtype=np.float64) * 500.0
        self._initialised = False

    @property
    def position(self) -> float:
        """Current filtered position."""
        return float(self._x[0])

    @property
    def velocity(self) -> float:
        """Current estimated velocity, in units per timestep."""
        return float(self._x[1])

    @property
    def value(self) -> float:
        """Alias for :attr:`position`, matching the EMA it replaces."""
        return self.position

    def update(self, measurement: float) -> float:
        """Advance one step and fold in a measurement. Returns the estimate."""
        z = float(measurement)

        if not self._initialised:
            # Starting from zero would make the filter crawl to the first real
            # value over many frames. Seed it instead.
            self._x[0] = z
            self._x[1] = 0.0
            self._P = np.eye(2, dtype=np.float64) * 1.0
            self._initialised = True
            return self.position

        # Predict
        self._x = self._F @ self._x
        self._P = self._F @ self._P @ self._F.T + self._Q

        # Correct
        # H @ x and H @ P @ H.T are 1-element arrays, not scalars, so take
        # the element explicitly rather than relying on float() coercion.
        y = z - (self._H @ self._x).item()                # innovation
        S = (self._H @ self._P @ self._H.T + self._R).item()
        K = (self._P @ self._H.T / S).reshape(2)         # Kalman gain
        self._x = self._x + K * y
        self._P = (np.eye(2) - np.outer(K, self._H)) @ self._P
        return self.position

    def reset(self) -> None:
        """Forget all state; the next update re-seeds the filter."""
        self._x = np.zeros(2, dtype=np.float64)
        self._P = np.eye(2, dtype=np.float64) * 500.0
        self._initialised = False


class KalmanPoint2D:
    """Two independent :class:`KalmanFilter1D`s, for an (x, y) landmark."""

    def __init__(self, process_noise: float = 0.05, measurement_noise: float = 36.0):
        self._x = KalmanFilter1D(process_noise, measurement_noise)
        self._y = KalmanFilter1D(process_noise, measurement_noise)

    def update(self, point: Tuple[float, float]) -> Tuple[int, int]:
        """Filter one point and return it rounded to integer pixels."""
        return (
            int(round(self._x.update(point[0]))),
            int(round(self._y.update(point[1]))),
        )

    @property
    def speed(self) -> float:
        """Magnitude of the estimated velocity, in pixels per frame."""
        return float(np.hypot(self._x.velocity, self._y.velocity))

    def reset(self) -> None:
        self._x.reset()
        self._y.reset()


class KalmanLandmarkSet:
    """A filter per coordinate of a full 21-landmark hand.

    Drop-in replacement for the per-landmark EMA bank in
    :class:`~src.hand_tracker.HandTracker`.
    """

    def __init__(
        self,
        count: int = 21,
        process_noise: float = 0.05,
        measurement_noise: float = 36.0,
    ) -> None:
        self.count = int(count)
        self._x = [KalmanFilter1D(process_noise, measurement_noise) for _ in range(self.count)]
        self._y = [KalmanFilter1D(process_noise, measurement_noise) for _ in range(self.count)]
        self._z = [KalmanFilter1D(process_noise, measurement_noise) for _ in range(self.count)]

    def update(
        self, points: Sequence[Tuple[float, float, float]]
    ) -> List[Tuple[int, int, float]]:
        """Filter a whole hand. Input and output are ``(x, y, z)`` triples.

        Raises:
            ValueError: If the wrong number of landmarks is supplied.
        """
        if len(points) != self.count:
            raise ValueError(
                f"Expected {self.count} landmarks, got {len(points)}."
            )
        out: List[Tuple[int, int, float]] = []
        for i, (px, py, pz) in enumerate(points):
            out.append(
                (
                    int(round(self._x[i].update(px))),
                    int(round(self._y[i].update(py))),
                    self._z[i].update(pz),
                )
            )
        return out

    def reset(self) -> None:
        for bank in (self._x, self._y, self._z):
            for f in bank:
                f.reset()
