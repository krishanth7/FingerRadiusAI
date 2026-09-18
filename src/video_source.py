"""
video_source.py - One interface over a live camera and a recorded file.

The rest of the application should not care where frames come from, but the
two sources genuinely differ: a file has a length, a position and a frame
rate you can seek within, and it ends; a camera has none of those and does
not. Rather than pretend they are the same, :class:`VideoSource` exposes the
file-only operations and reports honestly that they do nothing on a camera.

Playback control matters for a file. A recorded clip read as fast as the CPU
allows plays at several hundred frames a second, which is useless for
watching a gesture. :meth:`VideoSource.read` therefore paces itself to the
file's own frame rate unless asked not to.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

__all__ = ["VideoSource", "SourceInfo"]


@dataclass
class SourceInfo:
    """What is known about the open source."""

    kind: str                  # "camera" or "file"
    name: str
    width: int
    height: int
    fps: float
    frame_count: int           # 0 when unknown, which is always true of a camera

    @property
    def is_file(self) -> bool:
        return self.kind == "file"

    @property
    def duration_seconds(self) -> float:
        """Length in seconds, or 0.0 when unknown."""
        if self.frame_count and self.fps > 0:
            return self.frame_count / self.fps
        return 0.0


class VideoSource:
    """A camera index or a video file path, behind one interface.

    Args:
        source: An integer camera index, or a path to a video file. A string
            of digits is treated as a camera index, so ``"0"`` and ``0`` behave
            the same -- which is what a command-line argument needs.
        width: Requested capture width. Cameras only; ignored for files.
        height: Requested capture height. Cameras only.
        loop: Restart a file when it reaches the end. Ignored for cameras.
        realtime: Pace file playback to the file's frame rate. Turn it off to
            process a clip as fast as possible, which is what batch export
            wants.
        mirror: Flip horizontally. On by default for cameras, because a
            mirrored view is what makes a webcam feel usable; off by default
            for files, because a recording is not a mirror.

    Raises:
        FileNotFoundError: If a file path does not exist.
        RuntimeError: If the source cannot be opened.

    Example:
        >>> source = VideoSource("clip.mp4")          # doctest: +SKIP
        >>> ok, frame = source.read()                 # doctest: +SKIP
        >>> source.release()                          # doctest: +SKIP
    """

    def __init__(
        self,
        source: object = 0,
        width: int = 640,
        height: int = 480,
        loop: bool = True,
        realtime: bool = True,
        mirror: Optional[bool] = None,
    ) -> None:
        self._is_file = False
        resolved: object = source

        if isinstance(source, str):
            if source.isdigit():
                resolved = int(source)
            else:
                if not os.path.exists(source):
                    raise FileNotFoundError(f"Video file not found: {source}")
                self._is_file = True

        self._cap = cv2.VideoCapture(resolved)
        if not self._cap.isOpened():
            kind = "video file" if self._is_file else "camera"
            raise RuntimeError(f"Could not open {kind}: {source!r}")

        if not self._is_file:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        fps = float(self._cap.get(cv2.CAP_PROP_FPS) or 0.0)
        # Cameras frequently report 0 or a nonsense value; fall back to 30.
        if fps <= 1.0 or fps > 240.0:
            fps = 30.0

        self.info = SourceInfo(
            kind="file" if self._is_file else "camera",
            name=str(source),
            width=int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=fps,
            frame_count=int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT)) if self._is_file else 0,
        )

        self.loop = bool(loop)
        self.realtime = bool(realtime)
        self.mirror = (not self._is_file) if mirror is None else bool(mirror)

        self.paused = False
        self._position = 0
        self._last_frame: Optional[np.ndarray] = None
        self._next_due = 0.0
        self._step_once = False

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Return ``(ok, frame)``.

        While paused, the last frame is returned again so the UI keeps
        rendering rather than freezing on a stale buffer. ``ok`` is False only
        when the source is genuinely finished.
        """
        if self.paused and not self._step_once:
            if self._last_frame is not None:
                return True, self._last_frame.copy()
            return False, None

        self._step_once = False

        if self._is_file and self.realtime:
            self._wait_for_frame_time()

        ok, frame = self._cap.read()

        if not ok:
            if self._is_file and self.loop:
                self.seek_frame(0)
                ok, frame = self._cap.read()
                if not ok:
                    return False, None
            else:
                return False, None

        if self.mirror:
            frame = cv2.flip(frame, 1)

        self._position = int(self._cap.get(cv2.CAP_PROP_POS_FRAMES))
        self._last_frame = frame
        return True, frame

    def _wait_for_frame_time(self) -> None:
        """Sleep just long enough to hold the file's own frame rate."""
        period = 1.0 / self.info.fps
        now = time.perf_counter()
        if self._next_due == 0.0:
            self._next_due = now
        delay = self._next_due - now
        if delay > 0:
            time.sleep(delay)
        # Advance from the due time, not from now, so the clock does not drift
        # when a frame takes longer than its slot.
        self._next_due = max(self._next_due + period, now)

    # ------------------------------------------------------------------
    # Playback control -- meaningful for files, honest no-ops for cameras
    # ------------------------------------------------------------------
    def toggle_pause(self) -> bool:
        """Pause or resume. Returns the new paused state."""
        self.paused = not self.paused
        if not self.paused:
            self._next_due = 0.0
        return self.paused

    def step(self) -> None:
        """Advance exactly one frame while paused."""
        self._step_once = True

    def seek_frame(self, frame_index: int) -> bool:
        """Jump to a frame index. Returns False on a camera, which cannot seek."""
        if not self._is_file:
            return False
        target = max(0, min(int(frame_index), max(0, self.info.frame_count - 1)))
        ok = bool(self._cap.set(cv2.CAP_PROP_POS_FRAMES, target))
        self._position = target
        self._next_due = 0.0
        return ok

    def seek_relative(self, seconds: float) -> bool:
        """Jump forward or back by a number of seconds."""
        if not self._is_file:
            return False
        return self.seek_frame(self._position + int(seconds * self.info.fps))

    def seek_fraction(self, fraction: float) -> bool:
        """Jump to a point given as 0.0-1.0 through the file."""
        if not self._is_file:
            return False
        return self.seek_frame(int(self.info.frame_count * max(0.0, min(1.0, fraction))))

    # ------------------------------------------------------------------
    # Position
    # ------------------------------------------------------------------
    @property
    def position(self) -> int:
        """Index of the most recently read frame."""
        return self._position

    @property
    def progress(self) -> float:
        """How far through the file, 0.0-1.0. Always 0.0 for a camera."""
        if not self._is_file or not self.info.frame_count:
            return 0.0
        return min(1.0, self._position / self.info.frame_count)

    @property
    def is_file(self) -> bool:
        return self._is_file

    def describe(self) -> str:
        """One line summarising the source, for the console banner."""
        i = self.info
        if i.is_file:
            return (
                f"file '{os.path.basename(i.name)}' "
                f"{i.width}x{i.height} @ {i.fps:.1f}fps, "
                f"{i.frame_count} frames ({i.duration_seconds:.1f}s)"
            )
        return f"camera {i.name} {i.width}x{i.height} @ {i.fps:.1f}fps"

    def release(self) -> None:
        self._cap.release()

    def __enter__(self) -> "VideoSource":
        return self

    def __exit__(self, *exc) -> None:
        self.release()
