"""
gui.py - Tkinter desktop window with a settings panel and playback controls.

Tkinter rather than PyQt on purpose: it is in the Python standard library, so
the GUI adds no dependency to a project whose appeal is that it installs in
one line. PyQt would look slightly better and cost every user a 60 MB wheel.

The OpenCV window in ``main.py`` stays exactly as it was. This is a second
front end over the same engine, not a replacement -- the keyboard-driven
window is still the fastest way to use the tool, and this is for when you
want sliders.

Layout:

    +-----------------------------+----------------------+
    |                             |  Source              |
    |        video canvas         |  Gesture / status    |
    |                             |  Settings sliders    |
    |                             |  Theme picker        |
    +-----------------------------+  Export buttons      |
    |  playback: seek, pause      |                      |
    +-----------------------------+----------------------+

Frames are pulled on a Tk ``after`` callback rather than a thread: Tk is not
thread-safe, and marshalling images back to the main loop costs more than the
grab saves at these frame rates.
"""

from __future__ import annotations

import os
import sys
import time
from typing import Optional

import numpy as np

__all__ = ["FingerRadiusGUI", "launch", "tkinter_available"]


def tkinter_available() -> bool:
    """True when Tkinter can be imported.

    Some Python builds omit it -- notably slim Docker images and a few
    source builds without Tk headers present at compile time.
    """
    try:
        import tkinter  # noqa: F401
    except Exception:
        return False
    return True


def _install_hint() -> str:
    """Platform-specific advice for installing Tkinter."""
    if sys.platform.startswith("linux"):
        return "sudo apt install python3-tk     # Debian/Ubuntu\n    sudo dnf install python3-tkinter  # Fedora"
    if sys.platform == "darwin":
        return "brew install python-tk"
    return "Re-run the python.org installer and enable 'tcl/tk and IDLE'."


class FingerRadiusGUI:
    """Windowed front end over the tracking engine.

    Args:
        source: Camera index or video file path, as
            :class:`~src.video_source.VideoSource` accepts.
        theme: Initial colour theme name.
        use_kalman: Start with Kalman smoothing rather than the EMA.

    Raises:
        ImportError: If Tkinter is unavailable, with install instructions.
    """

    def __init__(
        self,
        source: object = 0,
        theme: str = "corporate",
        use_kalman: bool = True,
    ) -> None:
        if not tkinter_available():
            raise ImportError(
                "The GUI needs Tkinter, which this Python build does not have.\n"
                f"    {_install_hint()}\n"
                "The OpenCV window works without it: python main.py"
            )

        import tkinter as tk
        from tkinter import ttk

        # Imported here, not at module scope, so `import src.gui` stays cheap
        # and does not pull in MediaPipe for a caller that only wants
        # tkinter_available().
        from src.gestures import GestureRecognizer
        from src.hand_tracker import HandTracker
        from src.radius_calculator import RadiusCalculator
        from src.themes import THEME_NAMES, apply_theme
        from src.utils import CSVExporter, FPSCounter
        from src.video_source import VideoSource

        apply_theme(theme)

        self._tk = tk
        self._ttk = ttk
        self.source = VideoSource(source)
        self.tracker = HandTracker(max_num_hands=2)
        self.calculators = [RadiusCalculator() for _ in range(2)]
        self.recognizers = [GestureRecognizer() for _ in range(2)]
        self.fps_counter = FPSCounter()
        self.exporter = CSVExporter()

        self.root = tk.Tk()
        self.root.title("FingerRadiusAI")
        self.root.minsize(980, 620)
        self.root.configure(bg="#16171b")

        self._running = True
        self._frame_count = 0
        self._photo = None                 # a reference Tk will not keep itself
        self._latest_radii: list = []

        self.var_trails = tk.BooleanVar(value=True)
        self.var_kalman = tk.BooleanVar(value=use_kalman)
        self.var_audio = tk.BooleanVar(value=False)
        self.var_3d = tk.BooleanVar(value=False)
        self.var_theme = tk.StringVar(value=theme)
        self.var_smoothing = tk.DoubleVar(value=0.35)
        self.var_status = tk.StringVar(value="starting...")
        self.var_gesture = tk.StringVar(value="-")
        self.var_position = tk.DoubleVar(value=0.0)

        self._build_layout(THEME_NAMES)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

    # ------------------------------------------------------------------
    def _build_layout(self, theme_names) -> None:
        tk, ttk = self._tk, self._ttk

        outer = tk.Frame(self.root, bg="#16171b")
        outer.pack(fill="both", expand=True)

        # Video ------------------------------------------------------
        left = tk.Frame(outer, bg="#16171b")
        left.pack(side="left", fill="both", expand=True, padx=(10, 5), pady=10)

        self.canvas = tk.Label(left, bg="#0d0e11", text="Opening source...",
                               fg="#8a929e", anchor="center")
        self.canvas.pack(fill="both", expand=True)

        # Playback controls only make sense for a file, so they are only
        # created for one rather than shown greyed out.
        if self.source.is_file:
            bar = tk.Frame(left, bg="#16171b")
            bar.pack(fill="x", pady=(8, 0))

            self.btn_pause = tk.Button(bar, text="Pause", width=8,
                                       command=self._toggle_pause)
            self.btn_pause.pack(side="left")
            tk.Button(bar, text="< 5s", width=5,
                      command=lambda: self.source.seek_relative(-5)).pack(side="left", padx=4)
            tk.Button(bar, text="5s >", width=5,
                      command=lambda: self.source.seek_relative(5)).pack(side="left")
            tk.Button(bar, text="Step", width=5,
                      command=self.source.step).pack(side="left", padx=4)

            self.scrub = ttk.Scale(bar, from_=0.0, to=1.0, orient="horizontal",
                                   variable=self.var_position, command=self._on_scrub)
            self.scrub.pack(side="left", fill="x", expand=True, padx=8)
            self._scrubbing = False

        # Settings ---------------------------------------------------
        right = tk.Frame(outer, bg="#1e2026", width=280)
        right.pack(side="right", fill="y", padx=(5, 10), pady=10)
        right.pack_propagate(False)

        def heading(text: str) -> None:
            tk.Label(right, text=text, bg="#1e2026", fg="#8a929e",
                     font=("TkDefaultFont", 8, "bold"), anchor="w").pack(
                fill="x", padx=12, pady=(14, 4))

        tk.Label(right, text="FingerRadiusAI", bg="#1e2026", fg="#e6e8eb",
                 font=("TkDefaultFont", 13, "bold"), anchor="w").pack(
            fill="x", padx=12, pady=(12, 0))
        tk.Label(right, text=self.source.describe(), bg="#1e2026", fg="#6f7681",
                 anchor="w", wraplength=250, justify="left").pack(fill="x", padx=12)

        heading("STATUS")
        tk.Label(right, textvariable=self.var_status, bg="#1e2026", fg="#c8ccd2",
                 anchor="w").pack(fill="x", padx=12)
        tk.Label(right, textvariable=self.var_gesture, bg="#1e2026", fg="#3cc8b4",
                 font=("TkDefaultFont", 11, "bold"), anchor="w").pack(fill="x", padx=12)

        heading("TRACKING")
        for text, var, cb in [
            ("Motion trails", self.var_trails, None),
            ("Kalman smoothing", self.var_kalman, None),
            ("3D (depth-aware) radius", self.var_3d, None),
            ("Audio feedback", self.var_audio, self._toggle_audio),
        ]:
            tk.Checkbutton(right, text=text, variable=var, bg="#1e2026",
                           fg="#c8ccd2", selectcolor="#2a2d34",
                           activebackground="#1e2026", activeforeground="#e6e8eb",
                           anchor="w", command=cb).pack(fill="x", padx=10)

        heading("EMA SMOOTHING")
        ttk.Scale(right, from_=0.05, to=0.95, orient="horizontal",
                  variable=self.var_smoothing).pack(fill="x", padx=12)

        heading("THEME")
        theme_box = ttk.Combobox(right, values=list(theme_names),
                                 textvariable=self.var_theme, state="readonly")
        theme_box.pack(fill="x", padx=12)
        theme_box.bind("<<ComboboxSelected>>", self._on_theme)

        heading("EXPORT")
        for text, cmd in [
            ("Export CSV", self._export_csv),
            ("Build dashboard", self._export_dashboard),
            ("Render audio WAV", self._export_audio),
            ("Screenshot", self._screenshot),
        ]:
            tk.Button(right, text=text, command=cmd).pack(fill="x", padx=12, pady=2)

        tk.Label(right, text="v3.0", bg="#1e2026", fg="#565c66").pack(side="bottom", pady=8)

    # ------------------------------------------------------------------
    def _toggle_pause(self) -> None:
        paused = self.source.toggle_pause()
        self.btn_pause.config(text="Resume" if paused else "Pause")

    def _on_scrub(self, _value: str) -> None:
        self.source.seek_fraction(float(self.var_position.get()))

    def _on_theme(self, _event=None) -> None:
        from src.themes import apply_theme
        apply_theme(self.var_theme.get())

    def _toggle_audio(self) -> None:
        if self.var_audio.get() and not hasattr(self, "_audio"):
            from src.audio_feedback import AudioFeedback
            self._audio = AudioFeedback()
            print(f"[GUI] {self._audio.describe()}")

    # ------------------------------------------------------------------
    def _export_csv(self) -> None:
        self.exporter.export("radius_data.csv")

    def _export_dashboard(self) -> None:
        try:
            from src.dashboard import DashboardBuilder
            DashboardBuilder(self.exporter._rows).write("dashboard.html")
        except Exception as error:
            print(f"[GUI] Dashboard failed: {error}")

    def _export_audio(self) -> None:
        try:
            from src.audio_feedback import AudioFeedback
            audio = getattr(self, "_audio", None) or AudioFeedback()
            audio.render_wav(self._latest_radii or [100.0], "session_audio.wav")
        except Exception as error:
            print(f"[GUI] Audio export failed: {error}")

    def _screenshot(self) -> None:
        import cv2
        if getattr(self, "_last_bgr", None) is not None:
            name = f"screenshot_{int(time.time())}.png"
            cv2.imwrite(name, self._last_bgr)
            print(f"[GUI] Screenshot -> {name}")

    # ------------------------------------------------------------------
    def _tick(self) -> None:
        """Grab, process and draw one frame, then reschedule."""
        if not self._running:
            return

        import cv2
        from PIL import Image, ImageTk  # noqa: F401  (optional, checked below)

        ok, frame = self.source.read()
        if not ok:
            self.var_status.set("source ended")
            self.root.after(200, self._tick)
            return

        self.fps_counter.tick()
        self._frame_count += 1

        num_hands = self.tracker.process(frame)
        self._latest_radii = []
        gestures = []

        for index in range(num_hands):
            landmarks = self.tracker.get_landmarks(index)
            if landmarks is None:
                continue
            radii, wrist, status, deltas = self.calculators[index].compute(
                landmarks, self.tracker.get_landmarks_3d(index), self.var_3d.get()
            )
            result = self.recognizers[index].update(landmarks)
            gestures.append(f"{self.tracker.get_label(index)}: {result.gesture}")
            self._latest_radii.append(radii.get("Thumb-Index", 0.0))
            self.exporter.record(
                {f"{self.tracker.get_label(index)}_{k}": v for k, v in radii.items()},
                f"{self.tracker.get_label(index)}:{status}|{result.gesture}",
            )
            self.calculators[index].draw_radii(frame, landmarks, radii, wrist)

        self.tracker.draw_all(frame, self.var_trails.get())
        self._last_bgr = frame

        self.var_status.set(
            f"{self.fps_counter.fps:.1f} fps   frame {self._frame_count}   "
            f"{num_hands} hand(s)"
        )
        self.var_gesture.set("   ".join(gestures) if gestures else "-")

        if self.source.is_file and not getattr(self, "_scrubbing", False):
            self.var_position.set(self.source.progress)

        self._show(frame)
        # ~30 fps; Tk cannot do better than millisecond scheduling anyway.
        self.root.after(33, self._tick)

    def _show(self, frame_bgr: np.ndarray) -> None:
        """Push a BGR frame onto the canvas."""
        import cv2
        from PIL import Image, ImageTk

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        self._photo = ImageTk.PhotoImage(image=image)
        self.canvas.configure(image=self._photo, text="")

    # ------------------------------------------------------------------
    def run(self) -> None:
        """Start the event loop. Blocks until the window closes."""
        try:
            import PIL  # noqa: F401
        except ImportError as error:
            raise ImportError(
                "The GUI renders frames through Pillow. Install it with:\n"
                "    pip install pillow"
            ) from error

        print(f"[GUI] {self.source.describe()}")
        self.root.after(50, self._tick)
        self.root.mainloop()

    def close(self) -> None:
        """Stop the loop and release the camera or file."""
        self._running = False
        try:
            self.source.release()
            self.tracker.release()
        finally:
            self.root.destroy()


def launch(source: object = 0, theme: str = "corporate") -> None:
    """Open the GUI. Raises ImportError with instructions if Tk is missing."""
    FingerRadiusGUI(source=source, theme=theme).run()
