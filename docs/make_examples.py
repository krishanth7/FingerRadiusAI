"""
make_examples.py - Regenerate every image in the README.

Nothing here is a mockup. Each figure is produced by importing the same
modules the application uses and running them. Landmarks come from
``tests/synthetic_hands`` because CI has no camera; the recognition, the
filtering, the radius maths and the drawing are all the production code.

Run from the repository root:

    python docs/make_examples.py
"""

from __future__ import annotations

import os
import sys
import wave

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.audio_feedback import AudioFeedback, ToneMapper
from src.gestures import GestureRecognizer
from src.hand_tracker import FINGER_BONE_GROUPS, FINGER_TIPS, LandmarkIndex
from src.kalman import KalmanFilter1D
from src.radius_calculator import RadiusCalculator
from src.themes import THEME_NAMES, apply_theme
from src.utils import COLORS, ExponentialMovingAverage, FINGER_KEYS
from tests import synthetic_hands as hands

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "examples")
os.makedirs(OUT, exist_ok=True)


def fit_into(landmarks, width, height, top=56, bottom=24, side=20):
    """Translate and scale a hand so it fills its cell without clipping.

    Poses reach in different directions -- a thumbs-down extends well below
    the wrist -- so placing every wrist at the same spot clips some of them.
    Fitting the actual bounding box is what makes the grid uniform.
    """
    xs = [p[0] for p in landmarks]
    ys = [p[1] for p in landmarks]
    span_x = max(1, max(xs) - min(xs))
    span_y = max(1, max(ys) - min(ys))
    scale = min((width - 2 * side) / span_x, (height - top - bottom) / span_y)
    offset_x = side + (width - 2 * side - span_x * scale) / 2 - min(xs) * scale
    offset_y = top + (height - top - bottom - span_y * scale) / 2 - min(ys) * scale
    return [(int(x * scale + offset_x), int(y * scale + offset_y)) for x, y in landmarks]


def draw_hand(canvas, landmarks):
    """Draw a skeleton using the production bone groups and palette."""
    for key, bones in FINGER_BONE_GROUPS.items():
        color = COLORS[key]
        muted = tuple(max(0, min(255, int(c * 0.5))) for c in color)
        for a, b in bones:
            shade = COLORS["bone"] if a == 0 else muted
            cv2.line(canvas, landmarks[a], landmarks[b], shade, 2, cv2.LINE_AA)
    for a, b in [(5, 9), (9, 13), (13, 17)]:
        cv2.line(canvas, landmarks[a], landmarks[b], COLORS["bone"], 1, cv2.LINE_AA)
    for i, (x, y) in enumerate(landmarks):
        if i in FINGER_TIPS:
            color = COLORS[FINGER_KEYS[FINGER_TIPS.index(i)]]
            cv2.circle(canvas, (x, y), 7, color, 1, cv2.LINE_AA)
            cv2.circle(canvas, (x, y), 3, color, -1, cv2.LINE_AA)
        elif i == LandmarkIndex.WRIST:
            cv2.circle(canvas, (x, y), 5, COLORS["wrist"], 1, cv2.LINE_AA)
        else:
            cv2.circle(canvas, (x, y), 2, COLORS["joint"], -1, cv2.LINE_AA)


def gesture_grid(path=os.path.join(OUT, "gestures.png")):
    """Every gesture, drawn and then labelled with the recogniser's own verdict."""
    apply_theme("corporate")
    recognizer = GestureRecognizer(hold_frames=1)
    poses = [
        ("fist", hands.fist), ("open palm", hands.open_palm),
        ("peace", hands.peace), ("pointing", hands.pointing),
        ("thumbs up", hands.thumbs_up), ("thumbs down", hands.thumbs_down),
        ("OK", hands.ok_sign), ("pinch", hands.pinch),
    ]
    cell_w, cell_h, cols = 240, 270, 4
    rows = (len(poses) + cols - 1) // cols
    sheet = np.full((rows * cell_h + 44, cols * cell_w, 3), COLORS["bg_primary"], np.uint8)

    cv2.putText(sheet, "GESTURE LIBRARY - labels produced by GestureRecognizer",
                (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS["accent"], 1, cv2.LINE_AA)

    calculator = RadiusCalculator()
    for i, (name, builder) in enumerate(poses):
        cell = np.full((cell_h, cell_w, 3), COLORS["bg_secondary"], np.uint8)
        raw = builder(wrist=(cell_w // 2, cell_h - 60), scale=58.0)
        # Classify the geometry as built, then fit a copy for display --
        # the label must come from the pose, not from the framing.
        result = recognizer.classify(raw)
        landmarks = fit_into(raw, cell_w, cell_h)
        draw_hand(cell, landmarks)
        recognizer.reset()

        cv2.rectangle(cell, (0, 0), (cell_w - 1, cell_h - 1), COLORS["border"], 1)
        cv2.putText(cell, result.gesture.upper(), (12, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, COLORS["success"], 1, cv2.LINE_AA)
        cv2.putText(cell, f"confidence {result.confidence:.0%}", (12, 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, COLORS["text_tertiary"], 1, cv2.LINE_AA)
        flags = "".join("1" if result.fingers[k] else "0"
                        for k in ("thumb", "index", "middle", "ring", "pinky"))
        cv2.putText(cell, f"fingers {flags}", (12, cell_h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, COLORS["text_secondary"], 1, cv2.LINE_AA)

        r, c = divmod(i, cols)
        sheet[44 + r * cell_h: 44 + (r + 1) * cell_h, c * cell_w: (c + 1) * cell_w] = cell

    cv2.imwrite(path, sheet)
    print(f"  gestures.png      {os.path.getsize(path) // 1024} KiB")
    return path


def theme_strip(path=os.path.join(OUT, "themes.png")):
    """The same hand drawn once per theme, through the real palette switch."""
    recognizer = GestureRecognizer(hold_frames=1)
    calculator = RadiusCalculator()
    cell_w, cell_h = 280, 300
    sheet = np.zeros((cell_h + 40, cell_w * len(THEME_NAMES), 3), np.uint8)

    for i, theme in enumerate(THEME_NAMES):
        apply_theme(theme)
        cell = np.full((cell_h, cell_w, 3), COLORS["bg_primary"], np.uint8)
        landmarks = fit_into(hands.open_palm(wrist=(cell_w // 2, cell_h - 50), scale=62.0),
                             cell_w, cell_h, top=66)
        radii, wrist_radii, status, _ = calculator.compute(landmarks)
        draw_hand(cell, landmarks)
        calculator.draw_radii(cell, landmarks, radii, wrist_radii)
        cv2.rectangle(cell, (0, 0), (cell_w - 1, cell_h - 1), COLORS["border"], 2)
        cv2.putText(cell, theme.upper(), (14, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, COLORS["accent"], 1, cv2.LINE_AA)
        cv2.putText(cell, f"status: {status}", (14, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    0.38, COLORS["text_secondary"], 1, cv2.LINE_AA)
        sheet[40:40 + cell_h, i * cell_w:(i + 1) * cell_w] = cell

    apply_theme("corporate")
    sheet[:40] = COLORS["bg_primary"]
    cv2.putText(sheet, "THEMES - same frame, same code, four palettes",
                (16, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS["accent"], 1, cv2.LINE_AA)
    cv2.imwrite(path, sheet)
    print(f"  themes.png        {os.path.getsize(path) // 1024} KiB")
    return path


def kalman_chart(path=os.path.join(OUT, "kalman_vs_ema.png")):
    """Run both filters on the same noisy signal and plot what came out."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = 320
    t = np.arange(n)
    truth = np.full(n, 120.0)
    truth[120:150] = np.linspace(120, 230, 30)
    truth[150:] = 230.0
    noisy = truth + np.random.default_rng(7).normal(0, 8.0, n)

    ema_out = np.array([ExponentialMovingAverage(0.35).update(v) for v in noisy]) * 0
    ema = ExponentialMovingAverage(0.35)
    ema_out = np.array([ema.update(v) for v in noisy])
    kal = KalmanFilter1D()
    kal_out = np.array([kal.update(v) for v in noisy])

    def rmse(a):
        return float(np.sqrt(((a - truth) ** 2).mean()))

    fig, ax = plt.subplots(figsize=(11, 4.6), dpi=130)
    fig.patch.set_facecolor("#191a1e")
    ax.set_facecolor("#212328")
    ax.plot(t, noisy, color="#555b66", lw=0.8, label=f"raw landmark  RMSE {rmse(noisy):.2f}")
    ax.plot(t, truth, color="#e6e8eb", lw=1.6, ls="--", label="true position")
    ax.plot(t, ema_out, color="#eb9d4b", lw=1.8, label=f"EMA a=0.35  RMSE {rmse(ema_out):.2f}")
    ax.plot(t, kal_out, color="#3cc8b4", lw=1.8, label=f"Kalman      RMSE {rmse(kal_out):.2f}")
    ax.set_title("Kalman vs EMA on a hand that moves, then holds still",
                 color="#e6e8eb", fontsize=12)
    ax.set_xlabel("frame", color="#8a929e")
    ax.set_ylabel("radius (px)", color="#8a929e")
    ax.tick_params(colors="#8a929e")
    for spine in ax.spines.values():
        spine.set_color("#2e3138")
    ax.grid(color="#2e3138", lw=0.6)
    ax.legend(facecolor="#191a1e", edgecolor="#2e3138", labelcolor="#c8ccd2", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  kalman_vs_ema.png {os.path.getsize(path) // 1024} KiB  "
          f"(EMA {rmse(ema_out):.2f} -> Kalman {rmse(kal_out):.2f})")
    return path, rmse(noisy), rmse(ema_out), rmse(kal_out)


def audio_chart(path=os.path.join(OUT, "audio_mapping.png")):
    """Plot the real mapping and the real waveform read back from the WAV."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    wav_path = os.path.join(OUT, "example_session.wav")
    audio = AudioFeedback(ToneMapper(scale="pentatonic"))
    radii = [abs(150 + 140 * np.sin(i / 7.0)) for i in range(24)]
    audio.render_wav(radii, wav_path, note_duration=0.12)

    with wave.open(wav_path) as handle:
        rate = handle.getframerate()
        samples = np.frombuffer(handle.readframes(handle.getnframes()), dtype=np.int16)

    radius_axis = np.arange(0, 301)
    continuous = [ToneMapper(scale=None).frequency(r) for r in radius_axis]
    snapped = [ToneMapper(scale="pentatonic").frequency(r) for r in radius_axis]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6), dpi=130)
    fig.patch.set_facecolor("#191a1e")
    for ax in (ax1, ax2):
        ax.set_facecolor("#212328")
        ax.tick_params(colors="#8a929e")
        for spine in ax.spines.values():
            spine.set_color("#2e3138")
        ax.grid(color="#2e3138", lw=0.6)

    ax1.plot(radius_axis, continuous, color="#555b66", lw=1.4, ls="--",
             label="continuous (geometric)")
    ax1.plot(radius_axis, snapped, color="#3cc8b4", lw=1.8,
             label="snapped to pentatonic")
    ax1.set_title("Radius to pitch: equal steps in radius give equal musical intervals",
                  color="#e6e8eb", fontsize=12)
    ax1.set_xlabel("thumb-index radius (px)", color="#8a929e")
    ax1.set_ylabel("Hz", color="#8a929e")
    ax1.legend(facecolor="#191a1e", edgecolor="#2e3138", labelcolor="#c8ccd2", fontsize=9)

    seconds = np.arange(len(samples)) / rate
    ax2.plot(seconds, samples / 32768.0, color="#eb9d4b", lw=0.4)
    ax2.set_title(f"example_session.wav - {len(radii)} notes, "
                  f"{len(samples) / rate:.1f}s, read back from the file",
                  color="#e6e8eb", fontsize=11)
    ax2.set_xlabel("seconds", color="#8a929e")
    ax2.set_ylabel("amplitude", color="#8a929e")

    fig.tight_layout()
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  audio_mapping.png {os.path.getsize(path) // 1024} KiB")
    return path


def sample_dashboard(path=os.path.join(OUT, "sample_dashboard.html")):
    """Build a real Plotly report from a realistic session."""
    from src.dashboard import DashboardBuilder

    rng = np.random.default_rng(3)
    rows = []
    for i in range(420):
        t = i / 30.0
        open_close = 0.5 + 0.5 * np.sin(t * 0.9)
        rows.append({
            "timestamp": round(t, 4),
            "Left_Thumb-Index_2D": 45 + 190 * open_close + rng.normal(0, 3),
            "Left_Index-Middle_2D": 30 + 95 * open_close + rng.normal(0, 2),
            "Left_Middle-Ring_2D": 28 + 80 * open_close + rng.normal(0, 2),
            "Left_Ring-Pinky_2D": 26 + 70 * open_close + rng.normal(0, 2),
            "hand_status": "Left:" + ("Open" if open_close > 0.75 else
                                      "Pinch" if open_close < 0.15 else "Partial"),
        })
    DashboardBuilder(rows, title="Example session (420 frames)").write(path)
    return path


def main() -> int:
    print("Regenerating README examples from live code:")
    gesture_grid()
    theme_strip()
    kalman_chart()
    audio_chart()
    sample_dashboard()
    print(f"\nWritten to {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
