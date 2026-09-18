"""
main.py - FingerRadiusAI Entry Point
Multi-hand dashboard for hand finger radius tracking.

Usage:
    python main.py                          live camera
    python main.py --source clip.mp4        a recorded file
    python main.py --gui                    windowed Tkinter front end
    python main.py --theme cyberpunk        pick a colour theme
    python main.py --filter ema             use the old EMA instead of Kalman
    python main.py --audio                  sonify the thumb-index radius
    python main.py --onnx-model hand.onnx   run a user-supplied ONNX model
    python main.py --list-providers         report ONNX Runtime capability

Keys:
    Q / ESC  Quit          E  Export CSV        R  Reset buffers
    T        Trails        G  Graph             S  Screenshot
    D        2D / 3D       K  Kalman / EMA      C  Cycle theme
    A        Audio         P  Pause (file)      H  Dashboard HTML
    LEFT / RIGHT  Seek 5s (file)
"""

import argparse
import sys
import time

import cv2
import numpy as np

from src.audio_feedback import AudioFeedback, ToneMapper
from src.gestures import GestureRecognizer
from src.graph_visualizer import GraphVisualizer
from src.hand_tracker import HandTracker
from src.kalman import KalmanLandmarkSet
from src.onnx_backend import describe_runtime
from src.radius_calculator import RadiusCalculator
from src.themes import THEME_NAMES, apply_theme, current_theme, next_theme
from src.utils import (
    FPSCounter, CSVExporter, COLORS, FINGER_KEYS,
    draw_label, draw_hud_frame, draw_top_bar,
    draw_filled_rect, draw_divider, draw_progress_bar,
    draw_status_badge,
)
from src.video_source import VideoSource

CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
WINDOW_NAME = "FingerRadiusAI"
PANEL_WIDTH = 260


def _draw_hand_section(p, y, pw, hand_label, hand_status, pair_radii, hand_idx):
    """Draw a radius metrics section for one hand on the panel."""
    # Hand header
    label_color = COLORS["accent"] if hand_idx == 0 else COLORS["info"]
    cv2.putText(p, f"{hand_label.upper()} HAND", (16, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, label_color, 1, cv2.LINE_AA)

    # Status inline
    status_colors = {
        "Open": COLORS["success"], "Closed": COLORS["warning"],
        "Pinch": COLORS["thumb"], "Partial": COLORS["text_secondary"],
        "N/A": COLORS["text_tertiary"],
    }
    sc = status_colors.get(hand_status, COLORS["text_secondary"])
    cv2.putText(p, hand_status.upper(), (pw - 75, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, sc, 1, cv2.LINE_AA)
    y += 6
    draw_divider(p, y, 16, pw - 16, COLORS["divider"])
    y += 14

    pair_names = ["Thumb-Index", "Index-Middle", "Middle-Ring", "Ring-Pinky"]
    for i, name in enumerate(pair_names):
        color = COLORS[FINGER_KEYS[i]]
        val = pair_radii.get(name, 0)
        cv2.circle(p, (22, y), 3, color, -1, cv2.LINE_AA)
        parts = name.split("-")
        cv2.putText(p, f"{parts[0][:3]}-{parts[1][:3]}", (30, y + 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, COLORS["text_secondary"], 1, cv2.LINE_AA)
        cv2.putText(p, f"{val:.0f}", (pw - 50, y + 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, COLORS["text_primary"], 1, cv2.LINE_AA)
        y += 8
        draw_progress_bar(p, (22, y), pw - 44, 4, val, 300, color, border=False)
        y += 14
    return y


def create_panel(panel_h, fps, frame_count, num_hands,
                 hand_data, show_trails, show_graph, use_3d=False):
    """Create the professional side panel supporting multi-hand display."""
    pw = PANEL_WIDTH
    p = np.full((panel_h, pw, 3), COLORS["bg_secondary"], dtype=np.uint8)

    # Header
    draw_filled_rect(p, (0, 0), (pw, 50), COLORS["bg_primary"])
    cv2.putText(p, "FingerRadius", (16, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLORS["accent"], 1, cv2.LINE_AA)
    cv2.putText(p, "AI", (142, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLORS["text_primary"], 1, cv2.LINE_AA)
    cv2.putText(p, "Multi-Hand Analytics", (16, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, COLORS["text_tertiary"], 1, cv2.LINE_AA)
    draw_divider(p, 50, 0, pw, COLORS["border"])

    y = 68

    # System
    cv2.putText(p, "SYSTEM", (16, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, COLORS["text_tertiary"], 1, cv2.LINE_AA)
    y += 6
    draw_divider(p, y, 16, pw - 16, COLORS["divider"])
    y += 18

    fps_color = COLORS["success"] if fps >= 20 else COLORS["warning"]
    cv2.circle(p, (22, y - 4), 4, fps_color, -1, cv2.LINE_AA)
    cv2.putText(p, "FPS", (32, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, COLORS["text_label"], 1, cv2.LINE_AA)
    cv2.putText(p, f"{fps:.1f}", (pw - 60, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, fps_color, 1, cv2.LINE_AA)
    y += 20

    cv2.putText(p, "Hands", (32, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, COLORS["text_label"], 1, cv2.LINE_AA)
    h_color = COLORS["success"] if num_hands > 0 else COLORS["text_tertiary"]
    cv2.putText(p, f"{num_hands}", (pw - 50, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, h_color, 1, cv2.LINE_AA)
    y += 20

    # Mode (2D / 3D)
    mode_label = "3D" if use_3d else "2D"
    mode_color = COLORS["accent"] if use_3d else COLORS["success"]
    cv2.putText(p, "Mode", (32, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, COLORS["text_label"], 1, cv2.LINE_AA)
    cv2.putText(p, mode_label, (pw - 50, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, mode_color, 1, cv2.LINE_AA)
    y += 20

    cv2.putText(p, "Frame", (32, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, COLORS["text_label"], 1, cv2.LINE_AA)
    cv2.putText(p, f"{frame_count}", (pw - 70, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, COLORS["text_secondary"], 1, cv2.LINE_AA)
    y += 24

    # Hand sections
    for hi in range(min(num_hands, 2)):
        hd = hand_data[hi]
        y = _draw_hand_section(p, y, pw, hd["label"], hd["status"],
                               hd["pair_radii"], hi)
        y += 6

    if num_hands == 0:
        cv2.putText(p, "NO HANDS DETECTED", (16, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, COLORS["text_tertiary"], 1, cv2.LINE_AA)
        y += 6
        draw_divider(p, y, 16, pw - 16, COLORS["divider"])
        y += 20

    # Controls
    y = max(y, panel_h - 180)
    cv2.putText(p, "CONTROLS", (16, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, COLORS["text_tertiary"], 1, cv2.LINE_AA)
    y += 6
    draw_divider(p, y, 16, pw - 16, COLORS["divider"])
    y += 16

    controls = [
        ("Q", "Quit"), ("E", "Export CSV"), ("R", "Reset"),
        ("T", f"Trails {'ON' if show_trails else 'OFF'}"),
        ("G", f"Graph {'ON' if show_graph else 'OFF'}"),
        ("D", f"{'3D' if use_3d else '2D'} Radius"),
        ("S", "Screenshot"),
    ]
    for key, desc in controls:
        kw = 18
        draw_filled_rect(p, (20, y - 11), (20 + kw, y + 3), COLORS["bg_primary"])
        cv2.rectangle(p, (20, y - 11), (20 + kw, y + 3), COLORS["border"], 1)
        cv2.putText(p, key, (24, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, COLORS["text_primary"], 1, cv2.LINE_AA)
        cv2.putText(p, desc, (46, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.33, COLORS["text_tertiary"], 1, cv2.LINE_AA)
        y += 18

    # Footer
    draw_divider(p, panel_h - 25, 0, pw, COLORS["border"])
    cv2.putText(p, "v2.1  |  FingerRadiusAI", (16, panel_h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.28, COLORS["text_tertiary"], 1, cv2.LINE_AA)
    cv2.line(p, (pw - 1, 0), (pw - 1, panel_h), COLORS["border"], 1)

    return p


def parse_args(argv=None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="FingerRadiusAI - multi-hand finger radius tracking.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--source", default="0",
        help="Camera index or path to a video file. Default: 0 (first camera).",
    )
    parser.add_argument("--gui", action="store_true",
                        help="Open the Tkinter window instead of the OpenCV one.")
    parser.add_argument("--theme", default="corporate", choices=THEME_NAMES,
                        help="Colour theme. Default: corporate.")
    parser.add_argument("--filter", default="kalman", choices=("kalman", "ema"),
                        help="Landmark smoothing. Default: kalman.")
    parser.add_argument("--audio", action="store_true",
                        help="Sonify the thumb-index radius.")
    parser.add_argument("--scale", default="pentatonic",
                        help="Musical scale for --audio, or 'none' for a glide.")
    parser.add_argument("--no-loop", action="store_true",
                        help="Stop at the end of a video file instead of looping.")
    parser.add_argument("--fast", action="store_true",
                        help="Process a file as fast as possible, ignoring its frame rate.")
    parser.add_argument("--onnx-model", default=None,
                        help="Path to an ONNX hand model. None ships with this repo.")
    parser.add_argument("--list-providers", action="store_true",
                        help="Report ONNX Runtime providers and exit.")
    parser.add_argument("--export-dashboard", metavar="HTML", default=None,
                        help="On exit, write an interactive Plotly report here.")
    return parser.parse_args(argv)


def run_dashboard(exporter: CSVExporter, path: str) -> None:
    """Write the Plotly report, reporting any failure rather than raising."""
    try:
        from src.dashboard import DashboardBuilder
        DashboardBuilder(exporter._rows, title="FingerRadiusAI session").write(path)
    except Exception as error:
        print(f"[WARN] Dashboard not written: {error}")


def main(argv=None) -> int:
    args = parse_args(argv)

    if args.list_providers:
        print(describe_runtime())
        return 0

    apply_theme(args.theme)

    print("=" * 60)
    print("  FingerRadiusAI - Multi-Hand Tracker v3.0")
    print("=" * 60)

    if args.gui:
        from src.gui import FingerRadiusGUI
        FingerRadiusGUI(source=args.source, theme=args.theme,
                        use_kalman=args.filter == "kalman").run()
        return 0

    try:
        source = VideoSource(
            args.source,
            width=CAMERA_WIDTH, height=CAMERA_HEIGHT,
            loop=not args.no_loop, realtime=not args.fast,
        )
    except (FileNotFoundError, RuntimeError) as error:
        print(f"[ERROR] {error}")
        return 1

    print(f"Source : {source.describe()}")
    print(f"Theme  : {current_theme()}   Filter: {args.filter}")

    onnx_backend = None
    if args.onnx_model:
        try:
            from src.onnx_backend import OnnxLandmarkBackend
            onnx_backend = OnnxLandmarkBackend(args.onnx_model)
            print(f"ONNX   : {onnx_backend.describe()}")
            print(f"         {onnx_backend.benchmark(runs=20)}")
        except Exception as error:
            print(f"[WARN] ONNX backend unavailable, staying on MediaPipe: {error}")

    tracker = HandTracker(max_num_hands=2, min_detection_confidence=0.7,
                          min_tracking_confidence=0.6, trail_length=15,
                          smoothing_alpha=0.4)
    calculators = [RadiusCalculator(smoothing_alpha=0.35) for _ in range(2)]
    recognizers = [GestureRecognizer(hold_frames=3) for _ in range(2)]
    kalman_banks = [KalmanLandmarkSet() for _ in range(2)]
    graph_viz = GraphVisualizer(width=500, height=260, max_points=200, y_range=(0, 300))
    fps_counter = FPSCounter(window=30)
    csv_exporter = CSVExporter()

    audio = None
    if args.audio:
        scale = None if args.scale.lower() == "none" else args.scale
        audio = AudioFeedback(ToneMapper(scale=scale))
        print(f"Audio  : {audio.describe()}")

    frame_count = 0
    show_trails = True
    show_graph = True
    use_3d = False
    use_kalman = args.filter == "kalman"

    print("\nReady. Press Q or ESC to quit.\n")

    while True:
        ok, frame = source.read()
        if not ok:
            print("\n[INFO] Source finished.")
            break

        fps_counter.tick()
        frame_count += 1

        num_hands = tracker.process(frame)

        hand_data = []
        for hi in range(num_hands):
            lm = tracker.get_landmarks(hi)
            lm_3d = tracker.get_landmarks_3d(hi)
            if lm is None:
                continue

            # The Kalman bank runs on top of the tracker's own EMA rather than
            # replacing it in place, so switching filters mid-session needs no
            # re-initialisation of the tracker.
            if use_kalman and lm_3d is not None:
                filtered = kalman_banks[hi].update(lm_3d)
                lm = [(x, y) for x, y, _ in filtered]
                lm_3d = filtered

            pr, wr, status, dd = calculators[hi].compute(
                lm, landmarks_3d=lm_3d, use_3d=use_3d
            )
            gesture = recognizers[hi].update(lm)
            label = tracker.get_label(hi)
            graph_viz.update(pr, hand_idx=hi, label=label)

            if audio is not None and hi == 0:
                audio.update(pr.get("Thumb-Index", 0.0))

            mode_tag = "3D" if use_3d else "2D"
            csv_exporter.record(
                {f"{label}_{k}_{mode_tag}": v for k, v in pr.items()},
                f"{label}:{status}|{gesture.gesture}",
            )
            hand_data.append({
                "label": label, "status": status, "pair_radii": pr,
                "wrist_radii": wr, "depth_deltas": dd, "landmarks": lm,
                "gesture": gesture,
            })

        tracker.draw_all(frame, show_trails)
        for hi, hd in enumerate(hand_data):
            calculators[hi].draw_radii(
                frame, hd["landmarks"], hd["pair_radii"], hd["wrist_radii"],
                use_3d=use_3d, depth_deltas=hd.get("depth_deltas"),
            )

        fps = fps_counter.fps
        draw_hud_frame(frame)
        draw_top_bar(frame, fps, frame_count)

        status_colors = {
            "Open": COLORS["success"], "Closed": COLORS["warning"],
            "Pinch": COLORS["thumb"], "Partial": COLORS["text_secondary"],
        }
        for hi, hd in enumerate(hand_data):
            y_pos = 55 + hi * 28
            sc = status_colors.get(hd["status"], COLORS["text_secondary"])
            g = hd["gesture"]
            draw_status_badge(
                frame,
                f"{hd['label'].upper()}: {g.gesture.upper()} ({g.confidence:.0%})",
                (15, y_pos), sc,
            )

        if num_hands == 0:
            draw_status_badge(frame, "NO HANDS", (15, 55), COLORS["text_tertiary"])

        # Playback progress, for a file
        if source.is_file:
            bar_w = frame.shape[1] - 30
            draw_progress_bar(frame, (15, frame.shape[0] - 18), bar_w, 5,
                              source.progress * 100, 100, COLORS["accent"])

        if show_graph:
            graph_img = graph_viz.render(num_hands=num_hands)
            graph_img = cv2.resize(graph_img, (frame.shape[1], 260))
            video_col = np.vstack([frame, graph_img])
        else:
            video_col = frame

        panel = create_panel(video_col.shape[0], fps, frame_count, num_hands,
                             hand_data, show_trails, show_graph, use_3d=use_3d)
        composite = np.hstack([panel, video_col])
        cv2.imshow(WINDOW_NAME, composite)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            print("\nShutting down...")
            break
        elif key == ord('e'):
            csv_exporter.export("radius_data.csv")
        elif key == ord('r'):
            csv_exporter.clear()
            graph_viz = GraphVisualizer(width=500, height=260, max_points=200,
                                        y_range=(0, 300))
            for bank in kalman_banks:
                bank.reset()
            for rec in recognizers:
                rec.reset()
            print("[INFO] Buffers reset.")
        elif key == ord('t'):
            show_trails = not show_trails
        elif key == ord('g'):
            show_graph = not show_graph
        elif key == ord('d'):
            use_3d = not use_3d
            print(f"[INFO] Radius mode: {'3D (depth-aware)' if use_3d else '2D'}")
        elif key == ord('k'):
            use_kalman = not use_kalman
            for bank in kalman_banks:
                bank.reset()
            print(f"[INFO] Smoothing: {'Kalman' if use_kalman else 'EMA'}")
        elif key == ord('c'):
            print(f"[INFO] Theme: {next_theme()}")
        elif key == ord('a'):
            if audio is None:
                audio = AudioFeedback()
                print(f"[INFO] {audio.describe()}")
            else:
                audio.enabled = not audio.enabled
                print(f"[INFO] Audio {'on' if audio.enabled else 'off'}")
        elif key == ord('p'):
            print(f"[INFO] {'Paused' if source.toggle_pause() else 'Resumed'}")
        elif key == ord('h'):
            run_dashboard(csv_exporter, "dashboard.html")
        elif key == 81:                      # left arrow
            source.seek_relative(-5)
        elif key == 83:                      # right arrow
            source.seek_relative(5)
        elif key == ord('s'):
            fn = f"screenshot_{int(time.time())}.png"
            cv2.imwrite(fn, composite)
            print(f"[INFO] Screenshot -> {fn}")

    if args.export_dashboard:
        run_dashboard(csv_exporter, args.export_dashboard)
    if audio is not None and audio._history:
        audio.render_history("session_audio.wav")

    tracker.release()
    source.release()
    cv2.destroyAllWindows()
    print("FingerRadiusAI closed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
