"""
main.py - FingerRadiusAI Entry Point
Multi-hand Professional Dashboard for Hand Finger Radius Tracking

Controls:
    Q / ESC  - Quit       E - Export CSV      R - Reset
    T        - Trails     G - Toggle graph    S - Screenshot
"""

import sys
import time
import cv2
import numpy as np

from src.hand_tracker import HandTracker
from src.radius_calculator import RadiusCalculator
from src.graph_visualizer import GraphVisualizer
from src.utils import (
    FPSCounter, CSVExporter, COLORS, FINGER_KEYS,
    draw_label, draw_hud_frame, draw_top_bar,
    draw_filled_rect, draw_divider, draw_progress_bar,
    draw_status_badge,
)

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
                 hand_data, show_trails, show_graph):
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
    y = max(y, panel_h - 160)
    cv2.putText(p, "CONTROLS", (16, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, COLORS["text_tertiary"], 1, cv2.LINE_AA)
    y += 6
    draw_divider(p, y, 16, pw - 16, COLORS["divider"])
    y += 16

    controls = [
        ("Q", "Quit"), ("E", "Export CSV"), ("R", "Reset"),
        ("T", f"Trails {'ON' if show_trails else 'OFF'}"),
        ("G", f"Graph {'ON' if show_graph else 'OFF'}"),
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
    cv2.putText(p, "v2.0  |  FingerRadiusAI", (16, panel_h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.28, COLORS["text_tertiary"], 1, cv2.LINE_AA)
    cv2.line(p, (pw - 1, 0), (pw - 1, panel_h), COLORS["border"], 1)

    return p


def main():
    print("=" * 50)
    print("  FingerRadiusAI - Multi-Hand Tracker v2.0")
    print("=" * 50)
    print("Starting camera...")

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    if not cap.isOpened():
        print("[ERROR] Cannot open camera.")
        sys.exit(1)

    tracker = HandTracker(max_num_hands=2, min_detection_confidence=0.7,
                          min_tracking_confidence=0.6, trail_length=15,
                          smoothing_alpha=0.4)
    # One calculator per hand
    calculators = [RadiusCalculator(smoothing_alpha=0.35) for _ in range(2)]
    graph_viz = GraphVisualizer(width=500, height=260, max_points=200,
                                y_range=(0, 300))
    fps_counter = FPSCounter(window=30)
    csv_exporter = CSVExporter()

    frame_count = 0
    show_trails = True
    show_graph = True

    print("Camera ready. Show 1 or 2 hands! Press Q or ESC to quit.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        fps_counter.tick()
        frame_count += 1

        # Multi-hand detection
        num_hands = tracker.process(frame)

        # Compute radii for each hand
        hand_data = []
        for hi in range(num_hands):
            lm = tracker.get_landmarks(hi)
            if lm is not None:
                pr, wr, status = calculators[hi].compute(lm)
                label = tracker.get_label(hi)
                graph_viz.update(pr, hand_idx=hi, label=label)
                csv_exporter.record(
                    {f"{label}_{k}": v for k, v in pr.items()},
                    f"{label}:{status}"
                )
                hand_data.append({
                    "label": label, "status": status,
                    "pair_radii": pr, "wrist_radii": wr,
                    "landmarks": lm,
                })

        # Draw all hands
        tracker.draw_all(frame, show_trails)
        for hi, hd in enumerate(hand_data):
            calculators[hi].draw_radii(frame, hd["landmarks"],
                                        hd["pair_radii"], hd["wrist_radii"])

        fps = fps_counter.fps

        # Overlays
        draw_hud_frame(frame)
        draw_top_bar(frame, fps, frame_count)

        # Status badges for each hand
        for hi, hd in enumerate(hand_data):
            y_pos = 55 + hi * 28
            label_color = COLORS["accent"] if hi == 0 else COLORS["info"]
            status_colors = {
                "Open": COLORS["success"], "Closed": COLORS["warning"],
                "Pinch": COLORS["thumb"], "Partial": COLORS["text_secondary"],
            }
            sc = status_colors.get(hd["status"], COLORS["text_secondary"])
            draw_status_badge(frame,
                              f"{hd['label'].upper()}: {hd['status'].upper()}",
                              (15, y_pos), sc)

        if num_hands == 0:
            draw_status_badge(frame, "NO HANDS", (15, 55), COLORS["text_tertiary"])

        # Composite
        if show_graph:
            graph_img = graph_viz.render(num_hands=num_hands)
            graph_img = cv2.resize(graph_img, (CAMERA_WIDTH, 260))
            video_col = np.vstack([frame, graph_img])
        else:
            video_col = frame

        panel = create_panel(video_col.shape[0], fps, frame_count,
                             num_hands, hand_data, show_trails, show_graph)
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
            graph_viz = GraphVisualizer(width=500, height=260,
                                        max_points=200, y_range=(0, 300))
            print("[INFO] Data buffers reset.")
        elif key == ord('t'):
            show_trails = not show_trails
        elif key == ord('g'):
            show_graph = not show_graph
        elif key == ord('s'):
            fn = f"screenshot_{int(time.time())}.png"
            cv2.imwrite(fn, composite)
            print(f"[INFO] Screenshot -> {fn}")

    tracker.release()
    cap.release()
    cv2.destroyAllWindows()
    print("FingerRadiusAI closed.")


if __name__ == "__main__":
    main()
