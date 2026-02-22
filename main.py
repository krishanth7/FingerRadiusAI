"""
main.py - FingerRadiusAI Entry Point
Professional Corporate Dashboard for Hand Finger Radius Tracking

Controls:
    Q / ESC  - Quit
    E        - Export data to CSV
    R        - Reset buffers
    T        - Toggle trails
    G        - Toggle graph
    S        - Screenshot
"""

import sys
import time
import cv2
import numpy as np

from src.hand_tracker import HandTracker
from src.radius_calculator import RadiusCalculator
from src.graph_visualizer import GraphVisualizer
from src.utils import (
    FPSCounter, CSVExporter,
    COLORS, FINGER_KEYS,
    draw_label, draw_hud_frame, draw_top_bar,
    draw_filled_rect, draw_divider, draw_progress_bar,
    draw_heading, draw_caption, draw_status_badge,
)

CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
WINDOW_NAME = "FingerRadiusAI"
PANEL_WIDTH = 260


def create_panel(panel_h, fps, hand_status, frame_count,
                 pair_radii, show_trails, show_graph):
    """Create the professional side panel."""
    pw = PANEL_WIDTH
    p = np.full((panel_h, pw, 3), COLORS["bg_secondary"], dtype=np.uint8)

    # ── Header ──
    draw_filled_rect(p, (0, 0), (pw, 50), COLORS["bg_primary"])
    cv2.putText(p, "FingerRadius", (16, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLORS["accent"], 1, cv2.LINE_AA)
    cv2.putText(p, "AI", (142, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLORS["text_primary"], 1, cv2.LINE_AA)
    cv2.putText(p, "Hand Tracking Analytics", (16, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, COLORS["text_tertiary"], 1, cv2.LINE_AA)
    draw_divider(p, 50, 0, pw, COLORS["border"])

    y = 70

    # ── System Metrics Section ──
    cv2.putText(p, "SYSTEM", (16, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, COLORS["text_tertiary"], 1, cv2.LINE_AA)
    y += 6
    draw_divider(p, y, 16, pw - 16, COLORS["divider"])
    y += 20

    # FPS
    fps_color = COLORS["success"] if fps >= 20 else COLORS["warning"]
    cv2.circle(p, (22, y - 4), 4, fps_color, -1, cv2.LINE_AA)
    cv2.putText(p, "FPS", (32, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, COLORS["text_label"], 1, cv2.LINE_AA)
    cv2.putText(p, f"{fps:.1f}", (pw - 60, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, fps_color, 1, cv2.LINE_AA)
    y += 22

    # Frame
    cv2.putText(p, "Frame", (32, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, COLORS["text_label"], 1, cv2.LINE_AA)
    cv2.putText(p, f"{frame_count}", (pw - 70, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, COLORS["text_secondary"], 1, cv2.LINE_AA)
    y += 22

    # Status
    status_colors = {
        "Open": COLORS["success"], "Closed": COLORS["warning"],
        "Pinch": COLORS["thumb"], "Partial": COLORS["text_secondary"],
        "N/A": COLORS["text_tertiary"],
    }
    sc = status_colors.get(hand_status, COLORS["text_secondary"])
    cv2.putText(p, "Status", (32, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, COLORS["text_label"], 1, cv2.LINE_AA)
    # Status badge
    cv2.putText(p, hand_status.upper(), (pw - 80, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, sc, 1, cv2.LINE_AA)
    y += 30

    # ── Radius Metrics Section ──
    cv2.putText(p, "RADIUS METRICS", (16, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, COLORS["text_tertiary"], 1, cv2.LINE_AA)
    y += 6
    draw_divider(p, y, 16, pw - 16, COLORS["divider"])
    y += 16

    pair_names = ["Thumb-Index", "Index-Middle", "Middle-Ring", "Ring-Pinky"]
    for i, name in enumerate(pair_names):
        color = COLORS[FINGER_KEYS[i]]
        val = pair_radii.get(name, 0)

        # Color indicator dot
        cv2.circle(p, (22, y), 4, color, -1, cv2.LINE_AA)

        # Label
        parts = name.split("-")
        label = f"{parts[0][:3]}-{parts[1][:3]}"
        cv2.putText(p, label, (32, y + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, COLORS["text_secondary"], 1, cv2.LINE_AA)

        # Value
        cv2.putText(p, f"{val:.0f} px", (pw - 65, y + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, COLORS["text_primary"], 1, cv2.LINE_AA)
        y += 10

        # Progress bar
        draw_progress_bar(p, (22, y), pw - 44, 5, val, 300, color, border=False)
        y += 20

    y += 10

    # ── Controls Section ──
    cv2.putText(p, "CONTROLS", (16, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, COLORS["text_tertiary"], 1, cv2.LINE_AA)
    y += 6
    draw_divider(p, y, 16, pw - 16, COLORS["divider"])
    y += 18

    controls = [
        ("Q", "Quit"),
        ("E", "Export CSV"),
        ("R", "Reset"),
        ("T", f"Trails {'ON' if show_trails else 'OFF'}"),
        ("G", f"Graph {'ON' if show_graph else 'OFF'}"),
        ("S", "Screenshot"),
    ]
    for key, desc in controls:
        # Key badge
        kw = 18
        draw_filled_rect(p, (20, y - 11), (20 + kw, y + 3), COLORS["bg_primary"])
        cv2.rectangle(p, (20, y - 11), (20 + kw, y + 3), COLORS["border"], 1)
        cv2.putText(p, key, (24, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, COLORS["text_primary"], 1, cv2.LINE_AA)
        cv2.putText(p, desc, (46, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.33, COLORS["text_tertiary"], 1, cv2.LINE_AA)
        y += 20

    # ── Footer ──
    draw_divider(p, panel_h - 25, 0, pw, COLORS["border"])
    cv2.putText(p, "v1.0  |  FingerRadiusAI", (16, panel_h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.28, COLORS["text_tertiary"], 1, cv2.LINE_AA)

    # Right border
    cv2.line(p, (pw - 1, 0), (pw - 1, panel_h), COLORS["border"], 1)

    return p


def main():
    print("=" * 50)
    print("  FingerRadiusAI - Professional Hand Tracker")
    print("=" * 50)
    print("Starting camera...")

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    if not cap.isOpened():
        print("[ERROR] Cannot open camera.")
        sys.exit(1)

    tracker = HandTracker(max_num_hands=1, min_detection_confidence=0.7,
                          min_tracking_confidence=0.6, trail_length=15,
                          smoothing_alpha=0.4)
    calculator = RadiusCalculator(smoothing_alpha=0.35)
    graph_viz = GraphVisualizer(width=500, height=260, max_points=200,
                                y_range=(0, 300))
    fps_counter = FPSCounter(window=30)
    csv_exporter = CSVExporter()

    frame_count = 0
    show_trails = True
    show_graph = True
    hand_status = "N/A"
    pair_radii = {}
    wrist_radii = {}

    print("Camera ready. Press Q or ESC to quit.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        fps_counter.tick()
        frame_count += 1

        # Hand detection
        landmarks = tracker.process(frame)

        if landmarks is not None:
            pair_radii, wrist_radii, hand_status = calculator.compute(landmarks)
            graph_viz.update(pair_radii)
            csv_exporter.record(pair_radii, hand_status)

            tracker.draw_skeleton(frame)
            calculator.draw_radii(frame, landmarks, pair_radii, wrist_radii)
            if show_trails:
                tracker.draw_trails(frame)
        else:
            hand_status = "N/A"

        fps = fps_counter.fps

        # Professional overlays on video
        draw_hud_frame(frame)
        draw_top_bar(frame, fps, frame_count)
        calculator.draw_status(frame, hand_status, position=(15, 60))

        # Build composite: Panel | Video | Graph
        graph_h = 260 if show_graph else 0
        total_h = CAMERA_HEIGHT + graph_h

        if show_graph:
            graph_img = graph_viz.render()
            graph_img = cv2.resize(graph_img, (CAMERA_WIDTH, graph_h))
            video_col = np.vstack([frame, graph_img])
        else:
            video_col = frame

        panel = create_panel(video_col.shape[0], fps, hand_status,
                             frame_count, pair_radii, show_trails, show_graph)
        composite = np.hstack([panel, video_col])

        cv2.imshow(WINDOW_NAME, composite)

        # Keyboard
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
