"""
utils.py - Utility functions for FingerRadiusAI
Professional corporate-style UI helpers, smoothing, FPS, CSV export.
"""

import csv
import time
import math
from collections import deque
from typing import List, Tuple, Dict, Optional

import cv2
import numpy as np

# ── Corporate Color Palette (BGR) ──
COLORS = {
    "thumb":    (60, 180, 235),
    "index":    (180, 200, 60),
    "middle":   (220, 140, 60),
    "ring":     (160, 80, 200),
    "pinky":    (80, 120, 220),
    "bg_primary":   (25, 25, 30),
    "bg_secondary": (35, 35, 42),
    "bg_tertiary":  (45, 45, 55),
    "border":       (60, 60, 72),
    "border_light": (75, 75, 88),
    "divider":      (50, 50, 60),
    "text_primary":   (230, 232, 235),
    "text_secondary": (160, 165, 172),
    "text_tertiary":  (100, 105, 112),
    "text_label":     (130, 135, 142),
    "accent":    (200, 170, 60),
    "accent_bg": (50, 45, 35),
    "success":   (120, 200, 80),
    "warning":   (50, 140, 240),
    "danger":    (70, 80, 210),
    "info":      (200, 160, 60),
    "bone":         (70, 70, 80),
    "bone_light":   (90, 90, 105),
    "joint":        (110, 110, 125),
    "wrist":        (140, 145, 155),
}

FINGER_NAMES = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
FINGER_KEYS  = ["thumb", "index", "middle", "ring", "pinky"]


# ── Smoothing ──
class ExponentialMovingAverage:
    def __init__(self, alpha=0.3):
        self.alpha = alpha
        self._value = None

    def update(self, v):
        if self._value is None:
            self._value = v
        else:
            self._value = self.alpha * v + (1 - self.alpha) * self._value
        return self._value

    @property
    def value(self):
        return self._value if self._value is not None else 0.0

    def reset(self):
        self._value = None


class MovingAverageFilter:
    def __init__(self, window_size=5):
        self._buffer = deque(maxlen=window_size)

    def update(self, v):
        self._buffer.append(v)
        return sum(self._buffer) / len(self._buffer)

    @property
    def value(self):
        return sum(self._buffer) / len(self._buffer) if self._buffer else 0.0

    def reset(self):
        self._buffer.clear()


# ── FPS Counter ──
class FPSCounter:
    def __init__(self, window=30):
        self._timestamps = deque(maxlen=window)

    def tick(self):
        self._timestamps.append(time.perf_counter())

    @property
    def fps(self):
        if len(self._timestamps) < 2:
            return 0.0
        dt = self._timestamps[-1] - self._timestamps[0]
        return (len(self._timestamps) - 1) / dt if dt > 0 else 0.0


# ── CSV Exporter ──
class CSVExporter:
    def __init__(self):
        self._rows = []
        self._start_time = time.time()

    def record(self, radii, hand_status):
        row = {"timestamp": round(time.time() - self._start_time, 4)}
        row.update(radii)
        row["hand_status"] = hand_status
        self._rows.append(row)

    def export(self, filepath="radius_data.csv"):
        if not self._rows:
            print("[CSVExporter] No data to export.")
            return
        fieldnames = list(self._rows[0].keys())
        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self._rows)
        print(f"[CSVExporter] Exported {len(self._rows)} rows -> {filepath}")

    def clear(self):
        self._rows.clear()
        self._start_time = time.time()


def euclidean_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def euclidean_distance_3d(p1, p2):
    """3D Euclidean distance using (x, y, z) tuples."""
    return math.sqrt(
        (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2
    )


# ── Professional UI Drawing ──

def draw_filled_rect(img, pt1, pt2, color, alpha=1.0):
    if alpha >= 1.0:
        cv2.rectangle(img, pt1, pt2, color, -1)
    else:
        overlay = img.copy()
        cv2.rectangle(overlay, pt1, pt2, color, -1)
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def draw_rounded_rect(img, pt1, pt2, color, radius=8, thickness=-1, alpha=0.7):
    overlay = img.copy()
    x1, y1 = pt1
    x2, y2 = pt2
    r = min(radius, (x2 - x1) // 2, (y2 - y1) // 2)
    if r < 1:
        cv2.rectangle(overlay, pt1, pt2, color, thickness)
    else:
        cv2.rectangle(overlay, (x1 + r, y1), (x2 - r, y2), color, thickness)
        cv2.rectangle(overlay, (x1, y1 + r), (x2, y2 - r), color, thickness)
        cv2.circle(overlay, (x1 + r, y1 + r), r, color, thickness)
        cv2.circle(overlay, (x2 - r, y1 + r), r, color, thickness)
        cv2.circle(overlay, (x1 + r, y2 - r), r, color, thickness)
        cv2.circle(overlay, (x2 - r, y2 - r), r, color, thickness)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def draw_clean_line(img, pt1, pt2, color, thickness=1):
    cv2.line(img, pt1, pt2, color, thickness, cv2.LINE_AA)


def draw_dot(img, center, radius, color, filled=True):
    cv2.circle(img, center, radius, color, -1 if filled else 1, cv2.LINE_AA)


def draw_label(img, text, position, color=None, font_scale=0.45,
               thickness=1, bg=False, bg_color=None):
    if color is None:
        color = COLORS["text_primary"]
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = position
    if bg:
        if bg_color is None:
            bg_color = COLORS["bg_primary"]
        pad = 4
        draw_filled_rect(img, (x - pad, y - th - pad),
                         (x + tw + pad, y + baseline + pad), bg_color, alpha=0.75)
    cv2.putText(img, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)


def draw_heading(img, text, position, color=None, font_scale=0.55, thickness=1):
    if color is None:
        color = COLORS["text_primary"]
    cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, color, thickness, cv2.LINE_AA)


def draw_caption(img, text, position, color=None, font_scale=0.35):
    if color is None:
        color = COLORS["text_tertiary"]
    cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, color, 1, cv2.LINE_AA)


def draw_divider(img, y, x_start, x_end, color=None):
    if color is None:
        color = COLORS["divider"]
    cv2.line(img, (x_start, y), (x_end, y), color, 1)


def draw_progress_bar(img, position, width, height, value, max_value, color,
                      bg_color=None, border=True):
    x, y = position
    if bg_color is None:
        bg_color = COLORS["bg_primary"]
    cv2.rectangle(img, (x, y), (x + width, y + height), bg_color, -1)
    fill_w = int(width * min(value / max(max_value, 1), 1.0))
    if fill_w > 0:
        overlay = img.copy()
        cv2.rectangle(overlay, (x, y), (x + fill_w, y + height), color, -1)
        cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
    if border:
        cv2.rectangle(img, (x, y), (x + width, y + height), COLORS["border"], 1)


def draw_status_badge(img, text, position, color):
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = 0.45
    (tw, th), bl = cv2.getTextSize(text, font, fs, 1)
    x, y = position
    px, py = 8, 4
    bg = tuple(max(0, min(255, int(c * 0.3))) for c in color)
    draw_filled_rect(img, (x - px, y - th - py),
                     (x + tw + px, y + bl + py), bg, alpha=0.6)
    cv2.rectangle(img, (x - px, y - th - py),
                  (x - px + 3, y + bl + py), color, -1)
    cv2.putText(img, text, (x, y), font, fs, color, 1, cv2.LINE_AA)


def draw_hud_frame(img):
    h, w = img.shape[:2]
    c = COLORS["border_light"]
    L = 20
    cv2.line(img, (2, 2), (L, 2), c, 1, cv2.LINE_AA)
    cv2.line(img, (2, 2), (2, L), c, 1, cv2.LINE_AA)
    cv2.line(img, (w-3, 2), (w-L, 2), c, 1, cv2.LINE_AA)
    cv2.line(img, (w-3, 2), (w-3, L), c, 1, cv2.LINE_AA)
    cv2.line(img, (2, h-3), (L, h-3), c, 1, cv2.LINE_AA)
    cv2.line(img, (2, h-3), (2, h-L), c, 1, cv2.LINE_AA)
    cv2.line(img, (w-3, h-3), (w-L, h-3), c, 1, cv2.LINE_AA)
    cv2.line(img, (w-3, h-3), (w-3, h-L), c, 1, cv2.LINE_AA)


def draw_top_bar(img, fps, frame_count, recording=False):
    h, w = img.shape[:2]
    bar_h = 32
    draw_filled_rect(img, (0, 0), (w, bar_h), COLORS["bg_primary"], alpha=0.75)
    draw_divider(img, bar_h, 0, w, COLORS["border"])
    cv2.putText(img, "FINGER RADIUS AI", (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLORS["accent"], 1, cv2.LINE_AA)
    fps_color = COLORS["success"] if fps >= 20 else COLORS["warning"]
    cv2.putText(img, f"FPS {fps:.0f}", (w - 80, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, fps_color, 1, cv2.LINE_AA)
    dot_color = COLORS["success"] if fps >= 15 else COLORS["danger"]
    cv2.circle(img, (w - 90, 18), 4, dot_color, -1, cv2.LINE_AA)
    cv2.putText(img, f"F:{frame_count}", (w - 170, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, COLORS["text_tertiary"], 1, cv2.LINE_AA)
