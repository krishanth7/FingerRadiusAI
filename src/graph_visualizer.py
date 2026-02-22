"""
graph_visualizer.py - Professional real-time radius graph panel.
Corporate dashboard-style chart rendered with OpenCV.
"""

from collections import deque
from typing import Dict, Tuple

import cv2
import numpy as np

from src.utils import COLORS, FINGER_KEYS, draw_filled_rect, draw_divider


class GraphVisualizer:
    """Professional scrolling line chart for finger radius data."""

    def __init__(self, width=500, height=260, max_points=200, y_range=(0, 300)):
        self.width = width
        self.height = height
        self.max_points = max_points
        self.y_min, self.y_max = y_range
        self.margin_left = 52
        self.margin_right = 16
        self.margin_top = 42
        self.margin_bottom = 36
        self.plot_w = self.width - self.margin_left - self.margin_right
        self.plot_h = self.height - self.margin_top - self.margin_bottom
        self.pair_names = ["Thumb-Index", "Index-Middle", "Middle-Ring", "Ring-Pinky"]
        self.buffers: Dict[str, deque] = {
            name: deque(maxlen=max_points) for name in self.pair_names
        }

    def update(self, pair_radii: Dict[str, float]):
        for name in self.pair_names:
            self.buffers[name].append(pair_radii.get(name, 0.0))

    def render(self) -> np.ndarray:
        img = np.full((self.height, self.width, 3), COLORS["bg_primary"], dtype=np.uint8)
        self._draw_header(img)
        self._draw_plot_area(img)
        self._draw_grid(img)
        self._draw_axes_labels(img)
        self._draw_lines(img)
        self._draw_legend(img)
        return img

    def _draw_header(self, img):
        draw_filled_rect(img, (0, 0), (self.width, 34), COLORS["bg_secondary"])
        draw_divider(img, 34, 0, self.width, COLORS["border"])
        cv2.putText(img, "Radius Over Time", (self.margin_left, 23),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLORS["text_primary"], 1, cv2.LINE_AA)
        cv2.putText(img, "px", (self.width - 30, 23),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, COLORS["text_tertiary"], 1, cv2.LINE_AA)

    def _draw_plot_area(self, img):
        cv2.rectangle(img, (self.margin_left, self.margin_top),
                      (self.margin_left + self.plot_w, self.margin_top + self.plot_h),
                      COLORS["bg_secondary"], -1)
        cv2.rectangle(img, (self.margin_left, self.margin_top),
                      (self.margin_left + self.plot_w, self.margin_top + self.plot_h),
                      COLORS["border"], 1)

    def _val_to_y(self, val):
        ratio = max(0.0, min(1.0, (val - self.y_min) / (self.y_max - self.y_min + 1e-9)))
        return int(self.margin_top + self.plot_h * (1 - ratio))

    def _idx_to_x(self, idx, total):
        if total <= 1:
            return self.margin_left
        return int(self.margin_left + self.plot_w * idx / (total - 1))

    def _draw_grid(self, img):
        for i in range(6):
            y = self.margin_top + int(self.plot_h * i / 5)
            for x in range(self.margin_left + 1, self.margin_left + self.plot_w, 6):
                x_end = min(x + 3, self.margin_left + self.plot_w)
                cv2.line(img, (x, y), (x_end, y), (40, 40, 48), 1)

    def _draw_axes_labels(self, img):
        for i in range(6):
            y = self.margin_top + int(self.plot_h * i / 5)
            val = self.y_max - (self.y_max - self.y_min) * i / 5
            cv2.putText(img, f"{int(val)}", (6, y + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, COLORS["text_tertiary"], 1, cv2.LINE_AA)

    def _draw_lines(self, img):
        for pair_idx, name in enumerate(self.pair_names):
            buf = self.buffers[name]
            if len(buf) < 2:
                continue
            color = COLORS[FINGER_KEYS[pair_idx]]
            total = len(buf)
            pts = [(self._idx_to_x(i, total), self._val_to_y(v)) for i, v in enumerate(buf)]
            pts_np = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(img, [pts_np], False, color, 2, cv2.LINE_AA)
            if pts:
                last = pts[-1]
                cv2.circle(img, last, 4, color, 1, cv2.LINE_AA)
                cv2.circle(img, last, 2, color, -1, cv2.LINE_AA)
                val = list(buf)[-1]
                cv2.putText(img, f"{val:.0f}", (last[0] + 6, last[1] - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.28, color, 1, cv2.LINE_AA)

    def _draw_legend(self, img):
        draw_filled_rect(img, (0, self.height - 28), (self.width, self.height),
                         COLORS["bg_secondary"])
        draw_divider(img, self.height - 28, 0, self.width, COLORS["border"])
        spacing = self.plot_w // max(len(self.pair_names), 1)
        y_pos = self.height - 10
        for i, name in enumerate(self.pair_names):
            color = COLORS[FINGER_KEYS[i]]
            x = self.margin_left + i * spacing
            cv2.line(img, (x, y_pos), (x + 14, y_pos), color, 2, cv2.LINE_AA)
            cv2.circle(img, (x + 7, y_pos), 2, color, -1, cv2.LINE_AA)
            parts = name.split("-")
            cv2.putText(img, f"{parts[0][:3]}-{parts[1][:3]}", (x + 18, y_pos + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, COLORS["text_secondary"], 1, cv2.LINE_AA)
