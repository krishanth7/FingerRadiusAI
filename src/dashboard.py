"""
dashboard.py - Interactive Plotly report from recorded session data.

The CSV export answers "what were the numbers"; this answers "what happened".
It renders a self-contained HTML file -- Plotly is inlined, so the report
opens offline, can be emailed, and does not phone anywhere when opened.

Input is the row list :class:`~src.utils.CSVExporter` already collects, so
nothing new has to be recorded during a session.
"""

from __future__ import annotations

import csv
import os
from typing import Dict, Iterable, List, Optional, Sequence

__all__ = ["DashboardBuilder", "build_dashboard", "load_rows"]

#: Colour per finger pair, matched to the on-screen palette (given as RGB
#: here, because Plotly is RGB and OpenCV's BGR would come out wrong).
_SERIES_COLORS = [
    "#eb9d4b", "#3cc8b4", "#3c8cdc", "#c850a0", "#dc7850",
    "#7ec850", "#d4c04b", "#8a7ef0",
]


def load_rows(csv_path: str) -> List[Dict[str, str]]:
    """Read a CSV exported by :class:`~src.utils.CSVExporter`.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"No such CSV: {csv_path}")
    with open(csv_path, newline="") as handle:
        return list(csv.DictReader(handle))


def _numeric(value: object) -> Optional[float]:
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return out if out == out else None  # reject NaN


class DashboardBuilder:
    """Turns recorded rows into a standalone interactive HTML report.

    Args:
        rows: Dicts with a ``timestamp`` key, any number of numeric radius
            columns, and an optional ``hand_status``/``gesture`` column.
        title: Shown at the top of the report.

    Raises:
        ImportError: If Plotly is not installed, with the install command.

    Example:
        >>> rows = [{"timestamp": 0.0, "Left_Thumb-Index_2D": 120.0},
        ...         {"timestamp": 0.1, "Left_Thumb-Index_2D": 124.0}]
        >>> builder = DashboardBuilder(rows)      # doctest: +SKIP
        >>> builder.write("report.html")          # doctest: +SKIP
    """

    #: Columns that are metadata rather than measurements.
    META_COLUMNS = {"timestamp", "hand_status", "gesture", "frame", "theme"}

    def __init__(self, rows: Sequence[Dict[str, object]], title: str = "FingerRadiusAI session") -> None:
        try:
            import plotly.graph_objects as go  # noqa: F401
        except ImportError as error:  # pragma: no cover - environment dependent
            raise ImportError(
                "The dashboard needs Plotly. Install it with:\n"
                "    pip install plotly"
            ) from error

        if not rows:
            raise ValueError("No rows to plot -- record a session first.")

        self.rows = list(rows)
        self.title = title
        self.timestamps: List[float] = []
        for row in self.rows:
            t = _numeric(row.get("timestamp"))
            self.timestamps.append(0.0 if t is None else t)

        self.series: Dict[str, List[Optional[float]]] = {}
        for column in self.rows[0]:
            if column in self.META_COLUMNS:
                continue
            values = [_numeric(row.get(column)) for row in self.rows]
            # A column that never held a number is not a measurement.
            if any(v is not None for v in values):
                self.series[column] = values

        self.statuses: List[str] = [
            str(row.get("hand_status") or row.get("gesture") or "") for row in self.rows
        ]

    # ------------------------------------------------------------------
    def summary(self) -> Dict[str, Dict[str, float]]:
        """Per-series min / max / mean / stdev, skipping gaps."""
        out: Dict[str, Dict[str, float]] = {}
        for name, values in self.series.items():
            present = [v for v in values if v is not None]
            if not present:
                continue
            mean = sum(present) / len(present)
            variance = sum((v - mean) ** 2 for v in present) / len(present)
            out[name] = {
                "count": float(len(present)),
                "min": min(present),
                "max": max(present),
                "mean": mean,
                "stdev": variance ** 0.5,
            }
        return out

    def gesture_counts(self) -> Dict[str, int]:
        """How many frames each status or gesture held."""
        counts: Dict[str, int] = {}
        for status in self.statuses:
            for part in str(status).split("|"):
                part = part.split(":")[-1].strip()
                if part:
                    counts[part] = counts.get(part, 0) + 1
        return counts

    # ------------------------------------------------------------------
    def figure(self):
        """Build the Plotly figure: time series, distributions, gesture mix."""
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        counts = self.gesture_counts()
        fig = make_subplots(
            rows=2,
            cols=2,
            specs=[[{"colspan": 2}, None], [{"type": "box"}, {"type": "bar"}]],
            subplot_titles=(
                "Radius over time",
                "Distribution per measurement",
                "Frames per gesture",
            ),
            vertical_spacing=0.16,
            row_heights=[0.55, 0.45],
        )

        for i, (name, values) in enumerate(self.series.items()):
            color = _SERIES_COLORS[i % len(_SERIES_COLORS)]
            fig.add_trace(
                go.Scatter(
                    x=self.timestamps, y=values, name=name, mode="lines",
                    line=dict(color=color, width=1.6),
                    hovertemplate="%{y:.1f}px at %{x:.2f}s<extra>" + name + "</extra>",
                ),
                row=1, col=1,
            )
            fig.add_trace(
                go.Box(
                    y=[v for v in values if v is not None], name=name,
                    marker_color=color, showlegend=False, boxpoints=False,
                ),
                row=2, col=1,
            )

        if counts:
            ordered = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
            fig.add_trace(
                go.Bar(
                    x=[k for k, _ in ordered], y=[v for _, v in ordered],
                    marker_color="#3c8cdc", showlegend=False,
                    hovertemplate="%{y} frames<extra>%{x}</extra>",
                ),
                row=2, col=2,
            )

        duration = max(self.timestamps) if self.timestamps else 0.0
        fig.update_layout(
            title=dict(
                # A literal middle dot, not the HTML entity: Plotly's title
                # renderer does not decode entities and shows "&middot;" as
                # text. Confirmed by rendering the output in a browser.
                text=(
                    f"{self.title}<br>"
                    f"<span style='font-size:13px;color:#8a929e'>"
                    f"{len(self.rows)} frames \u00b7 {duration:.1f}s \u00b7 "
                    f"{len(self.series)} measurements</span>"
                ),
                x=0.5, xanchor="center", y=0.975, yanchor="top",
            ),
            template="plotly_dark",
            paper_bgcolor="#191a1e",
            plot_bgcolor="#212328",
            font=dict(family="Menlo, Consolas, monospace", size=12),
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.005, x=0,
                        font=dict(size=11)),
            margin=dict(l=60, r=30, t=150, b=50),
            height=860,
        )
        fig.update_xaxes(title_text="seconds", row=1, col=1, gridcolor="#2e3138")
        fig.update_yaxes(title_text="pixels", row=1, col=1, gridcolor="#2e3138")
        fig.update_yaxes(title_text="pixels", row=2, col=1, gridcolor="#2e3138")
        fig.update_yaxes(title_text="frames", row=2, col=2, gridcolor="#2e3138")
        return fig

    def write(self, path: str = "dashboard.html", open_browser: bool = False) -> str:
        """Write the report and return its path.

        Plotly is inlined rather than loaded from a CDN, so the file works
        offline and makes no network request when opened.
        """
        fig = self.figure()
        fig.write_html(path, include_plotlyjs=True, full_html=True)

        if open_browser:  # pragma: no cover - needs a desktop
            import webbrowser
            webbrowser.open(f"file://{os.path.abspath(path)}")

        print(f"[Dashboard] {len(self.rows)} frames -> {path}")
        return path


def build_dashboard(
    csv_path: str = "radius_data.csv",
    out_path: str = "dashboard.html",
    title: str = "FingerRadiusAI session",
    open_browser: bool = False,
) -> str:
    """Read a CSV and write the HTML report. Returns the report path."""
    return DashboardBuilder(load_rows(csv_path), title=title).write(out_path, open_browser)
