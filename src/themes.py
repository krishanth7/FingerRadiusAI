"""
themes.py - Switchable colour themes for the dashboard.

``src.utils.COLORS`` is imported by name all over the codebase, so a theme
switch cannot rebind it -- every module already holds a reference to the
original dict. Instead a theme is applied by mutating that dict in place,
which every holder of the reference sees immediately.

Colours are BGR, because OpenCV is BGR and converting at every draw call
would be a lot of work to gain nothing.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from src.utils import COLORS

__all__ = ["THEMES", "THEME_NAMES", "apply_theme", "current_theme", "next_theme"]

BGR = Tuple[int, int, int]


def _rgb(r: int, g: int, b: int) -> BGR:
    """Write colours in RGB (how everyone reads hex) and store them as BGR."""
    return (b, g, r)


#: Every theme supplies the full key set. A partial theme would leave stale
#: colours from whichever theme happened to be applied before it.
THEMES: Dict[str, Dict[str, BGR]] = {
    # The original look: dark slate with a warm amber accent.
    "corporate": {
        "thumb": (60, 180, 235), "index": (180, 200, 60), "middle": (220, 140, 60),
        "ring": (160, 80, 200), "pinky": (80, 120, 220),
        "bg_primary": (25, 25, 30), "bg_secondary": (35, 35, 42),
        "bg_tertiary": (45, 45, 55),
        "border": (60, 60, 72), "border_light": (75, 75, 88), "divider": (50, 50, 60),
        "text_primary": (230, 232, 235), "text_secondary": (160, 165, 172),
        "text_tertiary": (100, 105, 112), "text_label": (130, 135, 142),
        "accent": (200, 170, 60), "accent_bg": (50, 45, 35),
        "success": (120, 200, 80), "warning": (50, 140, 240),
        "danger": (70, 80, 210), "info": (200, 160, 60),
        "bone": (70, 70, 80), "bone_light": (90, 90, 105),
        "joint": (110, 110, 125), "wrist": (140, 145, 155),
    },
    # Saturated magenta and cyan on near-black.
    "cyberpunk": {
        "thumb": _rgb(255, 46, 136), "index": _rgb(0, 245, 212),
        "middle": _rgb(255, 214, 0), "ring": _rgb(181, 55, 242),
        "pinky": _rgb(0, 168, 255),
        "bg_primary": _rgb(10, 4, 20), "bg_secondary": _rgb(18, 8, 32),
        "bg_tertiary": _rgb(28, 14, 48),
        "border": _rgb(70, 30, 110), "border_light": _rgb(120, 50, 180),
        "divider": _rgb(48, 20, 78),
        "text_primary": _rgb(240, 230, 255), "text_secondary": _rgb(180, 150, 220),
        "text_tertiary": _rgb(120, 90, 160), "text_label": _rgb(150, 120, 190),
        "accent": _rgb(255, 46, 136), "accent_bg": _rgb(48, 10, 32),
        "success": _rgb(0, 245, 212), "warning": _rgb(255, 214, 0),
        "danger": _rgb(255, 46, 90), "info": _rgb(181, 55, 242),
        "bone": _rgb(80, 40, 120), "bone_light": _rgb(120, 60, 170),
        "joint": _rgb(150, 90, 200), "wrist": _rgb(0, 245, 212),
    },
    # Light, low-chroma, thin. For screenshots and print.
    "minimal": {
        "thumb": _rgb(66, 103, 178), "index": _rgb(40, 116, 92),
        "middle": _rgb(176, 122, 40), "ring": _rgb(126, 80, 154),
        "pinky": _rgb(180, 74, 74),
        "bg_primary": _rgb(248, 249, 250), "bg_secondary": _rgb(255, 255, 255),
        "bg_tertiary": _rgb(238, 241, 245),
        "border": _rgb(210, 214, 220), "border_light": _rgb(226, 230, 236),
        "divider": _rgb(232, 236, 241),
        "text_primary": _rgb(24, 30, 40), "text_secondary": _rgb(85, 95, 110),
        "text_tertiary": _rgb(140, 150, 165), "text_label": _rgb(110, 120, 135),
        "accent": _rgb(26, 86, 196), "accent_bg": _rgb(228, 237, 253),
        "success": _rgb(22, 122, 90), "warning": _rgb(176, 108, 6),
        "danger": _rgb(198, 40, 40), "info": _rgb(26, 86, 196),
        "bone": _rgb(186, 192, 200), "bone_light": _rgb(206, 212, 220),
        "joint": _rgb(150, 158, 170), "wrist": _rgb(90, 100, 115),
    },
    # Amber phosphor on dark brown, like a CRT terminal.
    "retro": {
        "thumb": _rgb(255, 176, 0), "index": _rgb(255, 214, 102),
        "middle": _rgb(230, 138, 0), "ring": _rgb(255, 235, 170),
        "pinky": _rgb(200, 110, 20),
        "bg_primary": _rgb(18, 12, 4), "bg_secondary": _rgb(28, 20, 8),
        "bg_tertiary": _rgb(40, 29, 12),
        "border": _rgb(92, 66, 20), "border_light": _rgb(128, 92, 28),
        "divider": _rgb(64, 46, 14),
        "text_primary": _rgb(255, 208, 112), "text_secondary": _rgb(206, 158, 72),
        "text_tertiary": _rgb(140, 104, 44), "text_label": _rgb(172, 128, 56),
        "accent": _rgb(255, 176, 0), "accent_bg": _rgb(56, 38, 10),
        "success": _rgb(154, 214, 60), "warning": _rgb(255, 176, 0),
        "danger": _rgb(232, 92, 40), "info": _rgb(255, 214, 102),
        "bone": _rgb(96, 70, 24), "bone_light": _rgb(130, 96, 34),
        "joint": _rgb(164, 122, 44), "wrist": _rgb(255, 208, 112),
    },
}

THEME_NAMES: List[str] = list(THEMES)

_current = "corporate"

#: The keys every theme must define, taken from the shipped palette.
_REQUIRED_KEYS = set(THEMES["corporate"])


def apply_theme(name: str) -> str:
    """Switch the palette in place and return the applied theme name.

    Raises:
        KeyError: If the theme is unknown, listing the ones that exist.

    Example:
        >>> from src.themes import apply_theme, current_theme
        >>> apply_theme("retro")
        'retro'
        >>> current_theme()
        'retro'
        >>> _ = apply_theme("corporate")
    """
    global _current
    if name not in THEMES:
        raise KeyError(
            f"Unknown theme {name!r}. Available: {', '.join(THEME_NAMES)}."
        )
    COLORS.update(THEMES[name])
    _current = name
    return name


def current_theme() -> str:
    """Return the name of the theme currently applied."""
    return _current


def next_theme() -> str:
    """Apply the next theme in the list and return its name.

    This is what the keyboard shortcut in the dashboard calls.
    """
    index = (THEME_NAMES.index(_current) + 1) % len(THEME_NAMES)
    return apply_theme(THEME_NAMES[index])


def validate() -> None:
    """Check every theme defines every colour key.

    A missing key would silently leave the previous theme's colour in place,
    producing a palette that is neither one theme nor the other.

    Raises:
        ValueError: Listing each theme's missing and unexpected keys.
    """
    problems = []
    for name, palette in THEMES.items():
        missing = _REQUIRED_KEYS - set(palette)
        extra = set(palette) - _REQUIRED_KEYS
        if missing or extra:
            problems.append(f"{name}: missing={sorted(missing)} unexpected={sorted(extra)}")
    if problems:
        raise ValueError("Theme key mismatch -- " + "; ".join(problems))
