"""Shared visual theme for the analysis dashboard.

Color tokens, figure-size constants, and the single place matplotlib styling is configured, so
plots.py and Altair charts both read from one source instead of hardcoding their own colors/sizes.
Leaf module: no imports from other viz/ modules.
"""
from pathlib import Path

import matplotlib as mpl
import matplotlib.font_manager as font_manager
import streamlit as st

# Okabe-Ito palette -- colorblind-safe.
COLOR_BEST = "#0072B2"
COLOR_AVG = "#56B4E9"
COLOR_TARGET = "#000000"
COLOR_SUCCESS = "#009E73"
COLOR_FAILURE = "#D55E00"
COLOR_STRUCTURAL_CHANGE = "#CC79A7"

CATEGORICAL_CYCLE = [
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#009E73",  # green
    "#D55E00",  # vermillion
    "#CC79A7",  # pink
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
    "#000000",  # black
]

SEQUENTIAL_CMAP = "viridis"
DIVERGING_CMAP = "PuOr"  # colorblind-safe alternative to RdYlGn

FIG_WIDE = (10, 3)
FIG_HALF = (8, 3)
FIG_SQUARE = (7, 5)
FIG_GRID_CELL = (4, 3)

SIDEBAR_LOGO_HEIGHT_REM = 10.5

_FONTS_DIR = Path(__file__).resolve().parents[2] / "resources" / "fonts"
_FONT_FAMILY_NAME = "JetBrains Mono"
_FALLBACK_FONT_FAMILY = "DejaVu Sans"


def _register_fonts() -> str:
    """Register the repo's JetBrains Mono files with matplotlib. Falls back to a bundled
    sans-serif font (never raises) if the files are missing, e.g. moved or renamed."""
    try:
        ttf_files = sorted(_FONTS_DIR.glob("JetBrainsMono-*.ttf"))
    except OSError:
        return _FALLBACK_FONT_FAMILY
    if not ttf_files:
        return _FALLBACK_FONT_FAMILY
    for ttf in ttf_files:
        try:
            font_manager.fontManager.addfont(str(ttf))
        except OSError:
            return _FALLBACK_FONT_FAMILY
    return _FONT_FAMILY_NAME


def apply_plot_style() -> None:
    """Set the matplotlib rcParams every plot in plots.py inherits: font, DPI, grid, spines,
    legend, and the categorical color cycle. Called once at plots.py's import time."""
    family = _register_fonts()
    mpl.rcParams["font.family"] = family
    mpl.rcParams["figure.dpi"] = 150
    mpl.rcParams["axes.grid"] = True
    mpl.rcParams["grid.alpha"] = 0.3
    mpl.rcParams["axes.spines.top"] = False
    mpl.rcParams["axes.spines.right"] = False
    mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color=CATEGORICAL_CYCLE)
    mpl.rcParams["legend.frameon"] = False
    mpl.rcParams["axes.titlesize"] = "medium"
    mpl.rcParams["axes.labelsize"] = "small"


def theme_type() -> str:
    """Return "light" or "dark". Defaults to "light" when Streamlit's theme context isn't
    available yet (first script run, mid theme-switch, or under AppTest)."""
    t = st.context.theme.type
    return t if t in ("light", "dark") else "light"
