"""Shared matplotlib styling so every chart in the repo reads as one system."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Fixed categorical order (never cycled): blue, orange, aqua, yellow, magenta, green, violet, red
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
STATUS = dict(good="#008300", warning="#eda100", serious="#eb6834", critical="#e34948")
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT2 = "#52514e"
GRID = "#e6e5e1"


def style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": TEXT2, "axes.titlecolor": TEXT,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.color": TEXT2, "ytick.color": TEXT2, "text.color": TEXT,
        "lines.linewidth": 2, "lines.markersize": 7,
        "legend.frameon": False, "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
        "axes.prop_cycle": matplotlib.cycler(color=SERIES),
    })
