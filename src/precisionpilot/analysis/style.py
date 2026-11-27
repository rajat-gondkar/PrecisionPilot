"""Shared figure style (static report figures, light surface).

Palette: validated reference categorical order; slots 1-3 only for co-plotted method
series (the all-pairs-safe subset). Baselines are neutral gray + direct labels.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE = "#ffffff"
TEXT = "#191919"
TEXT_2 = "#666666"
GRID = "#e6e6e6"
NEUTRAL = "#9c9c9c"
NEUTRAL_LIGHT = "#d9d9d9"

METHOD_COLORS = {
    "greedy-skip": "#257de0",     # slot 1 blue   - proposed
    "greedy-skip+ls": "#257de0",
    "greedy-dyn": "#257de0",
    "greedy-stop": "#257de0",
    "random": "#f47230",          # slot 2 orange
    "exhaustive": "#00b894",      # slot 3 aqua
}
METHOD_LABELS = {
    "greedy-skip": "Sensitivity-guided greedy (ours)",
    "greedy-stop": "Greedy, stop at first violation",
    "greedy-skip+ls": "Greedy + local search",
    "greedy-dyn": "Context-aware greedy (ours)",
    "random": "Random search (same evals)",
    "exhaustive": "Exhaustive (optimum)",
    "all-fp16": "All-FP16",
    "all-int8": "All-INT8",
}
SEQ_LIGHT = "#cce3fd"   # blue 100  -> FP16 cells
SEQ_DARK = "#1b64be"    # blue 550  -> INT8 cells
SEQ_MID = "#6aabf1"


def apply_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "sans-serif", "font.size": 10, "axes.titlesize": 12,
        "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.labelcolor": TEXT_2, "axes.edgecolor": GRID, "axes.linewidth": 0.8,
        "xtick.color": TEXT_2, "ytick.color": TEXT_2, "text.color": TEXT,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.axisbelow": True, "legend.frameon": False, "lines.linewidth": 2.0,
        "savefig.dpi": 200, "savefig.bbox": "tight",
    })
