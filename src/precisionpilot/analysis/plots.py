"""Report figures (outline §10). Every function reads result CSVs and writes a PNG."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from precisionpilot.analysis.style import (
    METHOD_COLORS,
    METHOD_LABELS,
    NEUTRAL,
    NEUTRAL_LIGHT,
    SEQ_DARK,
    SEQ_LIGHT,
    SURFACE,
    TEXT,
    TEXT_2,
    apply_style,
)
from precisionpilot.model import UNIT_NAMES


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


# 1 ------------------------------------------------------------------ sensitivity
def plot_sensitivity(sens: pd.DataFrame, path: Path, budget_pp: float | None = None) -> Path:
    apply_style()
    fig, ax = plt.subplots(figsize=(8, 3.6))
    drops = sens["drop_vs_fp16"].to_numpy()
    x = np.arange(len(sens))
    bars = ax.bar(x, drops, width=0.62, color="#257de0", edgecolor=SURFACE, linewidth=2)
    for b, d in zip(bars, drops):
        ax.annotate(f"{d:+.2f}", (b.get_x() + b.get_width() / 2, max(d, 0)),
                    xytext=(0, 3), textcoords="offset points", ha="center",
                    fontsize=8, color=TEXT_2)
    if budget_pp is not None:
        ax.axhline(budget_pp, color=TEXT_2, linestyle="--", linewidth=1)
        ax.annotate(f"budget {budget_pp:g} pp", (len(sens) - 0.5, budget_pp),
                    xytext=(0, 3), textcoords="offset points", ha="right", fontsize=8,
                    color=TEXT_2)
    ax.axhline(0, color=TEXT_2, linewidth=0.8)
    ax.set_xticks(x, sens["name"], rotation=30, ha="right")
    ax.set_ylabel("Top-1 drop vs all-FP16 (pp)")
    ax.set_title("Per-unit INT8 sensitivity (one unit INT8, rest FP16)")
    ax.grid(axis="x", visible=False)
    return _save(fig, path)


# 2 ------------------------------------------------------------------ cost vs quality
def plot_cost_vs_quality(results: pd.DataFrame, curve: pd.DataFrame, path: Path,
                         batch_size: int, budget_pp: float) -> Path:
    apply_style()
    r = results[(results.batch_size == batch_size) & (results.budget_pp == budget_pp)]
    c = curve[(curve.batch_size == batch_size) & (curve.budget_pp == budget_pp)]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(c.n_evals, c.gap_mean_pct, color=METHOD_COLORS["random"], marker="o",
            markersize=4, label="Random search (mean over seeds)")
    ax.fill_between(c.n_evals, np.maximum(c.gap_mean_pct - c.gap_std_pct, 0),
                    c.gap_mean_pct + c.gap_std_pct, color=METHOD_COLORS["random"], alpha=0.12,
                    linewidth=0)
    # greedy variants often coincide: hollow outer shapes + filled inner dot stay readable
    for method, marker, size, filled in (("greedy-skip+ls", "D", 190, False),
                                         ("greedy-stop", "s", 120, False),
                                         ("greedy-skip", "o", 55, True)):
        m = r[r.method == method]
        if len(m):
            col = METHOD_COLORS[method]
            ax.scatter(m.n_evals, m.gap_to_opt_pct, s=size, marker=marker, zorder=5,
                       facecolor=col if filled else "none", edgecolor=col,
                       linewidth=1.8, label=METHOD_LABELS[method])
    ex = r[r.method == "exhaustive"]
    if len(ex):
        ax.scatter(ex.n_evals, ex.gap_to_opt_pct, s=220, marker="*", zorder=5,
                   color=METHOD_COLORS["exhaustive"], edgecolor=SURFACE, linewidth=1.5,
                   label=METHOD_LABELS["exhaustive"])
    ax.set_xscale("log")
    ax.set_xlabel("Accuracy evaluations (log scale)")
    ax.set_ylabel("Latency gap to optimum (%)")
    ax.set_title(f"Search cost vs result quality (batch {batch_size}, budget {budget_pp:g} pp)")
    ax.set_ylim(bottom=min(-1, ax.get_ylim()[0]))
    ax.legend(loc="upper right", fontsize=8)
    return _save(fig, path)


# 3 ------------------------------------------------------------------ accuracy vs latency
def plot_scatter(acc_table: pd.DataFrame, lut_latency, reference_acc: float,
                 results: pd.DataFrame, path: Path, batch_size: int,
                 budget_pp: float) -> Path:
    apply_style()
    df = acc_table.copy()
    df["latency"] = [lut_latency(tuple(int(ch) for ch in c)) for c in df["config"]]
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.scatter(df.latency, df.val_acc, s=9, color=NEUTRAL_LIGHT, linewidth=0,
               label="All 1,024 configurations")
    ax.axhline(reference_acc - budget_pp, color=TEXT_2, linestyle="--", linewidth=1)
    ax.annotate(f"budget: FP32 − {budget_pp:g} pp", (df.latency.max(), reference_acc - budget_pp),
                xytext=(0, 4), textcoords="offset points", ha="right", fontsize=8, color=TEXT_2)
    r = results[(results.batch_size == batch_size) & (results.budget_pp == budget_pp)]
    for method in ("all-fp16", "all-int8"):
        m = r[r.method == method]
        if len(m):
            ax.scatter(m.est_latency_ms, m.val_acc, s=60, color=NEUTRAL, edgecolor=SURFACE,
                       linewidth=2, zorder=4)
            ax.annotate(METHOD_LABELS[method], (m.est_latency_ms.iloc[0], m.val_acc.iloc[0]),
                        xytext=(6, -10), textcoords="offset points", fontsize=8, color=TEXT)
    rnd = r[r.method == "random"]
    if len(rnd):
        ax.scatter(rnd.est_latency_ms, rnd.val_acc, s=40, color=METHOD_COLORS["random"],
                   edgecolor=SURFACE, linewidth=2, zorder=5, label=METHOD_LABELS["random"])
    for method, marker, size in (("exhaustive", "*", 220), ("greedy-skip", "o", 80)):
        m = r[r.method == method]
        if len(m):
            ax.scatter(m.est_latency_ms, m.val_acc, s=size, marker=marker, zorder=6,
                       color=METHOD_COLORS[method], edgecolor=SURFACE, linewidth=2,
                       label=METHOD_LABELS[method])
    ax.set_xlabel("Estimated latency (ms, lookup table)")
    ax.set_ylabel("Validation top-1 (%)")
    lo = max(df.val_acc.min(), reference_acc - max(8, 4 * budget_pp))
    ax.set_ylim(lo, reference_acc + 0.6)
    ax.set_title(f"Accuracy vs latency (batch {batch_size})")
    ax.legend(loc="lower right", fontsize=8)
    return _save(fig, path)


# 4 ------------------------------------------------------------------ heatmap
def plot_config_heatmap(results: pd.DataFrame, path: Path, method: str = "greedy-skip",
                        title: str | None = None) -> Path:
    apply_style()
    r = results[results.method == method].sort_values(["batch_size", "budget_pp"])
    if "gpu" in r and r["gpu"].nunique() > 1:
        labels = [f"{g} · bs {b} · {p:g} pp" for g, b, p in zip(r.gpu, r.batch_size, r.budget_pp)]
    else:
        labels = [f"bs {b} · {p:g} pp" for b, p in zip(r.batch_size, r.budget_pp)]
    grid = np.array([[int(ch) for ch in c] for c in r.config]) if len(r) else np.zeros((0, 10))
    fig, ax = plt.subplots(figsize=(8.2, 0.42 * max(len(r), 1) + 1.4))
    from matplotlib.colors import ListedColormap

    ax.imshow(grid, cmap=ListedColormap([SEQ_LIGHT, SEQ_DARK]), vmin=0, vmax=1, aspect="auto")
    for (i, j), v in np.ndenumerate(grid):
        ax.text(j, i, "INT8" if v else "FP16", ha="center", va="center", fontsize=7,
                color=SURFACE if v else TEXT)
    ax.set_xticks(range(len(UNIT_NAMES)), UNIT_NAMES, rotation=30, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xticks(np.arange(-0.5, len(UNIT_NAMES)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(labels)), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.grid(which="major", visible=False)
    ax.tick_params(which="minor", length=0)
    ax.set_title(title or f"Selected precision per unit — {METHOD_LABELS.get(method, method)}")
    ax.legend(handles=[Patch(color=SEQ_LIGHT, label="FP16"), Patch(color=SEQ_DARK, label="INT8")],
              loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)
    return _save(fig, path)


# 5 ------------------------------------------------------------------ LUT verification
def plot_lut_error(verify: pd.DataFrame, path: Path) -> Path:
    apply_style()
    fig, ax = plt.subplots(figsize=(6.0, 4.6))
    groups = list(verify.groupby("gpu")) if "gpu" in verify else [("", verify)]
    colors = ["#257de0", "#f47230", "#00b894"]
    for (gpu, g), col in zip(groups, colors):
        ax.scatter(g.measured_median_ms, g.estimated_ms, s=40, color=col, edgecolor=SURFACE,
                   linewidth=2, label=f"{gpu} (median |err| {g.error_pct.abs().median():.1f}%)")
    from matplotlib.ticker import FuncFormatter, LogLocator

    vals = pd.concat([verify.measured_median_ms, verify.estimated_ms])
    lim = [vals.min() / 1.25, vals.max() * 1.25]
    ax.plot(lim, lim, color=TEXT_2, linewidth=1, linestyle="--")
    ax.annotate("estimate = measured", (lim[1], lim[1]), xytext=(-4, -12),
                textcoords="offset points", ha="right", fontsize=8, color=TEXT_2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    fmt = FuncFormatter(lambda v, _: f"{v:g}")
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
        axis.set_major_formatter(fmt)
        axis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
    ax.set_xlabel("Measured full-model latency (ms)")
    ax.set_ylabel("Lookup-table estimate (ms)")
    ax.set_title("Latency estimate vs real measurement")
    ax.legend(loc="upper left", fontsize=8)
    return _save(fig, path)


# 6 ------------------------------------------------------------------ batch / GPU speedups
def plot_speedups(results: pd.DataFrame, path: Path, budget_pp: float) -> Path:
    apply_style()
    r = results[(results.budget_pp == budget_pp) &
                results.method.isin(["greedy-skip", "exhaustive", "all-int8"])]
    gpus = sorted(r.gpu.unique())
    bss = sorted(r.batch_size.unique())
    methods = ["greedy-skip", "exhaustive", "all-int8"]
    colors = {"greedy-skip": METHOD_COLORS["greedy-skip"],
              "exhaustive": METHOD_COLORS["exhaustive"], "all-int8": NEUTRAL}
    fig, axes = plt.subplots(1, len(gpus), figsize=(4.2 * len(gpus), 3.6), sharey=True,
                             squeeze=False)
    w = 0.26
    for ax, gpu in zip(axes[0], gpus):
        sub = r[r.gpu == gpu]
        for k, m in enumerate(methods):
            vals = [sub[(sub.batch_size == b) & (sub.method == m)].speedup_vs_fp16.mean()
                    for b in bss]
            ax.bar(np.arange(len(bss)) + (k - 1) * w, vals, width=w - 0.02, color=colors[m],
                   edgecolor=SURFACE, linewidth=2,
                   label=METHOD_LABELS[m] + (" (may break budget)" if m == "all-int8" else ""))
        ax.axhline(1.0, color=TEXT_2, linewidth=0.8)
        ax.set_xticks(range(len(bss)), [f"bs {b}" for b in bss])
        ax.set_title(gpu)
        ax.grid(axis="x", visible=False)
    axes[0][0].set_ylabel("Speedup vs all-FP16 (×)")
    axes[0][-1].legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)
    fig.suptitle(f"Estimated speedup by batch size and GPU (budget {budget_pp:g} pp)",
                 x=0.01, ha="left", fontsize=12, fontweight="bold", color=TEXT)
    fig.tight_layout()
    return _save(fig, path)
