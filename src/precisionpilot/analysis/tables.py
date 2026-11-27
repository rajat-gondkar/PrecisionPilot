"""Summary tables (CSV + Markdown) for the report."""

from __future__ import annotations

import pandas as pd

from precisionpilot.analysis.style import METHOD_LABELS

SUMMARY_METHODS = ["all-fp16", "all-int8", "greedy-stop", "greedy-skip", "greedy-skip+ls",
                   "greedy-dyn", "random", "exhaustive"]


def summary_table(results: pd.DataFrame) -> pd.DataFrame:
    """One row per (gpu, batch, budget, method); random aggregated over seeds."""
    r = results[results.method.isin(SUMMARY_METHODS)]
    keys = ["gpu", "batch_size", "budget_pp"]
    if "latency_model" in r.columns:
        keys.append("latency_model")
    if "budget_kind" in r.columns:
        keys.append("budget_kind")
    agg = (r.groupby([*keys, "method"], sort=False)
             .agg(val_acc=("val_acc", "mean"), drop_pp=("drop_pp", "mean"),
                  est_latency_ms=("est_latency_ms", "mean"),
                  speedup=("speedup_vs_fp16", "mean"), gap_pct=("gap_to_opt_pct", "mean"),
                  gap_std=("gap_to_opt_pct", "std"), n_evals=("n_evals", "mean"),
                  feasible_rate=("feasible", "mean"), config=("config", "first"),
                  model_mb=("model_mb", "mean"))
             .reset_index())
    agg["method_order"] = agg.method.map({m: i for i, m in enumerate(SUMMARY_METHODS)})
    agg = agg.sort_values([*keys, "method_order"])
    agg.loc[agg.method == "random", "config"] = "(varies)"
    return agg.drop(columns="method_order").reset_index(drop=True)


def to_markdown(df: pd.DataFrame, floatfmt: str = ".2f") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                cells.append("" if pd.isna(v) else format(v, floatfmt))
            else:
                cells.append(str(METHOD_LABELS.get(v, v)) if c == "method" else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"
