"""Tables that join search results with real full-engine measurements.

* :func:`memory_table`: memory of all-FP16 vs the selected mixed configuration vs all-INT8.
* :func:`latency_model_table`: *real* latency of the configuration each (latency model, method,
  budget) selected, so the additive and context-aware models can be compared on actual hardware.
* :func:`lut_error_summary`: estimate error of both latency models against real engines.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from precisionpilot.precision import cfg_to_str

FP16_CFG = cfg_to_str((0,) * 10)
INT8_CFG = cfg_to_str((1,) * 10)


def primary_model(res: pd.DataFrame) -> str:
    """The latency model the main figures use: ``context`` if present, else ``additive``."""
    if "latency_model" in res.columns and "context" in set(res["latency_model"]):
        return "context"
    return "additive"


def select_model(df: pd.DataFrame, model: str) -> pd.DataFrame:
    return df[df["latency_model"] == model] if "latency_model" in df.columns else df


def selected_method(res: pd.DataFrame) -> str:
    return "greedy-dyn" if "greedy-dyn" in set(res["method"]) else "greedy-skip"


def _lookup(verify: pd.DataFrame, gpu: str, bs: int, config: str) -> pd.Series | None:
    row = verify[(verify.gpu == gpu) & (verify.batch_size == bs) & (verify.config == config)]
    return row.iloc[0] if len(row) else None


def memory_table(res: pd.DataFrame, verify: pd.DataFrame, budget: float,
                 model: str | None = None) -> pd.DataFrame:
    """Rows: all-FP16, the configuration selected at ``budget``, all-INT8 (per GPU, batch size)."""
    model = model or primary_model(res)
    r = select_model(res, model)
    method = selected_method(r)
    rows = []
    for (gpu, bs), grp in r[np.isclose(r.budget_pp, budget)].groupby(["gpu", "batch_size"]):
        picks = {"all-FP16": grp[grp.method == "all-fp16"], "selected": grp[grp.method == method],
                 "all-INT8": grp[grp.method == "all-int8"]}
        base = None
        for name, sub in picks.items():
            if not len(sub):
                continue
            cfg = sub["config"].iloc[0]
            v = _lookup(verify, gpu, int(bs), cfg)
            row = {"gpu": gpu, "batch_size": int(bs), "name": name, "config": cfg,
                   "n_int8": cfg.count("1"), "model_mb": float(sub["model_mb"].iloc[0]),
                   "engine_mb": np.nan, "activation_memory_mb": np.nan,
                   "gpu_footprint_mb": np.nan, "measured_ms": np.nan}
            if v is not None:
                for k, col in (("engine_mb", "engine_mb"),
                               ("activation_memory_mb", "activation_memory_mb"),
                               ("gpu_footprint_mb", "gpu_footprint_mb"),
                               ("measured_ms", "measured_median_ms")):
                    if col in v.index and pd.notna(v[col]):
                        row[k] = float(v[col])
            rows.append(row)
            if name == "all-FP16":
                base = row
        if base:
            for row in rows[-3:]:
                if row["gpu"] == gpu and row["batch_size"] == int(bs):
                    for k in ("model_mb", "engine_mb", "activation_memory_mb", "gpu_footprint_mb"):
                        b = base[k]
                        row[f"{k}_saving_pct"] = (100.0 * (1 - row[k] / b)
                                                  if pd.notna(row[k]) and b else np.nan)
    return pd.DataFrame(rows)


def latency_model_table(res: pd.DataFrame, verify: pd.DataFrame) -> pd.DataFrame:
    """Real measured latency of each (model, method, budget, batch) selection (where verified)."""
    if "latency_model" not in res.columns:
        res = res.assign(latency_model="additive")
    keep = res[res.method.isin(["all-fp16", "all-int8", "greedy-skip", "greedy-dyn",
                                "greedy-skip+ls", "exhaustive"])]
    rows = []
    for r in keep.itertuples():
        v = _lookup(verify, r.gpu, int(r.batch_size), r.config)
        fp16 = _lookup(verify, r.gpu, int(r.batch_size), FP16_CFG)
        ms = float(v["measured_median_ms"]) if v is not None else np.nan
        base = float(fp16["measured_median_ms"]) if fp16 is not None else np.nan
        rows.append({"gpu": r.gpu, "batch_size": int(r.batch_size), "budget_pp": r.budget_pp,
                     "latency_model": r.latency_model, "method": r.method, "config": r.config,
                     "n_int8": r.n_int8, "drop_pp": r.drop_pp, "feasible": r.feasible,
                     "n_evals": r.n_evals, "est_latency_ms": r.est_latency_ms,
                     "measured_ms": ms, "real_speedup_vs_fp16": base / ms if ms else np.nan})
    return pd.DataFrame(rows)


def lut_error_summary(verify: pd.DataFrame) -> pd.DataFrame:
    """|error| of the additive table and (held-out) of the context-aware model, per batch size."""
    rows = []
    for (gpu, bs), g in verify.groupby(["gpu", "batch_size"]):
        row = {"gpu": gpu, "batch_size": int(bs), "n": len(g),
               "additive_mean_abs_pct": g.error_pct.abs().mean(),
               "additive_max_abs_pct": g.error_pct.abs().max()}
        if "ctx_error_pct" in g.columns:
            held = g[~g.ctx_trained.astype(bool)]
            row.update({"context_n_heldout": len(held),
                        "context_mean_abs_pct": held.ctx_error_pct.abs().mean(),
                        "context_max_abs_pct": held.ctx_error_pct.abs().max()})
        rows.append(row)
    return pd.DataFrame(rows)
