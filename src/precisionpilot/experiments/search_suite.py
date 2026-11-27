"""Experiments 2-5: every search method x batch size x accuracy budget, for one GPU.

Accuracy comes from the precomputed exhaustive table (hardware-independent, validation
subset). Latency comes from one of two models of that GPU, and every method runs under each:

* ``additive``: sum of per-unit lookup-table times (the original model),
* ``context``: context-aware model fitted to full-engine measurements (boundary costs).

Rows carry a ``latency_model`` column. Each method is still charged one
evaluation per distinct configuration it asks about; the estimated search time is
``n_evals x seconds-per-evaluation`` measured on the real model (``reference.json``).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from precisionpilot.latency.lut import LatencyTable
from precisionpilot.precision import all_fp16, all_int8
from precisionpilot.search.common import Evaluator, TableAccuracy, make_result
from precisionpilot.search.exhaustive import exhaustive_search, gap_to_optimum
from precisionpilot.search.greedy import greedy_dynamic, greedy_search
from precisionpilot.search.random_search import random_search

RANDOM_CURVE_EVALS = (5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200, 300, 500, 1024)


def _row(res, gpu, bs, budget, fp16_lat, opt_lat, sec_per_eval, seed=None, model="additive",
         size_fn=None) -> dict:
    size = size_fn(res.cfg) if size_fn else np.nan
    fp16_size = size_fn(all_fp16(len(res.config))) if size_fn else np.nan
    return {"gpu": gpu, "batch_size": bs, "budget_pp": budget, "latency_model": model,
            "method": res.method,
            "seed": seed, "config": res.config, "n_int8": res.config.count("1"),
            "model_mb": size, "size_saving_pct": 100.0 * (1 - size / fp16_size)
            if size_fn else np.nan,
            "val_acc": res.val_acc, "drop_pp": res.drop_pp, "est_latency_ms": res.latency,
            "speedup_vs_fp16": fp16_lat / res.latency if res.latency > 0 else np.nan,
            "gap_to_opt_pct": gap_to_optimum(res.latency, opt_lat),
            "n_evals": res.n_evals, "est_search_s": res.n_evals * sec_per_eval,
            "feasible": res.feasible}


def _run_cell(acc_table, reference_acc, lat, gpu, bs, budget, random_seeds, epsilon,
              sec_per_eval, n_units, log, model="additive", size_fn=None
              ) -> tuple[list[dict], list[dict]]:
    """All methods for one (batch size, budget) cell."""

    def new_ev() -> Evaluator:
        return Evaluator(acc_table, lat, reference_acc, budget)

    fp16_lat = lat(all_fp16(n_units))
    ex, _ = exhaustive_search(new_ev(), n_units)
    opt = ex.latency
    rows: list[dict] = []

    def add(res, seed=None) -> None:
        rows.append(_row(res, gpu, bs, budget, fp16_lat, opt, sec_per_eval, seed, model,
                         size_fn))

    add(make_result("all-fp16", new_ev(), all_fp16(n_units)))
    add(make_result("all-int8", new_ev(), all_int8(n_units)))
    g = greedy_search(new_ev(), n_units, epsilon, "skip")
    add(g)
    add(greedy_search(new_ev(), n_units, epsilon, "stop"))
    add(greedy_search(new_ev(), n_units, epsilon, "skip", local_search=True))
    add(greedy_dynamic(new_ev(), n_units, epsilon))
    for seed in range(random_seeds):
        add(random_search(new_ev(), g.n_evals, n_units, seed=seed), seed)
    add(ex)
    log(f"[{gpu}/{model}] bs={bs:<3} budget={budget:<4} greedy {g.config} {g.latency:.3f} ms "
        f"({g.n_evals} evals, gap {gap_to_optimum(g.latency, opt):+.1f}%) | "
        f"optimum {ex.config} {opt:.3f} ms")

    # Experiment 2: random-search quality as a function of its evaluation budget
    curve: list[dict] = []
    for n in RANDOM_CURVE_EVALS:
        gaps = [gap_to_optimum(random_search(new_ev(), n, n_units, seed=1000 + s).latency, opt)
                for s in range(random_seeds)]
        curve.append({"gpu": gpu, "batch_size": bs, "budget_pp": budget, "latency_model": model,
                      "n_evals": n,
                      "gap_mean_pct": float(np.mean(gaps)), "gap_std_pct": float(np.std(gaps)),
                      "gap_median_pct": float(np.median(gaps)),
                      "p_optimal": float(np.mean(np.isclose(gaps, 0.0)))})
    return rows, curve


def run_search_suite(acc_table: TableAccuracy, reference_acc: float, lut: LatencyTable,
                     gpu: str, batch_sizes: Sequence[int], budgets: Sequence[float],
                     random_seeds: int = 20, epsilon: float = 0.01,
                     sec_per_eval: float = 0.0, n_units: int = 10,
                     log=print, supplementary: Sequence[float] = (), ctx_model=None
                     ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``supplementary`` budgets are tagged ``budget_kind="supplementary"`` (post-hoc additions);
    all others are ``"pre-registered"``. With ``ctx_model`` (a ContextLatencyTable), every cell is
    run twice: under the additive lookup table and under the context-aware model."""
    models = {"additive": lut}
    if ctx_model is not None:
        models["context"] = ctx_model
    size_fn = lut.model_mb if "params" in lut.df.columns else None
    rows, curve_rows = [], []
    for model_name, model in models.items():
        for bs in batch_sizes:
            lat = model.latency_fn(bs)
            for budget in budgets:
                r, c = _run_cell(acc_table, reference_acc, lat, gpu, bs, budget, random_seeds,
                                 epsilon, sec_per_eval, n_units, log, model_name, size_fn)
                kind = "supplementary" if budget in set(supplementary) else "pre-registered"
                for d in (*r, *c):
                    d["budget_kind"] = kind
                rows += r
                curve_rows += c
    return pd.DataFrame(rows), pd.DataFrame(curve_rows)


def chosen_configs(results: pd.DataFrame, batch_size: int) -> list[tuple[str, str]]:
    """(config, label) pairs worth verifying on real hardware for one batch size.

    Covers every latency model in ``results`` (labels carry a ``[model]`` suffix when there is
    more than one). Duplicate configurations are measured once.
    """
    df = results[results["batch_size"] == batch_size]
    if "latency_model" not in df.columns:
        df = df.assign(latency_model="additive")
    models = list(dict.fromkeys(df["latency_model"]))
    out: list[tuple[str, str]] = []
    for method in ("all-fp16", "all-int8"):
        sub = df[df["method"] == method]
        if len(sub):
            out.append((sub["config"].iloc[0], method))
    for model in models:
        mdf = df[df["latency_model"] == model]
        tag = f"[{model}]" if len(models) > 1 else ""
        for budget, grp in mdf.groupby("budget_pp"):
            for method in ("greedy-skip", "greedy-dyn", "greedy-skip+ls", "exhaustive"):
                sub = grp[grp["method"] == method]
                if len(sub):
                    out.append((sub["config"].iloc[0], f"{method}@{budget:g}{tag}"))
            rnd = grp[grp["method"] == "random"].sort_values("est_latency_ms")
            if len(rnd):  # the median-latency random outcome is the typical one
                out.append((rnd["config"].iloc[len(rnd) // 2], f"random-median@{budget:g}{tag}"))
    seen, uniq = set(), []
    for c, label in out:
        if c not in seen:
            uniq.append((c, label))
            seen.add(c)
    return uniq
