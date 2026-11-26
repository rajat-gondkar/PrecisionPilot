"""Exhaustive search over all 2^N configurations: the ground-truth optimum (outline §7)."""

from __future__ import annotations

import pandas as pd

from precisionpilot.precision import all_configs, cfg_to_str
from precisionpilot.search.common import Evaluator, SearchResult, make_result


def exhaustive_search(ev: Evaluator, n_units: int = 10) -> tuple[SearchResult, pd.DataFrame]:
    rows = []
    for cfg in all_configs(n_units):
        acc = ev.accuracy(cfg)
        rows.append({"config": cfg_to_str(cfg), "n_int8": sum(cfg), "val_acc": acc,
                     "drop_pp": ev.drop(acc), "latency": ev.latency(cfg),
                     "feasible": ev.drop(acc) <= ev.budget_pp + 1e-9})
    table = pd.DataFrame(rows)
    best = ev.best_feasible_seen()
    return make_result("exhaustive", ev, best), table


def gap_to_optimum(latency: float, optimum_latency: float) -> float:
    """Latency gap in percent (0 = optimal)."""
    return 100.0 * (latency - optimum_latency) / optimum_latency


def pareto_front(table: pd.DataFrame, acc_col: str = "val_acc",
                 lat_col: str = "latency") -> pd.DataFrame:
    """Configurations not dominated in (higher accuracy, lower latency)."""
    df = table.sort_values([lat_col, acc_col], ascending=[True, False])
    keep, best_acc = [], -float("inf")
    for idx, row in df.iterrows():
        if row[acc_col] > best_acc + 1e-12:
            keep.append(idx)
            best_acc = row[acc_col]
    return table.loc[keep].sort_values(lat_col)
