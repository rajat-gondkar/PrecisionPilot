"""Per-unit sensitivity scan (outline §6.4 step 1; Experiment 1).

Quantize one unit at a time to INT8 while the rest stay FP16, and measure the validation
accuracy drop. Cost: 1 (all-FP16) + N evaluations.
"""

from __future__ import annotations

import pandas as pd

from precisionpilot.model import UNIT_NAMES
from precisionpilot.precision import FP16, INT8, all_fp16, cfg_to_str, one_hot
from precisionpilot.search.common import Evaluator


def sensitivity_scan(ev: Evaluator, n_units: int = 10, names=UNIT_NAMES) -> pd.DataFrame:
    base_cfg = all_fp16(n_units)
    base_acc = ev.accuracy(base_cfg)
    base_lat = ev.latency(base_cfg)
    rows = []
    for i in range(n_units):
        cfg = one_hot(i, n_units)
        acc = ev.accuracy(cfg)
        lat = ev.latency(cfg)
        rows.append({
            "unit": i, "name": names[i], "config": cfg_to_str(cfg), "acc": acc,
            "drop_vs_fp16": base_acc - acc, "drop_vs_ref": ev.reference_acc - acc,
            "latency_fp16_model": base_lat, "latency": lat,
            "latency_saved": base_lat - lat,
        })
    df = pd.DataFrame(rows)
    df.attrs["fp16_acc"] = base_acc
    return df


def score_units(sens: pd.DataFrame, epsilon: float = 0.01) -> pd.DataFrame:
    """Score = latency saved per accuracy point lost (outline §6.4 step 2).

    The accuracy drop is floored at ``epsilon`` (pp) so free or accuracy-*improving* units
    are ranked by their speedup instead of dividing by ~0. Units whose INT8 version is not
    faster (latency_saved <= 0) get a score of -inf and are never flipped.
    """
    df = sens.copy()
    drop = df["drop_vs_fp16"].clip(lower=epsilon)
    df["score"] = df["latency_saved"] / drop
    df.loc[df["latency_saved"] <= 0, "score"] = float("-inf")
    # tie-break: larger saving first, then lower unit index (deterministic)
    df = df.sort_values(["score", "latency_saved", "unit"], ascending=[False, False, True])
    df["rank"] = range(1, len(df) + 1)
    return df.reset_index(drop=True)


__all__ = ["sensitivity_scan", "score_units", "FP16", "INT8"]
