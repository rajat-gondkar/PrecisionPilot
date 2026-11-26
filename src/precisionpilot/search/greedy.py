"""Sensitivity-guided greedy search (outline §6.4) — the proposed method.

1. Sensitivity scan: 1 + N evaluations (all-FP16 and each unit alone in INT8).
2. Score units by *latency saved per accuracy point lost* on the target GPU / batch size.
3. Start from all-FP16 and flip units to INT8 in score order. When a flip would exceed the
   budget, back it off and either
     * ``"skip"`` (default) — continue with the next unit, or
     * ``"stop"`` — stop immediately (the cheapest variant).
   Units whose INT8 version is not faster are never flipped.
4. (Optional) pairwise-swap local search refinement, see :mod:`local_search`.

The final test-set check is done by the caller (experiments), never inside the search.
"""

from __future__ import annotations

from precisionpilot.precision import FP16, INT8, all_fp16
from precisionpilot.search.common import Evaluator, SearchResult, make_result
from precisionpilot.search.sensitivity import score_units, sensitivity_scan


def greedy_search(ev: Evaluator, n_units: int = 10, epsilon: float = 0.01,
                  on_violation: str = "skip", local_search: bool = False,
                  local_max_evals: int | None = None) -> SearchResult:
    if on_violation not in {"skip", "stop"}:
        raise ValueError("on_violation must be 'skip' or 'stop'")
    sens = sensitivity_scan(ev, n_units)
    ranked = score_units(sens, epsilon)
    cfg = list(all_fp16(n_units))
    trace = []
    for row in ranked.itertuples():
        if row.score == float("-inf"):
            break  # remaining units are not faster in INT8
        cand = list(cfg)
        cand[int(row.unit)] = INT8
        if ev.latency(cand) >= ev.latency(cfg):
            trace.append((int(row.unit), "no-gain"))
            continue
        if ev.feasible(cand):
            cfg = cand
            trace.append((int(row.unit), "flip"))
        else:
            trace.append((int(row.unit), "backoff"))
            if on_violation == "stop":
                break
    method = f"greedy-{on_violation}"
    extra = {"ranking": ranked["unit"].astype(int).tolist(),
             "scores": ranked["score"].tolist(), "trace": trace,
             "sensitivity_evals": n_units + 1}
    if local_search:
        from precisionpilot.search.local_search import pairwise_local_search

        before = tuple(cfg)
        cfg = list(pairwise_local_search(ev, tuple(cfg), max_evals=local_max_evals))
        method += "+ls"
        extra["local_search_changed"] = tuple(cfg) != before
    if not ev.feasible(cfg):  # can only happen if all-FP16 itself breaks the budget
        cfg = list(all_fp16(n_units))
    extra["fp16_units"] = [i for i, b in enumerate(cfg) if b == FP16]
    return make_result(method, ev, tuple(cfg), **extra)
