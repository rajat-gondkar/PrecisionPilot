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


def greedy_dynamic(ev: Evaluator, n_units: int = 10, epsilon: float = 0.01) -> SearchResult:
    """Context-aware greedy: re-score every unit in the *current* configuration at each step.

    The static greedy above ranks units once, by their isolated saving. With a latency model that
    is not additive (a unit's benefit depends on its neighbours, e.g. an INT8 stem only pays off
    once ``layer1.0`` is INT8 as well), that ranking is stale after the first flip. Here:

    1. sensitivity scan (1 + N evaluations) gives each unit's accuracy cost ``d_u``;
    2. at each step, for every unit still in FP16 and not banned, the *marginal* latency gain
       ``g_u = L(cfg) - L(cfg + u)`` is read from the latency model (free, no evaluation);
    3. the unit with the best ``g_u / max(d_u, eps)`` among those with ``g_u > 0`` is tried; if the
       budget still holds it is kept, otherwise it is banned;
    4. if no single flip gains anything, look one step ahead at adjacent *pairs* (a run of two INT8
       units pays the conversion cost once instead of twice);
    5. stop when neither a unit nor a pair has a positive marginal gain.

    Cost: 1 + N sensitivity evaluations plus one per flip or ban (a few dozen at most). With an
    additive latency model the order is identical to :func:`greedy_search`.
    """
    sens = sensitivity_scan(ev, n_units)
    cost = dict(zip(sens["unit"].astype(int), sens["drop_vs_fp16"].clip(lower=epsilon)))
    cfg = list(all_fp16(n_units))
    banned: set[int] = set()
    banned_pairs: set[tuple[int, int]] = set()
    trace = []
    while True:
        cur = ev.latency(cfg)
        cands = []
        for u in range(n_units):
            if cfg[u] == INT8 or u in banned:
                continue
            cand = list(cfg)
            cand[u] = INT8
            gain = cur - ev.latency(cand)
            if gain > 1e-12:
                cands.append((gain / cost[u], gain, -u, u, cand))
        if not cands:
            # No single flip helps. A run of two adjacent INT8 units pays the conversion cost
            # once instead of twice, so look one step ahead at adjacent pairs.
            for u in range(n_units - 1):
                if cfg[u] == INT8 or cfg[u + 1] == INT8 or (u, u + 1) in banned_pairs:
                    continue
                cand = list(cfg)
                cand[u] = cand[u + 1] = INT8
                gain = cur - ev.latency(cand)
                if gain > 1e-12:
                    cands.append((gain / (cost[u] + cost[u + 1]), gain, -u, (u, u + 1), cand))
            if not cands:
                break
            cands.sort(reverse=True)
            _, _, _, pair, cand = cands[0]
            if ev.feasible(cand):
                cfg = cand
                trace.append((pair, "flip-pair"))
            else:
                banned_pairs.add(pair)
                trace.append((pair, "backoff-pair"))
            continue
        cands.sort(reverse=True)
        _, _, _, u, cand = cands[0]
        if ev.feasible(cand):
            cfg = cand
            trace.append((u, "flip"))
        else:
            banned.add(u)
            trace.append((u, "backoff"))
    if not ev.feasible(cfg):
        cfg = list(all_fp16(n_units))
    return make_result("greedy-dyn", ev, tuple(cfg), trace=trace,
                       sensitivity_evals=n_units + 1,
                       fp16_units=[i for i, b in enumerate(cfg) if b == FP16])
