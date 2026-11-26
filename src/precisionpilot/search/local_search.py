"""Pairwise-swap local search (outline §6.4 step 5, optional refinement).

Neighbourhood of a configuration: flip one FP16 unit to INT8 ("add"), or swap one INT8 unit
back to FP16 while flipping one FP16 unit to INT8 ("swap"). Only neighbours with a *lower*
estimated latency are worth evaluating (latency comes from the lookup table, free), and they
are tried fastest-first. First improvement is accepted; repeat until no neighbour improves or
the evaluation cap is reached. The result is never worse than the start.
"""

from __future__ import annotations

from precisionpilot.precision import FP16, INT8, Config
from precisionpilot.search.common import Evaluator


def neighbours(cfg: Config) -> list[Config]:
    out = []
    int8 = [i for i, b in enumerate(cfg) if b == INT8]
    fp16 = [i for i, b in enumerate(cfg) if b == FP16]
    for j in fp16:
        c = list(cfg)
        c[j] = INT8
        out.append(tuple(c))
    for i in int8:
        for j in fp16:
            c = list(cfg)
            c[i], c[j] = FP16, INT8
            out.append(tuple(c))
    return out


def pairwise_local_search(ev: Evaluator, start: Config, max_evals: int | None = None) -> Config:
    current = tuple(start)
    if not ev.feasible(current):
        return current
    start_evals = ev.n_evals
    improved = True
    while improved:
        improved = False
        cur_lat = ev.latency(current)
        cands = [c for c in neighbours(current) if ev.latency(c) < cur_lat - 1e-12]
        cands.sort(key=ev.latency)
        for c in cands:
            if max_evals is not None and ev.n_evals - start_evals >= max_evals \
                    and not ev.is_cached(c):
                return current
            if ev.feasible(c):
                current = c
                improved = True
                break
    return current
