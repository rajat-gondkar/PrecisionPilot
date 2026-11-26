"""Random search baseline at a fixed evaluation budget (outline §7).

Draws distinct configurations uniformly at random, evaluates each, and returns the
lowest-latency feasible one. Given the *same* number of evaluations as the proposed method,
it shows whether the sensitivity guidance actually matters. If nothing sampled is feasible,
it falls back to all-FP16 (costing one more evaluation, reported honestly).
"""

from __future__ import annotations

import numpy as np

from precisionpilot.precision import all_fp16, int_to_cfg
from precisionpilot.search.common import Evaluator, SearchResult, make_result


def random_search(ev: Evaluator, n_evals: int, n_units: int = 10,
                  seed: int = 0) -> SearchResult:
    space = 2 ** n_units
    n = min(n_evals, space)
    rng = np.random.default_rng(seed)
    for idx in rng.choice(space, size=n, replace=False):
        ev.accuracy(int_to_cfg(int(idx), n_units))
    best = ev.best_feasible_seen()
    fallback = best is None
    if fallback:
        best = all_fp16(n_units)
    return make_result("random", ev, best, seed=seed, budget_evals=n_evals,
                       fallback_fp16=fallback)
