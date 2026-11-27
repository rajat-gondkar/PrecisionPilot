import numpy as np
import pandas as pd
import pytest

from precisionpilot.precision import all_configs, all_fp16, cfg_to_str
from precisionpilot.search.common import Evaluator
from precisionpilot.search.exhaustive import exhaustive_search, gap_to_optimum, pareto_front
from precisionpilot.search.greedy import greedy_search
from precisionpilot.search.local_search import neighbours, pairwise_local_search
from precisionpilot.search.random_search import random_search

N = 10


def make_problem(seed: int, interactions: bool = True):
    """Synthetic accuracy/latency oracles: additive drops + pairwise interactions."""
    rng = np.random.default_rng(seed)
    drops = rng.exponential(0.3, N)
    drops[0] += rng.uniform(1, 3)       # fragile stem
    drops[-1] += rng.uniform(0.5, 2)    # fragile head
    inter = rng.normal(0, 0.05, (N, N)) if interactions else np.zeros((N, N))
    savings = rng.uniform(-0.1, 1.0, N)
    base_acc, base_lat = 95.0, 10.0

    def acc(c):
        c = np.asarray(c)
        return base_acc - float(drops @ c) - float(c @ np.triu(inter, 1) @ c)

    def lat(c):
        return base_lat - float(savings @ np.asarray(c))

    return acc, lat, base_acc


def brute_force_opt(acc, lat, ref, budget):
    feas = [c for c in all_configs(N) if ref - acc(c) <= budget + 1e-9]
    return min(feas, key=lat)


@pytest.mark.parametrize("seed", range(25))
def test_properties_on_random_problems(seed):
    acc, lat, ref = make_problem(seed)
    budget = 1.0
    ev = Evaluator(acc, lat, ref, budget)
    g = greedy_search(ev)
    assert g.feasible and g.drop_pp <= budget + 1e-9
    assert g.n_evals <= 1 + 2 * N  # scan + at most one eval per unit

    ex, table = exhaustive_search(Evaluator(acc, lat, ref, budget))
    opt = brute_force_opt(acc, lat, ref, budget)
    assert ex.latency == pytest.approx(lat(opt))
    assert ex.n_evals == 1024 and len(table) == 1024
    assert g.latency >= ex.latency - 1e-9

    gl = greedy_search(Evaluator(acc, lat, ref, budget), local_search=True)
    assert gl.feasible and gl.latency <= g.latency + 1e-9
    assert gl.latency >= ex.latency - 1e-9


def test_greedy_is_optimal_on_additive_problem_with_free_units():
    drops = [3, 0, 0, 0, 0, 0, 0, 0, 0, 3]
    sav = [1] * N
    acc = lambda c: 95 - sum(d * b for d, b in zip(drops, c))  # noqa: E731
    lat = lambda c: 10 - sum(s * b for s, b in zip(sav, c))  # noqa: E731
    r = greedy_search(Evaluator(acc, lat, 95, 1.0))
    assert r.config == "0111111110"
    assert r.extra["fp16_units"] == [0, 9]


def test_greedy_never_flips_units_that_are_slower_in_int8():
    acc = lambda c: 95.0  # noqa: E731  (quantization is free)
    sav = [1, -1, 1, -1, 1, -1, 1, -1, 1, -1]
    lat = lambda c: 10 - sum(s * b for s, b in zip(sav, c))  # noqa: E731
    r = greedy_search(Evaluator(acc, lat, 95, 1.0))
    assert r.config == "1010101010"


def test_greedy_stop_vs_skip():
    # unit order by score: 1 (big saving), 2 (breaks budget), 3 (fine)
    drops = [5, 0.1, 0.95, 0.1, 5, 5, 5, 5, 5, 5]
    sav = [0.0, 3.0, 2.0, 0.15, 0, 0, 0, 0, 0, 0]
    acc = lambda c: 95 - sum(d * b for d, b in zip(drops, c))  # noqa: E731
    lat = lambda c: 10 - sum(s * b for s, b in zip(sav, c))  # noqa: E731
    skip = greedy_search(Evaluator(acc, lat, 95, 1.0), on_violation="skip")
    stop = greedy_search(Evaluator(acc, lat, 95, 1.0), on_violation="stop")
    assert skip.config[1] == "1" and skip.config[2] == "0" and skip.config[3] == "1"
    assert stop.config[1] == "1" and stop.config[3] == "0"
    assert stop.n_evals < skip.n_evals
    with pytest.raises(ValueError):
        greedy_search(Evaluator(acc, lat, 95, 1.0), on_violation="maybe")


def test_greedy_falls_back_when_nothing_is_feasible():
    r = greedy_search(Evaluator(lambda c: 90.0, lambda c: 10 - sum(c), 95, 1.0))
    assert r.config == cfg_to_str(all_fp16()) and not r.feasible


def test_random_search_reproducible_and_respects_budget():
    acc, lat, ref = make_problem(3)
    a = random_search(Evaluator(acc, lat, ref, 1.0), n_evals=21, seed=5)
    b = random_search(Evaluator(acc, lat, ref, 1.0), n_evals=21, seed=5)
    c = random_search(Evaluator(acc, lat, ref, 1.0), n_evals=21, seed=6)
    assert a.config == b.config and a.n_evals == b.n_evals
    assert a.n_evals in (21, 22)  # 22 only if it had to fall back to all-FP16
    assert a.feasible
    assert [h["config"] for h in a.history] != [h["config"] for h in c.history]


def test_random_search_fallback():
    r = random_search(Evaluator(lambda c: 95.0 if sum(c) == 0 else 0.0,
                                lambda c: 10 - sum(c), 95, 1.0), n_evals=5, seed=0)
    assert r.config == "0" * N and r.extra["fallback_fp16"]


def test_local_search_neighbourhood_and_improvement():
    cfg = (1, 0, 0) + (0,) * 7
    nb = neighbours(cfg)
    assert len(nb) == 9 + 9  # 9 adds + 1*9 swaps
    assert len(set(nb)) == len(nb) and cfg not in nb
    # greedy-like start that is improvable by a swap: unit 0 cheap-but-small, unit 1 big
    acc = lambda c: 95 - (0.6 * c[0] + 0.6 * c[1])  # noqa: E731  (budget 1 -> only one fits)
    lat = lambda c: 10 - (0.5 * c[0] + 2.0 * c[1])  # noqa: E731
    ev = Evaluator(acc, lat, 95, 1.0)
    out = pairwise_local_search(ev, cfg)
    assert out[1] == 1 and out[0] == 0
    # cap on evaluations is honoured
    ev2 = Evaluator(acc, lat, 95, 1.0)
    pairwise_local_search(ev2, cfg, max_evals=1)
    assert ev2.n_evals <= 2


def test_gap_and_pareto():
    assert gap_to_optimum(11, 10) == pytest.approx(10.0)
    t = pd.DataFrame({"val_acc": [95, 94, 93, 94.5], "latency": [10, 8, 7, 9.5]})
    pf = pareto_front(t)
    assert list(pf["latency"]) == [7, 8, 9.5, 10]
    t2 = pd.DataFrame({"val_acc": [95, 90], "latency": [5, 6]})
    assert list(pareto_front(t2)["latency"]) == [5]


# ---------------------------------------------------------------- context-aware greedy
def boundary_problem(n=10, boundary_cost=0.4):
    """Latency with conversion cost at FP16<->INT8 boundaries: an isolated INT8 unit is *slower*
    than FP16, but runs of INT8 units are faster."""
    drops = [0.05] * n
    f = [1.0] * n
    g = [0.5] * n

    def lat(c):
        seq = (0, *c, 0)
        bound = sum(1 for a, b in zip(seq[:-1], seq[1:]) if a != b)
        return sum(g[i] if c[i] else f[i] for i in range(n)) + boundary_cost * bound

    def acc(c):
        return 95.0 - sum(d * b for d, b in zip(drops, c))

    return acc, lat


def test_dynamic_greedy_finds_runs_that_static_greedy_cannot():
    from precisionpilot.search.greedy import greedy_dynamic

    acc, lat = boundary_problem()
    # each isolated flip: -0.5 (cheaper unit) + 2 * 0.4 boundary = +0.3 -> slower than FP16
    assert lat((1,) + (0,) * 9) > lat((0,) * 10)
    static = greedy_search(Evaluator(acc, lat, 95.0, 1.0))
    assert static.config == "0" * 10  # stuck: no single flip helps
    dyn = greedy_dynamic(Evaluator(acc, lat, 95.0, 1.0))
    # no single flip pays off, but an adjacent PAIR does (one conversion pair instead of two);
    # the pair look-ahead gets the search going and it then extends the run to all-INT8
    assert dyn.config == "1" * 10
    ex, _ = exhaustive_search(Evaluator(acc, lat, 95.0, 1.0))
    assert dyn.latency == pytest.approx(ex.latency)
    assert any(kind == "flip-pair" for _, kind in dyn.extra["trace"])


def test_dynamic_greedy_extends_a_run_and_enables_stem_style_units():
    from precisionpilot.search.greedy import greedy_dynamic

    # unit 0 alone is slower in INT8 (conversion cost), but beneficial next to an INT8 unit 1
    n = 10
    f = [1.0] * n
    g = [0.5] * n
    g[1] = 0.1  # unit 1 is so much faster in INT8 that flipping it alone already pays off

    def lat(c):
        seq = (0, *c, 0)
        bound = sum(1 for a, b in zip(seq[:-1], seq[1:]) if a != b)
        return sum(g[i] if c[i] else f[i] for i in range(n)) + 0.4 * bound

    acc = lambda c: 95.0 - 0.05 * sum(c)  # noqa: E731
    static = greedy_search(Evaluator(acc, lat, 95.0, 1.0))
    dyn = greedy_dynamic(Evaluator(acc, lat, 95.0, 1.0))
    ex, _ = exhaustive_search(Evaluator(acc, lat, 95.0, 1.0))
    assert dyn.latency <= static.latency + 1e-9
    assert dyn.latency == pytest.approx(ex.latency)       # reaches the optimum
    assert dyn.config[0] == "1" and dyn.config[1] == "1"  # stem-style unit joined the run


@pytest.mark.parametrize("seed", range(10))
def test_dynamic_greedy_feasible_and_matches_static_on_additive_problems(seed):
    from precisionpilot.search.greedy import greedy_dynamic

    acc, lat, ref = make_problem(seed, interactions=False)
    dyn = greedy_dynamic(Evaluator(acc, lat, ref, 1.0))
    static = greedy_search(Evaluator(acc, lat, ref, 1.0))
    assert dyn.feasible
    assert dyn.latency == pytest.approx(static.latency)   # same order when latency is additive
    assert dyn.n_evals <= 1 + N + 2 * N
