import math

import pytest

from precisionpilot.precision import cfg_to_str
from precisionpilot.search.common import Evaluator, TableAccuracy, make_result
from precisionpilot.search.sensitivity import score_units, sensitivity_scan


def additive_oracle(drops, savings, base_acc=95.0, base_lat=10.0):
    acc = lambda c: base_acc - sum(d for d, b in zip(drops, c) if b)  # noqa: E731
    lat = lambda c: base_lat - sum(s for s, b in zip(savings, c) if b)  # noqa: E731
    return acc, lat


def test_evaluator_counts_distinct_evals_only():
    acc, lat = additive_oracle([0.1] * 10, [0.5] * 10)
    calls = []
    ev = Evaluator(lambda c: calls.append(c) or acc(c), lat, reference_acc=95.0, budget_pp=1.0)
    ev.accuracy((0,) * 10)
    ev.accuracy((0,) * 10)
    ev.accuracy((1,) + (0,) * 9)
    assert ev.n_evals == 2 and len(calls) == 2 and len(ev.history) == 2
    assert ev.feasible((1,) * 10)  # drop 1.0 == budget -> feasible
    assert ev.n_evals == 3
    assert ev.best_feasible_seen() == (1,) * 10


def test_sensitivity_scan_costs_n_plus_one_and_measures_drops():
    drops = [3.0, 0.1, 0.0, 0.2, 0.1, 0.3, 0.1, 0.4, 0.2, 2.0]
    acc, lat = additive_oracle(drops, [1.0] * 10)
    ev = Evaluator(acc, lat, reference_acc=95.5, budget_pp=1.0)
    df = sensitivity_scan(ev)
    assert ev.n_evals == 11
    assert list(df["drop_vs_fp16"].round(6)) == drops
    assert df["drop_vs_ref"].iloc[0] == pytest.approx(3.5)
    assert df.attrs["fp16_acc"] == 95.0


def test_score_ranks_by_saving_per_drop_and_excludes_slower_units():
    drops = [3.0, 0.5, 0.0, 0.5, 0, 0, 0, 0, 0, 0]
    savings = [1.0, 2.0, 1.0, -0.5, 0, 0, 0, 0, 0, 0]
    acc, lat = additive_oracle(drops, savings)
    ev = Evaluator(acc, lat, 95.0, 1.0)
    ranked = score_units(sensitivity_scan(ev), epsilon=0.01)
    assert ranked.iloc[0]["unit"] == 2  # free + faster -> top
    assert ranked.iloc[1]["unit"] == 1  # 2.0/0.5 = 4 > 1/3
    slow = ranked[ranked["unit"] == 3].iloc[0]
    assert slow["score"] == -math.inf
    assert list(ranked["rank"]) == list(range(1, 11))


def test_table_accuracy_and_result():
    table = {cfg_to_str((0,) * 10): 95.0, cfg_to_str((1,) * 10): 90.0}
    ta = TableAccuracy(table)
    ev = Evaluator(ta, lambda c: 10.0 - sum(c), 95.0, 1.0)
    r = make_result("x", ev, (1,) * 10)
    assert r.drop_pp == 5.0 and not r.feasible and r.latency == 0.0
    with pytest.raises(KeyError):
        ta((0, 1) * 5)
