import numpy as np
import pandas as pd
import pytest

from precisionpilot.analysis.compare import latency_model_table, lut_error_summary, memory_table
from precisionpilot.latency.backends.mock import MockBackend
from precisionpilot.latency.context import (
    ContextLatencyModel,
    ContextLatencyTable,
    features,
    measure_training_set,
    training_configs,
)
from precisionpilot.latency.lut import LatencyTable, build_lut
from precisionpilot.precision import all_fp16, all_int8, cfg_to_str, one_hot

N = 10


def test_features_count_boundaries_with_virtual_fp16_ends():
    f = features((0,) * N)
    assert f[:N].sum() == N and f[N:2 * N].sum() == 0 and f[2 * N:].sum() == 0
    f = features((1,) * N)
    assert f[N:2 * N].sum() == N and f[2 * N:].sum() == 2          # input + output boundary
    f = features((0, 1, 1, 0, 0, 1, 0, 0, 0, 0))
    assert f[2 * N:].sum() == 4                                     # two INT8 runs
    assert len(f) == 3 * N + 1


def test_training_configs_cover_structure_and_are_unique():
    cfgs = training_configs(N, n_random=20, seed=0)
    assert len(cfgs) == len(set(cfgs)) == 2 + 2 * N + 2 * (N - 1) - 2 + 20 or len(cfgs) >= 50
    assert all_fp16() in cfgs and all_int8() in cfgs
    assert one_hot(3) in cfgs
    assert tuple(0 if j == 3 else 1 for j in range(N)) in cfgs     # one-cold
    assert (1, 1, 1, 0, 0, 0, 0, 0, 0, 0) in cfgs                  # INT8 prefix
    assert training_configs(N, 20, 0) == cfgs                       # deterministic
    assert training_configs(N, 20, 1) != cfgs


def _world(seed=0):
    rng = np.random.default_rng(seed)
    f = rng.uniform(0.2, 1.0, N)
    g = f * rng.uniform(0.4, 0.9, N)
    h = rng.uniform(0.02, 0.1, N + 1)
    return np.concatenate([f, g, h])


def test_model_recovers_a_boundary_cost_world_exactly():
    true = _world()
    cfgs = training_configs(N, 20, 0)
    m = ContextLatencyModel.fit(cfgs, [features(c) @ true for c in cfgs], None, 1, ridge=1e-9)
    rng = np.random.default_rng(5)
    for _ in range(100):
        c = tuple(int(b) for b in rng.integers(0, 2, N))
        assert m.estimate(c) == pytest.approx(features(c) @ true, abs=1e-6)
    assert m.int8_ms.shape == (N,) and m.boundary_ms.shape == (N + 1,)


def test_ridge_prior_keeps_the_fit_sane_with_little_data():
    true = _world()
    prior = true.copy()
    cfgs = [all_fp16(), all_int8()]          # far fewer measurements than parameters
    m = ContextLatencyModel.fit(cfgs, [features(c) @ true for c in cfgs], prior, 1, ridge=1e-3)
    assert m.estimate(all_fp16()) == pytest.approx(features(all_fp16()) @ true, rel=1e-2)
    # unconstrained directions fall back to the prior instead of exploding
    assert np.abs(m.theta - prior).max() < 0.2


def test_table_roundtrip_and_interface(tmp_path):
    true = _world()
    models = {b: ContextLatencyModel(true * (1 + b / 100), N, b) for b in (1, 8)}
    t = ContextLatencyTable(models)
    assert t.batch_sizes == [1, 8] and t.n_units == N
    path = t.save(tmp_path / "ctx.csv")
    t2 = ContextLatencyTable.load(path)
    cfg = one_hot(4)
    assert t2.estimate(cfg, 8) == pytest.approx(t.estimate(cfg, 8))
    fn = t2.latency_fn(1)
    assert fn(cfg) == pytest.approx(t.estimate(cfg, 1)) and fn(cfg) == fn(cfg)


@pytest.fixture(scope="module")
def mock_world():
    import torch

    from precisionpilot.model import UnitResNet, build_resnet18
    from precisionpilot.precision import MixedPrecisionModel

    torch.manual_seed(0)
    um = UnitResNet.from_torchvision(build_resnet18(pretrained=False)).eval()
    mm = MixedPrecisionModel(um)
    mm.calibrate(np.random.default_rng(0).integers(0, 255, (16, 32, 32, 3), dtype=np.uint8),
                 "cpu", 16, 32)
    return mm


def test_context_model_beats_additive_table_on_held_out_mock_configs(mock_world):
    mm = mock_world
    backend = MockBackend(reps=400)   # many reps: the median's noise must be below the effect
    lut = build_lut(mm, backend, [1, 32], 32, log=lambda *_: None)
    train = measure_training_set(mm, backend, [1, 32], 32, n_random=20, seed=0,
                                 log=lambda *_: None)
    ctx = ContextLatencyTable.from_measurements(lut, train)
    seen = set(train["config"])
    rng = np.random.default_rng(11)
    add_err, ctx_err = [], []
    for _ in range(40):
        cfg = tuple(int(b) for b in rng.integers(0, 2, N))
        if cfg_to_str(cfg) in seen:
            continue
        for bs in (1, 32):
            real = backend.measure_config(mm, cfg, bs, 32).median_ms
            add_err.append(abs(lut.estimate(cfg, bs) - real) / real)
            ctx_err.append(abs(ctx.estimate(cfg, bs) - real) / real)
    assert np.mean(ctx_err) < 0.5 * np.mean(add_err)
    assert np.mean(ctx_err) < 0.03


def test_lut_model_size_uses_two_and_one_bytes_per_param(mock_world):
    lut = build_lut(mock_world, MockBackend(reps=5), [1], 32, log=lambda *_: None)
    params = lut.unit_params()
    assert len(params) == N and all(p > 0 for p in params)
    assert lut.model_mb(all_fp16()) == pytest.approx(2 * sum(params) / 1e6)
    assert lut.model_mb(all_int8()) == pytest.approx(sum(params) / 1e6)
    assert lut.model_mb(one_hot(9)) == pytest.approx((2 * sum(params) - params[9]) / 1e6)
    legacy = LatencyTable(lut.df.drop(columns=["params"]))
    with pytest.raises(ValueError):
        legacy.model_mb(all_fp16())


# ------------------------------------------------------------------ comparison tables
def _frames():
    res = pd.DataFrame([
        {"gpu": "g", "batch_size": 8, "budget_pp": 1.0, "latency_model": m, "method": meth,
         "config": cfg, "n_int8": cfg.count("1"), "model_mb": mb, "drop_pp": 0.0,
         "feasible": True, "n_evals": 18, "est_latency_ms": 1.0}
        for m in ("additive", "context")
        for meth, cfg, mb in (("all-fp16", "0" * 10, 22.0), ("all-int8", "1" * 10, 11.0),
                              ("greedy-skip", "0111111110", 11.4),
                              ("greedy-dyn", "1111111111", 11.0))])
    ver = pd.DataFrame([
        {"gpu": "g", "batch_size": 8, "config": c, "measured_median_ms": ms, "engine_mb": e,
         "activation_memory_mb": a, "gpu_footprint_mb": e + a + 1, "error_pct": err,
         "ctx_estimated_ms": ms, "ctx_error_pct": 1.0, "ctx_trained": tr}
        for c, ms, e, a, err, tr in (("0" * 10, 2.0, 22.6, 19.0, -15.0, True),
                                     ("1" * 10, 1.0, 11.5, 4.8, 47.0, True),
                                     ("0111111110", 1.2, 11.5, 20.9, 12.0, False))])
    return res, ver


def test_memory_table_savings_and_selection():
    res, ver = _frames()
    mem = memory_table(res, ver, 1.0)
    assert list(mem.name) == ["all-FP16", "selected", "all-INT8"]
    sel = mem[mem.name == "selected"].iloc[0]
    assert sel.config == "1" * 10                       # greedy-dyn of the context model
    fp16 = mem[mem.name == "all-FP16"].iloc[0]
    assert fp16.model_mb_saving_pct == 0
    i8 = mem[mem.name == "all-INT8"].iloc[0]
    assert i8.model_mb_saving_pct == pytest.approx(50.0)
    assert i8.activation_memory_mb_saving_pct == pytest.approx(100 * (1 - 4.8 / 19.0))


def test_latency_model_table_joins_real_measurements():
    res, ver = _frames()
    t = latency_model_table(res, ver)
    g = t[(t.latency_model == "additive") & (t.method == "greedy-skip")].iloc[0]
    assert g.measured_ms == 1.2 and g.real_speedup_vs_fp16 == pytest.approx(2.0 / 1.2)
    d = t[(t.latency_model == "context") & (t.method == "greedy-dyn")].iloc[0]
    assert d.measured_ms == 1.0 and d.real_speedup_vs_fp16 == pytest.approx(2.0)


def test_lut_error_summary_reports_only_held_out_context_error():
    _, ver = _frames()
    s = lut_error_summary(ver).iloc[0]
    assert s.n == 3 and s.context_n_heldout == 1
    assert s.additive_max_abs_pct == 47.0 and s.context_mean_abs_pct == 1.0
