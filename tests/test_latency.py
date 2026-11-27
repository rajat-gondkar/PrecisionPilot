import numpy as np
import pytest
import torch

from precisionpilot.latency.backends.base import get_backend
from precisionpilot.latency.backends.mock import MockBackend
from precisionpilot.latency.lut import LatencyTable, build_lut, verify_configs
from precisionpilot.latency.timing import TimingStats, time_callable
from precisionpilot.model import UnitResNet, build_resnet18
from precisionpilot.precision import FP16, INT8, MixedPrecisionModel, all_fp16, all_int8, one_hot


@pytest.fixture(scope="module")
def mm():
    torch.manual_seed(0)
    um = UnitResNet.from_torchvision(build_resnet18(pretrained=False)).eval()
    m = MixedPrecisionModel(um)
    m.calibrate(np.random.default_rng(0).integers(0, 255, (16, 32, 32, 3), dtype=np.uint8),
                "cpu", 16, 64)
    return m


def test_timing_stats():
    s = TimingStats.from_samples(list(range(1, 101)))
    assert s.median_ms == pytest.approx(50.5)
    assert s.p95_ms == pytest.approx(95.05)
    assert s.n == 100 and s.min_ms == 1
    assert s.throughput(10) == pytest.approx(10 / 0.0505)
    with pytest.raises(ValueError):
        TimingStats.from_samples([])


def test_time_callable_counts_reps():
    calls = []
    s = time_callable(lambda: calls.append(1), warmup=3, reps=7)
    assert len(calls) == 10 and s.n == 7


def test_get_backend():
    assert isinstance(get_backend("mock"), MockBackend)
    with pytest.raises(ValueError):
        get_backend("nope")
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError):
            get_backend("torch")


def test_mock_is_deterministic_and_int8_faster_for_big_batches(mm):
    b = MockBackend()
    shape = (32, 256, 14, 14)
    a1 = b.measure_unit(mm, 6, INT8, shape)
    a2 = MockBackend().measure_unit(mm, 6, INT8, shape)
    assert a1 == a2
    assert a1.median_ms < b.measure_unit(mm, 6, FP16, shape).median_ms
    assert a1.p95_ms >= a1.median_ms


def test_mock_profiles_differ(mm):
    s = (8, 64, 56, 56)
    amp = MockBackend("mock-ampere").measure_unit(mm, 1, FP16, s).median_ms
    tur = MockBackend("mock-turing").measure_unit(mm, 1, FP16, s).median_ms
    assert amp != tur
    with pytest.raises(ValueError):
        MockBackend("mock-xyz")


def test_lut_build_estimate_roundtrip(mm, tmp_path):
    b = MockBackend(reps=20)
    lut = build_lut(mm, b, [1, 4], image_size=64, log=lambda *_: None)
    assert len(lut.df) == 2 * 10 * 2
    assert lut.batch_sizes == [1, 4] and lut.n_units == 10
    est = lut.estimate(one_hot(3), 4)
    manual = sum(lut.unit_ms(4, i, FP16) for i in range(10) if i != 3) + lut.unit_ms(4, 3, INT8)
    assert est == pytest.approx(manual)
    assert lut.latency_fn(4)(one_hot(3)) == pytest.approx(est)
    sav = lut.savings(4)
    assert sav[3] == pytest.approx(lut.estimate(all_fp16(), 4) - est)
    lut2 = LatencyTable.load(lut.save(tmp_path / "lut.csv"))
    assert lut2.estimate(all_int8(), 1) == pytest.approx(lut.estimate(all_int8(), 1))
    assert lut2.gpu == "mock-ampere" and lut2.backend == "mock"


def test_lut_rejects_bad_frame():
    import pandas as pd

    with pytest.raises(ValueError):
        LatencyTable(pd.DataFrame({"unit": [0]}))


def test_verify_reports_nonzero_but_bounded_error(mm):
    b = MockBackend(reps=20)
    lut = build_lut(mm, b, [4], image_size=64, log=lambda *_: None)
    df = verify_configs(mm, b, lut, [all_fp16(), all_int8(), one_hot(0)], 4, 64,
                        labels=["fp16", "int8", "stem"], log=lambda *_: None)
    assert list(df["label"]) == ["fp16", "int8", "stem"]
    assert (df["error_pct"].abs() > 0).all() and (df["error_pct"].abs() < 30).all()
    assert "weights_mb" in df.columns
    # all-INT8 weights are half the size of all-FP16 weights
    assert df["weights_mb"][1] == pytest.approx(df["weights_mb"][0] / 2)


# ------------------------------------------------------------------ real GPU / TensorRT
@pytest.mark.gpu
def test_torch_backend_runs(mm):
    b = get_backend("torch", warmup=2, reps=5)
    assert b.measure_config(mm, all_fp16(), 2, 64).median_ms > 0


@pytest.mark.trt
def test_tensorrt_unit_and_config(mm, tmp_path):
    b = get_backend("trt", warmup=2, reps=5, cache_dir=tmp_path)
    assert b.measure_unit(mm, 1, INT8, (2, 64, 16, 16)).median_ms > 0
    assert b.measure_config(mm, one_hot(0), 2, 64).median_ms > 0
    info = b.model_info(mm, all_int8(), 2, 64)
    assert info["engine_mb"] > 0 and info["gpu_footprint_mb"] > 0
    assert info["engine_mb"] < b.model_info(mm, all_fp16(), 2, 64)["engine_mb"]
    assert list(tmp_path.glob("*.engine"))  # engine cache populated


@pytest.mark.trt
def test_tensorrt_accuracy_matches_simulation(mm, tmp_path):
    from precisionpilot.evaluate import accuracy

    b = get_backend("trt", warmup=1, reps=2, cache_dir=tmp_path)
    rng = np.random.default_rng(1)
    x = rng.integers(0, 255, (64, 32, 32, 3), dtype=np.uint8)
    y = rng.integers(0, 10, 64)
    m = mm.cuda()
    for cfg in (all_fp16(), all_int8()):
        sim_pred_acc = accuracy(m.set_config(cfg), x, y, "cuda", 16, 64, torch.float16)
        trt_acc = b.accuracy(m, cfg, x, y, 16, 64)
        assert abs(sim_pred_acc - trt_acc) <= 10.0
