import numpy as np
import pytest
import torch

from precisionpilot.cli import main
from precisionpilot.config import DEFAULT_CONFIG, PROJECT_ROOT, apply_overrides, load_config
from precisionpilot.utils import (
    default_num_workers,
    get_device,
    gpu_slug,
    load_json,
    save_json,
    set_seed,
)


def test_default_config_loads_and_budget_is_fixed():
    cfg = load_config()
    assert DEFAULT_CONFIG.exists()
    assert cfg.budget.main_pp == 1.0
    assert cfg.budget.sweep_pp == [0.5, 1.0, 2.0]
    assert cfg.latency.batch_sizes == [1, 8, 32]


def test_overrides_parse_types():
    cfg = load_config(overrides=["train.epochs=2", "latency.backend=trt", "quant.act_percentile=99.9"])
    assert cfg.train.epochs == 2 and isinstance(cfg.train.epochs, int)
    assert cfg.latency.backend == "trt"
    assert cfg.quant.act_percentile == pytest.approx(99.9)


def test_unknown_key_rejected():
    with pytest.raises(KeyError):
        load_config(overrides=["train.not_a_key=1"])


def test_bad_override_format():
    with pytest.raises(ValueError):
        apply_overrides({}, ["no_equals_sign"])


def test_paths_resolve_relative_to_project_root():
    cfg = load_config()
    assert cfg.path("results_dir") == PROJECT_ROOT / "results"


def test_seed_reproducible():
    set_seed(1)
    a = (torch.rand(3), np.random.rand(3))
    set_seed(1)
    b = (torch.rand(3), np.random.rand(3))
    assert torch.equal(a[0], b[0]) and np.allclose(a[1], b[1])


def test_device_and_workers():
    assert get_device("cpu").type == "cpu"
    assert get_device().type in {"cpu", "cuda", "mps"}
    assert default_num_workers(3) == 3
    assert default_num_workers("auto") >= 0


def test_gpu_slug():
    assert gpu_slug("NVIDIA GeForce RTX 3050 Ti Laptop GPU") == "rtx3050ti"
    assert gpu_slug("Tesla T4") == "t4"


def test_json_roundtrip(tmp_path):
    p = save_json({"a": np.int64(3), "b": (1, 2)}, tmp_path / "x" / "y.json")
    assert load_json(p) == {"a": 3, "b": [1, 2]}


def test_cli_env_check(capsys):
    assert main(["env-check"]) == 0
    assert "torch" in capsys.readouterr().out
