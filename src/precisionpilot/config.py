"""Typed configuration loaded from YAML, with dotted-key CLI overrides."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.yaml"


@dataclass
class PathsCfg:
    data_dir: str = "data"
    checkpoint: str = "checkpoints/resnet18_cifar10.pt"
    results_dir: str = "results"
    engine_cache: str = ".engine_cache"


@dataclass
class DataCfg:
    image_size: int = 224
    num_workers: Any = "auto"
    splits_file: str = "data/splits.json"
    n_calib: int = 1000
    n_spare: int = 1000
    n_val: int = 5000
    n_val_subset: int = 1000
    n_test: int = 5000
    eval_batch_size: int = 128


@dataclass
class TrainCfg:
    epochs: int = 8
    batch_size: int = 128
    lr: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 5e-4
    label_smoothing: float = 0.1
    amp: bool = True


@dataclass
class QuantCfg:
    act_method: str = "percentile"
    act_percentile: float = 99.99
    calib_batches: int = 8


@dataclass
class BudgetCfg:
    main_pp: float = 1.0
    sweep_pp: list = field(default_factory=lambda: [0.5, 1.0, 2.0])
    extra_pp: list = field(default_factory=lambda: [0.1, 0.2, 0.3])


@dataclass
class LatencyCfg:
    backend: str = "mock"
    batch_sizes: list = field(default_factory=lambda: [1, 8, 32])
    warmup: int = 20
    reps: int = 100
    cooldown_s: float = 1.0


@dataclass
class SearchCfg:
    random_seeds: int = 20
    epsilon: float = 0.01
    local_search: bool = True


@dataclass
class Config:
    seed: int = 42
    paths: PathsCfg = field(default_factory=PathsCfg)
    data: DataCfg = field(default_factory=DataCfg)
    train: TrainCfg = field(default_factory=TrainCfg)
    quant: QuantCfg = field(default_factory=QuantCfg)
    budget: BudgetCfg = field(default_factory=BudgetCfg)
    latency: LatencyCfg = field(default_factory=LatencyCfg)
    search: SearchCfg = field(default_factory=SearchCfg)

    def path(self, name: str) -> Path:
        """Resolve a configured path relative to the project root (absolute paths kept)."""
        p = Path(getattr(self.paths, name)) if hasattr(self.paths, name) else Path(name)
        return p if p.is_absolute() else PROJECT_ROOT / p

    def resolve(self, rel: str | Path) -> Path:
        p = Path(rel)
        return p if p.is_absolute() else PROJECT_ROOT / p

    def to_dict(self) -> dict:
        return asdict(self)


def _build(cls, data: dict):
    kwargs = {}
    valid = {f.name: f for f in fields(cls)}
    for key, value in (data or {}).items():
        if key not in valid:
            raise KeyError(f"Unknown config key '{key}' for {cls.__name__}")
        ftype = valid[key].type
        default = getattr(cls(), key)
        if is_dataclass(default) and isinstance(value, dict):
            kwargs[key] = _build(type(default), value)
        else:
            kwargs[key] = value
        del ftype
    return cls(**kwargs)


def _parse_scalar(text: str) -> Any:
    return yaml.safe_load(text)


def apply_overrides(raw: dict, overrides: list[str] | None) -> dict:
    """Apply ``a.b=value`` overrides to a raw config dict."""
    raw = copy.deepcopy(raw)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must look like key.sub=value, got '{item}'")
        key, value = item.split("=", 1)
        node = raw
        parts = key.strip().split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = _parse_scalar(value)
    return raw


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> Config:
    path = Path(path) if path else DEFAULT_CONFIG
    raw = yaml.safe_load(path.read_text()) if path.exists() else {}
    raw = apply_overrides(raw or {}, overrides)
    return _build(Config, raw)
