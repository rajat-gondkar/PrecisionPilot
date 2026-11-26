"""Shared search infrastructure: accuracy oracles, a counting evaluator and result records.

Every search method talks to the same :class:`Evaluator`. It

* evaluates accuracy of a configuration (on the *validation* data only),
* counts **distinct** accuracy evaluations (the search-cost metric; repeats are cached),
* estimates latency from the lookup table (free: not counted as an evaluation),
* records wall-clock time and the full history.

Accuracy oracles are interchangeable: :class:`ModelAccuracy` runs the fake-quant model;
:class:`TableAccuracy` replays a precomputed table (e.g. the exhaustive 1,024-config table
computed once on a GPU). Both give identical search decisions; the table only saves time.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from precisionpilot.precision import Config, cfg_to_str, str_to_cfg

AccuracyFn = Callable[[Config], float]
LatencyFn = Callable[[Config], float]


class ModelAccuracy:
    """Accuracy of a MixedPrecisionModel configuration on fixed (x, y) data."""

    def __init__(self, mixed_model, x: np.ndarray, y: np.ndarray, device,
                 batch_size: int = 128, image_size: int = 224):
        self.mm = mixed_model
        self.x, self.y = x, y
        self.device = device
        self.batch_size = batch_size
        self.image_size = image_size

    def __call__(self, cfg: Config) -> float:
        import torch

        from precisionpilot.evaluate import accuracy

        self.mm.set_config(cfg)
        return accuracy(self.mm, self.x, self.y, self.device, self.batch_size,
                        self.image_size, input_dtype=torch.float16)


class TableAccuracy:
    """Accuracy looked up from a {config-string: accuracy} table."""

    def __init__(self, table: dict[str, float]):
        self.table = dict(table)

    @classmethod
    def from_csv(cls, path: str | Path, column: str = "val_acc") -> TableAccuracy:
        df = pd.read_csv(path, dtype={"config": str})
        return cls(dict(zip(df["config"], df[column])))

    def __call__(self, cfg: Config) -> float:
        key = cfg_to_str(cfg)
        if key not in self.table:
            raise KeyError(f"config {key} not in accuracy table")
        return float(self.table[key])


@dataclass
class EvalRecord:
    config: str
    acc: float
    latency: float
    n_evals: int
    t: float


class Evaluator:
    """Counts distinct accuracy evaluations and checks the accuracy budget."""

    def __init__(self, accuracy_fn: AccuracyFn, latency_fn: LatencyFn, reference_acc: float,
                 budget_pp: float):
        self.accuracy_fn = accuracy_fn
        self.latency_fn = latency_fn
        self.reference_acc = float(reference_acc)
        self.budget_pp = float(budget_pp)
        self._cache: dict[Config, float] = {}
        self.history: list[EvalRecord] = []
        self.t0 = time.perf_counter()

    @property
    def n_evals(self) -> int:
        return len(self._cache)

    def accuracy(self, cfg: Sequence[int]) -> float:
        cfg = tuple(int(b) for b in cfg)
        if cfg not in self._cache:
            acc = float(self.accuracy_fn(cfg))
            self._cache[cfg] = acc
            self.history.append(EvalRecord(cfg_to_str(cfg), acc, self.latency(cfg),
                                           self.n_evals, time.perf_counter() - self.t0))
        return self._cache[cfg]

    def latency(self, cfg: Sequence[int]) -> float:
        return float(self.latency_fn(tuple(cfg)))

    def drop(self, acc: float) -> float:
        return self.reference_acc - acc

    def feasible(self, cfg: Sequence[int]) -> bool:
        return self.drop(self.accuracy(cfg)) <= self.budget_pp + 1e-9

    def is_cached(self, cfg: Sequence[int]) -> bool:
        return tuple(cfg) in self._cache

    @property
    def wall_s(self) -> float:
        return time.perf_counter() - self.t0

    def best_feasible_seen(self) -> Config | None:
        """Lowest-latency feasible config among everything evaluated so far."""
        best, best_lat = None, float("inf")
        for cfg, acc in self._cache.items():
            if self.drop(acc) <= self.budget_pp + 1e-9:
                lat = self.latency(cfg)
                if lat < best_lat - 1e-12:
                    best, best_lat = cfg, lat
        return best


@dataclass
class SearchResult:
    method: str
    config: str
    val_acc: float
    drop_pp: float
    latency: float
    n_evals: int
    wall_s: float
    feasible: bool
    budget_pp: float
    extra: dict = field(default_factory=dict)
    history: list = field(default_factory=list)

    @property
    def cfg(self) -> Config:
        return str_to_cfg(self.config)

    def to_dict(self, with_history: bool = False) -> dict:
        d = asdict(self)
        if not with_history:
            d.pop("history")
        return d


def make_result(method: str, ev: Evaluator, cfg: Config | None, **extra) -> SearchResult:
    """Build a SearchResult; falls back to all-FP16 if a method found nothing feasible."""
    from precisionpilot.precision import all_fp16

    if cfg is None:
        cfg = all_fp16(len(ev.history[0].config) if ev.history else 10)
    acc = ev.accuracy(cfg)
    return SearchResult(
        method=method, config=cfg_to_str(cfg), val_acc=acc, drop_pp=ev.drop(acc),
        latency=ev.latency(cfg), n_evals=ev.n_evals, wall_s=ev.wall_s,
        feasible=ev.drop(acc) <= ev.budget_pp + 1e-9, budget_pp=ev.budget_pp, extra=extra,
        history=[asdict(h) for h in ev.history])
