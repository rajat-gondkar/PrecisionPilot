"""Context-aware latency model (fixes the non-additivity of the per-unit lookup table).

Why the plain lookup table fails
--------------------------------
A unit timed *alone* pays an FP16<->INT8 conversion at both ends. In a full engine, neighbouring
INT8 units keep the tensor in INT8 and fuse, so the table's sum overestimates INT8-heavy
configurations (up to +107 % on the T4, see DOCUMENTATION.md §7.7).

The model
---------
For a configuration ``c`` (bit ``c_u`` = 1 for INT8) with virtual FP16 neighbours before unit 0 and
after the last unit::

    L(c) = sum_u [ (1 - c_u) * f_u + c_u * g_u ]  +  sum_k  [ c_{k-1} != c_k ] * h_k

* ``f_u`` / ``g_u``: cost of unit ``u`` in FP16 / INT8 *inside a full engine*,
* ``h_k``: cost of an FP16<->INT8 boundary at position ``k`` (``k = 0 .. N``; 0 = network input,
  ``N`` = network output). Runs of INT8 units therefore pay conversions only at their ends.

``3N + 1`` parameters are fitted by ridge regression to a modest set of *full-engine* measurements
(about 60 configurations per batch size: all-FP16, all-INT8, every single-unit flip in both
directions, INT8 prefixes/suffixes and some random ones). The ridge prior is the isolated-unit
lookup table (``f``, ``g``) with zero boundary costs, so the fit degrades gracefully to the
additive table when data is scarce.

It is a model of *latency only*; accuracy still comes from real evaluations.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from precisionpilot.latency.lut import LatencyTable
from precisionpilot.precision import FP16, INT8, Config, cfg_to_str, str_to_cfg


def features(cfg: Sequence[int]) -> np.ndarray:
    """Feature vector of length 3N+1: [FP16 indicators (N), INT8 indicators (N), boundaries (N+1)]."""
    c = np.asarray(cfg, dtype=np.float64)
    seq = np.concatenate([[0.0], c, [0.0]])
    boundary = (seq[:-1] != seq[1:]).astype(np.float64)
    return np.concatenate([1.0 - c, c, boundary])


def training_configs(n_units: int = 10, n_random: int = 20, seed: int = 0) -> list[Config]:
    """Configurations to measure for fitting: informative, deduplicated, deterministic."""
    cfgs: list[Config] = [(FP16,) * n_units, (INT8,) * n_units]
    for u in range(n_units):  # single flips, both directions
        cfgs.append(tuple(INT8 if j == u else FP16 for j in range(n_units)))
        cfgs.append(tuple(FP16 if j == u else INT8 for j in range(n_units)))
    for k in range(1, n_units):  # INT8 prefixes / suffixes (contiguous runs)
        cfgs.append(tuple(INT8 if j < k else FP16 for j in range(n_units)))
        cfgs.append(tuple(INT8 if j >= k else FP16 for j in range(n_units)))
    rng = np.random.default_rng(seed)
    seen = set(cfgs)
    while n_random > 0:
        c = tuple(int(b) for b in rng.integers(0, 2, n_units))
        if c not in seen:
            cfgs.append(c)
            seen.add(c)
            n_random -= 1
    return list(dict.fromkeys(cfgs))


class ContextLatencyModel:
    """Fitted parameters for one batch size."""

    def __init__(self, theta: np.ndarray, n_units: int, batch_size: int):
        if len(theta) != 3 * n_units + 1:
            raise ValueError("theta has the wrong length")
        self.theta = np.asarray(theta, dtype=np.float64)
        self.n_units = n_units
        self.batch_size = batch_size

    @classmethod
    def fit(cls, cfgs: Sequence[Sequence[int]], latencies_ms: Sequence[float],
            prior: np.ndarray | None, batch_size: int, ridge: float = 1e-3) -> ContextLatencyModel:
        """Ridge fit: minimise ||X theta - y||^2 + lam ||theta - prior||^2.

        ``ridge`` is relative to the average diagonal of X^T X, so it is scale free.
        """
        n_units = len(cfgs[0])
        x = np.stack([features(c) for c in cfgs])
        y = np.asarray(latencies_ms, dtype=np.float64)
        p = x.shape[1]
        prior = np.zeros(p) if prior is None else np.asarray(prior, dtype=np.float64)
        gram = x.T @ x
        lam = ridge * float(np.trace(gram)) / p
        theta = np.linalg.solve(gram + lam * np.eye(p), x.T @ y + lam * prior)
        return cls(theta, n_units, batch_size)

    def estimate(self, cfg: Sequence[int]) -> float:
        return float(features(cfg) @ self.theta)

    @property
    def fp16_ms(self) -> np.ndarray:
        return self.theta[: self.n_units]

    @property
    def int8_ms(self) -> np.ndarray:
        return self.theta[self.n_units: 2 * self.n_units]

    @property
    def boundary_ms(self) -> np.ndarray:
        return self.theta[2 * self.n_units:]


class ContextLatencyTable:
    """One :class:`ContextLatencyModel` per batch size, with the LatencyTable-like interface."""

    def __init__(self, models: dict[int, ContextLatencyModel], gpu: str = "", backend: str = ""):
        self.models = dict(models)
        self.gpu, self.backend = gpu, backend

    @property
    def batch_sizes(self) -> list[int]:
        return sorted(self.models)

    @property
    def n_units(self) -> int:
        return next(iter(self.models.values())).n_units

    def estimate(self, cfg: Sequence[int], batch_size: int, stat: str = "median_ms") -> float:
        return self.models[batch_size].estimate(cfg)

    def latency_fn(self, batch_size: int, stat: str = "median_ms") -> Callable[[Config], float]:
        model = self.models[batch_size]
        cache: dict = {}

        def fn(cfg: Config) -> float:
            key = tuple(cfg)
            v = cache.get(key)
            if v is None:
                v = cache[key] = model.estimate(key)
            return v

        return fn

    # -- fitting from measurements ------------------------------------------
    @classmethod
    def from_measurements(cls, lut: LatencyTable, train: pd.DataFrame, ridge: float = 1e-3,
                          ) -> ContextLatencyTable:
        """``train``: columns config, batch_size, median_ms (full-engine measurements)."""
        models = {}
        for bs, grp in train.groupby("batch_size"):
            bs = int(bs)
            n = lut.n_units
            prior = np.concatenate([[lut.unit_ms(bs, u, FP16) for u in range(n)],
                                    [lut.unit_ms(bs, u, INT8) for u in range(n)],
                                    np.zeros(n + 1)])
            cfgs = [str_to_cfg(c) for c in grp["config"]]
            models[bs] = ContextLatencyModel.fit(cfgs, grp["median_ms"].to_numpy(), prior, bs,
                                                 ridge)
        return cls(models, lut.gpu, lut.backend)

    # -- io -----------------------------------------------------------------
    def to_frame(self) -> pd.DataFrame:
        rows = []
        for bs, m in self.models.items():
            n = m.n_units
            names = ([f"fp16_u{u}" for u in range(n)] + [f"int8_u{u}" for u in range(n)]
                     + [f"boundary_{k}" for k in range(n + 1)])
            rows += [{"batch_size": bs, "param": nm, "value_ms": float(v)}
                     for nm, v in zip(names, m.theta)]
        return pd.DataFrame(rows)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.to_frame().to_csv(path, index=False)
        return path

    @classmethod
    def load(cls, path: str | Path) -> ContextLatencyTable:
        df = pd.read_csv(path)
        models = {}
        for bs, grp in df.groupby("batch_size"):
            theta = grp["value_ms"].to_numpy()
            models[int(bs)] = ContextLatencyModel(theta, (len(theta) - 1) // 3, int(bs))
        return cls(models)


def measure_training_set(mm, backend, batch_sizes: Iterable[int], image_size: int,
                         n_random: int = 20, seed: int = 0, log=print) -> pd.DataFrame:
    """Full-engine latency of the training configurations, for every batch size."""
    rows = []
    cfgs = training_configs(mm.n_units, n_random, seed)
    for bs in batch_sizes:
        for i, cfg in enumerate(cfgs):
            st = backend.measure_config(mm, cfg, bs, image_size)
            rows.append({"gpu": backend.device_name, "backend": backend.name,
                         "batch_size": bs, "config": cfg_to_str(cfg), **st.to_dict()})
            if i % 10 == 0:
                log(f"context training: bs={bs} {i + 1}/{len(cfgs)} "
                    f"{cfg_to_str(cfg)} {st.median_ms:.3f} ms")
    return pd.DataFrame(rows)
