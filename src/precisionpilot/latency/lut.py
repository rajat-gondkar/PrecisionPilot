"""Per-unit latency lookup table (outline §6.3).

1. Measure every unit alone in FP16 and INT8, for each batch size, on one GPU.
2. Estimate a configuration's latency as the sum of its units' median latencies.
3. Verify chosen configurations with real full-model runs and report the estimation error
   (per-unit times do not add up exactly because of cross-unit layer fusion and the
   reformatting cost at FP16<->INT8 boundaries).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from pathlib import Path

import pandas as pd

from precisionpilot.model import UnitResNet, unit_input_shapes
from precisionpilot.precision import FP16, INT8, Config, cfg_to_str

PREC_NAME = {FP16: "fp16", INT8: "int8"}
PREC_CODE = {v: k for k, v in PREC_NAME.items()}


class LatencyTable:
    """Rows: (batch_size, unit, precision) -> timing stats in milliseconds."""

    def __init__(self, df: pd.DataFrame, gpu: str = "", backend: str = ""):
        required = {"batch_size", "unit", "precision", "median_ms", "p95_ms"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"latency table missing columns {missing}")
        self.df = df.reset_index(drop=True)
        self.gpu = gpu or (str(df["gpu"].iloc[0]) if "gpu" in df and len(df) else "")
        self.backend = backend or (str(df["backend"].iloc[0]) if "backend" in df and len(df)
                                   else "")
        self._index = {(int(r.batch_size), int(r.unit), str(r.precision)): r
                       for r in self.df.itertuples()}

    # -- io ------------------------------------------------------------------
    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.df.to_csv(path, index=False)
        return path

    @classmethod
    def load(cls, path: str | Path) -> LatencyTable:
        return cls(pd.read_csv(path))

    # -- queries -------------------------------------------------------------
    @property
    def batch_sizes(self) -> list[int]:
        return sorted(self.df["batch_size"].unique().tolist())

    @property
    def n_units(self) -> int:
        return int(self.df["unit"].max()) + 1

    def unit_ms(self, batch_size: int, unit: int, precision: int,
                stat: str = "median_ms") -> float:
        return float(getattr(self._index[(batch_size, unit, PREC_NAME[precision])], stat))

    def estimate(self, cfg: Sequence[int], batch_size: int, stat: str = "median_ms") -> float:
        return sum(self.unit_ms(batch_size, i, b, stat) for i, b in enumerate(cfg))

    def latency_fn(self, batch_size: int, stat: str = "median_ms") -> Callable[[Config], float]:
        """Fast estimator: base FP16 latency + sum of per-unit INT8 deltas."""
        n = self.n_units
        fp16 = [self.unit_ms(batch_size, i, FP16, stat) for i in range(n)]
        delta = [self.unit_ms(batch_size, i, INT8, stat) - fp16[i] for i in range(n)]
        base = sum(fp16)
        cache: dict = {}

        def fn(cfg: Config) -> float:
            key = tuple(cfg)
            v = cache.get(key)
            if v is None:
                v = cache[key] = base + sum(d for d, b in zip(delta, key) if b)
            return v

        return fn

    def savings(self, batch_size: int) -> list[float]:
        """FP16 - INT8 latency per unit (positive = INT8 faster)."""
        return [self.unit_ms(batch_size, i, FP16) - self.unit_ms(batch_size, i, INT8)
                for i in range(self.n_units)]


def build_lut(mm, backend, batch_sizes: Iterable[int], image_size: int = 224,
              log=print) -> LatencyTable:
    shapes_model = UnitResNet(mm.fp32_units, mm.names)
    device = next(mm.parameters()).device
    rows = []
    for bs in batch_sizes:
        shapes = unit_input_shapes(shapes_model, bs, image_size, device=device)
        for i, name in enumerate(mm.names):
            for prec in (FP16, INT8):
                st = backend.measure_unit(mm, i, prec, shapes[i])
                rows.append({"gpu": backend.device_name, "backend": backend.name,
                             "notes": backend.notes, "batch_size": bs, "unit": i, "name": name,
                             "precision": PREC_NAME[prec], "input_shape": "x".join(
                                 map(str, shapes[i])), **st.to_dict()})
                log(f"bs={bs:<3} unit {i} {name:<9} {PREC_NAME[prec]}: "
                    f"median {st.median_ms:.4f} ms  p95 {st.p95_ms:.4f} ms")
    return LatencyTable(pd.DataFrame(rows), backend.device_name, backend.name)


def verify_configs(mm, backend, table: LatencyTable, cfgs: Iterable[Sequence[int]],
                   batch_size: int, image_size: int = 224, labels: Iterable[str] | None = None,
                   log=print) -> pd.DataFrame:
    """Measure full-model latency for configs and compare with the LUT estimate."""
    cfgs = [tuple(c) for c in cfgs]
    labels = list(labels) if labels is not None else [""] * len(cfgs)
    rows = []
    for cfg, label in zip(cfgs, labels):
        st = backend.measure_config(mm, cfg, batch_size, image_size)
        est = table.estimate(cfg, batch_size)
        info = backend.model_info(mm, cfg, batch_size, image_size)
        row = {"gpu": backend.device_name, "backend": backend.name, "label": label,
               "config": cfg_to_str(cfg), "batch_size": batch_size, "estimated_ms": est,
               "measured_median_ms": st.median_ms, "measured_p95_ms": st.p95_ms,
               "throughput_ips": st.throughput(batch_size),
               "error_pct": 100.0 * (est - st.median_ms) / st.median_ms, **info}
        rows.append(row)
        log(f"bs={batch_size:<3} {cfg_to_str(cfg)} {label:<14} est {est:.3f} ms  "
            f"real {st.median_ms:.3f} ms  err {row['error_pct']:+.1f}%")
    return pd.DataFrame(rows)
