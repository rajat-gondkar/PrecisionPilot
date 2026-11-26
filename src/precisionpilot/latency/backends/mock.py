"""Deterministic analytic latency model for development / tests on machines without CUDA.

It is *not* a benchmark. It reproduces the qualitative effects the real study looks for, so
the whole pipeline can be exercised end-to-end on a Mac:

* compute-bound time ~ FLOPs / throughput, with INT8 throughput higher than FP16;
* per-kernel launch overhead (dominates at batch size 1, so INT8 helps less there);
* an extra fixed cost for INT8 kernels (Q/DQ, reformatting);
* in full configurations: a fusion discount and a reformat penalty at every FP16<->INT8
  boundary, so the additive lookup-table estimate has a realistic, non-zero error.

Two device profiles mimic an Ampere laptop GPU and a Turing T4.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import numpy as np
import torch
import torch.nn as nn

from precisionpilot.latency.backends.base import LatencyBackend
from precisionpilot.latency.timing import TimingStats
from precisionpilot.precision import INT8

PROFILES = {
    # tflops_fp16 / tflops_int8 effective; overheads in ms; bandwidth GB/s
    "mock-ampere": dict(tflops_fp16=9.0, tflops_int8=16.0, launch_ms=0.010, int8_fixed_ms=0.012,
                        bandwidth_gbs=180.0, fusion=0.93),
    "mock-turing": dict(tflops_fp16=14.0, tflops_int8=30.0, launch_ms=0.014, int8_fixed_ms=0.010,
                        bandwidth_gbs=280.0, fusion=0.91),
}


def _unit_cost_terms(unit: nn.Module, input_shape: Sequence[int]) -> tuple[float, int]:
    """(FLOPs, number of conv/linear kernels) for one unit at ``input_shape``."""
    flops, kernels = 0.0, 0
    hooks = []

    def conv_hook(m, inp, out):
        nonlocal flops, kernels
        k = m.kernel_size[0] * m.kernel_size[1]
        flops += 2.0 * out.numel() * (m.in_channels // m.groups) * k
        kernels += 1

    def lin_hook(m, inp, out):
        nonlocal flops, kernels
        flops += 2.0 * out.numel() * m.in_features
        kernels += 1

    for m in unit.modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(lin_hook))
    p = next(unit.parameters())
    with torch.no_grad():
        unit(torch.zeros(*input_shape, dtype=p.dtype, device=p.device))
    for h in hooks:
        h.remove()
    return flops, kernels


class MockBackend(LatencyBackend):
    name = "mock"

    def __init__(self, profile: str = "mock-ampere", warmup: int = 0, reps: int = 50,
                 cooldown_s: float = 0.0, seed: int = 0):
        super().__init__(warmup, reps, cooldown_s)
        if profile not in PROFILES:
            raise ValueError(f"unknown mock profile {profile}")
        self.profile = profile
        self.p = PROFILES[profile]
        self.seed = seed
        self._terms: dict = {}

    @property
    def device_name(self) -> str:
        return self.profile

    # -- model ---------------------------------------------------------------
    def _terms_for(self, mm, index: int, input_shape: Sequence[int]) -> tuple[float, int]:
        key = (id(mm), index, tuple(input_shape))
        if key not in self._terms:
            self._terms[key] = _unit_cost_terms(mm.fp32_units[index], input_shape)
        return self._terms[key]

    def _unit_ms(self, mm, index: int, precision: int, input_shape: Sequence[int]) -> float:
        flops, kernels = self._terms_for(mm, index, input_shape)
        p = self.p
        if precision == INT8:
            t = flops / (p["tflops_int8"] * 1e9) + kernels * p["launch_ms"] + p["int8_fixed_ms"]
        else:
            t = flops / (p["tflops_fp16"] * 1e9) + kernels * p["launch_ms"]
        return t * self._jitter(index, precision, input_shape)

    def _jitter(self, *key) -> float:
        h = hashlib.sha1(repr((self.profile, self.seed, key)).encode()).digest()
        return 1.0 + (int.from_bytes(h[:4], "little") / 2**32 - 0.5) * 0.04  # +-2 %

    def _samples(self, center_ms: float, *key) -> TimingStats:
        h = int.from_bytes(hashlib.sha1(repr(("s", self.profile, key)).encode()).digest()[:4],
                           "little")
        rng = np.random.default_rng(h)
        samples = center_ms * rng.lognormal(mean=0.0, sigma=0.03, size=self.reps)
        return TimingStats.from_samples(samples)

    # -- interface -----------------------------------------------------------
    def measure_unit(self, mm, index, precision, input_shape) -> TimingStats:
        return self._samples(self._unit_ms(mm, index, precision, input_shape),
                             "unit", index, precision, tuple(input_shape))

    def measure_config(self, mm, cfg, batch_size, image_size) -> TimingStats:
        from precisionpilot.model import UnitResNet, unit_input_shapes

        shapes = self._shapes(mm, batch_size, image_size, UnitResNet, unit_input_shapes)
        total = sum(self._unit_ms(mm, i, b, shapes[i]) for i, b in enumerate(cfg))
        total *= self.p["fusion"]
        prev = 0  # network input arrives as FP16/FP32
        for i, b in enumerate(cfg):
            if b != prev:
                elems = float(np.prod(shapes[i]))
                total += elems * 2 / (self.p["bandwidth_gbs"] * 1e9) * 1e3 + 0.004
            prev = b
        return self._samples(total, "cfg", tuple(cfg), batch_size, image_size)

    def _shapes(self, mm, batch_size, image_size, UnitResNet, unit_input_shapes):
        key = (id(mm), "shapes", batch_size, image_size)
        if key not in self._terms:
            self._terms[key] = unit_input_shapes(UnitResNet(mm.fp32_units, mm.names),
                                                 batch_size, image_size,
                                                 device=next(mm.parameters()).device)
        return self._terms[key]

    def model_info(self, mm, cfg, batch_size, image_size) -> dict:
        n_params = [sum(p.numel() for p in u.parameters()) for u in mm.fp32_units]
        size = sum(n * (1 if b == INT8 else 2) for n, b in zip(n_params, cfg))
        return {"weights_mb": size / 1e6}
