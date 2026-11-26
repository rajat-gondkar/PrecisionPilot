"""Plain PyTorch CUDA timing (sanity check for FP16; INT8 here is *simulated*, not real).

PyTorch has no fast INT8 convolution path on CUDA, so INT8 timings from this backend measure
the fake-quant simulation and must not be reported as INT8 latency. Use it to sanity-check the
FP16 numbers of the TensorRT backend and the timing harness itself.
"""

from __future__ import annotations

import warnings

import torch

from precisionpilot.latency.backends.base import LatencyBackend
from precisionpilot.latency.timing import TimingStats, cooldown, time_cuda
from precisionpilot.precision import INT8


class TorchCudaBackend(LatencyBackend):
    name = "torch"

    def __init__(self, warmup: int = 20, reps: int = 100, cooldown_s: float = 0.0):
        super().__init__(warmup, reps, cooldown_s)
        if not torch.cuda.is_available():
            raise RuntimeError("torch backend needs a CUDA GPU (use 'mock' on this machine)")
        torch.backends.cudnn.benchmark = True
        self._warned = False

    @property
    def device_name(self) -> str:
        return torch.cuda.get_device_name(0)

    def _warn_int8(self):
        if not self._warned:
            warnings.warn("torch backend: INT8 latency is simulated (fake-quant), not real.",
                          stacklevel=3)
            self._warned = True

    @torch.inference_mode()
    def measure_unit(self, mm, index, precision, input_shape) -> TimingStats:
        mm = mm.cuda()
        if precision == INT8:
            self._warn_int8()
            unit, dtype = mm.int8_units[index], torch.float32
        else:
            unit, dtype = mm.fp16_units[index], torch.float16
        x = torch.randn(*input_shape, device="cuda", dtype=dtype)
        stats = time_cuda(lambda: unit(x), self.warmup, self.reps)
        cooldown(self.cooldown_s)
        return stats

    @torch.inference_mode()
    def measure_config(self, mm, cfg, batch_size, image_size) -> TimingStats:
        if INT8 in cfg:
            self._warn_int8()
        mm = mm.cuda().set_config(cfg)
        x = torch.randn(batch_size, 3, image_size, image_size, device="cuda",
                        dtype=torch.float16)
        stats = time_cuda(lambda: mm(x), self.warmup, self.reps)
        cooldown(self.cooldown_s)
        return stats
