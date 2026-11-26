"""Latency backend interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from precisionpilot.latency.timing import TimingStats


class LatencyBackend(ABC):
    """Measures latency of single units and of full configurations.

    ``mm`` is a calibrated :class:`~precisionpilot.precision.MixedPrecisionModel`.
    """

    name: str = "base"
    notes: str = ""   # free-text backend details recorded in the LUT (library version, I/O mode)

    def __init__(self, warmup: int = 20, reps: int = 100, cooldown_s: float = 0.0):
        self.warmup = warmup
        self.reps = reps
        self.cooldown_s = cooldown_s

    @property
    @abstractmethod
    def device_name(self) -> str: ...

    @abstractmethod
    def measure_unit(self, mm, index: int, precision: int,
                     input_shape: Sequence[int]) -> TimingStats: ...

    @abstractmethod
    def measure_config(self, mm, cfg: Sequence[int], batch_size: int,
                       image_size: int) -> TimingStats: ...

    def model_info(self, mm, cfg: Sequence[int], batch_size: int,
                   image_size: int) -> dict:
        """Optional extra metrics (engine size, device memory). Default: none."""
        return {}


def get_backend(name: str, **kwargs) -> LatencyBackend:
    name = name.lower()
    if name == "mock":
        from precisionpilot.latency.backends.mock import MockBackend

        return MockBackend(**kwargs)
    if name == "torch":
        from precisionpilot.latency.backends.torch_cuda import TorchCudaBackend

        return TorchCudaBackend(**kwargs)
    if name in {"trt", "tensorrt"}:
        from precisionpilot.latency.backends.tensorrt import TensorRTBackend

        return TensorRTBackend(**kwargs)
    raise ValueError(f"unknown latency backend '{name}' (mock | torch | trt)")
