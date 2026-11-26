"""Timing protocol (outline §6.6): warm-up, repeated measurements, median and p95."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class TimingStats:
    median_ms: float
    p95_ms: float
    mean_ms: float
    std_ms: float
    min_ms: float
    n: int

    @classmethod
    def from_samples(cls, samples_ms: Sequence[float]) -> TimingStats:
        a = np.asarray(samples_ms, dtype=np.float64)
        if a.size == 0:
            raise ValueError("no timing samples")
        return cls(median_ms=float(np.median(a)), p95_ms=float(np.percentile(a, 95)),
                   mean_ms=float(a.mean()), std_ms=float(a.std()), min_ms=float(a.min()),
                   n=int(a.size))

    def throughput(self, batch_size: int) -> float:
        """Images per second at the median latency."""
        return batch_size / (self.median_ms / 1000.0) if self.median_ms > 0 else float("inf")

    def to_dict(self) -> dict:
        return asdict(self)


def time_callable(fn: Callable[[], None], warmup: int = 20, reps: int = 100,
                  sync: Callable[[], None] | None = None) -> TimingStats:
    """Wall-clock timing of ``fn`` with an optional device synchronisation."""
    for _ in range(warmup):
        fn()
    if sync:
        sync()
    samples = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        if sync:
            sync()
        samples.append((time.perf_counter() - t0) * 1000.0)
    return TimingStats.from_samples(samples)


def time_cuda(fn: Callable[[], None], warmup: int = 20, reps: int = 100) -> TimingStats:
    """GPU timing with CUDA events (one event pair per repetition)."""
    import torch

    stream = torch.cuda.current_stream()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
    for s, e in zip(starts, ends):
        s.record(stream)
        fn()
        e.record(stream)
    torch.cuda.synchronize()
    return TimingStats.from_samples([s.elapsed_time(e) for s, e in zip(starts, ends)])


def cooldown(seconds: float) -> None:
    """Pause between measurements to limit thermal throttling on laptops."""
    if seconds > 0:
        time.sleep(seconds)
