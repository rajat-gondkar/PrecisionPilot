"""Simulated (fake) INT8 quantization, TensorRT-compatible.

Scheme (matches TensorRT explicit quantization / NVIDIA's PTQ recipe):

* **Weights:** per-output-channel, symmetric, zero point 0, scale = max|w_c| / 127.
* **Activations:** per-tensor, symmetric, zero point 0, scale = amax / 127, where ``amax`` comes
  from calibration (``minmax`` = max |x|, or ``percentile`` = mean over batches of the p-th
  percentile of |x|).
* Integer range [-128, 127] so that the ops export to ONNX ``QuantizeLinear``/``DequantizeLinear``
  with an int8 zero point, which TensorRT consumes directly.

Quantizers sit on the *inputs* of every Conv/Linear (and optionally on the residual input of a
BasicBlock so TensorRT can fuse the add into the INT8 convolution).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

QMIN, QMAX = -128, 127
_MAX_SAMPLES = 1_000_000  # elements used per batch for percentile estimates


def _symmetric_scale(amax: torch.Tensor) -> torch.Tensor:
    return torch.clamp(amax, min=1e-8) / 127.0


def _native_fq_device(x: torch.Tensor) -> bool:
    """CPU / CUDA have fused fake-quant kernels; MPS (Apple) does not."""
    return x.device.type in {"cpu", "cuda"} or torch.onnx.is_in_onnx_export()


def _manual_fake_quant(x: torch.Tensor, scale: torch.Tensor | float) -> torch.Tensor:
    # identical maths: round-half-to-even, clamp to [QMIN, QMAX], dequantize
    return torch.clamp(torch.round(x / scale), QMIN, QMAX) * scale


def fake_quant_per_tensor(x: torch.Tensor, amax: float) -> torch.Tensor:
    scale = float(max(amax, 1e-8) / 127.0)
    if not _native_fq_device(x):
        return _manual_fake_quant(x, scale)
    return torch.fake_quantize_per_tensor_affine(x, scale, 0, QMIN, QMAX)


def weight_scale_per_channel(w: torch.Tensor, axis: int = 0) -> torch.Tensor:
    dims = [d for d in range(w.dim()) if d != axis]
    return _symmetric_scale(w.detach().abs().amax(dim=dims)).float()


def fake_quant_weight_per_channel(w: torch.Tensor, axis: int = 0,
                                  scale: torch.Tensor | None = None) -> torch.Tensor:
    if scale is None:
        scale = weight_scale_per_channel(w, axis)
    if not _native_fq_device(w):
        shape = [1] * w.dim()
        shape[axis] = -1
        return _manual_fake_quant(w, scale.to(w.device).view(shape))
    zp = torch.zeros_like(scale, dtype=torch.int32)
    return torch.fake_quantize_per_channel_affine(w, scale, zp, axis, QMIN, QMAX)


class _WeightQuantMixin:
    """Caches per-channel weight scales as a buffer so ONNX export sees *constant* scales
    (TensorRT rejects Q/DQ scales computed inside the graph)."""

    def prepare_weights(self) -> None:
        self.w_scale = weight_scale_per_channel(self._weight(), axis=0)
        # weights are static: cache their fake-quantized values for fast simulation
        self._wq_cache = fake_quant_weight_per_channel(self._weight().detach(), 0, self.w_scale)

    def _quant_weight(self) -> torch.Tensor:
        w = self._weight()
        cache = getattr(self, "_wq_cache", None)
        if cache is not None and not torch.onnx.is_in_onnx_export():
            if cache.device != w.device or cache.dtype != w.dtype:
                cache = self._wq_cache = cache.to(w.device, w.dtype)
            return cache
        scale = self.w_scale if self.w_scale is not None else None
        if scale is not None and scale.device != w.device:
            scale = scale.to(w.device)
        return fake_quant_weight_per_channel(w, 0, scale)


class ActQuantizer(nn.Module):
    """Per-tensor activation quantizer with a calibration mode.

    Modes: ``"calib"`` (pass-through, collect stats), ``"quant"`` (apply fake quant),
    ``"off"`` (pass-through).
    """

    def __init__(self, method: str = "percentile", percentile: float = 99.99):
        super().__init__()
        if method not in {"minmax", "percentile"}:
            raise ValueError(f"unknown calibration method {method}")
        self.method = method
        self.percentile = percentile
        self.mode = "off"
        self.register_buffer("amax", torch.tensor(0.0))
        self._stats: list[float] = []

    # -- calibration -------------------------------------------------------
    def reset(self) -> None:
        self._stats = []
        self.amax.zero_()

    def _observe(self, x: torch.Tensor) -> None:
        a = x.detach().abs().flatten().float()
        if self.method == "minmax":
            self._stats.append(float(a.max()))
        else:
            if a.numel() > _MAX_SAMPLES:  # deterministic strided subsample
                a = a[:: (a.numel() // _MAX_SAMPLES) + 1]
            self._stats.append(float(torch.quantile(a.cpu(), self.percentile / 100.0)))

    def finalize(self) -> float:
        if not self._stats:
            raise RuntimeError("ActQuantizer.finalize() called without calibration data")
        val = max(self._stats) if self.method == "minmax" else sum(self._stats) / len(self._stats)
        self.amax.fill_(val)
        self._stats = []
        return val

    # -- forward -----------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "calib":
            self._observe(x)
            return x
        if self.mode == "quant":
            return fake_quant_per_tensor(x, float(self.amax))
        return x

    def extra_repr(self) -> str:
        return f"method={self.method}, mode={self.mode}, amax={float(self.amax):.4f}"


class QuantConv2d(_WeightQuantMixin, nn.Module):
    """Conv2d with fake-quantized input (per-tensor) and weight (per-channel)."""

    def __init__(self, conv: nn.Conv2d, method: str = "percentile", percentile: float = 99.99):
        super().__init__()
        self.conv = conv
        self.input_quant = ActQuantizer(method, percentile)
        self.weight_quant_enabled = True
        self.register_buffer("w_scale", None)

    def _weight(self) -> torch.Tensor:
        return self.conv.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_quant(x)
        w = self.conv.weight
        if self.weight_quant_enabled and self.input_quant.mode == "quant":
            w = self._quant_weight()
        c = self.conv
        return F.conv2d(x, w, c.bias, c.stride, c.padding, c.dilation, c.groups)


class QuantLinear(_WeightQuantMixin, nn.Module):
    def __init__(self, fc: nn.Linear, method: str = "percentile", percentile: float = 99.99):
        super().__init__()
        self.fc = fc
        self.input_quant = ActQuantizer(method, percentile)
        self.weight_quant_enabled = True
        self.register_buffer("w_scale", None)

    def _weight(self) -> torch.Tensor:
        return self.fc.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_quant(x)
        w = self.fc.weight
        if self.weight_quant_enabled and self.input_quant.mode == "quant":
            w = self._quant_weight()
        return F.linear(x, w, self.fc.bias)


def quantizers(module: nn.Module) -> list[ActQuantizer]:
    return [m for m in module.modules() if isinstance(m, ActQuantizer)]


def prepare_weights(module: nn.Module) -> None:
    for m in module.modules():
        if isinstance(m, _WeightQuantMixin):
            m.prepare_weights()


def set_quant_mode(module: nn.Module, mode: str) -> None:
    for q in quantizers(module):
        q.mode = mode
