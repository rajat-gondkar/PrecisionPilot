"""Precision configurations and the switchable mixed-precision model.

A *configuration* is a tuple of 10 ints, one per unit: ``0 = FP16``, ``1 = INT8``.
It is written as a bit-string, unit 0 (stem) first, e.g. ``"0111111110"``.

:class:`MixedPrecisionModel` pre-builds, for every unit, an FP16 copy and a calibrated
fake-INT8 copy, so switching between all 1,024 configurations is free (no rebuilding).

Numerics in ``sim`` mode (accuracy simulation):
  * FP16 unit: half-precision weights and activations.
  * INT8 unit: input cast to FP32, fake-quantized conv/linear in FP32, output cast to FP16.
    This mirrors TensorRT, where INT8 kernels accumulate in INT32 and hand FP16 tensors to
    neighbouring FP16 layers.

In ``export`` mode the FP16 slots use FP32 weights (TensorRT's FP16 flag lowers them) and INT8
slots carry Q/DQ nodes, giving the explicit-quantization ONNX graph TensorRT expects.
"""

from __future__ import annotations

import copy
import itertools
from collections.abc import Iterable, Sequence

import numpy as np
import torch
import torch.nn as nn
from torchvision.models.resnet import BasicBlock

from precisionpilot.data import iterate_batches, preprocess
from precisionpilot.model import NUM_UNITS, UNIT_NAMES, Head, UnitResNet
from precisionpilot.quant.fakequant import (
    ActQuantizer,
    QuantConv2d,
    QuantLinear,
    prepare_weights,
    quantizers,
    set_quant_mode,
)
from precisionpilot.quant.fold import fold_bn

FP16, INT8 = 0, 1
Config = tuple[int, ...]


# --------------------------------------------------------------------------- config helpers
def cfg_to_str(cfg: Sequence[int]) -> str:
    return "".join(str(int(b)) for b in cfg)


def str_to_cfg(s: str) -> Config:
    if any(c not in "01" for c in s):
        raise ValueError(f"invalid config string {s!r}")
    return tuple(int(c) for c in s)


def cfg_to_int(cfg: Sequence[int]) -> int:
    return int(cfg_to_str(cfg), 2)


def int_to_cfg(i: int, n: int = NUM_UNITS) -> Config:
    if not 0 <= i < 2 ** n:
        raise ValueError("config index out of range")
    return str_to_cfg(format(i, f"0{n}b"))


def all_configs(n: int = NUM_UNITS) -> list[Config]:
    return [tuple(c) for c in itertools.product((0, 1), repeat=n)]


def all_fp16(n: int = NUM_UNITS) -> Config:
    return (FP16,) * n


def all_int8(n: int = NUM_UNITS) -> Config:
    return (INT8,) * n


def one_hot(i: int, n: int = NUM_UNITS) -> Config:
    return tuple(INT8 if j == i else FP16 for j in range(n))


def describe(cfg: Sequence[int], names: Sequence[str] = UNIT_NAMES) -> str:
    int8 = [n for n, b in zip(names, cfg) if b == INT8]
    return f"INT8[{', '.join(int8)}]" if int8 else "all-FP16"


# --------------------------------------------------------------------------- quant units
class QuantBasicBlock(nn.Module):
    """BasicBlock (BN already folded) with an optional quantizer on the residual branch."""

    def __init__(self, block: BasicBlock, method: str, percentile: float,
                 quantize_residual: bool = True):
        super().__init__()
        self.conv1 = QuantConv2d(block.conv1, method, percentile)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = QuantConv2d(block.conv2, method, percentile)
        self.downsample = None
        if block.downsample is not None:
            conv = next(m for m in block.downsample if isinstance(m, nn.Conv2d))
            self.downsample = QuantConv2d(conv, method, percentile)
        self.residual_quant = ActQuantizer(method, percentile) if quantize_residual else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        identity = self.downsample(x) if self.downsample is not None else x
        if self.residual_quant is not None:
            identity = self.residual_quant(identity)
        return self.relu(out + identity)


def quantize_unit(unit: nn.Module, method: str = "percentile", percentile: float = 99.99,
                  quantize_residual: bool = True) -> nn.Module:
    """Return a fake-INT8 copy of a (BN-folded) unit."""
    unit = copy.deepcopy(unit)
    if isinstance(unit, BasicBlock):
        return QuantBasicBlock(unit, method, percentile, quantize_residual)
    if isinstance(unit, Head):
        unit.fc = QuantLinear(unit.fc, method, percentile)
        return unit
    if isinstance(unit, nn.Sequential):
        for name, m in list(unit._modules.items()):
            if isinstance(m, nn.Conv2d):
                unit._modules[name] = QuantConv2d(m, method, percentile)
        return unit
    raise TypeError(f"don't know how to quantize unit of type {type(unit).__name__}")


# --------------------------------------------------------------------------- mixed model
class _Cast(nn.Module):
    def __init__(self, inner: nn.Module, in_dtype: torch.dtype, out_dtype: torch.dtype):
        super().__init__()
        self.inner = inner
        self.in_dtype = in_dtype
        self.out_dtype = out_dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.inner(x.to(self.in_dtype)).to(self.out_dtype)


@torch.no_grad()
def _round_to_half(module: nn.Module) -> nn.Module:
    for prm in module.parameters():
        prm.copy_(prm.half().float())
    return module


class MixedPrecisionModel(nn.Module):
    """Switchable FP16 / fake-INT8 ResNet built from a UnitResNet.

    Call :meth:`calibrate` once, then :meth:`set_config` for any configuration.
    """

    def __init__(self, model: UnitResNet, method: str = "percentile",
                 percentile: float = 99.99, quantize_residual: bool = True,
                 mode: str = "sim", fp16_impl: str = "native"):
        super().__init__()
        if mode not in {"sim", "export"}:
            raise ValueError("mode must be 'sim' or 'export'")
        if fp16_impl not in {"native", "emulate"}:
            raise ValueError("fp16_impl must be 'native' or 'emulate'")
        self.mode = mode
        # "emulate": FP32 compute with FP16-rounded weights and unit outputs. Only used on
        # CPU (smoke runs / tests), where native half-precision convolutions are very slow.
        self.fp16_impl = fp16_impl
        self.names = model.names
        folded = fold_bn(model)
        self.fp32_units = nn.ModuleList(folded.units)
        self.int8_units = nn.ModuleList(
            [quantize_unit(u, method, percentile, quantize_residual) for u in folded.units])
        if mode == "sim" and fp16_impl == "native":
            self.fp16_units = nn.ModuleList([copy.deepcopy(u).half() for u in folded.units])
        elif mode == "sim":
            self.fp16_units = nn.ModuleList([_round_to_half(copy.deepcopy(u))
                                             for u in folded.units])
        else:
            self.fp16_units = self.fp32_units
        self.config: Config = all_fp16(len(self.names))
        self.calibrated = False

    @property
    def n_units(self) -> int:
        return len(self.names)

    @property
    def io_dtype(self) -> torch.dtype:
        return torch.float16 if self.mode == "sim" else torch.float32

    def set_config(self, cfg: Sequence[int]) -> MixedPrecisionModel:
        cfg = tuple(int(b) for b in cfg)
        if len(cfg) != self.n_units or any(b not in (FP16, INT8) for b in cfg):
            raise ValueError(f"invalid config {cfg}")
        if INT8 in cfg and not self.calibrated:
            raise RuntimeError("calibrate() before using INT8 units")
        self.config = cfg
        return self

    def unit_module(self, i: int, precision: int) -> nn.Module:
        return self.int8_units[i] if precision == INT8 else self.fp16_units[i]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "export":
            for i, b in enumerate(self.config):
                x = self.unit_module(i, b)(x)
            return x
        x = x.to(torch.float16)
        for i, b in enumerate(self.config):
            x = self.run_unit(i, b, x)
        return x

    def run_unit(self, i: int, precision: int, x: torch.Tensor) -> torch.Tensor:
        """Run unit ``i`` at ``precision`` in sim mode (FP16 tensors in and out).

        Single code path shared by :meth:`forward` and the prefix-tree exhaustive evaluator,
        so both produce bit-identical results.
        """
        if precision == INT8:
            return self.int8_units[i](x.float()).half()
        if self.fp16_impl == "emulate":
            return self.fp16_units[i](x.float()).half()
        return self.fp16_units[i](x)

    def fp32_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reference FP32 forward through the BN-folded units."""
        x = x.float()
        for u in self.fp32_units:
            x = u(x)
        return x

    # -- calibration ---------------------------------------------------------
    @torch.inference_mode()
    def calibrate(self, x: np.ndarray, device, batch_size: int = 128, image_size: int = 224,
                  max_batches: int | None = None) -> dict[str, list[float]]:
        """Collect activation ranges for every INT8 unit from the FP32 activations.

        Ranges are configuration-independent: they are measured on the FP32 graph (the same
        thing TensorRT calibration does), so one calibration serves all 1,024 configs.
        """
        for u in self.int8_units:
            for q in quantizers(u):
                q.reset()
            set_quant_mode(u, "calib")
        dummy = np.zeros(len(x), dtype=np.int64)
        for bi, (xb, _) in enumerate(iterate_batches(x, dummy, batch_size)):
            if max_batches is not None and bi >= max_batches:
                break
            h = preprocess(xb, device, image_size)
            for fu, qu in zip(self.fp32_units, self.int8_units):
                qu(h)          # observe (pass-through in calib mode)
                h = fu(h)
        amax = {}
        for name, u in zip(self.names, self.int8_units):
            amax[name] = [q.finalize() for q in quantizers(u)]
            prepare_weights(u)
            set_quant_mode(u, "quant")
        self.calibrated = True
        return amax

    def export_copy(self) -> MixedPrecisionModel:
        """An ``export``-mode twin sharing calibration (FP32 weights + Q/DQ in INT8 slots)."""
        twin = copy.copy(self)
        nn.Module.__init__(twin)
        twin.mode = "export"
        twin.names = self.names
        twin.fp32_units = self.fp32_units
        twin.int8_units = self.int8_units
        twin.fp16_units = self.fp32_units
        twin.config = self.config
        twin.calibrated = self.calibrated
        return twin


def build_mixed_model(model: UnitResNet, cfg, calib_x: np.ndarray, device,
                      max_batches: int | None = None) -> MixedPrecisionModel:
    """Convenience: build + calibrate a MixedPrecisionModel from a Config object."""
    q = cfg.quant
    impl = "emulate" if torch.device(device).type == "cpu" else "native"
    mm = MixedPrecisionModel(model.cpu(), q.act_method, q.act_percentile,
                             fp16_impl=impl).to(device)
    mm.calibrate(calib_x, device, cfg.data.eval_batch_size, cfg.data.image_size,
                 max_batches=max_batches if max_batches is not None else q.calib_batches)
    return mm.eval()


def count_int8(cfg: Iterable[int]) -> int:
    return int(sum(cfg))
