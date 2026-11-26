"""ONNX export of full mixed-precision configurations and of individual units.

INT8 units are exported with ``QuantizeLinear``/``DequantizeLinear`` (Q/DQ) pairs on their
conv/linear inputs and weights; FP16 units are exported as plain FP32 graphs. Building the
TensorRT engine with both the FP16 and INT8 builder flags then yields exactly the requested
per-unit precisions (TensorRT *explicit quantization*).
"""

from __future__ import annotations

import copy
import warnings
from collections.abc import Sequence
from pathlib import Path

import torch
import torch.nn as nn

from precisionpilot.precision import INT8, MixedPrecisionModel, cfg_to_str

OPSET = 17


class _UnitWrapper(nn.Module):
    def __init__(self, unit: nn.Module):
        super().__init__()
        self.unit = unit

    def forward(self, x):
        return self.unit(x)


def _export(module: nn.Module, example: torch.Tensor, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Export a COPY on the CPU: `.cpu()` moves modules in place, which would pull the live
    # (GPU-resident) units off the GPU and break every later measurement / evaluation.
    module = copy.deepcopy(module).eval().cpu()
    example = example.cpu().float()
    with warnings.catch_warnings(), torch.no_grad():
        warnings.simplefilter("ignore")
        kwargs = dict(input_names=["input"], output_names=["output"], opset_version=OPSET,
                      do_constant_folding=True)
        try:
            torch.onnx.export(module, (example,), str(path), dynamo=False, **kwargs)
        except TypeError:  # older torch without the ``dynamo`` argument
            torch.onnx.export(module, (example,), str(path), **kwargs)
    return path


def export_config(mm: MixedPrecisionModel, cfg: Sequence[int], batch_size: int,
                  image_size: int, out_dir: str | Path) -> Path:
    """Export the full model for ``cfg`` (static batch size) to ``out_dir``."""
    twin = mm.export_copy().set_config(cfg)
    example = torch.zeros(batch_size, 3, image_size, image_size)
    return _export(twin, example, Path(out_dir) / f"full_{cfg_to_str(cfg)}_bs{batch_size}.onnx")


def export_unit(mm: MixedPrecisionModel, index: int, precision: int,
                input_shape: Sequence[int], out_dir: str | Path) -> Path:
    """Export one unit alone at the given precision and input shape."""
    unit = mm.int8_units[index] if precision == INT8 else mm.fp32_units[index]
    tag = "int8" if precision == INT8 else "fp16"
    example = torch.zeros(*input_shape)
    name = f"unit{index:02d}_{mm.names[index]}_{tag}_bs{input_shape[0]}.onnx"
    return _export(_UnitWrapper(unit), example, Path(out_dir) / name)


def count_qdq(path: str | Path) -> dict[str, int]:
    import onnx

    model = onnx.load(str(path))
    ops = [n.op_type for n in model.graph.node]
    return {"QuantizeLinear": ops.count("QuantizeLinear"),
            "DequantizeLinear": ops.count("DequantizeLinear"),
            "Conv": ops.count("Conv"), "Gemm": ops.count("Gemm") + ops.count("MatMul")}


def has_qdq(path: str | Path) -> bool:
    """True if the ONNX graph contains any Q/DQ node (i.e. needs the TensorRT INT8 flag)."""
    import onnx

    ops = {n.op_type for n in onnx.load(str(path)).graph.node}
    return "QuantizeLinear" in ops or "DequantizeLinear" in ops
