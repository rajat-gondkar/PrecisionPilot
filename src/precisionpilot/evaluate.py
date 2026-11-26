"""Top-1 accuracy evaluation for FP32 / FP16 / mixed-precision models."""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn

from precisionpilot.data import iterate_batches, preprocess


@torch.inference_mode()
def predict(model: nn.Module, x: np.ndarray, device: torch.device | str,
            batch_size: int = 128, image_size: int = 224,
            input_dtype: torch.dtype = torch.float32) -> np.ndarray:
    """Return predicted class indices for uint8 NHWC images."""
    model.eval()
    dummy_y = np.zeros(len(x), dtype=np.int64)
    preds = []
    for xb, _ in iterate_batches(x, dummy_y, batch_size):
        inp = preprocess(xb, device, image_size=image_size, dtype=input_dtype)
        logits = model(inp)
        preds.append(logits.argmax(1).cpu())
    return torch.cat(preds).numpy() if preds else np.zeros(0, dtype=np.int64)


def accuracy(model: nn.Module, x: np.ndarray, y: np.ndarray, device: torch.device | str,
             batch_size: int = 128, image_size: int = 224,
             input_dtype: torch.dtype = torch.float32) -> float:
    """Top-1 accuracy in percent."""
    if len(y) == 0:
        return float("nan")
    p = predict(model, x, device, batch_size, image_size, input_dtype)
    return float((p == y).mean() * 100.0)


def as_fp16(model: nn.Module) -> nn.Module:
    """A half-precision copy of ``model`` (the all-FP16 baseline)."""
    return copy.deepcopy(model).half().eval()


def baseline_accuracies(model: nn.Module, data, device, splits=("val", "test"),
                        batch_size: int = 128, image_size: int = 224) -> dict:
    """FP32 and all-FP16 top-1 for the requested splits."""
    out: dict = {}
    m16 = as_fp16(model).to(device)
    for split in splits:
        x, y = data.get(split)
        out[f"fp32_{split}"] = accuracy(model, x, y, device, batch_size, image_size)
        out[f"fp16_{split}"] = accuracy(m16, x, y, device, batch_size, image_size,
                                        input_dtype=torch.float16)
        out[f"n_{split}"] = int(len(y))
    return out
