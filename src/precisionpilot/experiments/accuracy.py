"""Hardware-independent accuracy experiments.

* :func:`reference_accuracies` — FP32 / all-FP16 / all-INT8 baselines + cost of one evaluation
* :func:`exhaustive_accuracy_table` — all 1,024 configs (prefix-tree evaluation)
* :func:`run_sensitivity` — Experiment 1
* :func:`final_test` — held-out test accuracy of selected configurations
"""

from __future__ import annotations

import time
from collections.abc import Iterable

import numpy as np
import pandas as pd
import torch

from precisionpilot.data import iterate_batches, preprocess
from precisionpilot.evaluate import accuracy
from precisionpilot.precision import (
    MixedPrecisionModel,
    all_configs,
    all_fp16,
    all_int8,
    cfg_to_str,
    str_to_cfg,
)
from precisionpilot.search.common import Evaluator, ModelAccuracy
from precisionpilot.search.sensitivity import sensitivity_scan
from precisionpilot.utils import load_json, save_json


@torch.inference_mode()
def fp32_accuracy(mm: MixedPrecisionModel, x, y, device, batch_size, image_size) -> float:
    correct = 0
    for xb, yb in iterate_batches(x, y, batch_size):
        logits = mm.fp32_forward(preprocess(xb, device, image_size))
        correct += int((logits.argmax(1).cpu() == yb).sum())
    return 100.0 * correct / max(1, len(y))


def reference_accuracies(ws, log=print) -> dict:
    mm, dev, bs, sz = ws.mm, ws.device, ws.batch_size, ws.image_size
    out: dict = {}
    for split in ("val_subset", "val", "test"):
        x, y = ws.data.get(split)
        out[f"fp32_{split}"] = fp32_accuracy(mm, x, y, dev, bs, sz)
        for name, cfg in (("fp16", all_fp16()), ("int8", all_int8())):
            out[f"{name}_{split}"] = accuracy(mm.set_config(cfg), x, y, dev, bs, sz,
                                              torch.float16)
        out[f"n_{split}"] = int(len(y))
        log(f"{split:>10}: FP32 {out[f'fp32_{split}']:.2f}  FP16 {out[f'fp16_{split}']:.2f}  "
            f"INT8 {out[f'int8_{split}']:.2f}  (n={len(y)})")
    # cost of one accuracy evaluation on the search subset (used to report search time)
    x, y = ws.data.get("val_subset")
    t0 = time.perf_counter()
    reps = 3
    for cfg in [(1, 0) * 5, (0, 1) * 5, all_int8()][:reps]:
        accuracy(mm.set_config(cfg), x, y, dev, bs, sz, torch.float16)
    out["sec_per_eval"] = (time.perf_counter() - t0) / reps
    out["device"] = str(dev)
    out["budget_reference"] = "fp32_val_subset"
    save_json(out, ws.accuracy_dir / "reference.json")
    return out


def load_reference(ws) -> dict:
    return load_json(ws.accuracy_dir / "reference.json")


@torch.inference_mode()
def exhaustive_accuracy_table(mm: MixedPrecisionModel, x: np.ndarray, y: np.ndarray, device,
                              batch_size: int = 128, image_size: int = 224,
                              log=print) -> pd.DataFrame:
    """Accuracy of all 2^N configurations via depth-first traversal of the prefix tree.

    Configurations sharing their first k unit precisions share those k activations, so each
    batch needs 2 + 4 + ... + 2^N unit forwards instead of N * 2^N. Uses the same
    :meth:`MixedPrecisionModel.run_unit` as the normal forward -> identical predictions.
    """
    n = mm.n_units
    correct = np.zeros(2 ** n, dtype=np.int64)
    t0 = time.perf_counter()
    n_batches = int(np.ceil(len(y) / batch_size))

    def dfs(level: int, h: torch.Tensor, code: int, labels: torch.Tensor) -> None:
        if level == n:
            correct[code] += int((h.argmax(1) == labels).sum())
            return
        for b in (0, 1):
            dfs(level + 1, mm.run_unit(level, b, h), (code << 1) | b, labels)

    for bi, (xb, yb) in enumerate(iterate_batches(x, y, batch_size)):
        h0 = preprocess(xb, device, image_size, dtype=torch.float16)
        dfs(0, h0, 0, yb.to(device))
        log(f"exhaustive table: batch {bi + 1}/{n_batches} "
            f"({time.perf_counter() - t0:.0f}s)")
    cfgs = all_configs(n)  # itertools order == binary counting order == `code`
    return pd.DataFrame({"config": [cfg_to_str(c) for c in cfgs],
                         "n_int8": [sum(c) for c in cfgs],
                         "val_acc": 100.0 * correct / len(y)})


def build_accuracy_table(ws, log=print) -> pd.DataFrame:
    x, y = ws.data.get("val_subset")
    t0 = time.perf_counter()
    df = exhaustive_accuracy_table(ws.mm, x, y, ws.device, ws.batch_size, ws.image_size, log)
    df.attrs["seconds"] = time.perf_counter() - t0
    path = ws.accuracy_dir / "accuracy_table.csv"
    df.to_csv(path, index=False)
    log(f"saved {path} ({df.attrs['seconds']:.0f}s)")
    return df


def run_sensitivity(ws, split: str = "val", log=print) -> pd.DataFrame:
    """Experiment 1: per-unit sensitivity on the full validation split."""
    x, y = ws.data.get(split)
    ref = fp32_accuracy(ws.mm, x, y, ws.device, ws.batch_size, ws.image_size)
    oracle = ModelAccuracy(ws.mm, x, y, ws.device, ws.batch_size, ws.image_size)
    ev = Evaluator(oracle, lambda c: 0.0, reference_acc=ref, budget_pp=0.0)
    df = sensitivity_scan(ev, ws.mm.n_units, ws.mm.names)
    df = df.drop(columns=[c for c in df.columns if c.startswith("latency")])
    df["fp32_acc"] = ref
    df["fp16_acc"] = df.attrs["fp16_acc"]
    df["split"] = split
    path = ws.accuracy_dir / "sensitivity.csv"
    df.to_csv(path, index=False)
    for r in df.itertuples():
        log(f"unit {r.unit} {r.name:<9} INT8 -> acc {r.acc:.2f}  drop {r.drop_vs_fp16:+.2f} pp")
    log(f"saved {path}")
    return df


def final_test(ws, configs: Iterable[str], labels: Iterable[str] | None = None,
               log=print) -> pd.DataFrame:
    """Held-out test accuracy (simulated INT8) of selected configurations."""
    x, y = ws.data.get("test")
    ref = fp32_accuracy(ws.mm, x, y, ws.device, ws.batch_size, ws.image_size)
    configs = list(dict.fromkeys(configs))
    labels = list(labels) if labels is not None else [""] * len(configs)
    rows = []
    for c, label in zip(configs, labels):
        acc = accuracy(ws.mm.set_config(str_to_cfg(c)), x, y, ws.device, ws.batch_size,
                       ws.image_size, torch.float16)
        rows.append({"config": c, "label": label, "test_acc": acc, "fp32_test_acc": ref,
                     "test_drop_pp": ref - acc})
        log(f"test {c} {label:<20} {acc:.2f}% (drop {ref - acc:+.2f} pp)")
    df = pd.DataFrame(rows)
    path = ws.accuracy_dir / "final_test.csv"
    if path.exists():  # merge with earlier runs (e.g. other GPUs' chosen configs)
        old = pd.read_csv(path, dtype={"config": str})
        df = pd.concat([old[~old["config"].isin(df["config"])], df], ignore_index=True)
    df.to_csv(path, index=False)
    return df
