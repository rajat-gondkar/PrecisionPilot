"""Experiment 6: lookup-table estimate vs real full-model latency (and real INT8 accuracy)."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from precisionpilot.experiments.search_suite import chosen_configs
from precisionpilot.latency.lut import LatencyTable, verify_configs
from precisionpilot.precision import str_to_cfg


def run_verify(ws, backend, lut: LatencyTable, search_results: pd.DataFrame,
               batch_sizes: Sequence[int], log=print, ctx=None,
               ctx_train: Sequence[str] = ()) -> pd.DataFrame:
    frames = []
    for bs in batch_sizes:
        pairs = chosen_configs(search_results, bs)
        df = verify_configs(ws.mm, backend, lut, [str_to_cfg(c) for c, _ in pairs], bs,
                            ws.image_size, labels=[lbl for _, lbl in pairs], log=log, ctx=ctx,
                            ctx_train=ctx_train)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def run_backend_accuracy(ws, backend, configs: Sequence[tuple[str, str]], batch_size: int = 32,
                         log=print) -> pd.DataFrame | None:
    """Test-set accuracy of the *real* engines (TensorRT only)."""
    if not hasattr(backend, "accuracy"):
        return None
    x, y = ws.data.get("test")
    rows = []
    for c, label in configs:
        acc = backend.accuracy(ws.mm, str_to_cfg(c), x, y, batch_size, ws.image_size)
        rows.append({"gpu": backend.device_name, "config": c, "label": label,
                     "engine_test_acc": acc})
        log(f"engine test acc {c} {label:<20} {acc:.2f}%")
    return pd.DataFrame(rows)
