"""Seeding, device selection, OS-safe defaults and small IO helpers."""

from __future__ import annotations

import json
import os
import platform
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(prefer: str | None = None) -> torch.device:
    """Pick a device: explicit preference, else cuda -> mps -> cpu."""
    if prefer and prefer != "auto":
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def default_num_workers(setting: Any = "auto") -> int:
    """DataLoader workers. Windows uses spawn (slow startup), so stay conservative."""
    if setting != "auto":
        return int(setting)
    cpus = os.cpu_count() or 1
    if platform.system() == "Windows":
        return min(2, cpus)
    return min(4, max(0, cpus - 1))


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(obj: Any, path: str | Path) -> Path:
    p = Path(path)
    ensure_dir(p.parent)
    p.write_text(json.dumps(obj, indent=2, default=_json_default))
    return p


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())


def _json_default(o: Any):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, tuple):
        return list(o)
    raise TypeError(f"Not JSON serializable: {type(o)}")


def gpu_name() -> str:
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "Apple MPS"
    return "CPU"


def gpu_slug(name: str | None = None) -> str:
    """Filesystem-safe short name for a device, e.g. 'rtx3050ti', 't4'."""
    name = (name or gpu_name()).lower()
    for key, slug in [("3050 ti", "rtx3050ti"), ("t4", "t4"), ("p100", "p100"),
                      ("mps", "mps"), ("cpu", "cpu")]:
        if key in name:
            return slug
    return "".join(c for c in name if c.isalnum())[:24] or "unknown"
