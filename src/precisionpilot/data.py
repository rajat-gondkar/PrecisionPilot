"""CIFAR-10 data: in-memory arrays, deterministic stratified splits, on-device preprocessing.

Design notes
------------
* The whole dataset (~180 MB as uint8) is held in memory and batches are moved to the device
  as uint8. Augmentation and the 32 -> 224 bilinear upscale run *on the device*, per batch.
  This avoids DataLoader worker processes entirely (no Windows ``spawn`` issues) and is much
  faster than resizing on the CPU.
* Splits (outline §6.5) are stratified per class, created once with a fixed seed and saved to
  ``data/splits.json`` so every machine (Mac / Windows / Kaggle) uses identical indices:

  ===========  ====================  ==============================================
  split        source                purpose
  ===========  ====================  ==============================================
  finetune     CIFAR train (48k)     fine-tuning the pretrained ResNet-18
  calib        CIFAR train (1k)      INT8 activation-range calibration
  spare        CIFAR train (1k)      monitoring during fine-tuning
  val          CIFAR test  (5k)      sensitivity scan + all searches
  val_subset   first 1k of val       fast search / exhaustive accuracy table
  test         CIFAR test  (5k)      final reported numbers ONLY
  ===========  ====================  ==============================================
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from precisionpilot.utils import load_json, save_json

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
NUM_CLASSES = 10
CIFAR10_CLASSES = ("airplane", "automobile", "bird", "cat", "deer",
                   "dog", "frog", "horse", "ship", "truck")
SPLIT_NAMES = ("finetune", "calib", "spare", "val", "val_subset", "test")
_TRAIN_SPLITS = {"finetune", "calib", "spare"}


# --------------------------------------------------------------------------- raw data
@dataclass
class Cifar10Arrays:
    """Raw CIFAR-10 as uint8 NHWC arrays plus int64 labels."""

    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray


def load_cifar10(data_dir: str | Path, download: bool = True) -> Cifar10Arrays:
    from torchvision.datasets import CIFAR10

    data_dir = Path(data_dir)
    train = CIFAR10(root=str(data_dir), train=True, download=download)
    test = CIFAR10(root=str(data_dir), train=False, download=download)
    return Cifar10Arrays(
        x_train=np.asarray(train.data, dtype=np.uint8),
        y_train=np.asarray(train.targets, dtype=np.int64),
        x_test=np.asarray(test.data, dtype=np.uint8),
        y_test=np.asarray(test.targets, dtype=np.int64),
    )


def make_fake_cifar10(n_train: int = 600, n_test: int = 400, seed: int = 0) -> Cifar10Arrays:
    """Tiny synthetic stand-in with learnable class structure (tests / smoke runs)."""
    rng = np.random.default_rng(seed)

    def gen(n: int):
        y = np.arange(n) % NUM_CLASSES
        rng.shuffle(y)
        base = rng.integers(0, 60, size=(n, 32, 32, 3))
        # class-dependent colour blob so a model can actually learn something
        for c in range(NUM_CLASSES):
            m = y == c
            base[m, (c * 3) % 24:(c * 3) % 24 + 8, :, c % 3] += 180
        return np.clip(base, 0, 255).astype(np.uint8), y.astype(np.int64)

    xtr, ytr = gen(n_train)
    xte, yte = gen(n_test)
    return Cifar10Arrays(xtr, ytr, xte, yte)


# --------------------------------------------------------------------------- splits
@dataclass
class Splits:
    seed: int
    indices: dict[str, list[int]] = field(default_factory=dict)

    def __getitem__(self, name: str) -> np.ndarray:
        return np.asarray(self.indices[name], dtype=np.int64)

    def source(self, name: str) -> str:
        return "train" if name in _TRAIN_SPLITS else "test"

    def sizes(self) -> dict[str, int]:
        return {k: len(v) for k, v in self.indices.items()}

    def to_dict(self) -> dict:
        return {"seed": self.seed, "sizes": self.sizes(), "indices": self.indices}

    @classmethod
    def from_dict(cls, d: dict) -> Splits:
        return cls(seed=d["seed"], indices={k: list(v) for k, v in d["indices"].items()})


def _stratified_take(labels: np.ndarray, pool: np.ndarray, n: int,
                     rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Take ``n`` indices from ``pool`` with (as near as possible) equal counts per class.

    Returns (taken, remaining_pool). Order of ``taken`` is shuffled.
    """
    if n > len(pool):
        raise ValueError(f"Requested {n} samples but only {len(pool)} remain")
    classes = np.unique(labels[pool])
    per_class = [pool[labels[pool] == c] for c in classes]
    base, extra = divmod(n, len(classes))
    taken = []
    for i, members in enumerate(per_class):
        k = min(len(members), base + (1 if i < extra else 0))
        taken.append(rng.choice(members, size=k, replace=False))
    taken = np.concatenate(taken)
    if len(taken) < n:  # class imbalance fallback: top up uniformly
        rest = np.setdiff1d(pool, taken)
        taken = np.concatenate([taken, rng.choice(rest, size=n - len(taken), replace=False)])
    rng.shuffle(taken)
    remaining = np.setdiff1d(pool, taken)
    return taken, remaining


def make_splits(y_train: np.ndarray, y_test: np.ndarray, seed: int, n_calib: int,
                n_spare: int, n_val: int, n_val_subset: int, n_test: int) -> Splits:
    rng = np.random.default_rng(seed)
    pool = np.arange(len(y_train))
    calib, pool = _stratified_take(y_train, pool, n_calib, rng)
    spare, pool = _stratified_take(y_train, pool, n_spare, rng)
    finetune = rng.permutation(pool)

    tpool = np.arange(len(y_test))
    val, tpool = _stratified_take(y_test, tpool, n_val, rng)
    test, _ = _stratified_take(y_test, tpool, n_test, rng)
    if n_val_subset > n_val:
        raise ValueError("n_val_subset must be <= n_val")
    val_subset, _ = _stratified_take(y_test, val, n_val_subset, rng)

    return Splits(seed=seed, indices={
        "finetune": finetune.tolist(), "calib": calib.tolist(), "spare": spare.tolist(),
        "val": val.tolist(), "val_subset": val_subset.tolist(), "test": test.tolist(),
    })


def load_or_create_splits(path: str | Path, arrays: Cifar10Arrays, seed: int, n_calib: int,
                          n_spare: int, n_val: int, n_val_subset: int, n_test: int,
                          force: bool = False) -> Splits:
    path = Path(path)
    want = {"calib": n_calib, "spare": n_spare, "val": n_val,
            "val_subset": n_val_subset, "test": n_test}
    if path.exists() and not force:
        splits = Splits.from_dict(load_json(path))
        sizes = splits.sizes()
        if splits.seed == seed and all(sizes.get(k) == v for k, v in want.items()):
            return splits
    splits = make_splits(arrays.y_train, arrays.y_test, seed, n_calib, n_spare,
                         n_val, n_val_subset, n_test)
    save_json(splits.to_dict(), path)
    return splits


# --------------------------------------------------------------------------- data access
class CifarData:
    """Raw arrays + splits; hands out (x, y) arrays per split name."""

    def __init__(self, arrays: Cifar10Arrays, splits: Splits):
        self.arrays = arrays
        self.splits = splits

    def get(self, name: str, limit: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        idx = self.splits[name]
        if limit is not None:
            idx = idx[:limit]
        if self.splits.source(name) == "train":
            return self.arrays.x_train[idx], self.arrays.y_train[idx]
        return self.arrays.x_test[idx], self.arrays.y_test[idx]


def build_data(cfg, fake: bool = False, fake_size: str = "normal") -> CifarData:
    """Load CIFAR-10 (or a synthetic stand-in) and the persisted splits for a Config.

    ``fake_size="tiny"`` gives the minimal synthetic set used by the smoke pipeline.
    """
    d = cfg.data
    if fake:
        if fake_size == "tiny":
            arrays = make_fake_cifar10(n_train=200, n_test=80, seed=cfg.seed)
            splits = make_splits(arrays.y_train, arrays.y_test, cfg.seed, n_calib=20,
                                 n_spare=20, n_val=40, n_val_subset=10, n_test=40)
            return CifarData(arrays, splits)
        arrays = make_fake_cifar10(seed=cfg.seed)
        n = len(arrays.y_test)
        splits = make_splits(arrays.y_train, arrays.y_test, cfg.seed,
                             n_calib=50, n_spare=50, n_val=n // 2,
                             n_val_subset=min(100, n // 2), n_test=n // 2)
        return CifarData(arrays, splits)
    arrays = load_cifar10(cfg.path("data_dir"))
    splits = load_or_create_splits(cfg.resolve(d.splits_file), arrays, cfg.seed, d.n_calib,
                                   d.n_spare, d.n_val, d.n_val_subset, d.n_test)
    return CifarData(arrays, splits)


# --------------------------------------------------------------------------- batching
def iterate_batches(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool = False,
                    seed: int = 0, drop_last: bool = False
                    ) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Yield CPU uint8 NCHW image batches and int64 labels."""
    n = len(y)
    order = np.random.default_rng(seed).permutation(n) if shuffle else np.arange(n)
    stop = n - (n % batch_size) if drop_last else n
    for start in range(0, stop, batch_size):
        sel = order[start:start + batch_size]
        xb = torch.from_numpy(np.ascontiguousarray(x[sel])).permute(0, 3, 1, 2)
        yb = torch.from_numpy(y[sel])
        yield xb, yb


def _augment(x: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
    """Random crop (pad 4, reflect) + horizontal flip at 32x32, on whatever device x is on."""
    n, _, h, w = x.shape
    padded = F.pad(x, (4, 4, 4, 4), mode="reflect")
    dev = x.device
    ox = torch.randint(0, 9, (n,), generator=generator).to(dev)
    oy = torch.randint(0, 9, (n,), generator=generator).to(dev)
    rows = (oy.view(n, 1) + torch.arange(h, device=dev).view(1, h))            # n,h
    cols = (ox.view(n, 1) + torch.arange(w, device=dev).view(1, w))            # n,w
    bidx = torch.arange(n, device=dev).view(n, 1, 1)
    out = padded.permute(0, 2, 3, 1)[bidx, rows.view(n, h, 1), cols.view(n, 1, w)]
    out = out.permute(0, 3, 1, 2)
    flip = (torch.rand(n, generator=generator) < 0.5).to(dev)
    out = torch.where(flip.view(n, 1, 1, 1), out.flip(3), out)
    return out.contiguous()


def preprocess(xb: torch.Tensor, device: torch.device | str, image_size: int = 224,
               augment: bool = False, dtype: torch.dtype = torch.float32,
               generator: torch.Generator | None = None) -> torch.Tensor:
    """uint8 NCHW (CPU) -> normalised float NCHW at ``image_size`` on ``device``."""
    x = xb.to(device, non_blocking=True).float().div_(255.0)
    if augment:
        x = _augment(x, generator)
    if image_size != x.shape[-1]:
        x = F.interpolate(x, size=(image_size, image_size), mode="bilinear",
                          align_corners=False)
    mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device).view(1, 3, 1, 1)
    x = (x - mean) / std
    return x.to(dtype)
