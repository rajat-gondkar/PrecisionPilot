"""ResNet-18 and its 10-unit view (outline §6.1).

Units (each gets one precision choice):

    0 stem      conv1 + bn1 + relu + maxpool
    1 layer1.0  BasicBlock      2 layer1.1  BasicBlock
    3 layer2.0  BasicBlock (downsample)   4 layer2.1
    5 layer3.0  BasicBlock (downsample)   6 layer3.1
    7 layer4.0  BasicBlock (downsample)   8 layer4.1
    9 head      avgpool + flatten + fc

10 units -> 2^10 = 1,024 configurations: small enough for exhaustive ground truth.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torchvision

UNIT_NAMES: tuple[str, ...] = (
    "stem",
    "layer1.0", "layer1.1",
    "layer2.0", "layer2.1",
    "layer3.0", "layer3.1",
    "layer4.0", "layer4.1",
    "head",
)
NUM_UNITS = len(UNIT_NAMES)


def build_resnet18(num_classes: int = 10, pretrained: bool = True) -> torchvision.models.ResNet:
    """torchvision ResNet-18 (optionally ImageNet weights) with a ``num_classes`` FC head."""
    weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = torchvision.models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


class Head(nn.Module):
    def __init__(self, avgpool: nn.Module, fc: nn.Linear):
        super().__init__()
        self.avgpool = avgpool
        self.fc = fc

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(torch.flatten(self.avgpool(x), 1))


class UnitResNet(nn.Module):
    """A ResNet-18 expressed as a sequence of 10 independently replaceable units.

    The units share parameters with the source torchvision model (no copy).
    """

    def __init__(self, units: nn.ModuleList, names: tuple[str, ...] = UNIT_NAMES):
        super().__init__()
        if len(units) != len(names):
            raise ValueError("units and names length mismatch")
        self.units = units
        self.names = names

    @classmethod
    def from_torchvision(cls, m: torchvision.models.ResNet) -> UnitResNet:
        stem = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool)
        blocks = [m.layer1[0], m.layer1[1], m.layer2[0], m.layer2[1],
                  m.layer3[0], m.layer3[1], m.layer4[0], m.layer4[1]]
        head = Head(m.avgpool, m.fc)
        return cls(nn.ModuleList([stem, *blocks, head]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for unit in self.units:
            x = unit(x)
        return x

    def __len__(self) -> int:
        return len(self.units)


@torch.no_grad()
def unit_input_shapes(model: UnitResNet, batch_size: int = 1, image_size: int = 224,
                      device: str = "cpu") -> list[tuple[int, ...]]:
    """Input tensor shape for every unit (needed to time / export units in isolation)."""
    was_training = model.training
    model.eval()
    shapes = []
    x = torch.zeros(batch_size, 3, image_size, image_size, device=device)
    p = next(model.parameters())
    x = x.to(p.dtype)
    for unit in model.units:
        shapes.append(tuple(x.shape))
        x = unit(x)
    model.train(was_training)
    return shapes


# --------------------------------------------------------------------------- checkpoints
def save_checkpoint(model: nn.Module, path: str | Path, meta: dict | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "meta": meta or {}}, path)
    return path


def load_resnet18_checkpoint(path: str | Path, num_classes: int = 10,
                             map_location: str | torch.device = "cpu"
                             ) -> tuple[torchvision.models.ResNet, dict]:
    ckpt = torch.load(Path(path), map_location=map_location, weights_only=False)
    model = build_resnet18(num_classes=num_classes, pretrained=False)
    state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    return model, ckpt.get("meta", {}) if isinstance(ckpt, dict) else {}


def load_model(cfg, device: str | torch.device = "cpu", allow_random: bool = False
               ) -> tuple[UnitResNet, dict]:
    """Load the fine-tuned checkpoint from the config as a UnitResNet in eval mode.

    ``allow_random=True`` falls back to an untrained network (smoke runs / tests only).
    """
    path = cfg.path("checkpoint")
    if path.exists():
        base, meta = load_resnet18_checkpoint(path, map_location="cpu")
    elif allow_random:
        base, meta = build_resnet18(pretrained=False), {"random_init": True}
    else:
        raise FileNotFoundError(
            f"Checkpoint not found at {path}. Train it on Kaggle "
            "(notebooks/kaggle_01_finetune.ipynb) and copy it there.")
    model = UnitResNet.from_torchvision(base).to(device).eval()
    return model, meta
