import numpy as np
import torch

from precisionpilot.config import load_config
from precisionpilot.data import build_data
from precisionpilot.evaluate import accuracy, as_fp16, baseline_accuracies
from precisionpilot.model import (
    NUM_UNITS,
    UNIT_NAMES,
    UnitResNet,
    build_resnet18,
    load_resnet18_checkpoint,
    unit_input_shapes,
)
from precisionpilot.train import train


def test_unit_view_has_ten_units_and_matches_torchvision():
    torch.manual_seed(0)
    base = build_resnet18(pretrained=False).eval()
    um = UnitResNet.from_torchvision(base).eval()
    assert len(um) == NUM_UNITS == 10 and um.names == UNIT_NAMES
    x = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        assert torch.allclose(base(x), um(x), atol=1e-6)


def test_unit_view_shares_parameters():
    base = build_resnet18(pretrained=False)
    um = UnitResNet.from_torchvision(base)
    assert um.units[0][0].weight is base.conv1.weight
    assert um.units[9].fc.weight is base.fc.weight


def test_unit_input_shapes():
    um = UnitResNet.from_torchvision(build_resnet18(pretrained=False))
    shapes = unit_input_shapes(um, batch_size=8, image_size=224)
    assert shapes[0] == (8, 3, 224, 224)
    assert shapes[1] == (8, 64, 56, 56)
    assert shapes[3] == (8, 64, 56, 56) and shapes[4] == (8, 128, 28, 28)
    assert shapes[9] == (8, 512, 7, 7)


def test_accuracy_function_perfect_and_fp16_copy():
    class Oracle(torch.nn.Module):
        def forward(self, x):  # predicts class from mean intensity of channel 0
            return torch.nn.functional.one_hot((x[:, 0].mean((1, 2)) > 0).long(), 10).float()

    x = np.zeros((8, 32, 32, 3), dtype=np.uint8)
    x[4:] = 255
    y = np.array([0] * 4 + [1] * 4)
    assert accuracy(Oracle(), x, y, "cpu", batch_size=3, image_size=32) == 100.0
    m = build_resnet18(pretrained=False)
    h = as_fp16(m)
    assert next(h.parameters()).dtype == torch.float16
    assert next(m.parameters()).dtype == torch.float32  # original untouched


def test_smoke_train_and_baselines(tmp_path):
    cfg = load_config(overrides=["train.epochs=1", "train.batch_size=32",
                                 "data.image_size=64", "data.eval_batch_size=64"])
    data = build_data(cfg, fake=True)
    ckpt = tmp_path / "m.pt"
    meta = train(cfg, data, torch.device("cpu"), ckpt, pretrained=False,
                 max_steps_per_epoch=3, log=lambda *_: None)
    assert ckpt.exists() and ckpt.with_suffix(".json").exists()
    assert meta["steps"] == 3 and len(meta["history"]) == 1
    model, m2 = load_resnet18_checkpoint(ckpt)
    assert m2["epochs"] == 1
    um = UnitResNet.from_torchvision(model).eval()
    res = baseline_accuracies(um, data, "cpu", splits=("val",), batch_size=64, image_size=64)
    assert 0.0 <= res["fp32_val"] <= 100.0
    # FP16 should agree closely with FP32
    assert abs(res["fp32_val"] - res["fp16_val"]) <= 2.0
