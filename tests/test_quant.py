import numpy as np
import pytest
import torch
import torch.nn as nn

from precisionpilot.data import preprocess
from precisionpilot.model import UnitResNet, build_resnet18
from precisionpilot.precision import (
    INT8,
    MixedPrecisionModel,
    all_configs,
    all_fp16,
    all_int8,
    cfg_to_int,
    cfg_to_str,
    describe,
    int_to_cfg,
    one_hot,
    str_to_cfg,
)
from precisionpilot.quant.fakequant import (
    ActQuantizer,
    fake_quant_per_tensor,
    fake_quant_weight_per_channel,
    quantizers,
)
from precisionpilot.quant.fold import fold_bn, fuse_conv_bn


# ---------------------------------------------------------------- config helpers
def test_config_roundtrip():
    for i in (0, 1, 513, 1023):
        c = int_to_cfg(i)
        assert cfg_to_int(c) == i and str_to_cfg(cfg_to_str(c)) == c
    assert len(all_configs()) == 1024 and len(set(all_configs())) == 1024
    assert cfg_to_str(one_hot(0)) == "1000000000"
    assert describe(all_fp16()) == "all-FP16"
    assert "stem" in describe(one_hot(0))
    with pytest.raises(ValueError):
        str_to_cfg("0120")
    with pytest.raises(ValueError):
        int_to_cfg(1024)


# ---------------------------------------------------------------- fake quant primitives
def test_per_tensor_error_bounded_by_half_scale():
    torch.manual_seed(0)
    x = torch.randn(10_000) * 3
    amax = float(x.abs().max())
    q = fake_quant_per_tensor(x, amax)
    scale = amax / 127
    assert (q - x).abs().max() <= scale / 2 + 1e-6


def test_fake_quant_is_idempotent_and_on_grid():
    x = torch.randn(1000)
    q = fake_quant_per_tensor(x, 2.0)
    assert torch.equal(fake_quant_per_tensor(q, 2.0), q)
    levels = torch.round(q / (2.0 / 127))
    assert torch.allclose(levels * (2.0 / 127), q, atol=1e-6)
    assert levels.min() >= -128 and levels.max() <= 127


def test_clipping_beyond_amax():
    q = fake_quant_per_tensor(torch.tensor([10.0, -10.0]), 1.0)
    assert torch.allclose(q, torch.tensor([1.0, -128 / 127]), atol=1e-6)


def test_per_channel_weight_scales():
    w = torch.zeros(3, 2, 1, 1)
    w[0], w[1], w[2] = 0.01, 1.0, 100.0
    w[:, 1] *= -0.5
    q = fake_quant_weight_per_channel(w)
    # each channel's max is exactly representable
    assert torch.allclose(q[:, 0], w[:, 0], rtol=1e-5)
    rel = ((q - w).abs() / w.abs().amax(dim=(1, 2, 3), keepdim=True)).max()
    assert rel <= 0.5 / 127 + 1e-6


def test_act_quantizer_calibration_methods():
    x = torch.cat([torch.linspace(-1, 1, 9999), torch.tensor([50.0])])
    for method, expected in [("minmax", 50.0), ("percentile", None)]:
        q = ActQuantizer(method, percentile=99.0)
        q.mode = "calib"
        out = q(x)
        assert torch.equal(out, x)  # calibration is pass-through
        amax = q.finalize()
        if expected:
            assert amax == pytest.approx(expected)
        else:
            assert amax < 1.5  # percentile ignores the outlier
    with pytest.raises(RuntimeError):
        ActQuantizer().finalize()
    with pytest.raises(ValueError):
        ActQuantizer("entropy")


# ---------------------------------------------------------------- BN folding
def test_fuse_conv_bn_exact():
    torch.manual_seed(0)
    conv = nn.Conv2d(3, 8, 3, padding=1, bias=False)
    bn = nn.BatchNorm2d(8)
    bn.running_mean.normal_()
    bn.running_var.uniform_(0.5, 2)
    bn.weight.data.normal_()
    bn.bias.data.normal_()
    bn.eval()
    x = torch.randn(2, 3, 8, 8)
    assert torch.allclose(bn(conv(x)), fuse_conv_bn(conv, bn)(x), atol=1e-5)


def test_fold_bn_whole_model_equivalent():
    torch.manual_seed(0)
    um = UnitResNet.from_torchvision(build_resnet18(pretrained=False)).eval()
    folded = fold_bn(um)
    assert not any(isinstance(m, nn.BatchNorm2d) for m in folded.modules())
    assert any(isinstance(m, nn.BatchNorm2d) for m in um.modules())  # original intact
    x = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        assert torch.allclose(um(x), folded(x), atol=1e-4)


# ---------------------------------------------------------------- mixed model
@pytest.fixture(scope="module")
def mixed():
    torch.manual_seed(0)
    um = UnitResNet.from_torchvision(build_resnet18(pretrained=False)).eval()
    mm = MixedPrecisionModel(um)
    calib = np.random.default_rng(0).integers(0, 255, (32, 32, 32, 3), dtype=np.uint8)
    amax = mm.calibrate(calib, "cpu", batch_size=16, image_size=64)
    x = preprocess(torch.from_numpy(calib[:4]).permute(0, 3, 1, 2), "cpu", 64)
    return um, mm, amax, x


def test_int8_requires_calibration():
    um = UnitResNet.from_torchvision(build_resnet18(pretrained=False)).eval()
    mm = MixedPrecisionModel(um)
    mm.set_config(all_fp16())
    with pytest.raises(RuntimeError):
        mm.set_config(one_hot(0))
    with pytest.raises(ValueError):
        mm.set_config((0, 1))


def test_calibration_produces_positive_ranges(mixed):
    _, mm, amax, _ = mixed
    assert set(amax) == set(mm.names)
    assert all(v > 0 for vals in amax.values() for v in vals)
    # stem: 1 conv; block w/o downsample: 2 convs + residual; head: 1 linear
    assert len(amax["stem"]) == 1 and len(amax["layer1.0"]) == 3
    assert len(amax["layer2.0"]) == 4 and len(amax["head"]) == 1
    assert all(q.mode == "quant" for u in mm.int8_units for q in quantizers(u))


@torch.no_grad()
def test_fp16_close_and_int8_noisier(mixed):
    um, mm, _, x = mixed
    ref = um(x)
    e16 = (mm.set_config(all_fp16())(x).float() - ref).abs().max()
    e8 = (mm.set_config(all_int8())(x).float() - ref).abs().max()
    assert e16 < 0.01 * ref.abs().max()
    assert e8 > e16
    assert e8 < 0.25 * ref.abs().max()
    assert torch.allclose(mm.fp32_forward(x), ref, atol=1e-4)


@torch.no_grad()
def test_config_only_changes_selected_units(mixed):
    _, mm, _, x = mixed
    base = mm.set_config(all_fp16())(x)
    head_only = mm.set_config(one_hot(9))(x)
    assert not torch.equal(base, head_only)
    assert mm.set_config(all_fp16())(x).dtype == torch.float16
    assert torch.equal(mm(x), base)  # switching back restores exactly


def test_quant_unit_is_independent_copy(mixed):
    um, mm, _, _ = mixed
    assert mm.int8_units[0][0].conv.weight is not um.units[0][0].weight


# ---------------------------------------------------------------- ONNX export
def test_onnx_qdq_only_in_int8_units(mixed, tmp_path):
    pytest.importorskip("onnx")
    from precisionpilot.export.onnx_export import count_qdq, export_config, export_unit

    _, mm, _, _ = mixed
    c0 = count_qdq(export_config(mm, all_fp16(), 1, 64, tmp_path))
    assert c0["QuantizeLinear"] == 0 and c0["Conv"] == 20
    c1 = count_qdq(export_config(mm, one_hot(0), 1, 64, tmp_path))
    assert c1["QuantizeLinear"] == 2  # stem: activation + weight
    call = count_qdq(export_config(mm, all_int8(), 1, 64, tmp_path))
    # 21 weight Q + 21 conv/fc input Q + 8 residual Q, minus 8 merged duplicates: the
    # residual (no-downsample blocks) / downsample input (downsample blocks) quantizes the
    # same tensor with the same scale as conv1's input, so the exporter shares one Q/DQ.
    assert call["QuantizeLinear"] == 42
    cu = count_qdq(export_unit(mm, 3, INT8, (1, 64, 16, 16), tmp_path))
    assert cu["QuantizeLinear"] == 6 and cu["Conv"] == 3
    # scales must be constants (no in-graph ReduceMax): TensorRT requirement
    import onnx

    ops = {n.op_type for n in onnx.load(str(export_config(mm, all_int8(), 1, 64,
                                                            tmp_path))).graph.node}
    assert "ReduceMax" not in ops
    # export mode leaves the simulation model untouched
    assert mm.mode == "sim"


def test_manual_fake_quant_matches_native():
    from precisionpilot.quant.fakequant import QMAX, QMIN, _manual_fake_quant

    torch.manual_seed(0)
    x = torch.randn(4096) * 2
    x[:4] = torch.tensor([0.5, 1.5, -0.5, -2.5]) * (1.0 / 127)  # exact ties
    native = torch.fake_quantize_per_tensor_affine(x, 1.0 / 127, 0, QMIN, QMAX)
    assert torch.equal(_manual_fake_quant(x, 1.0 / 127), native)
    w = torch.randn(8, 3, 3, 3)
    scale = w.abs().amax(dim=(1, 2, 3)) / 127
    native_w = torch.fake_quantize_per_channel_affine(
        w, scale, torch.zeros(8, dtype=torch.int32), 0, QMIN, QMAX)
    assert torch.allclose(_manual_fake_quant(w, scale.view(-1, 1, 1, 1)), native_w, atol=1e-7)
