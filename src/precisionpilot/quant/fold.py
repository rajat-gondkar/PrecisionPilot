"""Fold BatchNorm into the preceding convolution (inference-time, exact up to float error).

TensorRT performs this fusion itself; doing it before fake quantization makes the simulated
INT8 weights match what TensorRT actually quantizes.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
from torchvision.models.resnet import BasicBlock


@torch.no_grad()
def fuse_conv_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> nn.Conv2d:
    fused = nn.Conv2d(conv.in_channels, conv.out_channels, conv.kernel_size, conv.stride,
                      conv.padding, conv.dilation, conv.groups, bias=True,
                      padding_mode=conv.padding_mode).to(conv.weight.device, conv.weight.dtype)
    std = torch.sqrt(bn.running_var + bn.eps)
    scale = bn.weight / std
    fused.weight.copy_(conv.weight * scale.view(-1, 1, 1, 1))
    bias = conv.bias if conv.bias is not None else torch.zeros_like(bn.running_mean)
    fused.bias.copy_((bias - bn.running_mean) * scale + bn.bias)
    return fused


def _fold_sequential(seq: nn.Sequential) -> None:
    items = list(seq._modules.items())
    i = 0
    while i < len(items) - 1:
        (n1, m1), (n2, m2) = items[i], items[i + 1]
        if isinstance(m1, nn.Conv2d) and isinstance(m2, nn.BatchNorm2d):
            seq._modules[n1] = fuse_conv_bn(m1, m2)
            seq._modules[n2] = nn.Identity()
            i += 2
        else:
            i += 1


def fold_bn(module: nn.Module) -> nn.Module:
    """Return a deep copy of ``module`` (eval mode) with every Conv+BN pair folded."""
    m = copy.deepcopy(module).eval()
    for sub in m.modules():
        if isinstance(sub, BasicBlock):
            sub.conv1 = fuse_conv_bn(sub.conv1, sub.bn1)
            sub.bn1 = nn.Identity()
            sub.conv2 = fuse_conv_bn(sub.conv2, sub.bn2)
            sub.bn2 = nn.Identity()
        elif isinstance(sub, nn.Sequential):
            _fold_sequential(sub)
    return m
