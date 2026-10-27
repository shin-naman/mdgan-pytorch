"""Weight-initialisation schemes, selected by Config.init and applied in build_models.

  "pytorch"  PyTorch's defaults, untouched (the baseline; every run before this option used it).
             nn.Linear / nn.Conv1d: weights U(-1/sqrt(fan_in), 1/sqrt(fan_in)) (std 1/sqrt(3 fan_in)),
             biases U(-1/sqrt(fan_in), 1/sqrt(fan_in)).
  "chainer"  Chainer's defaults for L.Linear and L.Convolution1D/ND: weights LeCunNormal(scale=1),
             i.e. N(0, 1/fan_in) (std sqrt(3) x PyTorch's), biases 0. The papers do not name their
             framework; Chainer is the framework of the original MD-GAN group (assumption, unverified).

fan_in = in_features for Linear, in_channels * kernel_size for Conv1d. BatchNorm is left alone
(gamma 1, beta 0 in both frameworks). Draws use the current torch RNG, so a seeded build is
reproducible.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from mdgan.config import INIT_SCHEMES


def apply_init(module: nn.Module, scheme: str) -> nn.Module:
    if scheme not in INIT_SCHEMES:
        raise ValueError(f"unknown init scheme {scheme!r}; expected one of {INIT_SCHEMES}")
    if scheme == "pytorch":
        return module
    for m in module.modules():
        if isinstance(m, (nn.Linear, nn.Conv1d)):
            fan_in = m.weight[0].numel()
            with torch.no_grad():
                m.weight.normal_(0.0, 1.0 / math.sqrt(fan_in))
                if m.bias is not None:
                    m.bias.zero_()
    return module