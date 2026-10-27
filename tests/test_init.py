"""Tests for mdgan.models.init (Config.init).

apply_init is tested directly as a unit (models built with the default, then the scheme applied),
and build_models' wiring of config.init separately: build_models binds apply_init at import, so
testing only through it would leave the helper itself unchecked.
"""

import dataclasses
import math

import pytest
import torch
from torch import nn

from mdgan.config import Config
from mdgan.models import build_models
from mdgan.models.init import apply_init

CFG = Config(latent_dim=16, noise_dim=16, block_len=16, n_channels=3, hidden_dim=512)


def _built(init: str = "pytorch", seed: int = 0):
    torch.manual_seed(seed)
    return build_models(dataclasses.replace(CFG, init=init))


def _chainer(seed: int = 0):
    """Default-built models with the chainer scheme applied directly by apply_init."""
    nets = _built("pytorch", seed)
    for net in nets:
        apply_init(net, "chainer")
    return nets


def _layers(nets):
    return [m for net in nets for m in net.modules() if isinstance(m, (nn.Linear, nn.Conv1d))]


def test_pytorch_scheme_is_a_no_op():
    """The default leaves PyTorch's initialisation exactly as it was before this option existed."""
    torch.manual_seed(0)
    from mdgan.models import Critic, LatentTransitionGenerator, TrajectoryGenerator
    ref = (LatentTransitionGenerator(16, 16, hidden_dim=512), TrajectoryGenerator(16, 16, 16, 3, hidden_dim=512),
           Critic(16, 3, hidden_dim=512))
    for a, b in zip(ref, _built("pytorch"), strict=True):
        for (ka, va), (kb, vb) in zip(a.state_dict().items(), b.state_dict().items(), strict=True):
            assert ka == kb and torch.equal(va, vb), ka


def test_chainer_scheme_lecun_normal_weights_and_zero_biases():
    """N(0, 1/fan_in) weights, zero biases; fan_in = in_features or in_channels * kernel_size."""
    layers = _layers(_chainer())
    assert any(isinstance(m, nn.Conv1d) and m.kernel_size[0] > 1 for m in layers)     # the critic's input conv
    checked = 0
    for m in layers:
        fan_in = m.in_features if isinstance(m, nn.Linear) else m.in_channels * m.kernel_size[0]
        w = m.weight.detach()
        if w.numel() >= 5000:                                    # enough samples for a tight std check
            assert w.std().item() == pytest.approx(1 / math.sqrt(fan_in), rel=0.05), m
            assert abs(w.mean().item()) < 4 * w.std().item() / math.sqrt(w.numel())     # centred: within 4 s.e.
            checked += 1
        assert m.bias is not None and torch.all(m.bias == 0), m
    assert checked >= 5


def test_chainer_weights_are_sqrt3_wider_than_pytorch():
    pt, ch = _layers(_built("pytorch")), _layers(_chainer())
    big = [i for i, m in enumerate(pt) if m.weight.numel() >= 100_000]
    assert big
    for i in big:
        ratio = ch[i].weight.std().item() / pt[i].weight.std().item()
        assert ratio == pytest.approx(math.sqrt(3), rel=0.03)


def test_build_models_applies_config_init():
    """Wiring: Config(init="chainer") through build_models gives the chainer scheme."""
    layers = _layers(_built("chainer"))
    assert all(torch.all(m.bias == 0) for m in layers)
    big = max(layers, key=lambda m: m.weight.numel())
    assert big.weight.std().item() == pytest.approx(1 / math.sqrt(big.weight[0].numel()), rel=0.05)


def test_batchnorm_untouched_and_build_is_reproducible():
    a, b = _built("chainer", seed=5), _built("chainer", seed=5)
    bn = a[0].output_norm
    assert torch.all(bn.weight == 1) and torch.all(bn.bias == 0)
    for x, y in zip(a, b, strict=True):
        for (k, v), (_, w) in zip(x.state_dict().items(), y.state_dict().items(), strict=True):
            assert torch.equal(v, w), k


def test_unknown_scheme_rejected():
    with pytest.raises(ValueError, match="unknown init scheme"):
        apply_init(nn.Linear(2, 2), "xavier")
    with pytest.raises(ValueError):
        Config(init="xavier")