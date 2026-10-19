"""Hostile tests for the critic: the mistakes that would silently corrupt
WGAN-GP training.

Run: python -m pytest tests/D_G/test_dg_adversarial.py -v
"""

import copy

import pytest
import torch
from torch import nn

from mdgan.models.critic import Critic
from mdgan.models.latent_transition import LatentTransitionGenerator
from mdgan.models.trajectory_generator import TrajectoryGenerator

LAT, NR, M, C, H = 7, 3, 8, 3, 24
W = 2 * M


@pytest.fixture
def model():
    torch.manual_seed(99)
    return Critic(M, C, hidden_dim=H)


# 1. Shape traps

def test_batch_of_one_keeps_batch_dim():
    """squeeze() without an axis would turn (1, 1) into a scalar and break the
    per-sample GP. Output must be (1,)."""
    d = Critic(M, C, hidden_dim=H)
    s = d(torch.randn(1, W, C))
    assert s.shape == (1,)


def test_hidden_equals_window_len_does_not_confuse_squeeze():
    """If H == 1 the conv output is (B, 1, 1); squeeze(-1) must still leave (B, 1)
    for hidden1, not (B,)."""
    d = Critic(M, C, hidden_dim=1)
    assert d(torch.randn(3, W, C)).shape == (3,)


def test_square_window_shape_would_mask_a_transpose_bug():
    """When 2M == C a transposed window passes the shape check. Verify axis 1 is
    still time: swapping two time steps must change the score the same way as
    swapping those rows in the equivalent flattened affine."""
    d = Critic(block_len=2, n_channels=4, hidden_dim=H)      # 2M = C = 4
    w = torch.randn(1, 4, 4)
    w_swapped = w[:, [1, 0, 2, 3], :]
    Wmat = d.input_conv.weight.detach().permute(0, 2, 1).reshape(H, 16)
    b = d.input_conv.bias.detach()
    pre = []
    handle = d.input_conv.register_forward_hook(lambda m, i, o: pre.append(o.detach().squeeze(-1)))
    try:
        with torch.no_grad():
            d(w); d(w_swapped)
    finally:
        handle.remove()
    torch.testing.assert_close(pre[1], w_swapped.reshape(1, 16) @ Wmat.T + b, atol=1e-5, rtol=1e-5)
    assert not torch.allclose(pre[0], pre[1])


# 2. Things that would break the gradient penalty

def test_no_detach_or_inplace_on_input(model):
    """A detach() anywhere would zero d score / d window; an in-place op on a leaf
    that requires grad would raise. Both are checked by the GP path."""
    w = torch.randn(4, W, C, requires_grad=True)
    g, = torch.autograd.grad(model(w).sum(), w)
    assert g.abs().sum().item() > 0


def test_second_order_gradient_exists(model):
    """create_graph=True must produce a graph through which the GP can backprop.
    A LeakyReLU has zero second derivative, so the second-order signal comes
    through the weights; we check it is nonzero for at least one layer."""
    w = torch.randn(4, W, C, requires_grad=True)
    g, = torch.autograd.grad(model(w).sum(), w, create_graph=True)
    penalty = ((g.flatten(1).norm(dim=1) - 1) ** 2).mean()
    penalty.backward()
    assert any(p.grad is not None and p.grad.abs().sum().item() > 0 for p in model.parameters())


def test_no_module_couples_samples(model):
    assert not any(isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm,
                                  nn.InstanceNorm1d, nn.Dropout, nn.Dropout1d))
                   for m in model.modules())


def test_adding_batchnorm_would_be_detected_by_independence_test():
    """Sanity check on the test strategy itself: a critic WITH BatchNorm fails
    the per-sample independence property."""
    class Coupled(Critic):
        def __init__(self):
            super().__init__(M, C, hidden_dim=H)
            self.bn = nn.BatchNorm1d(H)

        def forward(self, window):
            h = self.input_conv(window.transpose(1, 2)).squeeze(-1)
            h = self.act(self.bn(h))
            h = self.act(self.hidden1(h))
            h = self.act(self.hidden2(h))
            h = self.act(self.hidden3(h))
            return self.output_layer(h).squeeze(-1)

    d = Coupled().train()
    w = torch.randn(4, W, C, requires_grad=True)
    g, = torch.autograd.grad(d(w)[0], w)
    assert g[1:].abs().sum().item() > 0      # coupling present, as expected


# 3. Dead or wrong paths

def test_zeroing_input_conv_makes_score_constant(model):
    with torch.no_grad():
        model.input_conv.weight.zero_()
        a = model(torch.randn(4, W, C))
        b = model(torch.randn(4, W, C))
    torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_each_hidden_layer_is_live(model):
    model(torch.randn(4, W, C)).sum().backward()
    for layer in (model.hidden1, model.hidden2, model.hidden3):
        assert layer.weight.grad.abs().sum().item() > 0


def test_score_depends_on_both_blocks(model):
    """The critic must use y_k AND y_{k+1}: perturbing only the second block
    (steps M..2M-1) must change the score, as must perturbing only the first."""
    w = torch.randn(1, W, C)
    with torch.no_grad():
        base = model(w)
        w1 = w.clone(); w1[:, :M] += 1.0
        w2 = w.clone(); w2[:, M:] += 1.0
        assert (model(w1) - base).abs().item() > 1e-6
        assert (model(w2) - base).abs().item() > 1e-6


# 4. Aliasing, copies, checkpoints

def test_output_does_not_alias_input_or_parameters(model):
    w = torch.randn(4, W, C)
    with torch.no_grad():
        s = model(w)
    s.add_(100.0)
    with torch.no_grad():
        torch.testing.assert_close(model(w), s - 100.0, atol=1e-5, rtol=0)


def test_deepcopy_is_independent(model):
    clone = copy.deepcopy(model)
    with torch.no_grad():
        model.output_layer.bias.add_(10.0)
        w = torch.randn(2, W, C)
        assert not torch.allclose(model(w), clone(w))


def test_whole_module_pickle_round_trip(model, tmp_path):
    path = tmp_path / "dg_module.pt"
    torch.save(model, path)
    restored = torch.load(path, map_location="cpu", weights_only=False)
    w = torch.randn(2, W, C)
    with torch.no_grad():
        torch.testing.assert_close(restored(w), model(w), rtol=0, atol=0)


def test_state_dicts_cannot_be_confused_with_generators():
    d = Critic(M, C, hidden_dim=H)
    gz = LatentTransitionGenerator(LAT, NR, hidden_dim=H)
    gy = TrajectoryGenerator(LAT, NR, M, C, hidden_dim=H)
    for other in (gz, gy):
        with pytest.raises(RuntimeError):
            d.load_state_dict(other.state_dict(), strict=True)
    # Key names collide with G_z (hidden1, hidden2, output_layer); non-strict is unsafe.
    assert set(d.state_dict()) & set(gz.state_dict())


def test_no_parameter_is_shared(model):
    ids = [id(p) for p in model.parameters()]
    assert len(ids) == len(set(ids))