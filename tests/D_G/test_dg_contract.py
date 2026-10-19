"""Architecture contract tests for MD-GAN's critic (D_G).

Pins the module to Kawada et al. (2022) ESI Fig. S2(C).

API:
    Critic(block_len=64, n_channels=3, hidden_dim=2048, negative_slope=0.2)
    model(window: (B, 2M, C)) -> (B,)

Run: python -m pytest tests/D_G/test_dg_contract.py -v
"""

import pytest
import torch
from torch import nn

from mdgan.models.critic import Critic

M, C, H = 8, 3, 24
W = 2 * M


@pytest.fixture
def model():
    torch.manual_seed(717)
    return Critic(block_len=M, n_channels=C, hidden_dim=H, negative_slope=0.17)


def _window(batch=8, block_len=M, n_channels=C):
    return torch.randn(batch, 2 * block_len, n_channels)


# 1. Layers and dimensions

def test_reference_layers_and_dimensions():
    d = Critic(block_len=64, n_channels=3, hidden_dim=48, negative_slope=0.17)
    assert isinstance(d.input_conv, nn.Conv1d)
    assert (d.input_conv.in_channels, d.input_conv.out_channels) == (3, 48)
    assert d.input_conv.kernel_size == (128,) and d.input_conv.stride == (128,)
    assert d.input_conv.padding == (0,) and d.input_conv.dilation == (1,)
    assert d.input_conv.bias is not None
    shapes = [(m.in_features, m.out_features) for m in (d.hidden1, d.hidden2, d.hidden3, d.output_layer)]
    assert shapes == [(48, 48), (48, 48), (48, 48), (48, 1)]
    assert isinstance(d.act, nn.LeakyReLU) and d.act.negative_slope == pytest.approx(0.17)
    assert d.window_len == 128
    assert sorted(n for n, _ in d.named_children()) == sorted(
        ["input_conv", "hidden1", "hidden2", "hidden3", "output_layer", "act"])


def test_default_width_matches_reference_design():
    d = Critic()
    assert d.input_conv.out_channels == 2048 and d.input_conv.kernel_size == (128,)
    assert [m.out_features for m in (d.hidden1, d.hidden2, d.hidden3)] == [2048] * 3
    assert d.output_layer.out_features == 1


def test_parameter_count_matches_closed_form():
    d = Critic()
    expected = (
        3 * 2048 * 128 + 2048            # input_conv
        + 3 * (2048 * 2048 + 2048)       # hidden1..3
        + 2048 * 1 + 1                   # output_layer
    )
    assert sum(p.numel() for p in d.parameters()) == expected


def test_no_normalization_dropout_or_buffers(model):
    """Hard constraint for WGAN-GP: the penalty is per-sample; nothing may couple samples."""
    kinds = {type(m) for m in model.modules()}
    assert kinds == {Critic, nn.Conv1d, nn.Linear, nn.LeakyReLU}
    assert list(model.buffers()) == []


def test_constructor_defaults_match_decisions():
    d = Critic()
    assert d.block_len == 64 and d.n_channels == 3 and d.hidden_dim == 2048
    assert d.act.negative_slope == pytest.approx(0.2)


@pytest.mark.parametrize("bad", [
    dict(block_len=0), dict(n_channels=-1), dict(hidden_dim=0),
    dict(block_len=True), dict(n_channels=3.0), dict(hidden_dim="2048"),
])
def test_invalid_constructor_arguments_rejected(bad):
    kwargs = dict(block_len=8, n_channels=3, hidden_dim=16)
    kwargs.update(bad)
    with pytest.raises(ValueError, match="positive integer"):
        Critic(**kwargs)


@pytest.mark.parametrize("block_len,n_channels,batch", [
    (64, 3, 2), (64, 9, 5), (16, 3, 64), (8, 3, 1), (1, 1, 3),
])
def test_independent_dimensions_and_batch(block_len, n_channels, batch):
    d = Critic(block_len, n_channels, hidden_dim=16)
    s = d(_window(batch, block_len, n_channels))
    assert s.shape == (batch,) and s.dtype == torch.float32 and torch.isfinite(s).all()


# 2. Data path

def test_forward_calls_layers_in_reference_order(model):
    order = []
    names = ["input_conv", "hidden1", "hidden2", "hidden3", "output_layer"]
    handles = [getattr(model, n).register_forward_hook(lambda m, i, o, n=n: order.append(n)) for n in names]
    try:
        with torch.no_grad():
            model(_window())
    finally:
        for h in handles:
            h.remove()
    assert order == names


def test_activation_called_four_times_at_hidden_width(model):
    """After input_conv, hidden1, hidden2, hidden3; never after output_layer."""
    calls = []
    handle = model.act.register_forward_hook(lambda m, i, o: calls.append(tuple(i[0].shape)))
    try:
        with torch.no_grad():
            model(_window(batch=5))
    finally:
        handle.remove()
    assert calls == [(5, H)] * 4


def test_input_conv_equals_affine_over_step_major_flattened_window(model):
    """Conv1d(kernel=stride=2M) on exactly 2M steps == one affine map.
    Weight layout (H, C, 2M) pairs with window (B, 2M, C) via permute(0, 2, 1)."""
    w = _window(batch=4)
    Wmat = model.input_conv.weight.detach().permute(0, 2, 1).reshape(H, W * C)   # (H, 2M*C)
    b = model.input_conv.bias.detach()
    expected = w.reshape(4, W * C) @ Wmat.T + b
    got = []
    handle = model.input_conv.register_forward_hook(lambda m, i, o: got.append(o.detach().squeeze(-1)))
    try:
        with torch.no_grad():
            model(w)
    finally:
        handle.remove()
    torch.testing.assert_close(got[0], expected, atol=1e-5, rtol=1e-5)


def test_output_is_bias_when_output_weights_are_zero(model):
    with torch.no_grad():
        model.output_layer.weight.zero_()
        s = model(_window(batch=3))
    torch.testing.assert_close(s, model.output_layer.bias.detach().expand(3), rtol=0, atol=0)


def test_score_is_unbounded_and_unsquashed(model):
    """No sigmoid/tanh/clamp on the Wasserstein score: scaling the input scales the
    pre-activation linearly, so the score must keep growing, not saturate."""
    w = _window(batch=4)
    with torch.no_grad():
        s1 = model(w)
        s2 = model(w * 1e3)
    assert torch.isfinite(s2).all()
    assert (s2 - s1).abs().max().item() > 1.0


# 3. Malformed input and purity

@pytest.mark.parametrize("problem", [
    "length_M", "length_3M", "length_4M", "wrong_channels", "rank_2", "rank_4", "transposed",
])
def test_malformed_inputs_raise(model, problem):
    if problem == "length_M":
        w = torch.randn(4, M, C)
    elif problem == "length_3M":
        w = torch.randn(4, 3 * M, C)
    elif problem == "length_4M":
        w = torch.randn(4, 4 * M, C)       # would give 2 conv output positions if unchecked
    elif problem == "wrong_channels":
        w = torch.randn(4, W, C + 1)
    elif problem == "rank_2":
        w = torch.randn(W, C)
    elif problem == "rank_4":
        w = torch.randn(4, W, C, 1)
    elif problem == "transposed":
        w = torch.randn(4, C, W)
    with pytest.raises(ValueError):
        model(w)


def test_forward_does_not_modify_input(model):
    w = _window()
    orig = w.clone()
    model(w)
    torch.testing.assert_close(w, orig, rtol=0, atol=0)


def test_forward_does_not_consume_global_rng(model):
    w = _window()
    before = torch.get_rng_state().clone()
    with torch.no_grad():
        model(w)
    assert torch.equal(before, torch.get_rng_state())


def test_eval_equals_train_and_repeatable(model):
    w = _window()
    model.train()
    with torch.no_grad():
        a = model(w)
    model.eval()
    with torch.no_grad():
        b, c = model(w), model(w)
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    torch.testing.assert_close(b, c, rtol=0, atol=0)


def test_every_step_and_axis_affects_score(model):
    """The critic must see the whole window: perturbing any (t, axis) changes the score."""
    w = _window(batch=1)
    with torch.no_grad():
        base = model(w)
        for t in (0, M - 1, M, W - 1):
            for a in range(C):
                w2 = w.clone()
                w2[0, t, a] += 3.0
                assert (model(w2) - base).abs().item() > 1e-7, (t, a)


# 4. Autograd, optimizer, state

def test_gradients_reach_input_and_every_parameter(model):
    model.train()
    w = _window().requires_grad_()
    model(w).sum().backward()
    assert w.grad is not None and torch.isfinite(w.grad).all() and w.grad.abs().sum().item() > 0
    for name, p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
        assert p.grad.abs().sum().item() > 0, name


def test_optimizer_step_updates_every_submodule(model):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    model(_window()).mean().backward()
    opt.step()
    changed = {n for n, p in model.named_parameters() if not torch.equal(before[n], p.detach())}
    for prefix in ("input_conv.", "hidden1.", "hidden2.", "hidden3.", "output_layer."):
        assert any(n.startswith(prefix) for n in changed), prefix


def test_state_dict_round_trip_and_key_names(model, tmp_path):
    model.eval()
    w = _window()
    with torch.no_grad():
        expected = model(w)
    path = tmp_path / "dg.pt"
    torch.save(model.state_dict(), path)
    state = torch.load(path, map_location="cpu", weights_only=True)
    assert set(state) == {
        "input_conv.weight", "input_conv.bias", "hidden1.weight", "hidden1.bias",
        "hidden2.weight", "hidden2.bias", "hidden3.weight", "hidden3.bias",
        "output_layer.weight", "output_layer.bias",
    }
    restored = Critic(M, C, hidden_dim=H, negative_slope=0.17)
    restored.load_state_dict(state, strict=True)
    restored.eval()
    with torch.no_grad():
        torch.testing.assert_close(restored(w), expected, rtol=0, atol=0)


def test_state_dict_of_different_config_is_rejected(model):
    with pytest.raises(RuntimeError):
        model.load_state_dict(Critic(M + 1, C, hidden_dim=H).state_dict(), strict=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not present")
def test_cuda_forward_backward():
    dev = torch.device("cuda")
    d = Critic(M, C, hidden_dim=H).to(dev)
    w = _window().to(dev).requires_grad_()
    d(w).sum().backward()
    assert w.grad is not None


@pytest.mark.skipif(not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available(),
                    reason="Apple MPS not present")
def test_mps_forward_backward():
    dev = torch.device("mps")
    d = Critic(M, C, hidden_dim=H).to(dev)
    w = _window().to(dev).requires_grad_()
    d(w).sum().backward()
    assert w.grad is not None