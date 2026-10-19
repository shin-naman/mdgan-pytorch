"""Critic properties required by WGAN-GP, and numerical regimes.

The gradient-penalty helper here is a REFERENCE implementation used to
exercise the module; the trainer's own gradient_penalty is tested in
tests/trainer and must agree with this one. Multi-network wiring lives in
tests/integration.

Run: python -m pytest tests/D_G/test_dg_numerics.py -v
"""

import pytest
import torch

from mdgan.models.critic import Critic

M, C, H = 8, 3, 24
W = 2 * M


@pytest.fixture
def model():
    torch.manual_seed(2023)
    return Critic(M, C, hidden_dim=H)


def _gp(critic, real, fake, lam=10.0):
    """Gradient penalty exactly as in Gulrajani et al. 2017 / Kawada 2022 eq. 5."""
    b = real.shape[0]
    eps = torch.rand(b, 1, 1, dtype=real.dtype)
    x_hat = (eps * real + (1 - eps) * fake).requires_grad_(True)
    score = critic(x_hat)
    grad, = torch.autograd.grad(score.sum(), x_hat, create_graph=True)
    norm = grad.flatten(1).norm(dim=1)
    return lam * ((norm - 1) ** 2).mean(), norm


# 1. Gradient penalty support

def test_gradient_penalty_is_computable_and_backprops_to_critic_weights(model):
    """The GP depends on d score / d window, which for a piecewise-linear critic
    is a product of weight matrices and LReLU slope masks. Biases enter only
    through the non-differentiable mask selection, so autograd connects the GP
    to every WEIGHT and to no BIAS (grad is None). Biases are trained by the
    Wasserstein term E[D(fake)] - E[D(real)], not by the penalty."""
    real, fake = torch.randn(16, W, C), torch.randn(16, W, C)
    gp, norm = _gp(model, real, fake)
    assert gp.requires_grad and torch.isfinite(gp)
    assert norm.shape == (16,) and (norm > 0).all()
    gp.backward()
    for name, p in model.named_parameters():
        if name.endswith(".weight"):
            assert p.grad is not None and torch.isfinite(p.grad).all(), name
            assert p.grad.abs().sum().item() > 0, name
        else:
            # GP contributes nothing to biases. Autograd reports that as None
            # (Linear: no edge) or as an all-zero tensor (Conv1d: fused backward
            # with a bias slot). Either is the same statement.
            assert p.grad is None or p.grad.abs().sum().item() == 0, name


def test_wasserstein_term_reaches_biases_where_gp_cannot(model):
    """The full critic loss trains every parameter EXCEPT output_layer.bias:
    d(E[D(fake)] - E[D(real)]) / d(bias_out) = 1 - 1 = 0, and the GP adds nothing.
    D is defined up to an additive constant under the Wasserstein objective."""
    real, fake = torch.randn(16, W, C), torch.randn(16, W, C)
    loss = model(fake).mean() - model(real).mean() + _gp(model, real, fake)[0]
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
        if name == "output_layer.bias":
            assert p.grad.abs().sum().item() == 0
        else:
            assert p.grad.abs().sum().item() > 0, name


def test_per_sample_gradients_are_independent(model):
    """d score[i] / d window[j] == 0 for i != j. This is what makes the per-sample
    penalty meaningful; any cross-sample coupling (e.g. BatchNorm) breaks it."""
    w = torch.randn(6, W, C, requires_grad=True)
    s = model(w)
    for i in range(6):
        g, = torch.autograd.grad(s[i], w, retain_graph=True)
        mask = torch.ones(6, dtype=torch.bool)
        mask[i] = False
        assert g[mask].abs().sum().item() == 0
        assert g[i].abs().sum().item() > 0


def test_score_sum_gradient_equals_stack_of_per_sample_gradients(model):
    """Justifies the usual trick grad(score.sum()) in the GP."""
    w = torch.randn(5, W, C, requires_grad=True)
    s = model(w)
    g_sum, = torch.autograd.grad(s.sum(), w, retain_graph=True)
    for i in range(5):
        g_i, = torch.autograd.grad(s[i], w, retain_graph=True)
        torch.testing.assert_close(g_sum[i], g_i[i], rtol=0, atol=0)


def test_input_gradient_norm_scale_at_init_is_recorded():
    """Documents the init regime at production width with unit-scale windows:
    ||d score / d window|| per sample. Relevant to lambda=10 and the sigma
    decision; not a requirement."""
    torch.manual_seed(0)
    d = Critic()
    w = torch.randn(64, 128, 3, requires_grad=True)
    g, = torch.autograd.grad(d(w).sum(), w)
    n = g.flatten(1).norm(dim=1)
    assert torch.isfinite(n).all()
    assert 1e-4 < n.median().item() < 1e2, n.median().item()


# 2. Numerical regimes

@pytest.mark.parametrize("scale", [1e2, 1e4, 1e6])
def test_large_windows_stay_finite(model, scale):
    with torch.no_grad():
        assert torch.isfinite(model(torch.randn(8, W, C) * scale)).all()


def test_zero_window_finite_score_and_gradients(model):
    w = torch.zeros(4, W, C, requires_grad=True)
    model(w).sum().backward()
    assert torch.isfinite(w.grad).all()
    for n, p in model.named_parameters():
        assert torch.isfinite(p.grad).all(), n


def test_batch_independence_and_permutation(model):
    w = torch.randn(8, W, C)
    perm = torch.randperm(8)
    with torch.no_grad():
        alone = model(w[:1])
        together = model(w)
        permuted = model(w[perm])
    torch.testing.assert_close(alone, together[:1], atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(permuted, together[perm], atol=1e-6, rtol=1e-5)


def test_fp64_round_trip(model):
    model.double()
    s = model(torch.randn(4, W, C, dtype=torch.float64))
    assert s.dtype == torch.float64 and torch.isfinite(s).all()


def test_bf16_autocast_smoke(model):
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        s = model(torch.randn(4, W, C))
    assert torch.isfinite(s.float()).all()


def test_estimation_and_training_batch_sizes(model):
    with torch.no_grad():
        assert model(torch.randn(64, W, C)).shape == (64,)
        assert model(torch.randn(8192, W, C)).shape == (8192,)