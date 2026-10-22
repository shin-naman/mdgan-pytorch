"""Tests for mdgan.losses.

Closed-form checks use a toy LINEAR critic D(x) = <w, x>: its input gradient is w
for every sample, so the gradient norm is ||w|| exactly and the penalty is
lambda * (||w|| - 1)^2. The real critic (D_G) is used for the properties that
depend on its architecture (which parameters the penalty reaches).

Run: python -m pytest tests/test_losses.py -v
     python -m pytest -m learning tests/test_losses.py -v     (critic learns to separate)
"""

import pytest
import torch
from torch import nn

from conftest import SMALL
from mdgan.losses import DEFAULT_GP_LAMBDA, critic_loss, generator_loss, gradient_penalty

LAT, M, C = SMALL["latent_dim"], SMALL["block_len"], SMALL["n_channels"]
W = 2 * M


class LinearCritic(nn.Module):
    """D(x) = <w, x>; records its last input."""

    def __init__(self, norm: float, seed: int = 0):
        super().__init__()
        w = torch.randn(W, C, generator=torch.Generator().manual_seed(seed))
        self.w = nn.Parameter(w / w.norm() * norm)
        self.last_input = None

    def forward(self, x):
        self.last_input = x.detach().clone()
        return (x * self.w).sum(dim=(1, 2))


def pair(batch=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(batch, W, C, generator=g) + 1.0, torch.randn(batch, W, C, generator=g)


# =============================================================================
class TestGradientPenalty:
    @pytest.mark.parametrize("norm", [0.5, 1.0, 2.0, 3.0])
    def test_closed_form_on_a_linear_critic(self, norm):
        real, fake = pair()
        gp, norms = gradient_penalty(LinearCritic(norm), real, fake, gp_lambda=10.0)
        torch.testing.assert_close(norms, torch.full((16,), norm), atol=1e-5, rtol=0)
        torch.testing.assert_close(gp, torch.tensor(10.0 * (norm - 1) ** 2), atol=1e-4, rtol=1e-5)

    def test_linear_in_lambda_and_default_is_10(self):
        real, fake = pair()
        d = LinearCritic(2.0)
        g = lambda lam: gradient_penalty(d, real, fake, gp_lambda=lam)[0]   # noqa: E731
        torch.testing.assert_close(g(5.0) * 2, g(10.0))
        assert DEFAULT_GP_LAMBDA == 10.0
        torch.testing.assert_close(gradient_penalty(d, real, fake)[0], g(10.0))

    def test_interpolate_lies_on_the_per_sample_segment(self):
        """x_hat_i = eps_i * real_i + (1 - eps_i) * fake_i, one eps per sample, eps in [0, 1)."""
        real, fake = pair(batch=32)
        d = LinearCritic(1.0)
        gradient_penalty(d, real, fake, generator=torch.Generator().manual_seed(1))
        ratio = (d.last_input - fake) / (real - fake)                     # = eps_i everywhere in sample i
        per_sample = ratio.flatten(1)
        torch.testing.assert_close(per_sample, per_sample[:, :1].expand_as(per_sample), atol=1e-4, rtol=1e-4)
        eps = per_sample[:, 0]
        assert (eps >= 0).all() and (eps < 1).all() and eps.std() > 0.1   # varies across samples

    def test_matches_the_reference_formula_with_the_same_eps(self, dg):
        real, fake = pair()
        gp, norms = gradient_penalty(dg, real, fake, generator=torch.Generator().manual_seed(7))
        eps = torch.rand(16, 1, 1, generator=torch.Generator().manual_seed(7))
        x = (eps * real + (1 - eps) * fake).requires_grad_(True)
        g, = torch.autograd.grad(dg(x).sum(), x)
        n = g.flatten(1).norm(dim=1)
        torch.testing.assert_close(norms, n, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(gp, 10.0 * ((n - 1) ** 2).mean(), rtol=1e-5, atol=1e-6)

    def test_reaches_every_critic_weight_and_no_bias(self, dg):
        """Piecewise-linear critic: grad_x D is a product of weights and slope masks.
        Autograd reports the bias case as None (Linear) or exact zero (Conv1d)."""
        gp, _ = gradient_penalty(dg, *pair())
        gp.backward()
        for name, p in dg.named_parameters():
            if name.endswith(".weight"):
                assert p.grad is not None and p.grad.abs().sum() > 0, name
            else:
                assert p.grad is None or p.grad.abs().sum() == 0, name

    def test_never_reaches_the_generators(self, three_nets):
        gz, gy, dg = three_nets
        fake = torch.cat([gy.generate(gz.unroll(torch.rand(16, LAT), 4)[-1])] * 2, dim=1)   # requires grad
        assert fake.requires_grad
        gp, _ = gradient_penalty(dg, torch.randn_like(fake), fake)
        gp.backward()
        assert all(p.grad is None for p in (*gz.parameters(), *gy.parameters()))

    def test_reproducible_with_generator_and_leaves_global_rng(self):
        real, fake = pair()
        d = LinearCritic(2.0)
        before = torch.get_rng_state().clone()
        gradient_penalty(d, real, fake, generator=torch.Generator().manual_seed(3)); a = d.last_input
        gradient_penalty(d, real, fake, generator=torch.Generator().manual_seed(3)); b = d.last_input
        gradient_penalty(d, real, fake, generator=torch.Generator().manual_seed(4)); c = d.last_input
        assert torch.equal(before, torch.get_rng_state())
        assert torch.equal(a, b) and not torch.equal(a, c)

    def test_norms_are_detached_per_sample(self, dg):
        _, norms = gradient_penalty(dg, *pair(batch=5))
        assert norms.shape == (5,) and not norms.requires_grad

    @pytest.mark.parametrize("shapes", [((4, W, C), (4, W, C + 1)), ((4, W, C), (3, W, C)), ((W, C), (W, C))])
    def test_shape_mismatch_rejected(self, dg, shapes):
        with pytest.raises(ValueError):
            gradient_penalty(dg, torch.randn(*shapes[0]), torch.randn(*shapes[1]))


# =============================================================================
class TestCriticLoss:
    def test_value_and_diagnostics_in_closed_form(self):
        """D = <w, .>, ||w|| = 2: loss = mean D(fake) - mean D(real) + lambda * (2 - 1)^2."""
        real, fake = pair()
        d = LinearCritic(2.0)
        loss, info = critic_loss(d, real, fake, gp_lambda=10.0)
        d_real, d_fake = d(real).mean(), d(fake).mean()
        torch.testing.assert_close(loss, d_fake - d_real + 10.0, atol=1e-4, rtol=1e-5)
        torch.testing.assert_close(info["wasserstein"], (d_real - d_fake).detach(), atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(info["gradient_penalty"], torch.tensor(10.0), atol=1e-4, rtol=1e-5)
        torch.testing.assert_close(info["grad_norm"], torch.tensor(2.0), atol=1e-5, rtol=0)
        assert not any(v.requires_grad for v in info.values())

    def test_sign_convention(self):
        """Minimizing the loss must push D(real) up and D(fake) down."""
        real, fake = pair()
        d = LinearCritic(1.0)                                             # unit norm: penalty is 0
        loss, _ = critic_loss(d, real, fake)
        loss.backward()
        expected = (fake.mean(0) - real.mean(0))                          # d loss / d w
        torch.testing.assert_close(d.w.grad, expected, atol=1e-4, rtol=1e-4)

    def test_detaches_fake_itself(self, three_nets):
        gz, gy, dg = three_nets
        fake = torch.cat([gy.generate(gz.unroll(torch.rand(16, LAT), 4)[-1])] * 2, dim=1)
        loss, _ = critic_loss(dg, torch.randn_like(fake), fake)          # caller forgot .detach()
        loss.backward()
        assert all(p.grad is None for p in (*gz.parameters(), *gy.parameters()))
        assert all(p.grad is not None for p in dg.parameters())

    def test_one_step_moves_every_critic_parameter_except_output_bias(self, dg):
        """output_layer.bias: d(E[D(fake)] - E[D(real)])/db = 0 and the penalty adds nothing."""
        opt = torch.optim.Adam(dg.parameters(), lr=1e-4, betas=(0.0, 0.9))
        before = {n: p.detach().clone() for n, p in dg.named_parameters()}
        loss, _ = critic_loss(dg, *pair(batch=32))
        opt.zero_grad(); loss.backward(); opt.step()
        for n, p in dg.named_parameters():
            moved = not torch.equal(before[n], p.detach())
            assert moved == (n != "output_layer.bias"), n


# =============================================================================
class TestGeneratorLoss:
    def test_value(self):
        _, fake = pair()
        d = LinearCritic(1.5)
        torch.testing.assert_close(generator_loss(d, fake), -d(fake).mean())

    def test_gradient_reaches_both_generators_with_the_opposite_sign_of_the_critic(self, three_nets):
        gz, gy, dg = three_nets
        fake = torch.cat([gy.generate(gz.unroll(torch.rand(16, LAT), 4)[-1])] * 2, dim=1)
        fake.retain_grad()
        generator_loss(dg, fake).backward()
        g_gen = fake.grad.clone()
        x = fake.detach().requires_grad_(True)
        g_critic, = torch.autograd.grad(dg(x).mean(), x)
        torch.testing.assert_close(g_gen, -g_critic, rtol=1e-5, atol=1e-7)
        for net in (gz, gy):
            for name, p in net.named_parameters():
                if name.endswith(".weight"):
                    assert p.grad is not None and p.grad.abs().sum() > 0, name

    def test_rejects_wrong_rank(self, dg):
        with pytest.raises(ValueError):
            generator_loss(dg, torch.randn(W, C))


# =============================================================================
@pytest.mark.learning
def test_critic_learns_to_separate_two_distributions(dg):
    """The most common silent GAN bug is a flipped sign in the critic loss. Train
    only the critic on real ~ N(0, 1) vs fake ~ N(0, 4) windows: its Wasserstein
    estimate must become clearly positive (D(real) > D(fake)), with input-gradient
    norms near 1. A flipped sign drives the estimate negative instead.
    For reference, f(x) = ||x|| is 1-Lipschitz and separates them by ~sqrt(48) ~ 6.9."""
    torch.manual_seed(0)
    opt = torch.optim.Adam(dg.parameters(), lr=1e-3, betas=(0.0, 0.9))
    g = torch.Generator().manual_seed(0)
    for _ in range(300):
        real = torch.randn(64, W, C, generator=g)
        fake = 2.0 * torch.randn(64, W, C, generator=g)
        loss, _ = critic_loss(dg, real, fake, generator=g)
        opt.zero_grad(); loss.backward(); opt.step()
    real = torch.randn(512, W, C, generator=g)
    fake = 2.0 * torch.randn(512, W, C, generator=g)
    _, info = critic_loss(dg, real, fake, generator=g)
    assert info["wasserstein"] > 1.0, info
    assert 0.5 < info["grad_norm"] < 1.5, info