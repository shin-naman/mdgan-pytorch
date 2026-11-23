"""Tests for mdgan.rewards (generator rewards; not in the papers, off by default).

The two statistics are checked on planted cases with known answers, the reward on a hand
construction that draws from a generator in the documented order, and the gradient claims
(which network each term reaches) directly.

Run: python -m pytest tests/test_rewards.py -v
"""

import numpy as np
import pytest
import torch

from conftest import SMALL
from mdgan.config import Config
from mdgan.ops import mean_operation as _center
from mdgan.rewards import (
    block_sum_correlations, block_sum_variance, generator_rewards, noise_use_share, reward_metric_names,
)
from mdgan.rollout import _initial_latents

LAT, M, C = SMALL["latent_dim"], SMALL["block_len"], SMALL["n_channels"]


def config(**kw) -> Config:
    return Config(**{**SMALL, "burn_in": 2, "reward_chains": 32, "window_offset_max_steps": 4 * M, **kw})


# =============================================================================
class TestNoiseUseShare:
    @pytest.mark.statistical
    @pytest.mark.parametrize("s", [0.1, 0.5, 0.9])
    def test_recovers_a_planted_share(self, s):
        """a = sqrt(1 - s) z + sqrt(s) e1, b likewise with e2: Corr(a, b) = 1 - s. With 40,000 chains the
        standard error of a correlation is below 0.005; the tolerance is 4 of them."""
        g = torch.Generator().manual_seed(0)
        z, e1, e2 = (torch.randn(40_000, 5, generator=g) for _ in range(3))
        share = noise_use_share((1 - s) ** 0.5 * z + s ** 0.5 * e1, (1 - s) ** 0.5 * z + s ** 0.5 * e2)
        assert share.shape == (5,)
        torch.testing.assert_close(share, torch.full((5,), s), rtol=0, atol=0.02)

    def test_zero_when_the_noise_is_ignored_and_per_dimension(self):
        g = torch.Generator().manual_seed(1)
        z, e = torch.randn(2000, 3, generator=g), torch.randn(2000, 3, generator=g)
        a, b = z.clone(), z.clone()
        b[:, 2] = e[:, 2]                                   # only dimension 2 is redrawn
        share = noise_use_share(a, b)
        torch.testing.assert_close(share[:2], torch.zeros(2), rtol=0, atol=1e-6)
        assert abs(float(share[2]) - 1) < 0.1

    def test_unchanged_by_shift_and_scale_of_either_argument(self):
        g = torch.Generator().manual_seed(2)
        a, b = torch.randn(500, 4, generator=g), torch.randn(500, 4, generator=g)
        torch.testing.assert_close(noise_use_share(3 * a + 7, 0.5 * b - 2), noise_use_share(a, b), rtol=0, atol=1e-5)

    def test_shapes_must_agree(self):
        with pytest.raises(ValueError, match="B, n_z"):
            noise_use_share(torch.zeros(4, 3), torch.zeros(4, 2))


# =============================================================================
class TestBlockSumCorrelations:
    def test_equals_numpy_corrcoef_per_lag(self):
        s = torch.randn(50, 6, 3, generator=torch.Generator().manual_seed(3), dtype=torch.float64).cumsum(1)
        got = block_sum_correlations(s, 4)
        expected = [np.corrcoef(s[:, :-k].reshape(-1).numpy(), s[:, k:].reshape(-1).numpy())[0, 1] for k in (1, 2, 3, 4)]
        np.testing.assert_allclose(got.numpy(), expected, rtol=1e-12)

    def test_equals_the_dataset_targets_on_one_block_grid(self):
        """The dataset's targets average the M grid positions; with exactly L blocks of data only position 0
        holds L blocks, and the two computations must agree."""
        from mdgan.data.dataset import WindowDataset
        from mdgan.data.preprocess import SystemData
        d = np.random.default_rng(4).normal(size=(30, 4 * M, 3)).cumsum(axis=1).astype(np.float32)
        ds = WindowDataset(SystemData(name="s", displacements=d, dt_ps=10.0, box_nm=np.full((30, 3), 4.75), meta={}),
                           block_len=M, step_skip=1, window_offset_max_steps=2 * M)
        sums = torch.from_numpy(d.astype(np.float64)).reshape(30, 4, M, 3).sum(2)
        np.testing.assert_allclose(block_sum_correlations(sums, 3).numpy(), ds.block_sum_correlations(4), atol=1e-5)

    @pytest.mark.parametrize("shape,lag", [((8, 4, 3), 4), ((8, 4, 3), 0), ((8, 4), 1)])
    def test_invalid_arguments(self, shape, lag):
        with pytest.raises(ValueError, match="max_lag"):
            block_sum_correlations(torch.zeros(shape), lag)


# =============================================================================
class TestGeneratorRewards:
    def test_matches_hand_construction_in_documented_draw_order(self, gz, gy):
        c = config(noise_use_min=0.9, corr_reward_blocks=4)
        targets = torch.tensor([-0.4, 0.1, 0.0])
        loss, stats = generator_rewards(gz, gy, c, targets, generator=torch.Generator().manual_seed(6))
        g = torch.Generator().manual_seed(6)
        z = gz.unroll(torch.rand(32, LAT, generator=g), 2, generator=g)[-1]
        a = gz(z, gz.sample_noise(32, generator=g))
        b = gz(z, gz.sample_noise(32, generator=g))
        share = noise_use_share(a, b)
        states = gz.unroll(z, 3, generator=g)
        y = torch.cat([gy.generate(s, generator=g) for s in states], dim=1)
        y = y - y.mean(0, keepdim=True)
        rho = block_sum_correlations(y.reshape(32, 4, M, C).sum(2), 3)
        penalty, error = torch.relu(0.9 - share).square().mean(), (rho - targets).square().sum()
        torch.testing.assert_close(loss, 100.0 * penalty + 10.0 * error, rtol=1e-6, atol=0)
        torch.testing.assert_close(stats["noise_use_share"], share.mean(), rtol=1e-6, atol=0)
        torch.testing.assert_close(stats["noise_use_penalty"], penalty, rtol=1e-6, atol=0)
        torch.testing.assert_close(stats["corr_error"], error, rtol=1e-6, atol=0)
        torch.testing.assert_close(stats["reward_loss"], loss, rtol=0, atol=0)
        assert tuple(stats) == reward_metric_names(c) and not any(v.requires_grad for v in stats.values())

    def test_weights_scale_their_terms(self, gz, gy):
        t = torch.zeros(2)
        both = lambda **kw: generator_rewards(gz, gy, config(noise_use_min=0.9, corr_reward_blocks=3, **kw), t,   # noqa: E731
                                              generator=torch.Generator().manual_seed(7))
        (base, s), (nu0, _), (cr0, _) = both(), both(noise_use_weight=0.0), both(corr_reward_weight=0.0)
        torch.testing.assert_close(nu0, 10.0 * s["corr_error"], rtol=1e-6, atol=0)
        torch.testing.assert_close(cr0, 100.0 * s["noise_use_penalty"], rtol=1e-6, atol=0)
        torch.testing.assert_close(base, nu0 + cr0, rtol=1e-6, atol=0)

    def test_noise_use_term_reaches_g_z_through_one_transition_only(self, gz, gy):
        """No gradient for G_Y; and the same gradient as the hand-built one-transition term from a detached
        z_c, so nothing flows back through the burn-in."""
        c = config(noise_use_min=0.99)
        loss, _ = generator_rewards(gz, gy, c, generator=torch.Generator().manual_seed(8))
        grads = torch.autograd.grad(loss, list(gz.parameters()), allow_unused=True)
        assert all(p.grad is None for p in gy.parameters())
        g = torch.Generator().manual_seed(8)
        with torch.no_grad():
            z = gz.unroll(torch.rand(32, LAT, generator=g), 2, generator=g)[-1]
        share = noise_use_share(gz(z, gz.sample_noise(32, generator=g)), gz(z, gz.sample_noise(32, generator=g)))
        ref = torch.autograd.grad(100.0 * torch.relu(0.99 - share).square().mean(), list(gz.parameters()), allow_unused=True)
        assert any(a is not None and a.abs().max() > 0 for a in grads)
        for a, b in zip(grads, ref, strict=True):
            assert (a is None) == (b is None)
            if a is not None:
                torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-9)

    def test_noise_use_term_stays_detached_from_the_burn_in_when_the_chain_carries_gradients(self, gz, gy):
        """With the correlation reward on, the burn-in is differentiated for that term. At correlation weight 0
        the gradient must still be the one-transition gradient of the noise-use term."""
        c = config(noise_use_min=0.99, corr_reward_blocks=3, corr_reward_weight=0.0)
        loss, _ = generator_rewards(gz, gy, c, torch.zeros(2), generator=torch.Generator().manual_seed(8))
        grads = torch.autograd.grad(loss, list(gz.parameters()), allow_unused=True)
        g = torch.Generator().manual_seed(8)
        with torch.no_grad():
            z = gz.unroll(torch.rand(32, LAT, generator=g), 2, generator=g)[-1]
        share = noise_use_share(gz(z, gz.sample_noise(32, generator=g)), gz(z, gz.sample_noise(32, generator=g)))
        ref = torch.autograd.grad(100.0 * torch.relu(0.99 - share).square().mean(), list(gz.parameters()), allow_unused=True)
        for a, b in zip(grads, ref, strict=True):
            torch.testing.assert_close(torch.zeros_like(b) if a is None else a, b, rtol=1e-5, atol=1e-9)

    def test_noise_use_term_is_zero_once_every_dimension_is_above_the_minimum(self, gz, gy):
        _, s = generator_rewards(gz, gy, config(noise_use_min=0.99), generator=torch.Generator().manual_seed(9))
        low = float(s["noise_use_share"]) / 1000                # far below every dimension's share
        loss, s = generator_rewards(gz, gy, config(noise_use_min=low), generator=torch.Generator().manual_seed(9))
        assert float(loss.detach()) == 0.0 and float(s["noise_use_penalty"]) == 0.0

    def test_correlation_term_reaches_both_generators_and_the_whole_chain(self, gz, gy):
        c = config(corr_reward_blocks=3)
        loss, _ = generator_rewards(gz, gy, c, torch.tensor([0.9, 0.9]), generator=torch.Generator().manual_seed(10))
        loss.backward()
        assert all(p.grad is not None and p.grad.abs().max() > 0 for p in gy.parameters())
        assert all(p.grad is not None for p in gz.parameters()) and gz.input_layer.weight.grad.abs().max() > 0

    def test_correlation_term_gradient_is_that_of_the_whole_chain(self, gz, gy):
        """Backpropagation through every transition, burn-in included: the gradient for both generators equals
        that of the hand-built term on an undetached chain."""
        c, targets = config(corr_reward_blocks=4), torch.tensor([-0.4, 0.1, 0.0])
        params = [*gz.parameters(), *gy.parameters()]
        loss, _ = generator_rewards(gz, gy, c, targets, generator=torch.Generator().manual_seed(10))
        grads = torch.autograd.grad(loss, params, allow_unused=True)
        g = torch.Generator().manual_seed(10)
        states = gz.unroll(gz.unroll(torch.rand(32, LAT, generator=g), 2, generator=g)[-1], 3, generator=g)
        y = torch.cat([gy.generate(s, generator=g) for s in states], dim=1)
        rho = block_sum_correlations((y - y.mean(0, keepdim=True)).reshape(32, 4, M, C).sum(2), 3)
        ref = torch.autograd.grad(10.0 * (rho - targets).square().sum(), params, allow_unused=True)
        scale = max(float(b.abs().max()) for b in ref if b is not None)
        for a, b in zip(grads, ref, strict=True):           # float32: a bias in front of a BatchNorm has gradient 0 +- 1e-5
            assert (a is None) == (b is None)
            if a is not None:
                torch.testing.assert_close(a, b, rtol=1e-3, atol=1e-4 * scale)

    def test_correlation_term_is_zero_at_its_own_targets_and_never_negative(self, gz, gy):
        c = config(corr_reward_blocks=4)
        g = torch.Generator().manual_seed(11)
        states = gz.unroll(gz.unroll(torch.rand(32, LAT, generator=g), 2, generator=g)[-1], 3, generator=g)
        y = torch.cat([gy.generate(s, generator=g) for s in states], dim=1)
        own = block_sum_correlations((y - y.mean(0, keepdim=True)).reshape(32, 4, M, C).sum(2), 3).detach()
        loss, s = generator_rewards(gz, gy, c, own, generator=torch.Generator().manual_seed(11))
        assert float(loss.detach()) < 1e-10 and float(s["corr_error"]) < 1e-10
        for shift in (-0.3, 0.3):
            assert float(generator_rewards(gz, gy, c, own + shift, generator=torch.Generator().manual_seed(11))[1]["reward_loss"]) > 0.2

    def test_mean_operation_switch_changes_the_blocks(self, gz, gy):
        t = torch.zeros(2)
        on = generator_rewards(gz, gy, config(corr_reward_blocks=3), t, generator=torch.Generator().manual_seed(12))[0]
        off = generator_rewards(gz, gy, config(corr_reward_blocks=3, mean_operation=False), t,
                                generator=torch.Generator().manual_seed(12))[0]
        assert float(on.detach()) != float(off.detach())

    def test_reproducible_and_leaves_global_rng(self, gz, gy):
        c, t = config(noise_use_min=0.5, corr_reward_blocks=3), torch.zeros(2)
        before = torch.get_rng_state().clone()
        a = generator_rewards(gz, gy, c, t, generator=torch.Generator().manual_seed(13))[0]
        b = generator_rewards(gz, gy, c, t, generator=torch.Generator().manual_seed(13))[0]
        assert torch.equal(a, b) and torch.equal(before, torch.get_rng_state())

    def test_metric_names(self):
        assert reward_metric_names(config()) == ()
        assert reward_metric_names(config(noise_use_min=0.5)) == ("noise_use_share", "noise_use_penalty", "reward_loss")
        assert reward_metric_names(config(corr_reward_blocks=2)) == ("corr_error", "reward_loss")

    def test_refuses_to_run_switched_off_or_without_targets(self, gz, gy):
        with pytest.raises(ValueError, match="no reward"):
            generator_rewards(gz, gy, config())
        for bad in (None, torch.zeros(3)):
            with pytest.raises(ValueError, match="corr_targets"):
                generator_rewards(gz, gy, config(corr_reward_blocks=3), bad)


# =============================================================================
# generator_rewards exactly as it was at commit 6e41f72 (Batch 8), before the variance reward existed. Kept
# verbatim as the reference that variance_reward_weight = 0 must reproduce bit for bit. Do not edit.
def _generator_rewards_6e41f72(gz, gy, config, corr_targets: torch.Tensor | None = None,
                      generator: torch.Generator | None = None) -> tuple[torch.Tensor, dict]:
    """(weighted sum of the enabled terms, {name: detached value} for reward_metric_names(config)).
    corr_targets: (corr_reward_blocks - 1,) on the models' device; needed when that reward is on."""
    n, blocks = config.reward_chains, config.corr_reward_blocks
    if not config.rewards_on:
        raise ValueError("no reward is enabled in this config")
    if blocks and (corr_targets is None or corr_targets.shape != (blocks - 1,)):
        raise ValueError(f"corr_targets must hold the {blocks - 1} target correlations")
    param = next(gz.parameters())

    def step(z):
        return gz(z, gz.sample_noise(n, device=z.device, dtype=z.dtype, generator=generator))

    z = _initial_latents(gz, n, generator, param.device, param.dtype)
    with torch.set_grad_enabled(blocks > 0):          # the noise-use term starts from a detached z_c
        for _ in range(config.burn_in):
            z = step(z)
    total, stats = 0.0, {}
    if config.noise_use_min > 0:
        share = noise_use_share(step(z.detach()), step(z.detach()))
        penalty = torch.relu(config.noise_use_min - share).square().mean()
        total = total + config.noise_use_weight * penalty
        stats["noise_use_share"], stats["noise_use_penalty"] = share.mean().detach(), penalty.detach()
    if blocks:
        states = [z]
        for _ in range(blocks - 1):
            states.append(step(states[-1]))
        y = torch.cat([gy.generate(s, generator=generator) for s in states], dim=1)        # (n, L*M, C)
        y = _center(y) if config.mean_operation else y
        sums = y.reshape(n, blocks, config.block_len, -1).sum(dim=2)                       # (n, L, C)
        error = (block_sum_correlations(sums, blocks - 1) - corr_targets).square().sum()
        total = total + config.corr_reward_weight * error
        stats["corr_error"] = error.detach()
    stats["reward_loss"] = total.detach()
    return total, stats


def _own_variance(gz, gy, c, seed):
    """V_gen of the side batch generator_rewards(..., generator=manual_seed(seed)) builds, by hand."""
    g = torch.Generator().manual_seed(seed)
    n, L = c.reward_chains, c.corr_reward_blocks
    z = gz.unroll(torch.rand(n, LAT, generator=g), c.burn_in, generator=g)[-1]
    if c.noise_use_min > 0:
        gz.sample_noise(n, generator=g), gz.sample_noise(n, generator=g)
    states = gz.unroll(z, L - 1, generator=g)
    y = torch.cat([gy.generate(s, generator=g) for s in states], dim=1)
    sums = (y - y.mean(0, keepdim=True)).reshape(n, L, M, C).sum(2)
    return sums, block_sum_variance(sums)


class TestBlockSumVariance:
    def test_equals_numpy_var_pooled(self):
        s = torch.randn(50, 6, 3, generator=torch.Generator().manual_seed(3), dtype=torch.float64) * 2.0 + 0.7
        np.testing.assert_allclose(float(block_sum_variance(s)), np.var(s.numpy()), rtol=1e-12)

    def test_equals_the_dataset_target_on_one_block_grid(self):
        from mdgan.data.dataset import WindowDataset
        from mdgan.data.preprocess import SystemData
        d = (np.random.default_rng(4).normal(size=(30, 4 * M, 3)) + 0.3).astype(np.float32)
        ds = WindowDataset(SystemData(name="s", displacements=d, dt_ps=10.0, box_nm=np.full((30, 3), 4.75), meta={}),
                           block_len=M, step_skip=1, scale_inputs=False, window_offset_max_steps=2 * M)
        sums = torch.from_numpy(d.astype(np.float64)).reshape(30, 4, M, 3).sum(2)
        np.testing.assert_allclose(float(block_sum_variance(sums)), ds.block_sum_variance(4), rtol=1e-6)

    def test_invalid_shape(self):
        with pytest.raises(ValueError, match="sums"):
            block_sum_variance(torch.zeros(8, 4))


class TestVarianceReward:
    """variance_reward_weight (not in the papers; off by default)."""

    @pytest.mark.parametrize("kw", [{"corr_reward_blocks": 4}, {"corr_reward_blocks": 4, "noise_use_min": 0.9},
                                    {"corr_reward_blocks": 2, "noise_use_min": 0.5, "corr_reward_weight": 3.0}],
                             ids=["corr", "corr_and_noise_use", "lag1_and_noise_use"])
    def test_weight_zero_is_the_batch8_reward_bit_for_bit(self, gz, gy, kw):
        """At the default weight the loss, every gradient, every logged value and the generator's state after the
        call equal those of generator_rewards at 6e41f72 exactly (torch.equal, not a tolerance)."""
        c, t = config(**kw), torch.linspace(-0.3, 0.1, kw["corr_reward_blocks"] - 1)
        params = [*gz.parameters(), *gy.parameters()]
        g_new, g_old = torch.Generator().manual_seed(21), torch.Generator().manual_seed(21)
        new, s_new = generator_rewards(gz, gy, c, t, generator=g_new)
        old, s_old = _generator_rewards_6e41f72(gz, gy, c, t, generator=g_old)
        assert torch.equal(new, old) and torch.equal(g_new.get_state(), g_old.get_state())
        assert list(s_new) == list(s_old) and all(torch.equal(s_new[k], s_old[k]) for k in s_new)
        for a, b in zip(torch.autograd.grad(new, params, allow_unused=True),
                        torch.autograd.grad(old, params, allow_unused=True), strict=True):
            assert (a is None and b is None) or torch.equal(a, b)

    def test_matches_hand_construction_and_draws_nothing_more(self, gz, gy):
        c = config(corr_reward_blocks=4, noise_use_min=0.9, variance_reward_weight=7.0)
        t, vt = torch.tensor([-0.4, 0.1, 0.0]), 0.37
        g = torch.Generator().manual_seed(22)
        loss, stats = generator_rewards(gz, gy, c, t, generator=g, variance_target=vt)
        g0 = torch.Generator().manual_seed(22)
        base, s0 = generator_rewards(gz, gy, config(corr_reward_blocks=4, noise_use_min=0.9), t, generator=g0)
        assert torch.equal(g.get_state(), g0.get_state())                     # the variance term drew nothing
        _, v = _own_variance(gz, gy, c, 22)
        penalty = torch.log(v / vt).square()
        torch.testing.assert_close(loss, base + 7.0 * penalty, rtol=1e-6, atol=0)
        torch.testing.assert_close(stats["block_var_ratio"], v / vt, rtol=1e-6, atol=0)
        torch.testing.assert_close(stats["variance_penalty"], penalty, rtol=1e-6, atol=0)
        assert all(torch.equal(stats[k], s0[k]) for k in s0 if k != "reward_loss")
        assert tuple(stats) == reward_metric_names(c) and not any(x.requires_grad for x in stats.values())

    def test_zero_at_the_target_and_symmetric_in_the_log_ratio(self, gz, gy):
        c = config(corr_reward_blocks=3, corr_reward_weight=0.0, variance_reward_weight=1.0)
        _, v = _own_variance(gz, gy, c, 23)
        v = float(v.detach())
        run = lambda target: generator_rewards(gz, gy, c, torch.zeros(2), generator=torch.Generator().manual_seed(23),  # noqa: E731
                                               variance_target=target)
        at, s_at = run(v)
        assert abs(float(at.detach())) < 1e-10 and abs(float(s_at["block_var_ratio"]) - 1) < 1e-6
        for f in (1.5, 2.0, 12.0):
            hi, s_hi = run(v * f)
            lo, s_lo = run(v / f)
            np.testing.assert_allclose(float(hi.detach()), float(lo.detach()), rtol=1e-4)
            np.testing.assert_allclose(float(hi.detach()), np.log(f) ** 2, rtol=1e-4)
            assert float(s_hi["block_var_ratio"]) < 1 < float(s_lo["block_var_ratio"])

    def test_gradient_reaches_both_generators(self, gz, gy):
        c = config(corr_reward_blocks=3, corr_reward_weight=0.0, variance_reward_weight=1.0)
        loss, _ = generator_rewards(gz, gy, c, torch.zeros(2), generator=torch.Generator().manual_seed(24),
                                    variance_target=10.0)
        loss.backward()
        assert all(p.grad is not None and p.grad.abs().max() > 0 for p in gy.parameters())
        assert gz.input_layer.weight.grad is not None and gz.input_layer.weight.grad.abs().max() > 0

    def test_gradient_is_that_of_the_hand_built_term(self, gz, gy):
        c = config(corr_reward_blocks=4, corr_reward_weight=0.0, variance_reward_weight=2.5)
        params = [*gz.parameters(), *gy.parameters()]
        loss, _ = generator_rewards(gz, gy, c, torch.zeros(3), generator=torch.Generator().manual_seed(25),
                                    variance_target=0.8)
        grads = torch.autograd.grad(loss, params, allow_unused=True)
        _, v = _own_variance(gz, gy, c, 25)
        ref = torch.autograd.grad(2.5 * torch.log(v / 0.8).square(), params, allow_unused=True)
        scale = max(float(b.abs().max()) for b in ref if b is not None)
        for a, b in zip(grads, ref, strict=True):
            assert (a is None) == (b is None)
            if a is not None:
                torch.testing.assert_close(a, b, rtol=1e-3, atol=1e-4 * scale)

    def test_metric_names(self):
        assert reward_metric_names(config(corr_reward_blocks=2, variance_reward_weight=1.0)) == \
            ("corr_error", "block_var_ratio", "variance_penalty", "reward_loss")
        assert reward_metric_names(config(corr_reward_blocks=2, noise_use_min=0.5, variance_reward_weight=0.0)) == \
            ("noise_use_share", "noise_use_penalty", "corr_error", "reward_loss")

    @pytest.mark.parametrize("bad", [None, 0.0, -1.0])
    def test_refuses_to_run_without_a_positive_target(self, gz, gy, bad):
        with pytest.raises(ValueError, match="variance_target"):
            generator_rewards(gz, gy, config(corr_reward_blocks=3, variance_reward_weight=1.0), torch.zeros(2),
                              variance_target=bad)



# =============================================================================
# generator_rewards and block_sum_variance exactly as they were at a1358ef (Batch 9), before the sub-block reward.
# The reference that subcorr_block_steps = 0 must reproduce bit for bit, the variance reward included. Do not edit.
def _block_sum_variance_a1358ef(sums: torch.Tensor) -> torch.Tensor:
    """Variance of block sums pooled over chains, blocks and axes (one mean removed). sums: (B, L, C).
    The torch twin of WindowDataset.block_sum_variance for one block grid."""
    if sums.dim() != 3:
        raise ValueError(f"sums must be (B, L, C); got {tuple(sums.shape)}")
    x = sums.reshape(-1)
    return (x - x.mean()).square().mean()


def _generator_rewards_a1358ef(gz, gy, config, corr_targets: torch.Tensor | None = None,
                      generator: torch.Generator | None = None,
                      variance_target: float | None = None) -> tuple[torch.Tensor, dict]:
    """(weighted sum of the enabled terms, {name: detached value} for reward_metric_names(config)).
    corr_targets: (corr_reward_blocks - 1,) on the models' device; needed when that reward is on.
    variance_target: V_target > 0 (WindowDataset.block_sum_variance); needed when the variance reward is on."""
    n, blocks = config.reward_chains, config.corr_reward_blocks
    if not config.rewards_on:
        raise ValueError("no reward is enabled in this config")
    if blocks and (corr_targets is None or corr_targets.shape != (blocks - 1,)):
        raise ValueError(f"corr_targets must hold the {blocks - 1} target correlations")
    if config.variance_reward_weight > 0 and not (variance_target is not None and variance_target > 0):
        raise ValueError("variance_target must be the positive target block-sum variance")
    param = next(gz.parameters())

    def step(z):
        return gz(z, gz.sample_noise(n, device=z.device, dtype=z.dtype, generator=generator))

    z = _initial_latents(gz, n, generator, param.device, param.dtype)
    with torch.set_grad_enabled(blocks > 0):          # the noise-use term starts from a detached z_c
        for _ in range(config.burn_in):
            z = step(z)
    total, stats = 0.0, {}
    if config.noise_use_min > 0:
        share = noise_use_share(step(z.detach()), step(z.detach()))
        penalty = torch.relu(config.noise_use_min - share).square().mean()
        total = total + config.noise_use_weight * penalty
        stats["noise_use_share"], stats["noise_use_penalty"] = share.mean().detach(), penalty.detach()
    if blocks:
        states = [z]
        for _ in range(blocks - 1):
            states.append(step(states[-1]))
        y = torch.cat([gy.generate(s, generator=generator) for s in states], dim=1)        # (n, L*M, C)
        y = _center(y) if config.mean_operation else y
        sums = y.reshape(n, blocks, config.block_len, -1).sum(dim=2)                       # (n, L, C)
        error = (block_sum_correlations(sums, blocks - 1) - corr_targets).square().sum()
        total = total + config.corr_reward_weight * error
        stats["corr_error"] = error.detach()
        if config.variance_reward_weight > 0:                 # the same sums: no forward pass, no draw
            ratio = _block_sum_variance_a1358ef(sums) / variance_target
            penalty = torch.log(ratio).square()
            total = total + config.variance_reward_weight * penalty
            stats["block_var_ratio"], stats["variance_penalty"] = ratio.detach(), penalty.detach()
    stats["reward_loss"] = total.detach()
    return total, stats


def _side_chain(gz, gy, c, seed):
    """The centred side-chain blocks generator_rewards(..., generator=manual_seed(seed)) decodes, by hand."""
    g = torch.Generator().manual_seed(seed)
    n, L = c.reward_chains, c.corr_reward_blocks
    z = gz.unroll(torch.rand(n, LAT, generator=g), c.burn_in, generator=g)[-1]
    if c.noise_use_min > 0:
        gz.sample_noise(n, generator=g), gz.sample_noise(n, generator=g)
    states = gz.unroll(z, L - 1, generator=g)
    y = torch.cat([gy.generate(s, generator=g) for s in states], dim=1)
    return y - y.mean(0, keepdim=True)


class TestSubblockReward:
    """subcorr_block_steps (not in the papers; off by default). SMALL's M = 8; sub-blocks of 2 steps, 8 per 2 blocks."""

    ON = dict(corr_reward_blocks=2, subcorr_block_steps=2, subcorr_lags=7, subcorr_weight=3.0)

    @pytest.mark.parametrize("kw", [{"corr_reward_blocks": 4}, {"corr_reward_blocks": 2, "noise_use_min": 0.5},
                                    {"corr_reward_blocks": 2, "corr_reward_weight": 0.0, "variance_reward_weight": 2.0},
                                    {"corr_reward_blocks": 4, "noise_use_min": 0.9, "variance_reward_weight": 0.5}],
                             ids=["corr", "lag1_noise", "variance_corr_off", "everything_batch9"])
    def test_off_is_the_batch9_reward_bit_for_bit(self, gz, gy, kw):
        """subcorr_block_steps = 0 (default): loss, gradients, logged values and the generator state after the call
        equal generator_rewards at a1358ef exactly, the variance reward included."""
        c, t = config(**kw), torch.linspace(-0.3, 0.1, kw["corr_reward_blocks"] - 1)
        vt = 0.7 if kw.get("variance_reward_weight") else None
        params = [*gz.parameters(), *gy.parameters()]
        g_new, g_old = torch.Generator().manual_seed(31), torch.Generator().manual_seed(31)
        new, s_new = generator_rewards(gz, gy, c, t, generator=g_new, variance_target=vt)
        old, s_old = _generator_rewards_a1358ef(gz, gy, c, t, generator=g_old, variance_target=vt)
        assert torch.equal(new, old) and torch.equal(g_new.get_state(), g_old.get_state())
        assert list(s_new) == list(s_old) and all(torch.equal(s_new[k], s_old[k]) for k in s_new)
        for a, b in zip(torch.autograd.grad(new, params, allow_unused=True),
                        torch.autograd.grad(old, params, allow_unused=True), strict=True):
            assert (a is None and b is None) or torch.equal(a, b)

    def test_matches_hand_construction_and_draws_nothing_more(self, gz, gy):
        c = config(**self.ON, noise_use_min=0.9)
        t, st = torch.tensor([-0.2]), torch.linspace(-0.3, 0.2, 7)
        g = torch.Generator().manual_seed(32)
        loss, stats = generator_rewards(gz, gy, c, t, generator=g, subcorr_targets=st)
        g0 = torch.Generator().manual_seed(32)
        base, s0 = generator_rewards(gz, gy, config(corr_reward_blocks=2, noise_use_min=0.9), t, generator=g0)
        assert torch.equal(g.get_state(), g0.get_state())
        y = _side_chain(gz, gy, c, 32)
        sub = y.reshape(32, 8, 2, C).sum(2)
        err = (block_sum_correlations(sub, 7) - st).square().sum()
        torch.testing.assert_close(loss, base + 3.0 * err, rtol=1e-6, atol=0)
        torch.testing.assert_close(stats["subcorr_error"], err, rtol=1e-6, atol=0)
        assert tuple(stats) == reward_metric_names(c) and not any(x.requires_grad for x in stats.values())

    def test_zero_at_its_own_targets_and_positive_away(self, gz, gy):
        c = config(**{**self.ON, "corr_reward_weight": 0.0})
        own = block_sum_correlations(_side_chain(gz, gy, c, 33).reshape(32, 8, 2, C).sum(2), 7).detach()
        run = lambda st: generator_rewards(gz, gy, c, torch.zeros(1), generator=torch.Generator().manual_seed(33),  # noqa: E731
                                           subcorr_targets=st)
        loss, s = run(own)
        assert float(loss.detach()) < 1e-10 and float(s["subcorr_error"]) < 1e-10
        assert float(run(own + 0.2)[0].detach()) == pytest.approx(3.0 * 7 * 0.04, rel=1e-4)

    def test_variance_term_uses_the_sub_block_sums(self, gz, gy):
        c = config(**{**self.ON, "corr_reward_weight": 0.0, "subcorr_weight": 0.0, "variance_reward_weight": 1.0})
        loss, stats = generator_rewards(gz, gy, c, torch.zeros(1), generator=torch.Generator().manual_seed(34),
                                        variance_target=0.5, subcorr_targets=torch.zeros(7))
        v = block_sum_variance(_side_chain(gz, gy, c, 34).reshape(32, 8, 2, C).sum(2))
        torch.testing.assert_close(stats["block_var_ratio"], (v / 0.5).detach(), rtol=1e-6, atol=0)
        torch.testing.assert_close(loss, torch.log(v / 0.5).square(), rtol=1e-5, atol=0)

    def test_gradient_reaches_g_y_and_the_transitions(self, gz, gy):
        c = config(**{**self.ON, "corr_reward_weight": 0.0})
        loss, _ = generator_rewards(gz, gy, c, torch.zeros(1), generator=torch.Generator().manual_seed(35),
                                    subcorr_targets=torch.full((7,), 0.5))
        loss.backward()
        assert all(p.grad is not None and p.grad.abs().max() > 0 for p in gy.parameters())
        assert gz.input_layer.weight.grad is not None and gz.input_layer.weight.grad.abs().max() > 0

    def test_metric_names_and_refusal_without_targets(self, gz, gy):
        assert reward_metric_names(config(**self.ON, variance_reward_weight=1.0)) == \
            ("corr_error", "subcorr_error", "block_var_ratio", "variance_penalty", "reward_loss")
        for bad in (None, torch.zeros(6)):
            with pytest.raises(ValueError, match="subcorr_targets"):
                generator_rewards(gz, gy, config(**self.ON), torch.zeros(1), subcorr_targets=bad)
