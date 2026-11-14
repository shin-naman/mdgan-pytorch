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
from mdgan.rewards import block_sum_correlations, generator_rewards, noise_use_share, reward_metric_names

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
