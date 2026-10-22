"""Tests for mdgan.rollout.

Reference tests rebuild each function's output by hand, drawing from a generator
in the documented order; equality pins the algorithm (which latents are decoded,
in what order, with which noise). Hook tests then check the structural claims
directly (j range, consecutive pair, block k decodes z_{c+k}).

Run: python -m pytest tests/test_rollout.py -v
"""

import pytest
import torch

from conftest import SMALL
from mdgan.rollout import make_fake_window, rollout

LAT, M, C = SMALL["latent_dim"], SMALL["block_len"], SMALL["n_channels"]


def _recorders(gz, gy):
    """Record every G_z output and every z fed to G_Y, in call order."""
    gz_out, gy_in = [], []
    h1 = gz.register_forward_hook(lambda m, i, o: gz_out.append(o.detach().clone()))
    h2 = gy.register_forward_pre_hook(lambda m, i: gy_in.append(i[0].detach().clone()))
    return gz_out, gy_in, (h1, h2)


# =============================================================================
class TestMakeFakeWindow:
    def test_matches_hand_construction_in_documented_draw_order(self, gz, gy):
        w = make_fake_window(gz, gy, 8, burn_in=2, pair_window=3, generator=torch.Generator().manual_seed(5))
        g = torch.Generator().manual_seed(5)
        j = int(torch.randint(2, 5, (1,), generator=g))
        states = gz.unroll(torch.rand(8, LAT, generator=g), j + 1, generator=g)
        ref = torch.cat([gy.generate(states[j], generator=g), gy.generate(states[j + 1], generator=g)], dim=1)
        torch.testing.assert_close(w, ref - ref.mean(0, keepdim=True), rtol=0, atol=0)

    def test_shape_and_mean_operation_switch(self, gz, gy):
        on = make_fake_window(gz, gy, 16, 1, 2, generator=torch.Generator().manual_seed(0))
        off = make_fake_window(gz, gy, 16, 1, 2, mean_operation=False, generator=torch.Generator().manual_seed(0))
        assert on.shape == off.shape == (16, 2 * M, C)
        torch.testing.assert_close(on.mean(0), torch.zeros(2 * M, C), atol=1e-6, rtol=0)
        assert off.mean(0).abs().max() > 1e-3
        torch.testing.assert_close(on, off - off.mean(0, keepdim=True), rtol=0, atol=0)

    def test_j_covers_exactly_burn_in_to_burn_in_plus_K_minus_1(self, gz, gy):
        """G_z runs j + 1 times per call, so the call count reveals j."""
        calls = []
        h = gz.register_forward_hook(lambda m, i, o: calls.append(1))
        g = torch.Generator().manual_seed(0)
        seen = set()
        try:
            with torch.no_grad():
                for _ in range(300):
                    calls.clear()
                    make_fake_window(gz, gy, 4, burn_in=2, pair_window=3, generator=g)
                    seen.add(len(calls) - 1)
        finally:
            h.remove()
        assert seen == {2, 3, 4}

    def test_decodes_two_consecutive_latents(self, gz, gy):
        gz_out, gy_in, hooks = _recorders(gz, gy)
        try:
            with torch.no_grad():
                make_fake_window(gz, gy, 8, burn_in=3, pair_window=2, generator=torch.Generator().manual_seed(1))
        finally:
            for h in hooks:
                h.remove()
        assert len(gy_in) == 2
        torch.testing.assert_close(gy_in[0], gz_out[-2], rtol=0, atol=0)    # z_j
        torch.testing.assert_close(gy_in[1], gz_out[-1], rtol=0, atol=0)    # z_{j+1} = G_z(z_j)

    def test_burn_in_zero_and_K_one_decodes_z0_and_z1(self, gz, gy):
        calls = []
        h = gz.register_forward_hook(lambda m, i, o: calls.append(1))
        try:
            make_fake_window(gz, gy, 4, burn_in=0, pair_window=1, generator=torch.Generator().manual_seed(0))
        finally:
            h.remove()
        assert len(calls) == 1

    def test_gradient_reaches_every_generator_weight(self, gz, gy):
        w = make_fake_window(gz, gy, 16, 3, 2)
        (w * torch.randn_like(w)).sum().backward()
        for net in (gz, gy):
            for name, p in net.named_parameters():
                if name.endswith(".weight"):
                    assert p.grad is not None and p.grad.abs().sum() > 0, name

    def test_reproducible_and_leaves_global_rng(self, gz, gy):
        before = torch.get_rng_state().clone()
        with torch.no_grad():
            a = make_fake_window(gz, gy, 8, 2, 3, generator=torch.Generator().manual_seed(4))
            b = make_fake_window(gz, gy, 8, 2, 3, generator=torch.Generator().manual_seed(4))
            c = make_fake_window(gz, gy, 8, 2, 3, generator=torch.Generator().manual_seed(5))
        assert torch.equal(before, torch.get_rng_state())
        assert torch.equal(a, b) and not torch.equal(a, c)

    @pytest.mark.parametrize("bad", [dict(batch=0), dict(burn_in=-1), dict(pair_window=0), dict(burn_in=1.0)])
    def test_invalid_arguments(self, gz, gy, bad):
        kw = dict(batch=4, burn_in=1, pair_window=2); kw.update(bad)
        with pytest.raises(ValueError):
            make_fake_window(gz, gy, **kw)


# =============================================================================
class TestRollout:
    def test_matches_hand_construction_in_documented_draw_order(self, gz, gy):
        N, n_blocks, c = 6, 4, 2
        out = rollout(gz, gy, N, n_blocks, c, generator=torch.Generator().manual_seed(9))
        g = torch.Generator().manual_seed(9)
        gz.eval(); gy.eval()
        with torch.no_grad():
            z = torch.rand(N, LAT, generator=g)
            for _ in range(c):
                z = gz(z, gz.sample_noise(N, generator=g))
            blocks = []
            for k in range(n_blocks):
                y = gy.generate(z, generator=g)
                blocks.append(y - y.mean(0, keepdim=True))
                if k + 1 < n_blocks:
                    z = gz(z, gz.sample_noise(N, generator=g))
        torch.testing.assert_close(out, torch.cat(blocks, dim=1), rtol=0, atol=0)

    def test_block_k_decodes_z_c_plus_k(self, gz, gy):
        c, n_blocks = 3, 5
        gz_out, gy_in, hooks = _recorders(gz, gy)
        try:
            rollout(gz, gy, 4, n_blocks, c, generator=torch.Generator().manual_seed(2))
        finally:
            for h in hooks:
                h.remove()
        assert len(gy_in) == n_blocks and len(gz_out) == c + n_blocks - 1
        for k in range(n_blocks):
            torch.testing.assert_close(gy_in[k], gz_out[c - 1 + k], rtol=0, atol=0)   # gz_out[i] = z_{i+1}

    def test_output_is_cpu_float32_detached_and_shaped(self, gz, gy):
        out = rollout(gz, gy, 5, 3, 1)
        assert out.shape == (5, 3 * M, C) and out.dtype == torch.float32
        assert out.device.type == "cpu" and not out.requires_grad and torch.isfinite(out).all()

    def test_mean_operation_switch(self, gz, gy):
        on = rollout(gz, gy, 16, 3, 1, generator=torch.Generator().manual_seed(0))
        off = rollout(gz, gy, 16, 3, 1, mean_operation=False, generator=torch.Generator().manual_seed(0))
        torch.testing.assert_close(on.mean(0), torch.zeros(3 * M, C), atol=1e-6, rtol=0)
        assert off.mean(0).abs().max() > 1e-3

    def test_runs_in_eval_mode_and_restores_modes(self, gz, gy):
        gz.train(); gy.eval()
        seen = []
        h = gz.register_forward_hook(lambda m, i, o: seen.append(m.training))
        try:
            rollout(gz, gy, 4, 2, 1)
        finally:
            h.remove()
        assert seen and not any(seen)
        assert gz.training is True and gy.training is False

    def test_modes_restored_even_on_error(self, gz, gy):
        gz.train(); gy.train()
        with pytest.raises(ValueError):
            rollout(gz, gy, 1, 2, 1)                 # batch of one: G_z's BatchNorm refuses
        assert gz.training and gy.training

    def test_reproducible_and_leaves_global_rng(self, gz, gy):
        before = torch.get_rng_state().clone()
        a = rollout(gz, gy, 4, 3, 2, generator=torch.Generator().manual_seed(1))
        b = rollout(gz, gy, 4, 3, 2, generator=torch.Generator().manual_seed(1))
        c = rollout(gz, gy, 4, 3, 2, generator=torch.Generator().manual_seed(2))
        assert torch.equal(before, torch.get_rng_state())
        assert torch.equal(a, b) and not torch.equal(a, c)

    def test_estimation_scale_ensemble(self, gz, gy):
        """8192 chains, the paper's estimation ensemble, in one batch."""
        out = rollout(gz, gy, 8192, 2, 1)
        assert out.shape == (8192, 2 * M, C) and torch.isfinite(out).all()

    def test_one_block_has_no_trailing_transition(self, gz, gy):
        calls = []
        h = gz.register_forward_hook(lambda m, i, o: calls.append(1))
        try:
            rollout(gz, gy, 4, 1, 2)
        finally:
            h.remove()
        assert len(calls) == 2                       # burn-in only

    @pytest.mark.parametrize("bad", [dict(n_chains=0), dict(n_blocks=0), dict(burn_in=-1), dict(n_blocks=2.0)])
    def test_invalid_arguments(self, gz, gy, bad):
        kw = dict(n_chains=4, n_blocks=2, burn_in=1); kw.update(bad)
        with pytest.raises(ValueError):
            rollout(gz, gy, **kw)