"""Tests for mdgan.models.latent_transition (G_z).

Replaces tests/G_z/*. One file per source module, grouped by behaviour.
Reference: Kawada et al. (2022) ESI Fig. S2(B); decisions in docs/decisions.md.

Run: python -m pytest tests/test_latent_transition.py -v
"""

import pytest
import torch
from torch import nn

from conftest import SMALL
from mdgan.models.latent_transition import LatentTransitionGenerator

LAT, NR, H = SMALL["latent_dim"], SMALL["noise_dim"], SMALL["hidden_dim"]


def inputs(batch=8):
    return torch.rand(batch, LAT), torch.rand(batch, NR)


def reference_forward(m, z, r):
    """The figure, written out with the module's own layers. Pins concat order,
    layer order and activation placement without hooks."""
    h = m.act(m.input_layer(torch.cat((z, r), dim=1)))
    h = m.act(m.hidden1(h))
    h = m.act(m.hidden2(h))
    return m.output_norm(m.output_layer(h))


# =============================================================================
class TestArchitecture:
    def test_reference_layers_and_dimensions(self):
        m = LatentTransitionGenerator(32, 11, hidden_dim=48, negative_slope=0.17)
        shapes = [(layer.in_features, layer.out_features)
                  for layer in (m.input_layer, m.hidden1, m.hidden2, m.output_layer)]
        assert shapes == [(43, 48), (48, 48), (48, 48), (48, 32)]
        assert isinstance(m.output_norm, nn.BatchNorm1d)
        assert m.output_norm.num_features == 32 and m.output_norm.affine is True
        assert isinstance(m.act, nn.LeakyReLU) and m.act.negative_slope == pytest.approx(0.17)
        assert sorted(n for n, _ in m.named_children()) == sorted(
            ["input_layer", "hidden1", "hidden2", "output_layer", "output_norm", "act"])

    def test_parameter_count_matches_closed_form(self):
        m = LatentTransitionGenerator(32, 32)
        expected = ((32 + 32) * 2048 + 2048          # input_layer
                    + 2 * (2048 * 2048 + 2048)       # hidden1, hidden2
                    + 2048 * 32 + 32                 # output_layer
                    + 2 * 32)                        # BatchNorm gamma, beta
        assert sum(p.numel() for p in m.parameters()) == expected

    def test_default_batchnorm_keeps_no_running_statistics(self):
        """The default VALUES live in Config (tests/test_config.py, TestNoDrift). This pins
        what the default means for the module: batch statistics, no running buffers."""
        m = LatentTransitionGenerator(LAT, NR, hidden_dim=H)
        assert m.use_running_stats is False
        assert m.output_norm.track_running_stats is False and m.output_norm.running_mean is None

    def test_forward_equals_reference(self, gz):
        """Both modes: the default BatchNorm uses batch statistics in eval too,
        so eval == train and both equal the hand-written forward."""
        z, r = inputs()
        with torch.no_grad():
            ref = reference_forward(gz, z, r)
            gz.train(); out_train = gz(z, r)
            gz.eval(); out_eval = gz(z, r)
        torch.testing.assert_close(out_train, ref, rtol=0, atol=0)
        torch.testing.assert_close(out_eval, ref, rtol=0, atol=0)

    def test_concat_order_is_z_then_r(self, gz):
        seen = []
        handle = gz.input_layer.register_forward_pre_hook(lambda m, i: seen.append(i[0].detach().clone()))
        z = torch.arange(8 * LAT, dtype=torch.float32).reshape(8, LAT) / 13
        r = 100 + torch.arange(8 * NR, dtype=torch.float32).reshape(8, NR)
        try:
            with torch.no_grad():
                gz(z, r)
        finally:
            handle.remove()
        torch.testing.assert_close(seen[0], torch.cat([z, r], dim=1), rtol=0, atol=0)
        assert not torch.equal(seen[0], torch.cat([r, z], dim=1))      # negative control

    @pytest.mark.parametrize("latent_dim,noise_dim,batch", [(32, 1, 2), (32, 16, 64), (3, 7, 8), (11, 2, 5)])
    def test_independent_dimensions(self, latent_dim, noise_dim, batch):
        m = LatentTransitionGenerator(latent_dim, noise_dim, hidden_dim=16)
        out = m(torch.rand(batch, latent_dim), torch.rand(batch, noise_dim))
        assert out.shape == (batch, latent_dim) and torch.isfinite(out).all()


# =============================================================================
class TestValidation:
    @pytest.mark.parametrize("bad", [
        dict(latent_dim=0), dict(latent_dim=-3), dict(noise_dim=0), dict(hidden_dim=0),
        dict(latent_dim=True), dict(noise_dim=2.0), dict(hidden_dim="24"),
    ])
    def test_invalid_constructor_arguments(self, bad):
        kw = dict(latent_dim=LAT, noise_dim=NR, hidden_dim=H)
        kw.update(bad)
        with pytest.raises(ValueError, match="positive integer"):
            LatentTransitionGenerator(**kw)

    @pytest.mark.parametrize("problem", ["wrong_z", "wrong_r", "batch_mismatch", "rank_1_z", "rank_3_r"])
    def test_malformed_inputs_raise(self, gz, problem):
        z, r = inputs()
        if problem == "wrong_z":
            z = z[:, :-1]
        elif problem == "wrong_r":
            r = r[:, :-1]
        elif problem == "batch_mismatch":
            r = r[:-1]
        elif problem == "rank_1_z":
            z = z[0]
        elif problem == "rank_3_r":
            r = r.unsqueeze(-1)
        with pytest.raises(ValueError):
            gz(z, r)

    def test_singleton_batch_rejected_in_both_modes(self, gz):
        """Batch statistics need at least two samples; the error is ours, not BatchNorm's."""
        z, r = inputs(batch=1)
        for mode in (gz.train, gz.eval):
            mode()
            with pytest.raises(ValueError, match="at least 2"):
                gz(z, r)


# =============================================================================
class TestBatchNormStabilization:
    """The output BatchNorm with batch statistics is the distribution-stabilization
    mechanism (decisions.md). These pin that it is active in both modes."""

    def test_output_is_centered_and_scaled_per_dimension(self, gz):
        """mean = 0 exactly; variance = v/(v+eps) with v the pre-BN batch variance
        (gamma = 1, beta = 0 at init)."""
        gz.eval()
        bn = gz.output_norm
        captured = []
        handle = bn.register_forward_pre_hook(lambda m, i: captured.append(i[0].detach()))
        z, r = inputs(batch=256)
        try:
            with torch.no_grad():
                out = gz(z, r)
        finally:
            handle.remove()
        v = captured[0].var(0, unbiased=False)
        torch.testing.assert_close(out.mean(0), torch.zeros(LAT), atol=1e-5, rtol=0)
        torch.testing.assert_close(out.var(0, unbiased=False), v / (v + bn.eps), atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize("scale", [1e4])
    def test_output_scale_is_invariant_to_input_scale(self, gz, scale):
        gz.eval()
        z, r = inputs(batch=64)
        with torch.no_grad():
            out = gz(z * scale, r)
        assert torch.isfinite(out).all()
        torch.testing.assert_close(out.mean(0), torch.zeros(LAT), atol=1e-3, rtol=0)
        assert (out.var(0, unbiased=False) > 0.99).all() and out.abs().max() < 20

    def test_collapsed_batch_gives_finite_output_and_gradients(self, gz):
        """Identical rows -> zero batch variance -> BatchNorm eps path, not NaN."""
        gz.train()
        z = torch.full((16, LAT), 0.3, requires_grad=True)
        r = torch.full((16, NR), 0.6)
        out = gz(z, r)
        assert torch.isfinite(out).all()
        torch.testing.assert_close(out, torch.zeros_like(out), atol=1e-4, rtol=0)
        (out * torch.randn_like(out)).sum().backward()
        assert torch.isfinite(z.grad).all()
        assert all(torch.isfinite(p.grad).all() for p in gz.parameters())

    def test_long_rollout_stays_bounded_at_init(self, gz):
        """200 transitions (more than the 125 + burn-in needed for 80 ns) without
        any training: the ensemble must neither drift nor collapse."""
        gz.eval()
        with torch.no_grad():
            states = gz.unroll(torch.rand(64, LAT), steps=200)
        assert torch.isfinite(states).all()
        per_step_std = states[1:].std(dim=1)
        assert 0.1 < per_step_std.min() and per_step_std.max() < 5.0


# =============================================================================
class TestRandomness:
    def test_sample_noise_shape_range_and_reproducibility(self, gz):
        a = gz.sample_noise(4096, generator=torch.Generator().manual_seed(0))
        b = gz.sample_noise(4096, generator=torch.Generator().manual_seed(0))
        assert a.shape == (4096, NR) and a.dtype == torch.float32
        assert a.min() >= 0 and a.max() < 1
        assert torch.equal(a, b)

    def test_unroll_reproducible_with_generator_and_leaves_global_rng(self, gz):
        gz.eval()
        z0 = torch.rand(5, LAT)                  # drawn before the snapshot
        before = torch.get_rng_state().clone()
        with torch.no_grad():
            a = gz.unroll(z0, 8, generator=torch.Generator().manual_seed(1))
            b = gz.unroll(z0, 8, generator=torch.Generator().manual_seed(1))
            c = gz.unroll(z0, 8, generator=torch.Generator().manual_seed(2))
        assert torch.equal(before, torch.get_rng_state())
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        assert not torch.equal(a, c)

    def test_noise_and_state_both_affect_next_state(self, gz):
        z, r = inputs()
        with torch.no_grad():
            base = gz(z, r)
            assert not torch.allclose(gz(z, r + 0.5), base)
            assert not torch.allclose(gz(z + 0.5, r), base)


# =============================================================================
class TestUnroll:
    def test_shape_includes_z0_and_zero_steps(self, gz):
        gz.eval()
        z0 = torch.rand(5, LAT)
        with torch.no_grad():
            states = gz.unroll(z0, 11)
            only = gz.unroll(z0, 0)
        assert states.shape == (12, 5, LAT) and torch.isfinite(states).all()
        torch.testing.assert_close(states[0], z0, rtol=0, atol=0)
        assert only.shape == (1, 5, LAT)

    def test_negative_steps_rejected(self, gz):
        with pytest.raises(ValueError):
            gz.unroll(torch.rand(5, LAT), -1)

    def test_full_unroll_backpropagates_to_z0_and_all_weights(self, gz):
        """Decision: full BPTT through the burn-in. A detach anywhere zeroes z0.grad."""
        gz.train()
        z0 = torch.rand(8, LAT, requires_grad=True)
        states = gz.unroll(z0, 8)
        (states[-1] * torch.randn_like(states[-1])).sum().backward()
        assert z0.grad is not None and z0.grad.abs().sum() > 0
        for name, p in gz.named_parameters():
            assert p.grad is not None and p.grad.abs().sum() > 0, name


# =============================================================================
class TestGradientsAndState:
    def test_gradients_reach_inputs_and_all_parameters(self, gz):
        gz.train()
        z, r = inputs()
        z.requires_grad_(); r.requires_grad_()
        out = gz(z, r)
        (out * torch.randn_like(out)).sum().backward()
        for name, t in (("z", z), ("r", r)):
            assert t.grad is not None and t.grad.abs().sum() > 0, name
        for name, p in gz.named_parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0, name

    def test_state_dict_keys_and_round_trip(self, gz, tmp_path):
        gz.eval()
        z, r = inputs()
        with torch.no_grad():
            expected = gz(z, r)
        path = tmp_path / "gz.pt"
        torch.save(gz.state_dict(), path)
        state = torch.load(path, map_location="cpu", weights_only=True)
        assert set(state) == {
            "input_layer.weight", "input_layer.bias", "hidden1.weight", "hidden1.bias",
            "hidden2.weight", "hidden2.bias", "output_layer.weight", "output_layer.bias",
            "output_norm.weight", "output_norm.bias",
        }
        restored = LatentTransitionGenerator(LAT, NR, hidden_dim=H)
        restored.load_state_dict(state, strict=True)
        restored.eval()
        with torch.no_grad():
            torch.testing.assert_close(restored(z, r), expected, rtol=0, atol=0)

    def test_state_dict_of_different_config_is_rejected(self, gz):
        with pytest.raises(RuntimeError):
            gz.load_state_dict(LatentTransitionGenerator(LAT + 1, NR, hidden_dim=H).state_dict(), strict=True)


# =============================================================================
@pytest.mark.ablation
class TestAblationRunningStats:
    """use_running_stats=True: conventional BatchNorm. Not the baseline; kept so
    the two modes can be compared. Not interchangeable with the default after training."""

    @pytest.fixture
    def gz_rs(self):
        torch.manual_seed(717)
        return LatentTransitionGenerator(LAT, NR, hidden_dim=H, use_running_stats=True)

    def test_train_updates_statistics_and_eval_freezes_them(self, gz_rs):
        bn = gz_rs.output_norm
        assert bn.track_running_stats is True
        gz_rs.train()
        n0 = bn.num_batches_tracked.item()
        for _ in range(3):
            gz_rs(*inputs())
        assert bn.num_batches_tracked.item() == n0 + 3
        gz_rs.eval()
        mean, var = bn.running_mean.clone(), bn.running_var.clone()
        with torch.no_grad():
            gz_rs(*inputs())
        torch.testing.assert_close(bn.running_mean, mean, rtol=0, atol=0)
        torch.testing.assert_close(bn.running_var, var, rtol=0, atol=0)

    def test_eval_is_sample_independent_and_allows_singleton(self, gz_rs):
        gz_rs.train()
        for _ in range(3):
            gz_rs(*inputs())
        gz_rs.eval()
        z, r = inputs(batch=8)
        with torch.no_grad():
            alone, together = gz_rs(z[:1], r[:1]), gz_rs(z, r)[:1]
        torch.testing.assert_close(alone, together, atol=1e-6, rtol=1e-5)
        gz_rs.train()
        with pytest.raises(ValueError, match="at least 2"):
            gz_rs(z[:1], r[:1])

    def test_modes_differ_after_loading_the_same_weights(self):
        torch.manual_seed(5)
        bs = LatentTransitionGenerator(LAT, NR, hidden_dim=H, use_running_stats=False)
        rs = LatentTransitionGenerator(LAT, NR, hidden_dim=H, use_running_stats=True)
        rs.load_state_dict(bs.state_dict(), strict=False)
        bs.eval(); rs.eval()
        z0 = torch.rand(32, LAT)
        with torch.no_grad():
            a = bs.unroll(z0, 3, generator=torch.Generator().manual_seed(3))
            b = rs.unroll(z0, 3, generator=torch.Generator().manual_seed(3))
        assert not torch.allclose(a[1:], b[1:])