"""Tests for mdgan.models.trajectory_generator (G_Y).

Replaces tests/G_Y/*. Reference: Kawada et al. (2022) ESI Fig. S2(A).
Tests that pair G_Y with G_z (fake-window construction, rollout length) live
in tests/integration until rollout.py exists; state-dict cross-load tests move
to test_checkpoint.py when checkpoint.py exists.

Run: python -m pytest tests/test_trajectory_generator.py -v
"""

import pytest
import torch
from conftest import SMALL
from torch import nn

from mdgan.models.trajectory_generator import ResidualAffineBlock, TrajectoryGenerator

LAT, NR, M, C, H = (SMALL[k] for k in ("latent_dim", "noise_dim", "block_len", "n_channels", "hidden_dim"))


def inputs(batch=8):
    return torch.rand(batch, LAT), torch.rand(batch, NR), torch.randn(batch, M, C)


def reference_forward(m, z, r, u):
    """Fig. S2(A) written out with the module's own layers: u through the kernel-1
    conv, flattened step-major; [r', z, u']; Affine+LReLU; residual; LReLU; Affine;
    reshape step-major."""
    b = z.shape[0]
    u_mixed = m.noise_mix(u.transpose(1, 2)).transpose(1, 2).reshape(b, M * C)
    h = m.input_act(m.input_layer(torch.cat((r, z, u_mixed), dim=1)))
    h = m.post_act(m.residual(h))
    return m.output_layer(h).reshape(b, M, C)


# =============================================================================
class TestArchitecture:
    def test_reference_layers_and_dimensions(self):
        m = TrajectoryGenerator(32, 11, block_len=64, n_channels=3, hidden_dim=48, negative_slope=0.17)
        conv = m.noise_mix
        assert isinstance(conv, nn.Conv1d)
        assert (conv.in_channels, conv.out_channels, conv.kernel_size, conv.stride) == (3, 3, (1,), (1,))
        assert conv.padding == (0,) and conv.bias is not None
        shapes = [(l.in_features, l.out_features) for l in
                  (m.input_layer, m.residual.fc1, m.residual.fc2, m.output_layer)]
        assert shapes == [(11 + 32 + 192, 48), (48, 48), (48, 48), (48, 192)]
        for act in (m.input_act, m.post_act, m.residual.act):
            assert isinstance(act, nn.LeakyReLU) and act.negative_slope == pytest.approx(0.17)
        assert sorted(n for n, _ in m.named_children()) == sorted(
            ["noise_mix", "input_layer", "input_act", "residual", "post_act", "output_layer"])
        assert sorted(n for n, _ in m.residual.named_children()) == ["act", "fc1", "fc2"]

    def test_parameter_count_matches_closed_form(self):
        m = TrajectoryGenerator(32, 32)
        n_in = 32 + 32 + 192
        expected = (3 * 3 + 3 + n_in * 2048 + 2048 + 2 * (2048 * 2048 + 2048) + 2048 * 192 + 192)
        assert sum(p.numel() for p in m.parameters()) == expected

    def test_defaults_match_decisions(self):
        m = TrajectoryGenerator(32, 32)
        assert (m.block_len, m.n_channels, m.hidden_dim, m.flat_dim) == (64, 3, 2048, 192)
        assert m.input_act.negative_slope == pytest.approx(0.2)
        assert list(m.buffers()) == []                       # no running statistics of any kind

    def test_forward_equals_reference_in_both_modes(self, gy):
        z, r, u = inputs()
        with torch.no_grad():
            ref = reference_forward(gy, z, r, u)
            gy.train(); a = gy(z, r, u)
            gy.eval(); b = gy(z, r, u)
        torch.testing.assert_close(a, ref, rtol=0, atol=0)
        torch.testing.assert_close(b, ref, rtol=0, atol=0)

    def test_concat_order_is_rprime_then_z_then_mixed_noise(self, gy):
        seen = []
        handle = gy.input_layer.register_forward_pre_hook(lambda m, i: seen.append(i[0].detach().clone()))
        z = torch.arange(8 * LAT, dtype=torch.float32).reshape(8, LAT) / 13
        r = 100 + torch.arange(8 * NR, dtype=torch.float32).reshape(8, NR)
        u = torch.randn(8, M, C)
        try:
            with torch.no_grad():
                gy(z, r, u)
                u_mixed = gy.noise_mix(u.transpose(1, 2)).transpose(1, 2).reshape(8, M * C)
        finally:
            handle.remove()
        torch.testing.assert_close(seen[0], torch.cat([r, z, u_mixed], dim=1))
        assert not torch.allclose(seen[0], torch.cat([z, r, u_mixed], dim=1))     # negative control

    @pytest.mark.parametrize("latent_dim,noise_dim,block_len,n_channels,batch",
                             [(32, 32, 64, 3, 2), (32, 1, 64, 9, 5), (4, 7, 16, 3, 64), (7, 3, 8, 3, 1), (1, 1, 1, 1, 3)])
    def test_independent_dimensions_and_batch(self, latent_dim, noise_dim, block_len, n_channels, batch):
        m = TrajectoryGenerator(latent_dim, noise_dim, block_len, n_channels, hidden_dim=16)
        y = m(torch.rand(batch, latent_dim), torch.rand(batch, noise_dim), torch.randn(batch, block_len, n_channels))
        assert y.shape == (batch, block_len, n_channels) and torch.isfinite(y).all()


# =============================================================================
class TestNoisePathAndLayout:
    """The three tests a channel-major bug would fail. Such a bug trains fine and
    is physically wrong, which is why these matter more than most."""

    def test_noise_mix_is_the_same_cxc_affine_at_every_step(self, gy):
        u = torch.randn(4, M, C)
        W = gy.noise_mix.weight.detach()[:, :, 0]
        b = gy.noise_mix.bias.detach()
        with torch.no_grad():
            got = gy.noise_mix(u.transpose(1, 2)).transpose(1, 2)
        torch.testing.assert_close(got, u @ W.T + b, atol=1e-6, rtol=1e-5)
        u2 = u.clone(); u2[:, 2] += 5.0                      # perturb one step only
        with torch.no_grad():
            got2 = gy.noise_mix(u2.transpose(1, 2)).transpose(1, 2)
        mask = torch.ones(M, dtype=torch.bool); mask[2] = False
        torch.testing.assert_close(got2[:, mask], got[:, mask], rtol=0, atol=0)
        assert not torch.allclose(got2[:, 2], got[:, 2])

    def test_mix_noise_flatten_is_step_major(self, gy):
        with torch.no_grad():
            gy.noise_mix.weight.copy_(torch.eye(C).unsqueeze(-1))
            gy.noise_mix.bias.zero_()
            u = torch.randn(4, M, C)
            flat = gy._mix_noise(u)
        torch.testing.assert_close(flat, u.reshape(4, M * C), rtol=0, atol=0)
        assert not torch.equal(flat, u.transpose(1, 2).reshape(4, M * C))

    def test_output_reshape_is_step_major(self, gy):
        z, r, u = inputs(batch=2)
        flat = []
        handle = gy.output_layer.register_forward_hook(lambda m, i, o: flat.append(o.detach()))
        try:
            with torch.no_grad():
                y = gy(z, r, u)
        finally:
            handle.remove()
        for t, a in [(0, 0), (3, 1), (M - 1, C - 1)]:
            torch.testing.assert_close(y[:, t, a], flat[0][:, t * C + a], rtol=0, atol=0)

    def test_square_block_shape_still_treats_axis_1_as_time(self):
        """When M == C a transposed u passes the shape check (which is why SMALL uses
        M != C). Swapping two TIME steps must permute the mixed rows, not mix them."""
        m = TrajectoryGenerator(LAT, NR, block_len=3, n_channels=3, hidden_dim=H)
        u = torch.randn(2, 3, 3)
        with torch.no_grad():
            a = m._mix_noise(u).reshape(2, 3, 3)
            b = m._mix_noise(u[:, [1, 0, 2], :]).reshape(2, 3, 3)
        torch.testing.assert_close(b, a[:, [1, 0, 2], :], rtol=0, atol=0)

    def test_noise_at_step_t_only_enters_through_flat_slice_t(self, gy):
        u = torch.randn(1, M, C, requires_grad=True)
        cols = []
        handle = gy.input_layer.register_forward_pre_hook(lambda m, i: cols.append(i[0]))
        try:
            gy(torch.rand(1, LAT), torch.rand(1, NR), u)
        finally:
            handle.remove()
        offset, t = NR + LAT, 5
        sel = cols[0][0, offset + t * C: offset + (t + 1) * C].sum()
        grad = torch.autograd.grad(sel, u)[0][0]
        mask = torch.ones(M, dtype=torch.bool); mask[t] = False
        assert grad[t].abs().sum() > 0 and grad[mask].abs().sum() == 0

    def test_zeroing_noise_mix_kills_u_but_not_z_or_rprime(self, gy):
        with torch.no_grad():
            gy.noise_mix.weight.zero_(); gy.noise_mix.bias.zero_()
            z, r, u = inputs(batch=4)
            base = gy(z, r, u)
            assert torch.equal(gy(z, r, torch.randn(4, M, C)), base)
            assert not torch.equal(gy(z + 1, r, u), base)
            assert not torch.equal(gy(z, r + 1, u), base)


# =============================================================================
class TestResidualBlock:
    def test_identity_plus_branch_and_zero_branch_is_identity(self):
        block = ResidualAffineBlock(width=16, negative_slope=0.2)
        x = torch.randn(5, 16)
        with torch.no_grad():
            torch.testing.assert_close(block(x), x + block.fc2(block.act(block.fc1(x))))
            block.fc2.weight.zero_(); block.fc2.bias.zero_()
            torch.testing.assert_close(block(x), x, rtol=0, atol=0)

    def test_jacobian_is_identity_when_branch_is_zero(self):
        block = ResidualAffineBlock(width=6, negative_slope=0.2)
        with torch.no_grad():
            block.fc2.weight.zero_(); block.fc2.bias.zero_()
        jac = torch.autograd.functional.jacobian(block, torch.randn(1, 6))[0, :, 0, :]
        torch.testing.assert_close(jac, torch.eye(6), rtol=0, atol=0)

    def test_no_activation_after_the_add_inside_the_block(self):
        """The LReLU after the skip-add is the main stack's post_act, not the block's."""
        block = ResidualAffineBlock(width=4, negative_slope=0.2)
        with torch.no_grad():
            block.fc2.weight.zero_(); block.fc2.bias.zero_()
            x = -torch.ones(2, 4)
            torch.testing.assert_close(block(x), x, rtol=0, atol=0)   # a LReLU would scale by 0.2

    def test_branch_is_live_at_init(self, gy):
        y = gy(*inputs(batch=4))
        (y * torch.randn_like(y)).sum().backward()
        assert gy.residual.fc1.weight.grad.abs().sum() > 0
        assert gy.residual.fc2.weight.grad.abs().sum() > 0

    def test_rejects_width_mismatch(self):
        with pytest.raises(RuntimeError):
            ResidualAffineBlock(width=8, negative_slope=0.2)(torch.randn(2, 9))


# =============================================================================
class TestBehaviour:
    def test_output_is_bias_when_output_weights_are_zero(self, gy):
        """Pins output_layer as the last op: nothing acts on y after it."""
        with torch.no_grad():
            gy.output_layer.weight.zero_()
            y = gy(*inputs(batch=3))
        torch.testing.assert_close(y, gy.output_layer.bias.detach().reshape(1, M, C).expand(3, M, C), rtol=0, atol=0)

    def test_output_independent_of_batch_composition_and_order(self, gy):
        """The behavioural statement of 'no normalization layer'."""
        z, r, u = inputs(batch=8)
        perm = torch.randperm(8)
        with torch.no_grad():
            alone = gy(z[:1], r[:1], u[:1])
            together = gy(z, r, u)
            permuted = gy(z[perm], r[perm], u[perm])
        torch.testing.assert_close(alone, together[:1], atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(permuted, together[perm], atol=1e-6, rtol=1e-5)

    @pytest.mark.parametrize("which", ["z", "r_prime", "u"])
    def test_each_input_affects_output(self, gy, which):
        z, r, u = inputs()
        with torch.no_grad():
            base = gy(z, r, u)
            other = gy(z + 2.0 if which == "z" else z, r + 2.0 if which == "r_prime" else r,
                       u + 2.0 if which == "u" else u)
        assert (base - other).abs().max() > 1e-6

    def test_decoder_is_stochastic_given_fixed_latent(self, gy):
        """Same z, different noise -> different trajectories; p(Y|z) is not a point."""
        z = torch.rand(1, LAT).expand(64, LAT)
        with torch.no_grad():
            y = gy.generate(z, generator=torch.Generator().manual_seed(0))
        assert y.std(dim=0).mean() > 1e-4

    def test_collapsed_zero_inputs_give_finite_output_and_gradients(self, gy):
        z = torch.zeros(4, LAT, requires_grad=True)
        y = gy(z, torch.zeros(4, NR), torch.zeros(4, M, C))
        assert torch.isfinite(y).all()
        (y * torch.randn_like(y)).sum().backward()
        assert torch.isfinite(z.grad).all() and all(torch.isfinite(p.grad).all() for p in gy.parameters())


# =============================================================================
class TestValidation:
    @pytest.mark.parametrize("bad", [
        dict(latent_dim=0), dict(noise_dim=-1), dict(block_len=0), dict(n_channels=0),
        dict(hidden_dim=0), dict(latent_dim=True), dict(block_len=2.0), dict(n_channels="3"),
    ])
    def test_invalid_constructor_arguments(self, bad):
        kw = dict(SMALL); kw.update(bad)
        with pytest.raises(ValueError, match="positive integer"):
            TrajectoryGenerator(**kw)

    @pytest.mark.parametrize("problem", [
        "wrong_z", "wrong_r", "wrong_u_steps", "wrong_u_channels", "u_transposed",
        "batch_mismatch_z", "batch_mismatch_u", "rank_1_z", "rank_2_u", "rank_3_z", "dtype_mismatch",
    ])
    def test_malformed_inputs_raise(self, gy, problem):
        z, r, u = inputs()
        if problem == "wrong_z": z = z[:, :-1]
        elif problem == "wrong_r": r = r[:, :-1]
        elif problem == "wrong_u_steps": u = u[:, :-1]
        elif problem == "wrong_u_channels": u = u[:, :, :-1]
        elif problem == "u_transposed": u = torch.randn(8, C, M)
        elif problem == "batch_mismatch_z": z = z[:-1]
        elif problem == "batch_mismatch_u": u = u[:-1]
        elif problem == "rank_1_z": z = z[0]
        elif problem == "rank_2_u": u = u[0]
        elif problem == "rank_3_z": z = z.unsqueeze(-1)
        elif problem == "dtype_mismatch": u = u.double()
        with pytest.raises((RuntimeError, ValueError, TypeError, IndexError)):
            gy(z, r, u)


# =============================================================================
class TestRandomness:
    def test_sample_noise_shapes_ranges_and_dtype(self, gy):
        r, u = gy.sample_noise(4096, dtype=torch.float64, generator=torch.Generator().manual_seed(0))
        assert r.shape == (4096, NR) and u.shape == (4096, M, C)
        assert r.dtype == u.dtype == torch.float64
        assert r.min() >= 0 and r.max() < 1
        assert abs(u.mean()) < 0.05 and abs(u.std() - 1) < 0.05

    def test_generate_reproducible_with_generator_and_leaves_global_rng(self, gy):
        z = torch.rand(5, LAT)
        before = torch.get_rng_state().clone()
        with torch.no_grad():
            a = gy.generate(z, generator=torch.Generator().manual_seed(1))
            b = gy.generate(z, generator=torch.Generator().manual_seed(1))
            c = gy.generate(z, generator=torch.Generator().manual_seed(2))
        assert torch.equal(before, torch.get_rng_state())
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        assert not torch.equal(a, c) and a.shape == (5, M, C)

    def test_generate_matches_forward_with_same_noise(self, gy):
        z = torch.rand(5, LAT)
        with torch.no_grad():
            y_gen = gy.generate(z, generator=torch.Generator().manual_seed(9))
            r, u = gy.sample_noise(5, generator=torch.Generator().manual_seed(9))
            y_fwd = gy(z, r, u)
        torch.testing.assert_close(y_gen, y_fwd, rtol=0, atol=0)


# =============================================================================
class TestGradientsAndState:
    def test_gradients_reach_all_inputs_and_parameters(self, gy):
        gy.train()
        z, r, u = inputs()
        for t in (z, r, u):
            t.requires_grad_()
        y = gy(z, r, u)
        (y * torch.randn_like(y)).sum().backward()
        for name, t in (("z", z), ("r_prime", r), ("u", u)):
            assert t.grad is not None and t.grad.abs().sum() > 0, name
        for name, p in gy.named_parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0, name

    def test_state_dict_keys_and_round_trip(self, gy, tmp_path):
        gy.eval()
        z, r, u = inputs()
        with torch.no_grad():
            expected = gy(z, r, u)
        path = tmp_path / "gy.pt"
        torch.save(gy.state_dict(), path)
        state = torch.load(path, map_location="cpu", weights_only=True)
        assert set(state) == {
            "noise_mix.weight", "noise_mix.bias", "input_layer.weight", "input_layer.bias",
            "residual.fc1.weight", "residual.fc1.bias", "residual.fc2.weight", "residual.fc2.bias",
            "output_layer.weight", "output_layer.bias",
        }
        restored = TrajectoryGenerator(**SMALL)
        restored.load_state_dict(state, strict=True)
        restored.eval()
        with torch.no_grad():
            torch.testing.assert_close(restored(z, r, u), expected, rtol=0, atol=0)

    def test_state_dict_of_different_config_is_rejected(self, gy):
        other = TrajectoryGenerator(LAT, NR, M + 1, C, hidden_dim=H)
        with pytest.raises(RuntimeError):
            gy.load_state_dict(other.state_dict(), strict=True)