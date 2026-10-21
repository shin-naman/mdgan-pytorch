"""Tests for mdgan.models.critic (D_G).

Replaces tests/D_G/*. Reference: Kawada et al. (2022) ESI Fig. S2(C).
The gradient-penalty reference implementation and the GP-reaches-weights test
live in tests/integration until losses.py exists; cross-module state-dict
tests move to test_checkpoint.py when checkpoint.py exists.

Run: python -m pytest tests/test_critic.py -v
"""

import pytest
import torch
from torch import nn

from conftest import SMALL
from mdgan.models.critic import Critic

M, C, H = SMALL["block_len"], SMALL["n_channels"], SMALL["hidden_dim"]
W = 2 * M


def window(batch=8):
    return torch.randn(batch, W, C)


def reference_forward(m, w):
    """Fig. S2(C) written out with the module's own layers."""
    h = m.act(m.input_conv(w.transpose(1, 2)).squeeze(-1))
    h = m.act(m.hidden1(h))
    h = m.act(m.hidden2(h))
    h = m.act(m.hidden3(h))
    return m.output_layer(h).squeeze(-1)


# =============================================================================
class TestArchitecture:
    def test_reference_layers_and_dimensions(self):
        d = Critic(block_len=64, n_channels=3, hidden_dim=48, negative_slope=0.17)
        conv = d.input_conv
        assert isinstance(conv, nn.Conv1d)
        assert (conv.in_channels, conv.out_channels, conv.kernel_size, conv.stride) == (3, 48, (128,), (128,))
        assert conv.padding == (0,) and conv.dilation == (1,) and conv.bias is not None
        shapes = [(l.in_features, l.out_features) for l in (d.hidden1, d.hidden2, d.hidden3, d.output_layer)]
        assert shapes == [(48, 48), (48, 48), (48, 48), (48, 1)]
        assert isinstance(d.act, nn.LeakyReLU) and d.act.negative_slope == pytest.approx(0.17)
        assert d.window_len == 128
        assert sorted(n for n, _ in d.named_children()) == sorted(
            ["input_conv", "hidden1", "hidden2", "hidden3", "output_layer", "act"])

    def test_parameter_count_matches_closed_form(self):
        expected = 3 * 2048 * 128 + 2048 + 3 * (2048 * 2048 + 2048) + 2048 + 1
        assert sum(p.numel() for p in Critic().parameters()) == expected

    def test_defaults_match_decisions(self):
        d = Critic()
        assert (d.block_len, d.n_channels, d.hidden_dim, d.window_len) == (64, 3, 2048, 128)
        assert d.act.negative_slope == pytest.approx(0.2)
        assert list(d.buffers()) == []

    def test_forward_equals_reference_in_both_modes(self, dg):
        w = window()
        with torch.no_grad():
            ref = reference_forward(dg, w)
            dg.output_layer.bias -= ref.median()          # make scores straddle zero
            ref = reference_forward(dg, w)
            assert (ref < 0).any() and (ref > 0).any()
            dg.train(); a = dg(w)
            dg.eval(); b = dg(w)
        torch.testing.assert_close(a, ref, rtol=0, atol=0)
        torch.testing.assert_close(b, ref, rtol=0, atol=0)

    def test_input_conv_equals_affine_over_step_major_flattened_window(self, dg):
        """Conv1d(kernel = stride = 2M) on exactly 2M steps is one affine map.
        Weight layout (H, C, 2M) pairs with the window (B, 2M, C) via permute(0, 2, 1)."""
        w = window(batch=4)
        Wmat = dg.input_conv.weight.detach().permute(0, 2, 1).reshape(H, W * C)
        b = dg.input_conv.bias.detach()
        with torch.no_grad():
            got = dg.input_conv(w.transpose(1, 2)).squeeze(-1)
        torch.testing.assert_close(got, w.reshape(4, W * C) @ Wmat.T + b, atol=1e-5, rtol=1e-5)

    @pytest.mark.parametrize("block_len,n_channels,batch", [(64, 3, 2), (64, 9, 5), (16, 3, 64), (8, 3, 1), (1, 1, 3)])
    def test_independent_dimensions_and_batch(self, block_len, n_channels, batch):
        d = Critic(block_len, n_channels, hidden_dim=16)
        s = d(torch.randn(batch, 2 * block_len, n_channels))
        assert s.shape == (batch,) and torch.isfinite(s).all()


# =============================================================================
class TestScore:
    def test_output_is_bias_when_output_weights_are_zero(self, dg):
        """Pins output_layer as the last op. The bias is set NEGATIVE on purpose:
        an activation after the output (LeakyReLU, ReLU, sigmoid, tanh) changes
        negative values, while for positive values LeakyReLU is the identity."""
        with torch.no_grad():
            dg.output_layer.weight.zero_()
            dg.output_layer.bias.fill_(-1.5)
            s = dg(window(batch=3))
        torch.testing.assert_close(s, torch.full((3,), -1.5), rtol=0, atol=0)

    def test_score_is_unbounded_and_unsquashed(self, dg):
        """A Wasserstein score must not pass through sigmoid/tanh/clamp: scaling the
        input must keep moving it. A sigmoid on the output is a classic WGAN bug."""
        w = window(batch=4)
        with torch.no_grad():
            s1, s2 = dg(w), dg(w * 1e3)
        assert torch.isfinite(s2).all() and (s2 - s1).abs().max() > 1.0

    def test_every_step_and_axis_affects_score(self, dg):
        w = window(batch=1)
        with torch.no_grad():
            base = dg(w)
            for t in (0, M - 1, M, W - 1):
                for a in range(C):
                    w2 = w.clone(); w2[0, t, a] += 3.0
                    assert (dg(w2) - base).abs() > 1e-7, (t, a)

    def test_score_depends_on_both_blocks(self, dg):
        """The critic must judge the transition y_k -> y_{k+1}, so both halves matter."""
        w = window(batch=1)
        with torch.no_grad():
            base = dg(w)
            w1 = w.clone(); w1[:, :M] += 1.0
            w2 = w.clone(); w2[:, M:] += 1.0
            assert (dg(w1) - base).abs() > 1e-6 and (dg(w2) - base).abs() > 1e-6


# =============================================================================
class TestShapeTraps:
    def test_batch_of_one_keeps_batch_dim(self, dg):
        """squeeze() without an axis would scalarize (1, 1) and break the per-sample GP."""
        assert dg(window(batch=1)).shape == (1,)

    def test_hidden_dim_one_does_not_confuse_squeeze(self):
        assert Critic(M, C, hidden_dim=1)(window(batch=3)).shape == (3,)

    def test_square_window_still_treats_axis_1_as_time(self):
        """When 2M == C a transposed window passes the shape check. Verify against
        the flattened-affine form that axis 1 is time."""
        d = Critic(block_len=2, n_channels=4, hidden_dim=H)          # 2M = C = 4
        w = torch.randn(1, 4, 4)[:, [1, 0, 2, 3], :]
        Wmat = d.input_conv.weight.detach().permute(0, 2, 1).reshape(H, 16)
        with torch.no_grad():
            got = d.input_conv(w.transpose(1, 2)).squeeze(-1)
        torch.testing.assert_close(got, w.reshape(1, 16) @ Wmat.T + d.input_conv.bias.detach(), atol=1e-5, rtol=1e-5)

    @pytest.mark.parametrize("problem", ["length_M", "length_3M", "length_4M", "wrong_channels",
                                         "rank_2", "rank_4", "transposed"])
    def test_malformed_inputs_raise(self, dg, problem):
        """length_4M is the dangerous one: it would yield two conv positions if unchecked."""
        w = {"length_M": torch.randn(4, M, C), "length_3M": torch.randn(4, 3 * M, C),
             "length_4M": torch.randn(4, 4 * M, C), "wrong_channels": torch.randn(4, W, C + 1),
             "rank_2": torch.randn(W, C), "rank_4": torch.randn(4, W, C, 1),
             "transposed": torch.randn(4, C, W)}[problem]
        with pytest.raises(ValueError):
            dg(w)

    @pytest.mark.parametrize("bad", [dict(block_len=0), dict(n_channels=-1), dict(hidden_dim=0),
                                     dict(block_len=True), dict(n_channels=3.0), dict(hidden_dim="2048")])
    def test_invalid_constructor_arguments(self, bad):
        kw = dict(block_len=M, n_channels=C, hidden_dim=H); kw.update(bad)
        with pytest.raises(ValueError, match="positive integer"):
            Critic(**kw)


# =============================================================================
class TestWGANGPRequirements:
    """Properties of the module that the gradient penalty depends on."""

    def test_per_sample_gradients_are_independent(self, dg):
        """d score[i] / d window[j] == 0 for i != j: what makes a per-sample penalty meaningful."""
        w = torch.randn(6, W, C, requires_grad=True)
        s = dg(w)
        for i in range(6):
            g, = torch.autograd.grad(s[i], w, retain_graph=True)
            mask = torch.ones(6, dtype=torch.bool); mask[i] = False
            assert g[mask].abs().sum() == 0 and g[i].abs().sum() > 0

    def test_independence_test_detects_batchnorm(self):
        """Positive control: a critic WITH BatchNorm fails the property above."""
        class Coupled(Critic):
            def __init__(self):
                super().__init__(M, C, hidden_dim=H)
                self.bn = nn.BatchNorm1d(H)

            def forward(self, w):
                h = self.act(self.bn(self.input_conv(w.transpose(1, 2)).squeeze(-1)))
                for layer in (self.hidden1, self.hidden2, self.hidden3):
                    h = self.act(layer(h))
                return self.output_layer(h).squeeze(-1)

        w = torch.randn(4, W, C, requires_grad=True)
        g, = torch.autograd.grad(Coupled().train()(w)[0], w)
        assert g[1:].abs().sum() > 0

    def test_score_sum_gradient_equals_per_sample_gradients(self, dg):
        """Justifies grad(score.sum()) in the GP."""
        w = torch.randn(5, W, C, requires_grad=True)
        s = dg(w)
        g_sum, = torch.autograd.grad(s.sum(), w, retain_graph=True)
        for i in range(5):
            g_i, = torch.autograd.grad(s[i], w, retain_graph=True)
            torch.testing.assert_close(g_sum[i], g_i[i], rtol=0, atol=0)

    def test_second_order_gradient_exists(self, dg):
        """create_graph=True must give a graph the penalty can backprop through."""
        w = torch.randn(4, W, C, requires_grad=True)
        g, = torch.autograd.grad(dg(w).sum(), w, create_graph=True)
        ((g.flatten(1).norm(dim=1) - 1) ** 2).mean().backward()
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in dg.parameters())

    def test_wasserstein_term_trains_every_parameter_except_output_bias(self, dg):
        """d(E[D(fake)] - E[D(real)]) / d(b_out) = 1 - 1 = 0: gauge freedom (decisions.md)."""
        real, fake = torch.randn(16, W, C), torch.randn(16, W, C)
        (dg(fake).mean() - dg(real).mean()).backward()
        for name, p in dg.named_parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all(), name
            if name == "output_layer.bias":
                assert p.grad.abs().sum() == 0
            else:
                assert p.grad.abs().sum() > 0, name


# =============================================================================
class TestGradientsAndState:
    def test_gradients_reach_input_and_every_parameter(self, dg):
        w = window().requires_grad_()
        (dg(w) * torch.randn(8)).sum().backward()
        assert w.grad is not None and w.grad.abs().sum() > 0
        for name, p in dg.named_parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0, name

    def test_state_dict_keys_and_round_trip(self, dg, tmp_path):
        w = window()
        with torch.no_grad():
            expected = dg(w)
        path = tmp_path / "dg.pt"
        torch.save(dg.state_dict(), path)
        state = torch.load(path, map_location="cpu", weights_only=True)
        assert set(state) == {
            "input_conv.weight", "input_conv.bias", "hidden1.weight", "hidden1.bias",
            "hidden2.weight", "hidden2.bias", "hidden3.weight", "hidden3.bias",
            "output_layer.weight", "output_layer.bias",
        }
        restored = Critic(M, C, hidden_dim=H)
        restored.load_state_dict(state, strict=True)
        with torch.no_grad():
            torch.testing.assert_close(restored(w), expected, rtol=0, atol=0)

    def test_state_dict_of_different_config_is_rejected(self, dg):
        with pytest.raises(RuntimeError):
            dg.load_state_dict(Critic(M + 1, C, hidden_dim=H).state_dict(), strict=True)