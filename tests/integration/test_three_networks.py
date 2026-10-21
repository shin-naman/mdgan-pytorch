"""Integration: the three networks wired as the trainer will wire them, plus
tests parked here until their home module exists:

  - fake-window construction and rollout length  -> tests/test_rollout.py   (needs rollout.py)
  - gradient-penalty reference implementation    -> tests/test_losses.py    (needs losses.py)
  - cross-module state-dict hazards              -> tests/test_checkpoint.py (needs checkpoint.py)

Run: python -m pytest tests/integration -v
"""

import pytest
import torch
from conftest import SMALL

from mdgan.ops import mean_operation

LAT, NR, M, C, H = (SMALL[k] for k in ("latent_dim", "noise_dim", "block_len", "n_channels", "hidden_dim"))
W = 2 * M


def _gp(critic, real, fake, lam=10.0):
    """Gradient penalty exactly as in Gulrajani et al. 2017 / Kawada 2022 eq. 5.
    The single reference copy; losses.gradient_penalty must agree with it."""
    b = real.shape[0]
    eps = torch.rand(b, 1, 1, dtype=real.dtype)
    x_hat = (eps * real + (1 - eps) * fake).requires_grad_(True)
    score = critic(x_hat)
    grad, = torch.autograd.grad(score.sum(), x_hat, create_graph=True)
    norm = grad.flatten(1).norm(dim=1)
    return lam * ((norm - 1) ** 2).mean(), norm


def _fake_window(gz, gy, batch=16, c=3, K=2):
    """Decided pairing: burn in c, unroll K more, decode (z_j, z_{j+1}), center."""
    states = gz.unroll(torch.rand(batch, LAT), steps=c + K)
    j = c + 1
    return mean_operation(torch.cat([gy.generate(states[j]), gy.generate(states[j + 1])], dim=1))


# --- training steps ---------------------------------------------------------------------------

def test_critic_step_wgan_gp_loss_and_update(dg):
    """One full critic step: L = E[D(fake)] - E[D(real)] + GP; backward; Adam step.
    output_layer.bias has identically zero gradient under WGAN-GP (gauge freedom)."""
    opt = torch.optim.Adam(dg.parameters(), lr=1e-4, betas=(0.0, 0.9))
    real, fake = torch.randn(32, W, C), torch.randn(32, W, C) * 2.0
    before = {n: p.detach().clone() for n, p in dg.named_parameters()}
    loss = dg(fake).mean() - dg(real).mean() + _gp(dg, real, fake)[0]
    opt.zero_grad(); loss.backward(); opt.step()
    assert torch.isfinite(loss)
    for n, p in dg.named_parameters():
        if n == "output_layer.bias":
            assert torch.equal(before[n], p.detach())
        else:
            assert not torch.equal(before[n], p.detach()), n


def test_generator_step_backprops_through_critic_into_both_generators(three_nets):
    gz, gy, dg = three_nets
    for n in three_nets:
        n.train()
    fake = _fake_window(gz, gy)
    assert fake.shape == (16, W, C)
    torch.testing.assert_close(fake.mean(0), torch.zeros(W, C), atol=1e-5, rtol=0)
    (-dg(fake).mean()).backward()
    for net, name in ((gz, "G_z"), (gy, "G_Y")):
        for pn, p in net.named_parameters():
            if pn.endswith(".weight"):
                assert p.grad is not None and p.grad.abs().sum() > 0, f"{name}.{pn}"
    assert dg.hidden1.weight.grad is not None       # critic gets gradient too; trainer must not step it


def test_critic_step_with_detached_fake_leaves_generators_untouched(three_nets):
    gz, gy, dg = three_nets
    fake = _fake_window(gz, gy).detach()
    real = torch.randn(16, W, C)
    (dg(fake).mean() - dg(real).mean() + _gp(dg, real, fake)[0]).backward()
    assert all(p.grad is None for p in gz.parameters())
    assert all(p.grad is None for p in gy.parameters())
    assert all(p.grad is not None for p in dg.parameters())


def test_gradient_penalty_on_centered_fake_window(three_nets):
    """Decision: mean operation BEFORE interpolation."""
    gz, gy, dg = three_nets
    fake = _fake_window(gz, gy).detach()
    gp, _ = _gp(dg, torch.randn(16, W, C), fake)
    gp.backward()
    assert torch.isfinite(gp) and dg.input_conv.weight.grad is not None


# --- parked: -> test_losses.py ---------------------------------------------------------------

def test_gradient_penalty_reaches_critic_weights_and_no_bias(dg):
    """For a piecewise-linear critic d score/d window is a product of weights and
    slope masks; biases enter only through the non-differentiable mask selection.
    Autograd reports that as None (Linear) or exact zero (Conv1d fused backward)."""
    gp, norm = _gp(dg, torch.randn(16, W, C), torch.randn(16, W, C))
    assert torch.isfinite(gp) and norm.shape == (16,) and (norm > 0).all()
    gp.backward()
    for name, p in dg.named_parameters():
        if name.endswith(".weight"):
            assert p.grad is not None and p.grad.abs().sum() > 0, name
        else:
            assert p.grad is None or p.grad.abs().sum() == 0, name


# --- parked: -> test_rollout.py ----------------------------------------------------------------

def test_gz_to_gy_pairing_produces_2m_window_and_backprops_to_gz(three_nets):
    gz, gy, _ = three_nets
    gz.train(); gy.train()
    window = _fake_window(gz, gy)
    assert window.shape == (16, W, C) and torch.isfinite(window).all()
    (window * torch.randn_like(window)).sum().backward()
    for name, p in gz.named_parameters():
        if name.endswith(".weight"):
            assert p.grad is not None and p.grad.abs().sum() > 0, name
    for name, p in gy.named_parameters():
        assert p.grad is not None and p.grad.abs().sum() > 0, name


def test_consecutive_latents_decode_to_different_blocks(three_nets):
    """If G_Y ignored z, consecutive blocks with the same noise would be identical."""
    gz, gy, _ = three_nets
    gz.eval(); gy.eval()
    states = gz.unroll(torch.rand(8, LAT), steps=4, generator=torch.Generator().manual_seed(0))
    r, u = gy.sample_noise(8, generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        assert (gy(states[3], r, u) - gy(states[4], r, u)).abs().max() > 1e-6


def test_cumsum_of_generated_displacements_gives_trajectories_from_origin(gy):
    with torch.no_grad():
        d = gy.generate(torch.rand(4, LAT), generator=torch.Generator().manual_seed(0))
    pos = torch.cumsum(d, dim=1)
    torch.testing.assert_close(pos[:, 0], d[:, 0], rtol=0, atol=0)
    torch.testing.assert_close(pos[:, -1], d.sum(dim=1), atol=1e-5, rtol=1e-5)


def test_estimation_rollout_length_for_80ns():
    """80 ns at 10 ps and n_s = 1 is 8000 steps = 125 blocks of M = 64."""
    assert 8000 % 64 == 0 and 8000 // 64 == 125


def test_estimation_scale_batch(three_nets):
    """8192 chains is the paper's estimation ensemble; no batch-size assumption anywhere."""
    gz, gy, dg = three_nets
    with torch.no_grad():
        z = gz.unroll(torch.rand(8192, LAT), steps=1)[-1]
        y = gy.generate(z)
        s = dg(torch.cat([y, y], dim=1))
    assert z.shape == (8192, LAT) and y.shape == (8192, M, C) and s.shape == (8192,)


# --- parked: -> test_checkpoint.py -------------------------------------------------------------

def test_cross_module_state_dicts_are_rejected_strictly_and_unsafe_non_strictly(three_nets):
    """Modules share key names (input_layer.*, hidden1.*, output_layer.*). strict=True
    rejects a cross-load; strict=False silently copies same-shaped tensors and raises
    on the rest. The checkpoint loader must therefore be strict-only."""
    gz, gy, dg = three_nets
    for a, b in ((gy, gz), (gz, gy), (dg, gz), (dg, gy)):
        with pytest.raises(RuntimeError):
            a.load_state_dict(b.state_dict(), strict=True)
    assert set(gz.state_dict()) & set(gy.state_dict())
    assert gz.state_dict()["input_layer.bias"].shape == gy.state_dict()["input_layer.bias"].shape
    with pytest.raises(RuntimeError, match="size mismatch"):
        gy.load_state_dict(gz.state_dict(), strict=False)
