"""Numerical regimes G_Y meets in a real run, and the G_z -> G_Y pairing.

Run: python -m pytest tests/G_Y/test_gy_numerics.py -v
"""

import pytest
import torch

from mdgan.models.latent_transition import LatentTransitionGenerator
from mdgan.models.trajectory_generator import TrajectoryGenerator

LAT, NR, M, C, H = 7, 3, 8, 3, 24


@pytest.fixture
def model():
    torch.manual_seed(2023)
    return TrajectoryGenerator(LAT, NR, M, C, hidden_dim=H)


# 1. Batch independence (no BatchNorm -> no coupling, ever)

def test_identical_rows_give_identical_outputs(model):
    z = torch.rand(1, LAT).expand(16, LAT)
    r = torch.rand(1, NR).expand(16, NR)
    u = torch.randn(1, M, C).expand(16, M, C)
    with torch.no_grad():
        y = model(z, r, u)
    torch.testing.assert_close(y, y[:1].expand_as(y), rtol=0, atol=0)


def test_output_independent_of_batch_composition_and_order(model):
    z, r, u = torch.rand(8, LAT), torch.rand(8, NR), torch.randn(8, M, C)
    perm = torch.randperm(8)
    with torch.no_grad():
        alone = model(z[:1], r[:1], u[:1])
        together = model(z, r, u)
        permuted = model(z[perm], r[perm], u[perm])
    torch.testing.assert_close(alone, together[:1], atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(permuted, together[perm], atol=1e-6, rtol=1e-5)


def test_singleton_batch_supported_in_both_modes(model):
    z, r, u = torch.rand(1, LAT), torch.rand(1, NR), torch.randn(1, M, C)
    for mode in (model.train, model.eval):
        mode()
        with torch.no_grad():
            assert model(z, r, u).shape == (1, M, C)


# 2. Scale and dtype

@pytest.mark.parametrize("scale", [1e2, 1e4, 1e6])
def test_large_inputs_stay_finite_and_scale_linearly_at_most(model, scale):
    """No normalization in G_Y, so output grows with input; it must not overflow fp32."""
    z, r, u = torch.rand(16, LAT) * scale, torch.rand(16, NR), torch.randn(16, M, C)
    with torch.no_grad():
        y = model(z, r, u)
    assert torch.isfinite(y).all()
    assert y.abs().max().item() < 1e3 * scale


def test_collapsed_zero_inputs_give_finite_output_and_gradients(model):
    z = torch.zeros(4, LAT, requires_grad=True)
    r = torch.zeros(4, NR)
    u = torch.zeros(4, M, C)
    y = model(z, r, u)
    assert torch.isfinite(y).all()
    (y * torch.randn_like(y)).sum().backward()
    assert torch.isfinite(z.grad).all()
    for n, p in model.named_parameters():
        assert torch.isfinite(p.grad).all(), n


def test_fp64_round_trip(model):
    model.double()
    z, r, u = (torch.rand(4, LAT, dtype=torch.float64), torch.rand(4, NR, dtype=torch.float64),
               torch.randn(4, M, C, dtype=torch.float64))
    with torch.no_grad():
        y = model(z, r, u)
    assert y.dtype == torch.float64 and torch.isfinite(y).all()
    r2, u2 = model.sample_noise(4, dtype=torch.float64)
    assert r2.dtype == u2.dtype == torch.float64


def test_bf16_autocast_forward_on_cpu(model):
    """Mixed precision may be used on CUDA later; make sure nothing in the path
    is dtype-fragile. Only a smoke test on CPU."""
    z, r, u = torch.rand(4, LAT), torch.rand(4, NR), torch.randn(4, M, C)
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        y = model(z, r, u)
    assert torch.isfinite(y.float()).all()


def test_output_scale_at_init_is_recorded():
    """Documents the init regime at production width with unit-scale inputs:
    the generator starts emitting O(0.01-1) values. Relevant to the input-scaling
    decision (normalize displacements by sigma). Not a requirement."""
    torch.manual_seed(0)
    gy = TrajectoryGenerator(32, 32).eval()
    z = torch.rand(256, 32)
    with torch.no_grad():
        y = gy.generate(z, generator=torch.Generator().manual_seed(0))
    s = y.std().item()
    assert 1e-3 < s < 10.0, s


# 3. Estimation-scale batch

def test_estimation_scale_batch(model):
    z = torch.rand(8192, LAT)
    with torch.no_grad():
        y = model.generate(z, generator=torch.Generator().manual_seed(0))
    assert y.shape == (8192, M, C) and torch.isfinite(y).all()


# 4. Pairing with G_z (the training data path)

def _pair():
    torch.manual_seed(0)
    gz = LatentTransitionGenerator(LAT, NR, hidden_dim=H)
    gy = TrajectoryGenerator(LAT, NR, M, C, hidden_dim=H)
    return gz, gy


def test_gz_to_gy_pairing_produces_2m_window_and_backprops_to_gz():
    gz, gy = _pair()
    gz.train(); gy.train()
    z0 = torch.rand(16, LAT)
    c, K = 3, 2
    states = gz.unroll(z0, steps=c + K)              # (c+K+1, B, LAT)
    j = c + 1
    window = torch.cat([gy.generate(states[j]), gy.generate(states[j + 1])], dim=1)
    assert window.shape == (16, 2 * M, C)
    assert torch.isfinite(window).all()
    (window * torch.randn_like(window)).sum().backward()
    for name, p in gz.named_parameters():
        if name.endswith(".weight"):
            assert p.grad is not None and p.grad.abs().sum().item() > 0, name
    for name, p in gy.named_parameters():
        assert p.grad is not None and p.grad.abs().sum().item() > 0, name


def test_consecutive_latents_decode_to_different_blocks():
    """y_j and y_{j+1} must differ: if G_Y ignored z, consecutive blocks with the
    same noise would be identical, and the critic could never learn transitions."""
    gz, gy = _pair()
    gz.eval(); gy.eval()
    states = gz.unroll(torch.rand(8, LAT), steps=4, generator=torch.Generator().manual_seed(0))
    r, u = gy.sample_noise(8, generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        y_a = gy(states[3], r, u)
        y_b = gy(states[4], r, u)
    assert (y_a - y_b).abs().max().item() > 1e-6


def test_cumsum_of_generated_displacements_gives_trajectories_from_origin():
    """Downstream contract: positions = cumsum over the time axis, starting at 0."""
    _, gy = _pair()
    with torch.no_grad():
        d = gy.generate(torch.rand(4, LAT), generator=torch.Generator().manual_seed(0))
    pos = torch.cumsum(d, dim=1)
    assert pos.shape == d.shape
    torch.testing.assert_close(pos[:, 0], d[:, 0], rtol=0, atol=0)
    torch.testing.assert_close(pos[:, -1], d.sum(dim=1), atol=1e-5, rtol=1e-5)


def test_estimation_rollout_length_for_80ns():
    """80 ns at 10 ps sampling and n_s=1 is 8000 steps = 125 blocks of M=64.
    The rollout loop will rely on this arithmetic; pin it."""
    steps_per_block = 64
    total_steps = int(80e3 / 10)
    assert total_steps % steps_per_block == 0
    assert total_steps // steps_per_block == 125