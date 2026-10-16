"""Numerical edge cases and rollout behavior for G_z.

These are the situations G_z can legitimately encounter over a 300k-iteration
run: collapsed batches, large-magnitude activations, long unrolls, large
estimation batches, and non-float32 dtypes. They are NOT a license for the
module to "handle" NaNs; divergence detection lives in the trainer.

Run: python -m pytest tests/G_z/test_gz_numerics.py -v
"""

import pytest
import torch
from torch import nn

from mdgan.models.latent_transition import LatentTransitionGenerator


@pytest.fixture
def model():
    torch.manual_seed(2023)
    return LatentTransitionGenerator(latent_dim=7, noise_dim=3, hidden_dim=24)


# 1. Degenerate batches

def test_identical_samples_in_batch_do_not_produce_nan(model):
    """Collapsed latent ensemble -> zero batch variance -> BatchNorm eps path.

    Output must be finite (and ~0 since every row is identical), not NaN.
    """
    model.eval()
    z = torch.full((16, 7), 0.3)
    r = torch.full((16, 3), 0.6)
    with torch.no_grad():
        out = model(z, r)
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out, torch.zeros_like(out), atol=1e-4, rtol=0)


def test_collapsed_batch_still_has_finite_gradients(model):
    model.train()
    z = torch.full((16, 7), 0.3, requires_grad=True)
    r = torch.full((16, 3), 0.6)
    out = model(z, r)
    (out * torch.randn_like(out)).sum().backward()
    assert torch.isfinite(z.grad).all()
    for name, p in model.named_parameters():
        assert torch.isfinite(p.grad).all(), name


# 2. Large-magnitude inputs (weights drifting, not NaNs)

@pytest.mark.parametrize("scale", [1e2, 1e4, 1e6])
def test_large_magnitude_inputs_stay_finite_and_normalized(model, scale):
    """BatchNorm makes the output scale-invariant; nothing overflows in fp32."""
    model.eval()
    z = torch.rand(64, 7) * scale
    r = torch.rand(64, 3)
    with torch.no_grad():
        out = model(z, r)
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out.mean(0), torch.zeros(7), atol=1e-3, rtol=0)
    assert out.abs().max().item() < 20.0


def test_fp64_round_trip(model):
    model.double().eval()
    z, r = torch.rand(8, 7, dtype=torch.float64), torch.rand(8, 3, dtype=torch.float64)
    with torch.no_grad():
        out = model(z, r)
    assert out.dtype == torch.float64 and torch.isfinite(out).all()


# 3. Unroll (burn-in) behavior

def test_unroll_shape_and_includes_initial_state(model):
    model.eval()
    z0 = torch.rand(5, 7)
    with torch.no_grad():
        states = model.unroll(z0, steps=11)
    assert states.shape == (12, 5, 7)
    torch.testing.assert_close(states[0], z0, rtol=0, atol=0)
    assert torch.isfinite(states).all()


def test_unroll_zero_steps_returns_only_z0(model):
    z0 = torch.rand(5, 7)
    with torch.no_grad():
        states = model.unroll(z0, steps=0)
    assert states.shape == (1, 5, 7)


def test_unroll_rejects_negative_steps(model):
    with pytest.raises(ValueError):
        model.unroll(torch.rand(5, 7), steps=-1)


def test_unroll_is_reproducible_with_explicit_generator(model):
    """Estimation runs must be reproducible from a seed for paper comparisons."""
    model.eval()
    z0 = torch.rand(5, 7)
    with torch.no_grad():
        a = model.unroll(z0, steps=8, generator=torch.Generator().manual_seed(1))
        b = model.unroll(z0, steps=8, generator=torch.Generator().manual_seed(1))
        c = model.unroll(z0, steps=8, generator=torch.Generator().manual_seed(2))
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert not torch.equal(a, c)


def test_unroll_does_not_consume_global_rng_when_generator_given(model):
    model.eval()
    z0 = torch.rand(5, 7)  # must be drawn BEFORE the snapshot
    before = torch.get_rng_state().clone()
    with torch.no_grad():
        model.unroll(z0, steps=4, generator=torch.Generator().manual_seed(0))
    assert torch.equal(before, torch.get_rng_state())


def test_unroll_without_generator_uses_global_rng(model):
    """Documents the default: no generator -> global RNG advances (seedable
    via torch.manual_seed, which the trainer will own)."""
    model.eval()
    z0 = torch.rand(5, 7)
    before = torch.get_rng_state().clone()
    with torch.no_grad():
        model.unroll(z0, steps=4)
    assert not torch.equal(before, torch.get_rng_state())


def test_full_unroll_backpropagates_to_z0_and_all_parameters(model):
    """Decision: full BPTT through the burn-in (docs/decisions.md).

    A detach() anywhere in the chain would zero the gradient at z0.
    """
    model.train()
    z0 = torch.rand(8, 7, requires_grad=True)
    states = model.unroll(z0, steps=8)
    loss = (states[-1] * torch.randn_like(states[-1])).sum()
    loss.backward()
    assert z0.grad is not None and z0.grad.abs().sum().item() > 0
    for name, p in model.named_parameters():
        if name.endswith(".weight"):
            assert p.grad is not None and p.grad.abs().sum().item() > 0, name


def test_long_rollout_stays_bounded_at_init(model):
    """125 generation steps for 80 ns at n_s=1, plus burn-in. The output
    BatchNorm must keep the ensemble bounded without any training."""
    model.eval()
    z0 = torch.rand(64, 7)
    with torch.no_grad():
        states = model.unroll(z0, steps=140)
    assert torch.isfinite(states).all()
    per_step_std = states[1:].std(dim=1)  # (steps, latent_dim)
    assert per_step_std.max().item() < 5.0
    assert per_step_std.min().item() > 0.1


def test_unroll_in_running_stats_mode_warns_nothing_but_differs_from_batch_mode():
    """Document that the two BN modes are not interchangeable after training."""
    torch.manual_seed(5)
    bs = LatentTransitionGenerator(7, 3, hidden_dim=24, use_running_stats=False)
    rs = LatentTransitionGenerator(7, 3, hidden_dim=24, use_running_stats=True)
    rs.load_state_dict(bs.state_dict(), strict=False)  # shared affine weights
    z0 = torch.rand(32, 7)
    gen = torch.Generator().manual_seed(3)
    bs.eval()
    rs.eval()
    with torch.no_grad():
        a = bs.unroll(z0, 3, generator=torch.Generator().manual_seed(3))
        b = rs.unroll(z0, 3, generator=gen)
    assert not torch.allclose(a[1:], b[1:])


# 4. Estimation-scale batch

def test_estimation_scale_batch_forward(model):
    """Batch 8192 is the paper's estimation ensemble. Small hidden width keeps
    this cheap; the point is no batch-size assumption anywhere."""
    model.eval()
    z = torch.rand(8192, 7)
    r = torch.rand(8192, 3)
    with torch.no_grad():
        out = model(z, r)
    assert out.shape == (8192, 7)
    assert torch.isfinite(out).all()


# 5. Wiring sanity after the rewrite

def test_no_unexpected_submodules(model):
    kinds = {type(m) for m in model.modules()}
    assert kinds <= {LatentTransitionGenerator, nn.Sequential, nn.Linear,
                     nn.LeakyReLU, nn.BatchNorm1d}

# 6. Sample noise

def test_sample_noise_shape_range_and_reproducibility(model):
    r1 = model.sample_noise(4096, generator=torch.Generator().manual_seed(0))
    r2 = model.sample_noise(4096, generator=torch.Generator().manual_seed(0))
    assert r1.shape == (4096, 3)
    assert r1.min() >= 0 and r1.max() < 1
    assert torch.equal(r1, r2)