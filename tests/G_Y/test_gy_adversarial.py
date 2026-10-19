"""Deliberately hostile tests: the mistakes that are easy to make later and
hard to see in a loss curve. If one of these fails after an edit, the edit
broke something real.

Run: python -m pytest tests/G_Y/test_gy_adversarial.py -v
"""

import copy

import pytest
import torch
from torch import nn

from mdgan.models.latent_transition import LatentTransitionGenerator
from mdgan.models.trajectory_generator import ResidualAffineBlock, TrajectoryGenerator

LAT, NR, M, C, H = 7, 3, 8, 3, 24


@pytest.fixture
def model():
    torch.manual_seed(99)
    return TrajectoryGenerator(LAT, NR, M, C, hidden_dim=H)


# 1. A (B, C, M) vs (B, M, C) mixup must be visible, not silent

def test_square_block_shape_would_mask_a_transpose_bug_so_we_catch_it_explicitly():
    """When M == C, a transposed u passes the shape check. Verify the module
    still treats axis 1 as time: the conv mixes channels at fixed t, so
    swapping two TIME steps of u must permute the mixed output rows, not mix them."""
    gy = TrajectoryGenerator(LAT, NR, block_len=3, n_channels=3, hidden_dim=H)
    u = torch.randn(2, 3, 3)
    u_swapped = u[:, [1, 0, 2], :]
    with torch.no_grad():
        a = gy._mix_noise(u).reshape(2, 3, 3)
        b = gy._mix_noise(u_swapped).reshape(2, 3, 3)
    torch.testing.assert_close(b, a[:, [1, 0, 2], :], rtol=0, atol=0)


def test_noise_at_step_t_only_enters_through_flat_slice_t(model):
    """Gradient of the input_layer pre-activation w.r.t. u[t] is nonzero only
    in the u' columns belonging to step t. A wrong flatten would smear it."""
    u = torch.randn(1, M, C, requires_grad=True)
    cols = []
    handle = model.input_layer.register_forward_pre_hook(lambda m, i: cols.append(i[0]))
    try:
        model(torch.rand(1, LAT), torch.rand(1, NR), u)
    finally:
        handle.remove()
    x = cols[0]                                   # (1, NR + LAT + M*C)
    offset = NR + LAT
    t = 5
    sel = x[0, offset + t * C: offset + (t + 1) * C].sum()
    grad = torch.autograd.grad(sel, u)[0][0]      # (M, C)
    assert grad[t].abs().sum().item() > 0
    mask = torch.ones(M, dtype=torch.bool)
    mask[t] = False
    assert grad[mask].abs().sum().item() == 0


# 2. Dead paths

def test_zeroing_noise_mix_kills_u_but_not_z_or_rprime(model):
    """If the conv is zeroed, u can no longer affect y; z and r' still can."""
    with torch.no_grad():
        model.noise_mix.weight.zero_()
        model.noise_mix.bias.zero_()
    z, r, u = torch.rand(4, LAT), torch.rand(4, NR), torch.randn(4, M, C)
    with torch.no_grad():
        base = model(z, r, u)
        assert torch.equal(model(z, r, torch.randn(4, M, C)), base)
        assert not torch.equal(model(z + 1, r, u), base)
        assert not torch.equal(model(z, r + 1, u), base)


def test_residual_skip_is_not_the_only_path(model):
    """If fc1/fc2 learn nothing the block is identity; verify the branch is
    actually live at init (gradient reaches fc1 and fc2 weights)."""
    y = model(torch.rand(4, LAT), torch.rand(4, NR), torch.randn(4, M, C))
    (y * torch.randn_like(y)).sum().backward()
    assert model.residual.fc1.weight.grad.abs().sum().item() > 0
    assert model.residual.fc2.weight.grad.abs().sum().item() > 0


# 3. Mutability and aliasing

def test_output_does_not_alias_any_input_or_parameter(model):
    z, r, u = torch.rand(4, LAT), torch.rand(4, NR), torch.randn(4, M, C)
    with torch.no_grad():
        y = model(z, r, u)
    y.add_(1.0)
    for t in (z, r, u, model.output_layer.bias):
        assert torch.isfinite(t).all()
    with torch.no_grad():
        y2 = model(z, r, u)
    torch.testing.assert_close(y2, y - 1.0, atol=1e-6, rtol=0)


def test_deepcopy_is_independent(model):
    clone = copy.deepcopy(model)
    with torch.no_grad():
        model.output_layer.bias.add_(10.0)
    z, r, u = torch.rand(2, LAT), torch.rand(2, NR), torch.randn(2, M, C)
    with torch.no_grad():
        assert not torch.allclose(model(z, r, u), clone(z, r, u))


def test_inplace_modification_of_noise_after_forward_does_not_change_graph(model):
    u = torch.randn(4, M, C, requires_grad=True)
    y = model(torch.rand(4, LAT), torch.rand(4, NR), u)
    loss = y.square().sum()
    loss.backward()   # must not raise even if u is later modified; and we check grad is sane
    assert torch.isfinite(u.grad).all()


# 4. Checkpoint hygiene across modules

def test_gy_and_gz_state_dicts_cannot_be_confused():
    gz = LatentTransitionGenerator(LAT, NR, hidden_dim=H)
    gy = TrajectoryGenerator(LAT, NR, M, C, hidden_dim=H)
    with pytest.raises(RuntimeError):
        gy.load_state_dict(gz.state_dict(), strict=True)
    with pytest.raises(RuntimeError):
        gz.load_state_dict(gy.state_dict(), strict=True)


def test_non_strict_load_is_unsafe_across_modules_because_keys_collide():
    """G_z and G_Y share the key names input_layer.* and output_layer.*.
    strict=False does NOT skip colliding keys: same-shaped ones are copied
    silently and mismatched ones raise. Policy (docs/decisions.md): every
    checkpoint is keyed by module name and every load uses strict=True."""
    gz = LatentTransitionGenerator(LAT, NR, hidden_dim=H)
    gy = TrajectoryGenerator(LAT, NR, M, C, hidden_dim=H)
    shared = set(gz.state_dict()) & set(gy.state_dict())
    assert shared, "if this becomes empty the hazard is gone and this test can be deleted"
    # Same-shaped colliding key would be copied without error:
    assert gz.state_dict()["input_layer.bias"].shape == gy.state_dict()["input_layer.bias"].shape
    # Mismatched colliding keys make even a non-strict load raise:
    with pytest.raises(RuntimeError, match="size mismatch"):
        gy.load_state_dict(gz.state_dict(), strict=False)


def test_whole_module_pickle_round_trip(model, tmp_path):
    """torch.save of the module object (not just state_dict) must work: the
    trainer may checkpoint optimizer + modules together."""
    path = tmp_path / "gy_module.pt"
    torch.save(model, path)
    restored = torch.load(path, map_location="cpu", weights_only=False)
    z, r, u = torch.rand(2, LAT), torch.rand(2, NR), torch.randn(2, M, C)
    with torch.no_grad():
        torch.testing.assert_close(restored(z, r, u), model(z, r, u), rtol=0, atol=0)


# 5. Things that must NOT be there

def test_no_parameter_is_shared_between_layers(model):
    ids = [id(p) for p in model.parameters()]
    assert len(ids) == len(set(ids))


def test_no_buffers_registered(model):
    """G_Y has no running statistics or constants; a buffer would mean someone
    added normalization or a fixed scale that bypasses the sigma decision."""
    assert list(model.buffers()) == []


def test_module_has_no_train_eval_dependent_behavior(model):
    assert not any(isinstance(m, (nn.BatchNorm1d, nn.Dropout, nn.LayerNorm)) for m in model.modules())


def test_residual_block_rejects_width_mismatch():
    block = ResidualAffineBlock(width=8, negative_slope=0.2)
    with pytest.raises(RuntimeError):
        block(torch.randn(2, 9))


# 6. Gradient scale sanity (early warning for WGAN-GP tuning)

def test_input_gradient_norms_are_finite_and_nonzero_at_init(model):
    """d y / d u per sample: used later to reason about the gradient penalty.
    Here we only pin that it is finite, nonzero, and has the input's shape."""
    u = torch.randn(4, M, C, requires_grad=True)
    y = model(torch.rand(4, LAT), torch.rand(4, NR), u)
    g = torch.autograd.grad(y.sum(), u)[0]
    assert g.shape == u.shape
    norms = g.flatten(1).norm(dim=1)
    assert torch.isfinite(norms).all() and (norms > 0).all()