"""Architecture contract tests for MD-GAN's G_Y (TrajectoryGenerator).

Pins the module to Kawada et al. (2022) ESI Fig. S2(A). Any change to the
wiring, the concatenation order, or the flatten order must fail here.

API:
    TrajectoryGenerator(latent_dim, noise_dim, block_len=64, n_channels=3,
                        hidden_dim=2048, negative_slope=0.2)
    model(z: (B, n_z), r_prime: (B, n_r), u: (B, M, C)) -> (B, M, C)
    model.sample_noise(batch, *, device, dtype, generator) -> (r_prime, u)
    model.generate(z, generator) -> (B, M, C)

Run: python -m pytest tests/G_Y/test_gy_contract.py -v
"""

import pytest
import torch
from torch import nn

from mdgan.models.trajectory_generator import ResidualAffineBlock, TrajectoryGenerator

LAT, NR, M, C, H = 7, 3, 8, 3, 24


@pytest.fixture
def model():
    torch.manual_seed(717)
    return TrajectoryGenerator(latent_dim=LAT, noise_dim=NR, block_len=M,
                               n_channels=C, hidden_dim=H, negative_slope=0.17)


def _inputs(batch=8, latent_dim=LAT, noise_dim=NR, block_len=M, n_channels=C):
    return (torch.rand(batch, latent_dim),
            torch.rand(batch, noise_dim),
            torch.randn(batch, block_len, n_channels))


# 1. Layers and dimensions

def test_reference_layers_and_dimensions():
    gy = TrajectoryGenerator(32, 11, block_len=64, n_channels=3, hidden_dim=48, negative_slope=0.17)
    assert isinstance(gy.noise_mix, nn.Conv1d)
    assert (gy.noise_mix.in_channels, gy.noise_mix.out_channels) == (3, 3)
    assert gy.noise_mix.kernel_size == (1,) and gy.noise_mix.stride == (1,)
    assert gy.noise_mix.padding == (0,) and gy.noise_mix.bias is not None
    assert isinstance(gy.residual, ResidualAffineBlock)
    shapes = [(m.in_features, m.out_features) for m in
              (gy.input_layer, gy.residual.fc1, gy.residual.fc2, gy.output_layer)]
    assert shapes == [(11 + 32 + 192, 48), (48, 48), (48, 48), (48, 192)]
    for act in (gy.input_act, gy.post_act, gy.residual.act):
        assert isinstance(act, nn.LeakyReLU)
        assert act.negative_slope == pytest.approx(0.17)
    assert sorted(n for n, _ in gy.named_children()) == sorted(
        ["noise_mix", "input_layer", "input_act", "residual", "post_act", "output_layer"])
    assert sorted(n for n, _ in gy.residual.named_children()) == ["act", "fc1", "fc2"]


def test_default_width_matches_reference_design():
    gy = TrajectoryGenerator(latent_dim=32, noise_dim=32)
    assert gy.input_layer.in_features == 32 + 32 + 192
    assert [m.out_features for m in (gy.input_layer, gy.residual.fc1, gy.residual.fc2)] == [2048] * 3
    assert gy.output_layer.out_features == 192 == gy.flat_dim


def test_parameter_count_matches_closed_form():
    """Catches an accidentally added or dropped layer."""
    gy = TrajectoryGenerator(32, 32)
    n_in = 32 + 32 + 192
    expected = (
        3 * 3 + 3                              # conv 1x1 weight + bias
        + n_in * 2048 + 2048                   # input_layer
        + 2 * (2048 * 2048 + 2048)             # residual fc1, fc2
        + 2048 * 192 + 192                     # output_layer
    )
    assert sum(p.numel() for p in gy.parameters()) == expected


def test_no_normalization_or_stochastic_layers(model):
    kinds = {type(m) for m in model.modules()}
    assert kinds == {TrajectoryGenerator, ResidualAffineBlock, nn.Conv1d, nn.Linear, nn.LeakyReLU}


@pytest.mark.parametrize("bad", [
    dict(latent_dim=0), dict(noise_dim=-1), dict(block_len=0), dict(n_channels=0),
    dict(hidden_dim=0), dict(latent_dim=True), dict(block_len=2.0), dict(n_channels="3"),
])
def test_invalid_constructor_arguments_rejected(bad):
    kwargs = dict(latent_dim=7, noise_dim=3, block_len=8, n_channels=3, hidden_dim=16)
    kwargs.update(bad)
    with pytest.raises(ValueError, match="positive integer"):
        TrajectoryGenerator(**kwargs)


@pytest.mark.parametrize("latent_dim,noise_dim,block_len,n_channels,batch", [
    (32, 32, 64, 3, 2), (32, 1, 64, 9, 5), (4, 7, 16, 3, 64), (7, 3, 8, 3, 1), (1, 1, 1, 1, 3),
])
def test_independent_dimensions_and_batch(latent_dim, noise_dim, block_len, n_channels, batch):
    gy = TrajectoryGenerator(latent_dim, noise_dim, block_len, n_channels, hidden_dim=16)
    z, r, u = _inputs(batch, latent_dim, noise_dim, block_len, n_channels)
    y = gy(z, r, u)
    assert y.shape == (batch, block_len, n_channels)
    assert y.dtype == z.dtype and torch.isfinite(y).all()


# 2. Data path: order of operations, concat order, flatten order

def test_forward_calls_layers_in_reference_order(model):
    order = []
    names = ["noise_mix", "input_layer", "input_act", "residual", "post_act", "output_layer"]
    handles = [getattr(model, n).register_forward_hook(lambda m, i, o, n=n: order.append(n))
               for n in names]
    try:
        with torch.no_grad():
            model(*_inputs())
    finally:
        for h in handles:
            h.remove()
    assert order == names


def test_concat_order_is_rprime_then_z_then_mixed_noise(model):
    """Fig. S2(A), left to right: r'_i, z_i, u'_i."""
    seen = []
    handle = model.input_layer.register_forward_pre_hook(lambda m, inp: seen.append(inp[0].detach().clone()))
    z = torch.arange(8 * LAT, dtype=torch.float32).reshape(8, LAT) / 13
    r = 100 + torch.arange(8 * NR, dtype=torch.float32).reshape(8, NR)
    u = torch.randn(8, M, C)
    try:
        with torch.no_grad():
            model(z, r, u)
            u_mixed = model.noise_mix(u.transpose(1, 2)).transpose(1, 2).reshape(8, M * C)
    finally:
        handle.remove()
    torch.testing.assert_close(seen[0], torch.cat([r, z, u_mixed], dim=1))
    # Negative control: the other plausible order is NOT what was seen.
    assert not torch.allclose(seen[0], torch.cat([z, r, u_mixed], dim=1))


def test_noise_mix_is_the_same_cxc_affine_at_every_step(model):
    """Kernel-1 Conv1d == u[t] @ W^T + b independently per step; no temporal mixing."""
    u = torch.randn(4, M, C)
    W = model.noise_mix.weight.detach()[:, :, 0]
    b = model.noise_mix.bias.detach()
    with torch.no_grad():
        got = model.noise_mix(u.transpose(1, 2)).transpose(1, 2)
    torch.testing.assert_close(got, u @ W.T + b, atol=1e-6, rtol=1e-5)
    # Perturbing step 2 must not change any other step after the conv.
    u2 = u.clone()
    u2[:, 2] += 5.0
    with torch.no_grad():
        got2 = model.noise_mix(u2.transpose(1, 2)).transpose(1, 2)
    mask = torch.ones(M, dtype=torch.bool)
    mask[2] = False
    torch.testing.assert_close(got2[:, mask], got[:, mask], rtol=0, atol=0)
    assert not torch.allclose(got2[:, 2], got[:, 2])


def test_mix_noise_flatten_is_step_major(model):
    """With noise_mix set to identity, _mix_noise(u) must equal u.reshape(B, M*C)
    (index = step*C + axis). A (B, C, M)-ordered flatten would fail this."""
    with torch.no_grad():
        model.noise_mix.weight.copy_(torch.eye(C).unsqueeze(-1))
        model.noise_mix.bias.zero_()
        u = torch.randn(4, M, C)
        flat = model._mix_noise(u)
    torch.testing.assert_close(flat, u.reshape(4, M * C), rtol=0, atol=0)
    assert not torch.equal(flat, u.transpose(1, 2).reshape(4, M * C))


def test_output_reshape_is_step_major(model):
    """y[:, t, a] comes from output_layer flat index t*C + a."""
    z, r, u = _inputs(batch=2)
    flat = []
    handle = model.output_layer.register_forward_hook(lambda m, i, o: flat.append(o.detach()))
    try:
        with torch.no_grad():
            y = model(z, r, u)
    finally:
        handle.remove()
    for t, a in [(0, 0), (3, 1), (M - 1, C - 1)]:
        torch.testing.assert_close(y[:, t, a], flat[0][:, t * C + a], rtol=0, atol=0)


def test_output_is_bias_when_output_weights_are_zero(model):
    """Pins that output_layer is the LAST op: nothing acts on y after it."""
    with torch.no_grad():
        model.output_layer.weight.zero_()
        y = model(*_inputs(batch=3))
    expected = model.output_layer.bias.detach().reshape(1, M, C).expand(3, M, C)
    torch.testing.assert_close(y, expected, rtol=0, atol=0)


# 3. Residual block

def test_residual_block_is_identity_plus_branch():
    torch.manual_seed(0)
    block = ResidualAffineBlock(width=16, negative_slope=0.2)
    x = torch.randn(5, 16)
    with torch.no_grad():
        branch = block.fc2(block.act(block.fc1(x)))
        torch.testing.assert_close(block(x), x + branch)
        block.fc2.weight.zero_()
        block.fc2.bias.zero_()
        torch.testing.assert_close(block(x), x, rtol=0, atol=0)


def test_residual_block_gradient_has_identity_component():
    """d(out)/d(x) = I + J_branch: with the branch zeroed the gradient is exactly I."""
    block = ResidualAffineBlock(width=6, negative_slope=0.2)
    with torch.no_grad():
        block.fc2.weight.zero_()
        block.fc2.bias.zero_()
    x = torch.randn(1, 6, requires_grad=True)
    jac = torch.autograd.functional.jacobian(lambda v: block(v), x)[0, :, 0, :]
    torch.testing.assert_close(jac, torch.eye(6), rtol=0, atol=0)


def test_no_activation_after_residual_add_inside_block():
    """The LReLU after the skip-add belongs to the main stack (post_act), not the block."""
    block = ResidualAffineBlock(width=4, negative_slope=0.2)
    with torch.no_grad():
        block.fc2.weight.zero_()
        block.fc2.bias.zero_()
        x = -torch.ones(2, 4)
        torch.testing.assert_close(block(x), x, rtol=0, atol=0)  # a LReLU would scale by 0.2


# 4. Malformed input and purity

@pytest.mark.parametrize("problem", [
    "wrong_z", "wrong_r", "wrong_u_steps", "wrong_u_channels", "u_transposed",
    "batch_mismatch_z", "batch_mismatch_u", "rank_1_z", "rank_2_u", "rank_3_z", "dtype_mismatch",
])
def test_malformed_inputs_raise(model, problem):
    z, r, u = _inputs()
    if problem == "wrong_z":
        z = z[:, :-1]
    elif problem == "wrong_r":
        r = r[:, :-1]
    elif problem == "wrong_u_steps":
        u = u[:, :-1]
    elif problem == "wrong_u_channels":
        u = u[:, :, :-1]
    elif problem == "u_transposed":
        u = torch.randn(8, C, M)   # (B, C, M) instead of (B, M, C); M != C here so it must fail
    elif problem == "batch_mismatch_z":
        z = z[:-1]
    elif problem == "batch_mismatch_u":
        u = u[:-1]
    elif problem == "rank_1_z":
        z = z[0]
    elif problem == "rank_2_u":
        u = u[0]
    elif problem == "rank_3_z":
        z = z.unsqueeze(-1)
    elif problem == "dtype_mismatch":
        u = u.double()
    with pytest.raises((RuntimeError, ValueError, TypeError, IndexError)):
        model(z, r, u)


def test_forward_does_not_modify_inputs(model):
    z, r, u = _inputs()
    originals = (z.clone(), r.clone(), u.clone())
    model(z, r, u)
    for a, b in zip((z, r, u), originals):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_forward_does_not_consume_global_rng(model):
    z, r, u = _inputs()
    before = torch.get_rng_state().clone()
    with torch.no_grad():
        model(z, r, u)
    assert torch.equal(before, torch.get_rng_state())


# 5. Determinism, stochasticity, noise helpers

def test_eval_equals_train_and_repeatable(model):
    z, r, u = _inputs()
    model.train()
    with torch.no_grad():
        a = model(z, r, u)
    model.eval()
    with torch.no_grad():
        b = model(z, r, u)
        c = model(z, r, u)
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    torch.testing.assert_close(b, c, rtol=0, atol=0)


@pytest.mark.parametrize("which", ["z", "r_prime", "u"])
def test_each_input_affects_output(model, which):
    z, r, u = _inputs()
    with torch.no_grad():
        base = model(z, r, u)
        other = model(z + 2.0 if which == "z" else z,
                      r + 2.0 if which == "r_prime" else r,
                      u + 2.0 if which == "u" else u)
    assert (base - other).abs().max().item() > 1e-6


def test_decoder_is_stochastic_given_fixed_latent(model):
    """Same z, different noise -> different trajectories. A G_Y that ignores its
    noise would collapse p(Y|z) to a point and could not model diffusion."""
    z = torch.rand(1, LAT).expand(64, LAT)
    with torch.no_grad():
        y = model.generate(z, generator=torch.Generator().manual_seed(0))
    assert y.std(dim=0).mean().item() > 1e-4


def test_sample_noise_shapes_ranges_and_dtype_device(model):
    r, u = model.sample_noise(4096, dtype=torch.float64, generator=torch.Generator().manual_seed(0))
    assert r.shape == (4096, NR) and u.shape == (4096, M, C)
    assert r.dtype == u.dtype == torch.float64
    assert r.min() >= 0 and r.max() < 1
    assert abs(u.mean().item()) < 0.05 and abs(u.std().item() - 1) < 0.05


def test_sample_noise_rejects_bad_batch(model):
    with pytest.raises((RuntimeError, ValueError)):
        model.sample_noise(-1)


def test_generate_reproducible_with_generator_and_leaves_global_rng(model):
    z = torch.rand(5, LAT)
    before = torch.get_rng_state().clone()
    with torch.no_grad():
        a = model.generate(z, generator=torch.Generator().manual_seed(1))
        b = model.generate(z, generator=torch.Generator().manual_seed(1))
        c = model.generate(z, generator=torch.Generator().manual_seed(2))
    assert torch.equal(before, torch.get_rng_state())
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert not torch.equal(a, c)
    assert a.shape == (5, M, C)


def test_generate_without_generator_advances_global_rng(model):
    z = torch.rand(5, LAT)
    before = torch.get_rng_state().clone()
    with torch.no_grad():
        model.generate(z)
    assert not torch.equal(before, torch.get_rng_state())


def test_generate_matches_forward_with_same_noise(model):
    z = torch.rand(5, LAT)
    with torch.no_grad():
        y_gen = model.generate(z, generator=torch.Generator().manual_seed(9))
        r, u = model.sample_noise(5, generator=torch.Generator().manual_seed(9))
        y_fwd = model(z, r, u)
    torch.testing.assert_close(y_gen, y_fwd, rtol=0, atol=0)


# 6. Autograd, optimizer, state

def test_gradients_reach_all_inputs_and_parameters(model):
    model.train()
    z, r, u = _inputs()
    for t in (z, r, u):
        t.requires_grad_()
    y = model(z, r, u)
    (y * torch.randn_like(y)).sum().backward()
    for name, t in (("z", z), ("r_prime", r), ("u", u)):
        assert t.grad is not None and torch.isfinite(t.grad).all(), name
        assert t.grad.abs().sum().item() > 0, f"zero gradient for {name}"
    for name, p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
        assert p.grad.abs().sum().item() > 0, f"zero gradient: {name}"


def test_gradient_flows_through_generate_to_z(model):
    z = torch.rand(4, LAT, requires_grad=True)
    y = model.generate(z, generator=torch.Generator().manual_seed(0))
    y.square().sum().backward()
    assert z.grad is not None and z.grad.abs().sum().item() > 0


def test_optimizer_step_updates_every_submodule(model):
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    y = model(*_inputs())
    (y - torch.randn_like(y)).square().mean().backward()
    opt.step()
    changed = {n for n, p in model.named_parameters() if not torch.equal(before[n], p.detach())}
    for prefix in ("noise_mix.", "input_layer.", "residual.fc1.", "residual.fc2.", "output_layer."):
        assert any(n.startswith(prefix) for n in changed), prefix


def test_state_dict_round_trip_and_key_names(model, tmp_path):
    model.eval()
    z, r, u = _inputs()
    with torch.no_grad():
        expected = model(z, r, u)
    path = tmp_path / "gy.pt"
    torch.save(model.state_dict(), path)
    state = torch.load(path, map_location="cpu", weights_only=True)
    assert set(state) == {
        "noise_mix.weight", "noise_mix.bias", "input_layer.weight", "input_layer.bias",
        "residual.fc1.weight", "residual.fc1.bias", "residual.fc2.weight", "residual.fc2.bias",
        "output_layer.weight", "output_layer.bias",
    }
    restored = TrajectoryGenerator(LAT, NR, M, C, hidden_dim=H, negative_slope=0.17)
    restored.load_state_dict(state, strict=True)
    restored.eval()
    with torch.no_grad():
        torch.testing.assert_close(restored(z, r, u), expected, rtol=0, atol=0)


def test_state_dict_of_different_config_is_rejected(model):
    other = TrajectoryGenerator(LAT, NR, M + 1, C, hidden_dim=H)
    with pytest.raises(RuntimeError):
        model.load_state_dict(other.state_dict(), strict=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not present")
def test_cuda_forward_backward():
    dev = torch.device("cuda")
    gy = TrajectoryGenerator(LAT, NR, M, C, hidden_dim=H).to(dev)
    z, r, u = (t.to(dev).requires_grad_() for t in _inputs())
    y = gy(z, r, u)
    y.square().mean().backward()
    assert y.device.type == "cuda" and u.grad is not None
    assert gy.generate(z.detach()).device.type == "cuda"


@pytest.mark.skipif(not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available(),
                    reason="Apple MPS not present")
def test_mps_forward_backward():
    dev = torch.device("mps")
    gy = TrajectoryGenerator(LAT, NR, M, C, hidden_dim=H).to(dev)
    z, r, u = (t.to(dev).requires_grad_() for t in _inputs())
    y = gy(z, r, u)
    (y * torch.randn_like(y)).mean().backward()
    assert y.device.type == "mps" and u.grad is not None
    assert gy.generate(z.detach()).device.type == "mps"