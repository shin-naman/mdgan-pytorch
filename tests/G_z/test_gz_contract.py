"""Adversarial contract tests for MD-GAN's G_z.

Assumed production API:
    from mdgan.models.latent_transition import LatentTransitionGenerator
    LatentTransitionGenerator(latent_dim, noise_dim, hidden_dim=2048,
                              negative_slope=0.2)
    model(z: Tensor[B, latent_dim], r: Tensor[B, noise_dim])
        -> Tensor[B, latent_dim]

These tests are independent of G_Y, GROMACS, and model training quality.
Run: python -m pytest tests/test_gz_contract.py -v
"""

import pytest
import torch
from torch import nn

from mdgan.models.latent_transition import LatentTransitionGenerator


@pytest.fixture
def model():
    # Small hidden width makes adversarial tests fast; it tests the same wiring.
    torch.manual_seed(717)
    return LatentTransitionGenerator(
        latent_dim=7, noise_dim=3, hidden_dim=24, negative_slope=0.17
    )


def _inputs(batch: int = 8, latent_dim: int = 7, noise_dim: int = 3):
    # The paper's r input is uniform random noise in [0, 1].
    return torch.rand(batch, latent_dim), torch.rand(batch, noise_dim)


# 1. Architecture and dimensional contracts

def test_reference_network_layers_and_dimensions():
    """Catch swapped dimensions, missing layers, wrong activations/normalizer."""
    gz = LatentTransitionGenerator(32, 11, hidden_dim=48, negative_slope=0.17)
    layers = list(gz.network.children())
    expected_types = [
        nn.Linear, nn.LeakyReLU,
        nn.Linear, nn.LeakyReLU,
        nn.Linear, nn.LeakyReLU,
        nn.Linear, nn.BatchNorm1d,
    ]
    assert len(layers) == len(expected_types)
    assert [type(layer) for layer in layers] == expected_types
    linear_shapes = [(layer.in_features, layer.out_features)
                     for layer in layers if isinstance(layer, nn.Linear)]
    assert linear_shapes == [(43, 48), (48, 48), (48, 48), (48, 32)]
    for activation in (layer for layer in layers if isinstance(layer, nn.LeakyReLU)):
        assert activation.negative_slope == pytest.approx(0.17)
    assert layers[-1].num_features == 32
    assert layers[-1].track_running_stats is True


def test_default_width_matches_reference_design():
    """Inspect production width without an expensive production-size forward pass."""
    gz = LatentTransitionGenerator(latent_dim=32, noise_dim=11)
    linears = [m for m in gz.modules() if isinstance(m, nn.Linear)]
    assert len(linears) == 4
    assert [layer.out_features for layer in linears] == [2048, 2048, 2048, 32]


@pytest.mark.parametrize("latent_dim,noise_dim,batch", [
    (32, 1, 2), (32, 16, 64), (32, 37, 3), (3, 7, 8), (11, 2, 5),
])
def test_independent_latent_noise_and_batch_dimensions(latent_dim, noise_dim, batch):
    """No hard-coded 'r must make input width 64' or batch-size assumptions."""
    gz = LatentTransitionGenerator(latent_dim, noise_dim, hidden_dim=24)
    z, r = _inputs(batch, latent_dim, noise_dim)
    result = gz(z, r)
    assert result.shape == (batch, latent_dim)
    assert result.dtype == z.dtype
    assert torch.isfinite(result).all()


def test_concat_is_featurewise_and_in_z_then_r_order(model):
    """A forward pre-hook catches a silently reversed cat((r,z)) or dim error."""
    first_layer = next(m for m in model.network if isinstance(m, nn.Linear))
    seen = []
    handle = first_layer.register_forward_pre_hook(
        lambda module, inputs: seen.append(inputs[0].detach().clone())
    )
    try:
        z = torch.arange(8 * 7, dtype=torch.float32).reshape(8, 7) / 13
        r = 100 + torch.arange(8 * 3, dtype=torch.float32).reshape(8, 3)
        model.eval()
        model(z, r)
    finally:
        handle.remove()
    assert len(seen) == 1
    torch.testing.assert_close(seen[0], torch.cat([z, r], dim=1))


# 2. Failure behavior for malformed input

@pytest.mark.parametrize("problem", [
    "wrong_z_features", "wrong_r_features", "different_batches",
    "rank_1_z", "rank_3_r", "dtype_mismatch",
])
def test_malformed_inputs_do_not_silently_produce_a_state(model, problem):
    """PyTorch RuntimeErrors are acceptable; silent acceptance is not."""
    model.eval()
    z, r = _inputs()
    if problem == "wrong_z_features":
        z = z[:, :-1]
    elif problem == "wrong_r_features":
        r = r[:, :-1]
    elif problem == "different_batches":
        r = r[:-1]
    elif problem == "rank_1_z":
        z = z[0]
    elif problem == "rank_3_r":
        r = r.unsqueeze(-1)
    elif problem == "dtype_mismatch":
        r = r.double()
    with pytest.raises((RuntimeError, ValueError, TypeError, IndexError)):
        model(z, r)


def test_forward_does_not_modify_callers_inputs(model):
    z, r = _inputs()
    original_z, original_r = z.clone(), r.clone()
    model(z, r)
    torch.testing.assert_close(z, original_z)
    torch.testing.assert_close(r, original_r)


# 3. Randomness contracts

def test_fixed_inputs_produce_exactly_repeatable_outputs_in_eval(model):
    """Hidden torch.rand/dropout inside forward would violate explicit-r API."""
    model.eval()
    z, r = _inputs()
    with torch.no_grad():
        out1 = model(z, r)
        out2 = model(z, r)
    torch.testing.assert_close(out1, out2, rtol=0, atol=0)


def test_forward_does_not_consume_torch_random_state(model):
    """All transition randomness should arrive explicitly via r."""
    model.eval()
    z, r = _inputs()
    rng_before = torch.get_rng_state().clone()
    with torch.no_grad():
        model(z, r)
    rng_after = torch.get_rng_state()
    assert torch.equal(rng_before, rng_after)


def test_changing_noise_changes_next_state(model):
    """Catch a forward pass that ignores r; not a claim about trained dynamics."""
    model.eval()
    z, r = _inputs()
    with torch.no_grad():
        original = model(z, r)
        perturbed = model(z, r + 2.0)
    assert (original - perturbed).abs().max().item() > 1e-6


def test_changing_latent_state_changes_next_state(model):
    """Catch a forward pass that ignores z."""
    model.eval()
    z, r = _inputs()
    with torch.no_grad():
        original = model(z, r)
        perturbed = model(z + 2.0, r)
    assert (original - perturbed).abs().max().item() > 1e-6


# 4. Autograd and minimal optimizer integration

def test_gradients_reach_both_inputs_and_every_trainable_layer(model):
    """Detect detach(), no_grad(), dead branches, or unused affine layers."""
    model.train()
    z, r = _inputs()
    z.requires_grad_()
    r.requires_grad_()
    output = model(z, r)
    # A plain output.mean() is a BAD test with final BatchNorm: it is ~0.
    weights = torch.randn_like(output)
    loss = (output * weights).sum()
    loss.backward()
    for name, tensor in [("z", z), ("r", r)]:
        assert tensor.grad is not None, f"no gradient for {name}"
        assert torch.isfinite(tensor.grad).all(), f"nonfinite {name} gradient"
        assert tensor.grad.abs().sum().item() > 0, f"zero gradient for {name}"
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, f"unused parameter: {name}"
        assert torch.isfinite(parameter.grad).all(), f"bad gradient: {name}"
        # A bias immediately before BatchNorm may have ~zero gradient
        # because BatchNorm removes the batch mean; that is not a bug.
        if name.endswith(".weight"):
            assert parameter.grad.abs().sum().item() > 0, f"zero weight gradient: {name}"


def test_optimizer_step_really_updates_model_parameters(model):
    """Smoke-level integration: forward + backward + Adam update."""
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    z, r = _inputs()
    optimizer.zero_grad(set_to_none=True)
    output = model(z, r)
    target = torch.randn_like(output)
    loss = (output - target).square().mean()
    assert torch.isfinite(loss)
    loss.backward()
    optimizer.step()
    changed = [name for name, p in model.named_parameters()
               if not torch.equal(before[name], p.detach())]
    assert changed, "optimizer.step() did not change any learnable parameters"
    # More discriminating than just checking the BatchNorm bias changed:
    assert any(name.startswith("network.0.") for name in changed), changed
    assert any(name.startswith("network.6.") for name in changed), changed


# 5. BatchNorm correctness and inference behavior

def test_training_mode_updates_batchnorm_statistics(model):
    bn = next(m for m in model.modules() if isinstance(m, nn.BatchNorm1d))
    model.train()
    old_count = bn.num_batches_tracked.item()
    z, r = _inputs(batch=8)
    model(z, r)
    assert bn.num_batches_tracked.item() == old_count + 1


def test_eval_mode_keeps_batchnorm_statistics_fixed(model):
    bn = next(m for m in model.modules() if isinstance(m, nn.BatchNorm1d))
    model.train()
    for _ in range(3):
        model(*_inputs())
    model.eval()
    before_mean = bn.running_mean.clone()
    before_var = bn.running_var.clone()
    with torch.no_grad():
        model(*_inputs())
    torch.testing.assert_close(before_mean, bn.running_mean, rtol=0, atol=0)
    torch.testing.assert_close(before_var, bn.running_var, rtol=0, atol=0)


def test_eval_state_is_independent_of_other_samples_in_batch(model):
    """Important when inference is chunked into different batch sizes."""
    model.train()
    for _ in range(3):
        model(*_inputs())
    model.eval()
    z, r = _inputs(batch=8)
    with torch.no_grad():
        isolated = model(z[:1], r[:1])
        together = model(z, r)[:1]
    torch.testing.assert_close(isolated, together, atol=1e-6, rtol=1e-5)


def test_singleton_batch_is_supported_in_eval_but_not_train(model):
    """KNOWN BatchNorm1d limitation, not automatically a G_z implementation bug."""
    z, r = _inputs(batch=1)
    model.eval()
    with torch.no_grad():
        assert model(z, r).shape == (1, 7)
    model.train()
    with pytest.raises(ValueError, match="more than 1 value per channel"):
        model(z, r)


# 6. State management and short rollouts (not a stationarity test)

def test_model_checkpoint_state_dict_restores_exact_eval_predictions(model, tmp_path):
    model.train()
    for _ in range(3):
        model(*_inputs())
    model.eval()
    z, r = _inputs()
    with torch.no_grad():
        expected = model(z, r)
    path = tmp_path / "gz_weights.pt"
    torch.save(model.state_dict(), path)
    restored = LatentTransitionGenerator(7, 3, hidden_dim=24, negative_slope=0.17)
    restored.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    restored.eval()
    with torch.no_grad():
        actual = restored(z, r)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_sixteen_successive_transitions_preserve_shapes_and_finiteness(model):
    """Catches accumulation of NaNs due to forward mechanics, not learned physics."""
    model.eval()
    z, _ = _inputs(batch=5)
    with torch.no_grad():
        for _ in range(16):
            r = torch.rand(5, 3)
            z = model(z, r)
            assert z.shape == (5, 7)
            assert torch.isfinite(z).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not present")
def test_cuda_forward_and_backward():
    device = torch.device("cuda")
    gz = LatentTransitionGenerator(7, 3, hidden_dim=24).to(device)
    gz.train()
    z = torch.rand(8, 7, device=device, requires_grad=True)
    r = torch.rand(8, 3, device=device, requires_grad=True)
    out = gz(z, r)
    out.square().mean().backward()
    assert out.device.type == "cuda"
    assert z.grad is not None and r.grad is not None


@pytest.mark.skipif(not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available(),
                    reason="Apple MPS not present")
def test_mps_forward_and_backward():
    device = torch.device("mps")
    gz = LatentTransitionGenerator(7, 3, hidden_dim=24).to(device)
    gz.train()
    z = torch.rand(8, 7, device=device, requires_grad=True)
    r = torch.rand(8, 3, device=device, requires_grad=True)
    out = gz(z, r)
    (out * torch.randn_like(out)).mean().backward()
    assert out.device.type == "mps"
    assert z.grad is not None and r.grad is not None
