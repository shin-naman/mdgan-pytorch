"""Accelerator smoke test (marker `gpu`; excluded by default).

One parametrized test per device replaces the six per-module CUDA/MPS tests:
the full training data path on the device (unroll -> decode pair -> mean
operation -> critic -> generator loss backward -> gradient penalty double
backward), plus one CPU/device numerical agreement check on the critic.
Smoke only: no performance or bit-exactness claims (docs/testing.md).

Run: python -m pytest -m gpu tests/test_devices.py -v
"""

import pytest
import torch

from conftest import SMALL
from mdgan.ops import mean_operation
from mdgan.rollout import make_fake_window, rollout

pytestmark = pytest.mark.gpu

_MPS = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
DEVICES = [
    pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not present")),
    pytest.param("mps", marks=pytest.mark.skipif(not _MPS, reason="Apple MPS not present")),
]


@pytest.mark.parametrize("device", DEVICES)
def test_training_data_path_runs_on_device(device, three_nets):
    gz, gy, dg = (net.to(device) for net in three_nets)
    for net in (gz, gy, dg):
        net.train()

    states = gz.unroll(torch.rand(16, SMALL["latent_dim"], device=device), steps=4)
    fake = mean_operation(torch.cat([gy.generate(states[3]), gy.generate(states[4])], dim=1))
    score = dg(fake)
    assert fake.device.type == device and score.shape == (16,)

    (-score.mean()).backward()                                        # generator step
    for net in (gz, gy):
        for name, p in net.named_parameters():
            if name.endswith(".weight"):
                assert p.grad is not None and torch.isfinite(p.grad).all(), name

    x = fake.detach().requires_grad_(True)                            # gradient penalty
    g, = torch.autograd.grad(dg(x).sum(), x, create_graph=True)
    ((g.flatten(1).norm(dim=1) - 1) ** 2).mean().backward()
    assert torch.isfinite(dg.input_conv.weight.grad).all()


@pytest.mark.parametrize("device", DEVICES)
def test_critic_agrees_with_cpu(device, dg):
    w = torch.randn(8, 2 * SMALL["block_len"], SMALL["n_channels"])
    with torch.no_grad():
        cpu = dg(w)
        dev = dg.to(device)(w.to(device)).cpu()
    torch.testing.assert_close(dev, cpu, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("device", DEVICES)
def test_one_cpu_generator_drives_models_on_device(device, three_nets):
    """Noise is drawn on the generator's device and moved: the same CPU seed gives
    the same noise on any device, and training/estimation code accepts a CPU generator."""
    gz, gy, _ = (net.to(device) for net in three_nets)
    r_dev = gz.sample_noise(4, device=device, generator=torch.Generator().manual_seed(0))
    r_cpu = gz.sample_noise(4, generator=torch.Generator().manual_seed(0))
    assert r_dev.device.type == device
    torch.testing.assert_close(r_dev.cpu(), r_cpu, rtol=0, atol=0)
    rp_dev, u_dev = gy.sample_noise(4, device=device, generator=torch.Generator().manual_seed(1))
    rp_cpu, u_cpu = gy.sample_noise(4, generator=torch.Generator().manual_seed(1))
    torch.testing.assert_close(u_dev.cpu(), u_cpu, rtol=0, atol=0)
    w = make_fake_window(gz, gy, 8, 2, 2, generator=torch.Generator().manual_seed(2))
    assert w.device.type == device
    out = rollout(gz, gy, 8, 2, 1, generator=torch.Generator().manual_seed(3))
    assert out.device.type == "cpu" and torch.isfinite(out).all()