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