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


@pytest.mark.parametrize("device", DEVICES)
def test_gradient_penalty_with_a_cpu_generator(device, dg):
    """Found on the first CHTC job: eps was requested from a CPU generator directly on the GPU."""
    from mdgan.losses import gradient_penalty
    dg = dg.to(device)
    real = torch.randn(8, 2 * SMALL["block_len"], SMALL["n_channels"], device=device)
    gp, norms = gradient_penalty(dg, real, torch.randn_like(real), generator=torch.Generator().manual_seed(0))
    assert gp.device.type == device and torch.isfinite(gp)


@pytest.mark.parametrize("device", DEVICES)
def test_trainer_runs_on_device(device, tmp_path):
    """The real training loop on the device, as a cluster job runs it: one CPU generator,
    critic and generator steps, checkpoints, a snapshot and an 8-chain evaluation."""
    import json

    import numpy as np

    from mdgan.config import Config
    from mdgan.data.dataset import WindowDataset
    from mdgan.data.preprocess import SystemData
    from mdgan.evaluate import reference_msd
    from mdgan.train import Trainer

    d = np.random.default_rng(0).normal(scale=0.05, size=(6, 200, 3)).astype(np.float32)
    system = SystemData(name="smoke", displacements=d, dt_ps=10.0, box_nm=np.full((6, 3), 5.0), meta={})
    c = Config(latent_dim=7, noise_dim=3, block_len=8, n_channels=3, hidden_dim=24, batch_size=8, n_critic=2,
               burn_in=2, pair_window=2, checkpoint_every=2, n_iterations=4, estimation_chains=8,
               estimation_time_ps=1000.0, err_window_ps=(20.0, 900.0), diffusion_fit_ps=(250.0, 500.0))
    out = tmp_path / "run"
    t = Trainer(c, WindowDataset(system, 8, 1), out, device=device, reference=reference_msd(system), log_every=1)
    assert t.run() == "done"
    assert t.gz.output_layer.weight.device.type == device
    evals = [json.loads(line) for line in (out / "evaluations.jsonl").read_text().splitlines()]
    assert [e["iteration"] for e in evals] == [2, 4] and all(np.isfinite(e["err_percent"]) for e in evals)