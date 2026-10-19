"""Integration: the three networks wired as the trainer will wire them.

These test the data path BETWEEN modules (generator step, critic step with
detached fake, GP on the centered fake window), not any single module. When
the trainer exists, its gradient_penalty replaces _gp here.

Run: python -m pytest tests/integration -v
"""

import torch

from mdgan.models.critic import Critic
from mdgan.models.latent_transition import LatentTransitionGenerator
from mdgan.models.trajectory_generator import TrajectoryGenerator
from mdgan.ops import mean_operation

LAT, NR, M, C, H = 7, 3, 8, 3, 24
W = 2 * M


def _gp(critic, real, fake, lam=10.0):
    """Gradient penalty exactly as in Gulrajani et al. 2017 / Kawada 2022 eq. 5."""
    b = real.shape[0]
    eps = torch.rand(b, 1, 1, dtype=real.dtype)
    x_hat = (eps * real + (1 - eps) * fake).requires_grad_(True)
    score = critic(x_hat)
    grad, = torch.autograd.grad(score.sum(), x_hat, create_graph=True)
    norm = grad.flatten(1).norm(dim=1)
    return lam * ((norm - 1) ** 2).mean(), norm


def _nets():
    torch.manual_seed(0)
    return (LatentTransitionGenerator(LAT, NR, hidden_dim=H),
            TrajectoryGenerator(LAT, NR, M, C, hidden_dim=H),
            Critic(M, C, hidden_dim=H))


def _fake_window(gz, gy, batch=16, c=3, K=2):
    """Decided pairing: burn in c, unroll K more, decode (z_j, z_{j+1}), center."""
    states = gz.unroll(torch.rand(batch, LAT), steps=c + K)
    j = c + 1
    return mean_operation(torch.cat([gy.generate(states[j]), gy.generate(states[j + 1])], dim=1))


def test_critic_step_wgan_gp_loss_and_update():
    """One full critic step: L = E[D(fake)] - E[D(real)] + GP; backward; Adam step."""
    model = Critic(M, C, hidden_dim=H)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.0, 0.9))
    real = torch.randn(32, W, C)
    fake = torch.randn(32, W, C) * 2.0
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    loss = model(fake).mean() - model(real).mean() + _gp(model, real, fake)[0]
    opt.zero_grad()
    loss.backward()
    opt.step()
    assert torch.isfinite(loss)
        # output_layer.bias has identically zero gradient under WGAN-GP (gauge freedom);
    # every other parameter must move.
    for n, p in model.named_parameters():
        if n == "output_layer.bias":
            assert torch.equal(before[n], p.detach())
        else:
            assert not torch.equal(before[n], p.detach()), n


# Full three-network path

def test_generator_step_backprops_through_critic_into_both_generators():
    """Generator loss -E[D(G)] with the mean operation applied to the fake window."""
    gz, gy, d = _nets()
    for n in (gz, gy, d):
        n.train()
    fake = _fake_window(gz, gy)
    assert fake.shape == (16, W, C)
    torch.testing.assert_close(fake.mean(0), torch.zeros(W, C), atol=1e-5, rtol=0)
    loss_g = -d(fake).mean()
    loss_g.backward()
    for net, name in ((gz, "G_z"), (gy, "G_Y")):
        for pn, p in net.named_parameters():
            if pn.endswith(".weight"):
                assert p.grad is not None and p.grad.abs().sum().item() > 0, f"{name}.{pn}"
    # Critic params receive gradient too; the trainer must simply not step them here.
    assert d.hidden1.weight.grad is not None


def test_critic_step_with_detached_fake_leaves_generators_untouched():
    gz, gy, d = _nets()
    fake = _fake_window(gz, gy).detach()
    real = torch.randn(16, W, C)
    loss_d = d(fake).mean() - d(real).mean() + _gp(d, real, fake)[0]
    loss_d.backward()
    assert all(p.grad is None for p in gz.parameters())
    assert all(p.grad is None for p in gy.parameters())
    assert all(p.grad is not None for p in d.parameters())


def test_gradient_penalty_on_centered_fake_window():
    """Decision: mean operation BEFORE interpolation. The interpolate must be a
    leaf with requires_grad so autograd.grad(create_graph=True) works."""
    gz, gy, d = _nets()
    fake = _fake_window(gz, gy).detach()
    gp, _ = _gp(d, torch.randn(16, W, C), fake)
    gp.backward()
    assert torch.isfinite(gp) and d.input_conv.weight.grad is not None