"""Integration: the three networks wired together as the trainer wires them
(fake window -> losses -> gradients). These move to test_train.py with the trainer.

Run: python -m pytest tests/integration -v
"""

import torch
from conftest import SMALL

from mdgan.losses import critic_loss, generator_loss, gradient_penalty
from mdgan.rollout import make_fake_window

LAT, NR, M, C, H = (SMALL[k] for k in ("latent_dim", "noise_dim", "block_len", "n_channels", "hidden_dim"))
W = 2 * M


# --- training steps ---------------------------------------------------------------------------

def test_generator_step_backprops_through_critic_into_both_generators(three_nets):
    gz, gy, dg = three_nets
    for n in three_nets:
        n.train()
    fake = make_fake_window(gz, gy, 16, burn_in=3, pair_window=2)
    assert fake.shape == (16, W, C)
    torch.testing.assert_close(fake.mean(0), torch.zeros(W, C), atol=1e-5, rtol=0)
    generator_loss(dg, fake).backward()
    for net, name in ((gz, "G_z"), (gy, "G_Y")):
        for pn, p in net.named_parameters():
            if pn.endswith(".weight"):
                assert p.grad is not None and p.grad.abs().sum() > 0, f"{name}.{pn}"
    assert dg.hidden1.weight.grad is not None       # critic gets gradient too; trainer must not step it


def test_critic_step_leaves_generators_untouched(three_nets):
    """critic_loss detaches the fake window itself; no .detach() needed here."""
    gz, gy, dg = three_nets
    fake = make_fake_window(gz, gy, 16, burn_in=3, pair_window=2)
    real = torch.randn(16, W, C)
    critic_loss(dg, real, fake)[0].backward()
    assert all(p.grad is None for p in gz.parameters())
    assert all(p.grad is None for p in gy.parameters())
    assert all(p.grad is not None for p in dg.parameters())


def test_gradient_penalty_on_centered_fake_window(three_nets):
    """Decision: mean operation BEFORE interpolation."""
    gz, gy, dg = three_nets
    fake = make_fake_window(gz, gy, 16, burn_in=3, pair_window=2).detach()
    gp, _ = gradient_penalty(dg, torch.randn(16, W, C), fake)
    gp.backward()
    assert torch.isfinite(gp) and dg.input_conv.weight.grad is not None