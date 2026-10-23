"""WGAN-GP losses for MD-GAN (Gulrajani et al. 2017; Kawada et al. 2022 eq. 5).

    L_critic    = E[D(fake)] - E[D(real)] + lambda * E[(||grad_x D(x_hat)|| - 1)^2]
    L_generator = -E[D(fake)]

x_hat = eps * real + (1 - eps) * fake with eps ~ U[0, 1) drawn once PER SAMPLE and
broadcast over the window's steps and axes, so x_hat lies on the straight line
between one real and one fake window.

Conventions (docs/decisions.md):
  - The critic step never updates the generators. critic_loss detaches `fake`
    itself, so this holds no matter what the caller passes.
  - The penalty is a critic-only term: the interpolate is detached and made a
    fresh leaf, so no gradient of the penalty reaches the generators either.
  - The mean operation is NOT applied here; `fake` arrives already centered
    (rollout.make_fake_window), so the penalty interpolates the centered window.
  - Gradients of the penalty reach every critic WEIGHT and no bias: for a
    piecewise-linear critic grad_x D is a product of weights and slope masks.

All randomness (eps) goes through an optional explicit torch.Generator, drawn on the
generator's device and moved to the data's, so a CPU generator works with a GPU critic.
"""

from __future__ import annotations

import torch

from mdgan.models.latent_transition import _draw_device

DEFAULT_GP_LAMBDA = 10.0


def _check_pair(real: torch.Tensor, fake: torch.Tensor) -> None:
    if real.dim() != 3 or real.shape != fake.shape:
        raise ValueError(f"real and fake must both be (B, 2M, C) and equal; got {tuple(real.shape)} "
                         f"and {tuple(fake.shape)}")


def gradient_penalty(critic, real: torch.Tensor, fake: torch.Tensor,
                     gp_lambda: float = DEFAULT_GP_LAMBDA,
                     generator: torch.Generator | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (penalty, per-sample gradient norms (B,)).

    The penalty is differentiable w.r.t. the critic's parameters (create_graph=True).
    The norms are detached, for logging.
    """
    _check_pair(real, fake)
    b = real.shape[0]
    # Drawn on the generator's device, then moved: a CPU generator must work with a GPU critic
    # (the same rule as every noise sampler; see models.latent_transition._draw_device).
    eps = torch.rand(b, 1, 1, device=_draw_device(generator, real.device), dtype=real.dtype,
                     generator=generator).to(real.device)
    x_hat = (eps * real.detach() + (1 - eps) * fake.detach()).requires_grad_(True)
    score = critic(x_hat)
    grad, = torch.autograd.grad(score.sum(), x_hat, create_graph=True)
    norm = grad.flatten(1).norm(dim=1)
    return gp_lambda * ((norm - 1) ** 2).mean(), norm.detach()


def critic_loss(critic, real: torch.Tensor, fake: torch.Tensor,
                gp_lambda: float = DEFAULT_GP_LAMBDA,
                generator: torch.Generator | None = None) -> tuple[torch.Tensor, dict]:
    """Return (loss to minimize, diagnostics).

    diagnostics (all detached scalars): "wasserstein" = E[D(real)] - E[D(fake)],
    the critic's estimate of the Wasserstein-1 distance (rises as the critic
    separates the two; should fall as the generator improves); "gradient_penalty";
    "grad_norm" = mean of the per-sample input-gradient norms (target 1).
    """
    _check_pair(real, fake)
    fake = fake.detach()
    d_real = critic(real).mean()
    d_fake = critic(fake).mean()
    gp, norms = gradient_penalty(critic, real, fake, gp_lambda, generator)
    loss = d_fake - d_real + gp
    return loss, {
        "wasserstein": (d_real - d_fake).detach(),
        "gradient_penalty": gp.detach(),
        "grad_norm": norms.mean(),
    }


def generator_loss(critic, fake: torch.Tensor) -> torch.Tensor:
    """-E[D(fake)]. Gradients flow through the critic into G_Y and G_z; the
    trainer steps only the generators' optimizer on this loss."""
    if fake.dim() != 3:
        raise ValueError(f"fake must be (B, 2M, C); got {tuple(fake.shape)}")
    return -critic(fake).mean()