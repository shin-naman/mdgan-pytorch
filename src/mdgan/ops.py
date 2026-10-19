"""Differentiable, parameter-free operations used in training and estimation."""

import torch


def mean_operation(x: torch.Tensor, batch_dim: int = 0) -> torch.Tensor:
    """Subtract the ensemble (batch) mean at every step and axis.

    Kawada et al. (2023), eq. (9): r'_{abg} = r_{abg} - (1/N) sum_a r_{abg},
    where a indexes the trajectory, b the step, g the axis. Encodes the
    domain knowledge that diffusion from the origin has zero mean position
    at every time.

    Applied by the trainer to the GENERATED 2M-step displacement window
    before the critic and before gradient-penalty interpolation, inside the
    autograd graph. Never applied to real data or to latents. Because cumsum
    over time is linear, centering displacements per step is identical to
    centering positions per time.
    """
    return x - x.mean(dim=batch_dim, keepdim=True)