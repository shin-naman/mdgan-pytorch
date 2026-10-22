"""Generating trajectories with G_z and G_Y.

make_fake_window  training: one 2M-step window per chain, shown to the critic.
rollout           estimation: long trajectories, block after block.

Both start each chain at z_0 ~ U[0,1)^{n_z} and discard the first `burn_in` (= c)
latent states: "trajectories are not generated from latent variables that have
time-evolved for less than c times" (2022 ESI S1). Outputs are displacements in
the units the models were trained in (sigma units); evaluate multiplies by sigma.

Mean operation (2023 eq. 9): subtract the ensemble mean over chains at every step
and axis. Applied to the fake window in training and to every generated block in
estimation, as the paper does for its 8192 estimation trajectories.

Batch coupling: G_z's output BatchNorm uses batch statistics (decisions.md), so
chains interact through the normalization. rollout therefore processes all chains
as ONE batch (the paper's 8192); splitting them into chunks would change results.

All randomness goes through an optional explicit torch.Generator, in a fixed order
(documented per function), so a run is reproducible from a seed.
"""

from __future__ import annotations

from contextlib import contextmanager

import torch

from mdgan._util import positive_int
from mdgan.ops import mean_operation as _center


def _non_negative_int(name, value):
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _initial_latents(gz, batch, generator, device, dtype):
    return torch.rand(batch, gz.latent_dim, device=device, dtype=dtype, generator=generator)


def make_fake_window(gz, gy, batch: int, burn_in: int, pair_window: int, *,
                     mean_operation: bool = True,
                     generator: torch.Generator | None = None) -> torch.Tensor:
    """One generated 2M-step window per chain: (batch, 2M, C), differentiable.

    1. j ~ uniform integer in [burn_in, burn_in + pair_window - 1]   (one j per call)
    2. z_0 ~ U[0, 1)^{n_z}
    3. z_1 .. z_{j+1} by G_z (full BPTT; gradients reach z_0's descendants and G_z)
    4. y_j = G_Y(z_j), y_{j+1} = G_Y(z_{j+1}); window = [y_j, y_{j+1}] along time
    5. mean operation over the batch if enabled
    Random draws happen in exactly this order from `generator`.
    Runs in whatever train/eval mode the modules are in (the trainer sets train).
    """
    positive_int("batch", batch)
    _non_negative_int("burn_in", burn_in)
    positive_int("pair_window", pair_window)
    param = next(gz.parameters())
    j = int(torch.randint(burn_in, burn_in + pair_window, (1,), generator=generator))
    z0 = _initial_latents(gz, batch, generator, param.device, param.dtype)
    states = gz.unroll(z0, j + 1, generator=generator)                  # (j + 2, B, n_z)
    window = torch.cat([gy.generate(states[j], generator=generator),
                        gy.generate(states[j + 1], generator=generator)], dim=1)
    return _center(window) if mean_operation else window


@contextmanager
def _eval_mode(*modules):
    flags = [m.training for m in modules]
    try:
        for m in modules:
            m.eval()
        yield
    finally:
        for m, f in zip(modules, flags, strict=True):
            m.train(f)


@torch.no_grad()
def rollout(gz, gy, n_chains: int, n_blocks: int, burn_in: int, *,
            mean_operation: bool = True,
            generator: torch.Generator | None = None) -> torch.Tensor:
    """Long generated trajectories: (n_chains, n_blocks * M, C) displacements, on CPU.

    z_0 ~ U[0,1); z_1..z_c by G_z (burn-in, not decoded); then for k = 0..n_blocks-1:
    block k = G_Y(z_{c+k}) (centered if mean_operation), then z_{c+k+1} = G_z(z_{c+k}).
    Random draws in order: z_0; the c burn-in r's; per block, G_Y's (r', u) then
    G_z's r. Runs in eval mode and restores each module's mode afterwards.
    Output is moved to CPU block by block, so device memory holds one block at a time.
    """
    positive_int("n_chains", n_chains)
    positive_int("n_blocks", n_blocks)
    _non_negative_int("burn_in", burn_in)
    param = next(gz.parameters())
    out = torch.empty(n_chains, n_blocks * gy.block_len, gy.n_channels, dtype=param.dtype)
    with _eval_mode(gz, gy):
        z = _initial_latents(gz, n_chains, generator, param.device, param.dtype)
        for _ in range(burn_in):
            z = gz(z, gz.sample_noise(n_chains, device=z.device, dtype=z.dtype, generator=generator))
        for k in range(n_blocks):
            y = gy.generate(z, generator=generator)
            if mean_operation:
                y = _center(y)
            out[:, k * gy.block_len:(k + 1) * gy.block_len] = y.cpu()
            if k + 1 < n_blocks:
                z = gz(z, gz.sample_noise(n_chains, device=z.device, dtype=z.dtype, generator=generator))
    return out