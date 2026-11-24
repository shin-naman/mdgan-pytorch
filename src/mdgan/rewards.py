"""Generator rewards: noise use, block-sum correlations, block-sum variance. NOT in the papers; off by default.

Three terms can be added to the generator loss (Config.noise_use_min, Config.corr_reward_blocks,
Config.variance_reward_weight). All are computed on one side batch of Config.reward_chains chains, separate
from the window shown to the critic:

    z_0 ~ U[0,1), z_1 .. z_c by G_z (c = burn_in)

Noise use (noise_use_min = s > 0). From z_c (detached) two next latents a, b are made with independent
noise. Per latent dimension d, share_d = 1 - Corr(a_d, b_d) over the chains: the part of z_{c+1,d} set by
the fresh noise and not by z_c (0: the noise is ignored; 1: the state is forgotten in one step). The term is
    noise_use_weight * mean_d relu(s - share_d)^2
which is zero once every dimension uses its noise at least that much. Gradients reach G_z through that one
transition only.

Block-sum correlations (corr_reward_blocks = L >= 2). The chain continues to z_{c+L-1}; the L blocks
G_Y(z_c) .. G_Y(z_{c+L-1}) are decoded, centred like the fake window (mean operation), and summed over
their M steps. rho_k is the correlation of block sums k blocks apart, pooled over chains, blocks and axes.
The term is
    corr_reward_weight * sum_{k=1}^{L-1} (rho_k - target_k)^2
with the targets measured once from the training region (WindowDataset.block_sum_correlations). It is a sum
of squares on purpose: a form that can go negative is minimised by distorting single chains. Gradients
reach G_Y and, through every transition from z_0 on, G_z.

Sub-block correlations (subcorr_block_steps = s > 0, subcorr_lags = K; needs the side chain). The same decoded,
centred L blocks are cut into L*M/s consecutive s-step sub-blocks and summed; rho_k (k = 1..K) is pooled over
chains, positions and axes as for the blocks. The term is
    subcorr_weight * sum_k (rho_k - subcorr_target_k)^2
with targets from the training region cut the same way (WindowDataset.subblock_sum_correlations). A block
lag-1 correlation is a fixed combination of the sub-block correlations at lags 1 .. 2M/s - 1, so at s = M/4 lags
1-7 constrain the block lag 1 and the variance of the block sums. No further forward pass or draw.

Block-sum variance (variance_reward_weight = w > 0; needs the correlation reward's side chain, at any weight). On the SAME block sums as
the correlations (no further forward pass, no further draw), V_gen = their variance pooled over chains, blocks
and axes. The term is
    w * (log(V_gen / V_target))^2
with V_target measured once from the training region (WindowDataset.block_sum_variance, on the trajectories
and block grids the correlation targets use). With the sub-block reward on, V is the variance of the s-step
sub-block sums instead, generated and target alike (WindowDataset.subblock_sum_variance). Symmetric in the log-ratio: a factor 2 too large costs as much
as a factor 2 too small. Gradients reach G_Y and G_z as for the correlations. At w = 0 nothing is computed,
so a run without it is the run before the term existed.

Random draws, in order, from the one training generator: z_0; the c burn-in r's; (noise use) two r's;
(correlations) L - 1 r's, then G_Y's (r', u) per block. The variance term draws nothing. With every reward off
nothing is drawn and the trainer never calls this module, so a default run is unchanged.

The side batch passes through G_z in training mode like every training batch: with use_running_stats its
transitions also enter the BatchNorm running statistics.
"""

from __future__ import annotations

import torch

from mdgan.ops import mean_operation as _center
from mdgan.rollout import _initial_latents


def noise_use_share(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """(n_z,) per-dimension 1 - Corr(a_d, b_d) over the batch, for two (B, n_z) next latents that share
    their previous state and differ in the noise."""
    if a.dim() != 2 or a.shape != b.shape:
        raise ValueError(f"a and b must both be (B, n_z) and equal; got {tuple(a.shape)} and {tuple(b.shape)}")
    a, b = a - a.mean(dim=0), b - b.mean(dim=0)
    return 1.0 - (a * b).mean(dim=0) / ((a * a).mean(dim=0) * (b * b).mean(dim=0)).sqrt()


def block_sum_correlations(sums: torch.Tensor, max_lag: int) -> torch.Tensor:
    """(max_lag,) correlations of block sums at lags 1 .. max_lag, pooled over chains, blocks and axes
    (means removed per lag). sums: (B, L, C) with L > max_lag. The torch twin of
    WindowDataset.block_sum_correlations for one block grid."""
    if sums.dim() != 3 or not 1 <= max_lag < sums.shape[1]:
        raise ValueError(f"sums must be (B, L, C) with L > max_lag = {max_lag}; got {tuple(sums.shape)}")
    rho = []
    for k in range(1, max_lag + 1):
        x, y = sums[:, :-k].reshape(-1), sums[:, k:].reshape(-1)
        x, y = x - x.mean(), y - y.mean()
        rho.append((x * y).mean() / ((x * x).mean() * (y * y).mean()).sqrt())
    return torch.stack(rho)


def block_sum_variance(sums: torch.Tensor) -> torch.Tensor:
    """Variance of block sums pooled over chains, blocks and axes (one mean removed). sums: (B, L, C).
    The torch twin of WindowDataset.block_sum_variance for one block grid."""
    if sums.dim() != 3:
        raise ValueError(f"sums must be (B, L, C); got {tuple(sums.shape)}")
    x = sums.reshape(-1)
    return (x - x.mean()).square().mean()


def reward_targets(config, dataset) -> dict:
    """Every target the enabled rewards need, measured once from the training region, with its sample count:
    corr and corr_pairs (block lags 1..L-1), subcorr and subcorr_pairs (sub-block lags 1..K), variance and variance_count (of the
    block sums, or of the sub-block sums when subcorr_block_steps > 0). Only the enabled terms' keys."""
    out, s = {}, config.subcorr_block_steps
    if config.corr_reward_blocks:
        out["corr"] = dataset.block_sum_correlations(config.corr_reward_blocks)
        out["corr_pairs"] = dataset.block_pair_counts(config.corr_reward_blocks)
    if s:
        out["subcorr"] = dataset.subblock_sum_correlations(s, config.subcorr_lags)
        out["subcorr_pairs"] = dataset.subblock_pair_counts(s, config.subcorr_lags)
    if config.variance_reward_weight > 0:
        if s:
            out["variance"], out["variance_count"] = dataset.subblock_sum_variance(s)
        else:
            out["variance"] = dataset.block_sum_variance(config.corr_reward_blocks)
            out["variance_count"] = dataset.block_sum_count(config.corr_reward_blocks)
    return out


def reward_metric_names(config) -> tuple[str, ...]:
    """The extra columns of metrics.jsonl for this config (none with every reward off). Unweighted values,
    then reward_loss, the weighted sum that was added to the generator loss. block_var_ratio = V_gen / V_target
    and variance_penalty = log(V_gen / V_target)^2 appear only when the variance reward is on."""
    names = ()
    if config.noise_use_min > 0:
        names += ("noise_use_share", "noise_use_penalty")
    if config.corr_reward_blocks:
        names += ("corr_error",)
    if config.subcorr_block_steps:
        names += ("subcorr_error",)
    if config.variance_reward_weight > 0:
        names += ("block_var_ratio", "variance_penalty")
    return names + ("reward_loss",) if names else ()


def generator_rewards(gz, gy, config, corr_targets: torch.Tensor | None = None,
                      generator: torch.Generator | None = None,
                      variance_target: float | None = None,
                      subcorr_targets: torch.Tensor | None = None) -> tuple[torch.Tensor, dict]:
    """(weighted sum of the enabled terms, {name: detached value} for reward_metric_names(config)).
    corr_targets: (corr_reward_blocks - 1,) on the models' device; needed when that reward is on.
    variance_target: V_target > 0 (reward_targets(...)["variance"]); needed when the variance reward is on.
    subcorr_targets: (subcorr_lags,) on the models' device; needed when the sub-block reward is on."""
    n, blocks = config.reward_chains, config.corr_reward_blocks
    if not config.rewards_on:
        raise ValueError("no reward is enabled in this config")
    if blocks and (corr_targets is None or corr_targets.shape != (blocks - 1,)):
        raise ValueError(f"corr_targets must hold the {blocks - 1} target correlations")
    if config.variance_reward_weight > 0 and not (variance_target is not None and variance_target > 0):
        raise ValueError("variance_target must be the positive target block-sum variance")
    if config.subcorr_block_steps and (subcorr_targets is None or subcorr_targets.shape != (config.subcorr_lags,)):
        raise ValueError(f"subcorr_targets must hold the {config.subcorr_lags} target sub-block correlations")
    param = next(gz.parameters())

    def step(z):
        return gz(z, gz.sample_noise(n, device=z.device, dtype=z.dtype, generator=generator))

    z = _initial_latents(gz, n, generator, param.device, param.dtype)
    with torch.set_grad_enabled(blocks > 0):          # the noise-use term starts from a detached z_c
        for _ in range(config.burn_in):
            z = step(z)
    total, stats = 0.0, {}
    if config.noise_use_min > 0:
        share = noise_use_share(step(z.detach()), step(z.detach()))
        penalty = torch.relu(config.noise_use_min - share).square().mean()
        total = total + config.noise_use_weight * penalty
        stats["noise_use_share"], stats["noise_use_penalty"] = share.mean().detach(), penalty.detach()
    if blocks:
        states = [z]
        for _ in range(blocks - 1):
            states.append(step(states[-1]))
        y = torch.cat([gy.generate(s, generator=generator) for s in states], dim=1)        # (n, L*M, C)
        y = _center(y) if config.mean_operation else y
        sums = y.reshape(n, blocks, config.block_len, -1).sum(dim=2)                       # (n, L, C)
        error = (block_sum_correlations(sums, blocks - 1) - corr_targets).square().sum()
        total = total + config.corr_reward_weight * error
        stats["corr_error"] = error.detach()
        if config.subcorr_block_steps:                        # the same centred blocks, cut finer: no pass, no draw
            sub = y.reshape(n, blocks * config.block_len // config.subcorr_block_steps, config.subcorr_block_steps,
                            -1).sum(dim=2)                                                  # (n, L*M/s, C)
            sub_error = (block_sum_correlations(sub, config.subcorr_lags) - subcorr_targets).square().sum()
            total = total + config.subcorr_weight * sub_error
            stats["subcorr_error"] = sub_error.detach()
            sums = sub                                        # the variance term, if on, is that of the sub-blocks
        if config.variance_reward_weight > 0:                 # the same sums: no forward pass, no draw
            ratio = block_sum_variance(sums) / variance_target
            penalty = torch.log(ratio).square()
            total = total + config.variance_reward_weight * penalty
            stats["block_var_ratio"], stats["variance_penalty"] = ratio.detach(), penalty.detach()
    stats["reward_loss"] = total.detach()
    return total, stats
