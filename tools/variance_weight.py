"""Choose variance_reward_weight: the weight at which the variance term's gradient on the generators has the size
of the correlation term's, on a freshly initialised model (iteration 0).

For each model seed s (G_z, G_Y built exactly as Trainer builds them: torch.manual_seed(s), build_models) and each
side-batch draw d, generator_rewards is called twice on the same side batch (same generator seed):
    A: corr_reward_weight as in the config (10), variance weight 0      -> the correlation term as trained
    B: corr_reward_weight 0, variance weight 1                            -> log(V_gen / V_target)^2 alone
The noise-use reward stays switched on with weight 0, so the side batch is drawn in the training order and the
noise-use term contributes nothing. The gradient norm is taken over all G_z and G_Y parameters together, and the
weight that equalises the two is ||grad A|| / ||grad B||. The median over seeds and draws is printed, rounded to
two significant figures, as the value to put in the config.

    PYTHONPATH=src python tools/variance_weight.py configs/batch8_c2_both_full.json data/cache/PE.npz
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
from pathlib import Path

import numpy as np
import torch


def grad_norm(loss, params) -> float:
    grads = torch.autograd.grad(loss, params, allow_unused=True)
    return math.sqrt(sum(float(g.square().sum()) for g in grads if g is not None))


def main(argv=None) -> int:
    from mdgan.config import Config
    from mdgan.data.dataset import WindowDataset
    from mdgan.models import build_models
    from mdgan.rewards import generator_rewards
    from mdgan.train import load_system

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("config", help="config JSON (the batch-8 c2_both_full config)")
    ap.add_argument("data", help="preprocessed cache .npz")
    ap.add_argument("--seeds", default="0,1,2", help="model seeds (comma list)")
    ap.add_argument("--draws", type=int, default=3, help="side batches per model seed")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args(argv)

    cfg = Config.from_dict({**Config().to_dict(), **json.loads(Path(args.config).read_text())})
    if not cfg.corr_reward_blocks:
        raise SystemExit("the config has no correlation reward; the variance term needs its side chains")
    ds = WindowDataset(load_system(args.data), cfg.block_len, cfg.step_skip, cfg.phase_mode, cfg.scale_inputs,
                       cfg.window_offset_max_steps, cfg.holdout_trajectories, cfg.critic_blocks,
                       region_start_steps=cfg.region_start_steps)
    targets = torch.as_tensor(ds.block_sum_correlations(cfg.corr_reward_blocks), dtype=torch.float32, device=args.device)
    v_target = ds.block_sum_variance(cfg.corr_reward_blocks)
    corr_cfg = dataclasses.replace(cfg, noise_use_weight=0.0, variance_reward_weight=0.0)
    var_cfg = dataclasses.replace(cfg, noise_use_weight=0.0, corr_reward_weight=0.0, variance_reward_weight=1.0)
    print(f"config {args.config}: corr_reward_blocks {cfg.corr_reward_blocks}, corr_reward_weight {cfg.corr_reward_weight}, "
          f"reward_chains {cfg.reward_chains}, region {ds.region_steps} steps, {ds.n_trajectories} trajectories")
    print(f"targets: rho {np.round(targets.cpu().numpy(), 4).tolist()}, V_target {v_target:.4f} (sigma units)")
    print(f"{'seed':>4} {'draw':>4} {'V_gen/V_t':>9} {'corr_err':>9} {'|g corr x w_c|':>14} {'|g log^2|':>10} {'weight':>8}")

    weights = []
    for seed in (int(s) for s in args.seeds.split(",")):
        torch.manual_seed(seed)                                   # as Trainer: the models' initial weights
        gz, gy, _ = build_models(cfg)
        gz, gy = gz.to(args.device).train(), gy.to(args.device).train()
        params = [*gz.parameters(), *gy.parameters()]
        for draw in range(args.draws):
            gseed = 1000 * seed + draw
            a, sa = generator_rewards(gz, gy, corr_cfg, targets, generator=torch.Generator().manual_seed(gseed))
            b, sb = generator_rewards(gz, gy, var_cfg, targets, generator=torch.Generator().manual_seed(gseed),
                                      variance_target=v_target)
            assert torch.equal(sa["corr_error"], sb["corr_error"]), "the two calls must see the same side batch"
            na, nb = grad_norm(a, params), grad_norm(b, params)
            weights.append(na / nb)
            print(f"{seed:>4} {draw:>4} {float(sb['block_var_ratio']):>9.3f} {float(sa['corr_error']):>9.4f} "
                  f"{na:>14.4g} {nb:>10.4g} {na / nb:>8.3g}")
    med = float(np.median(weights))
    rounded = float(f"{med:.2g}")
    print(f"weight: median {med:.4g} (range {min(weights):.3g} - {max(weights):.3g} over {len(weights)} draws); "
          f"use variance_reward_weight = {rounded:g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
