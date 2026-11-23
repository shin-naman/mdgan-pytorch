"""Choose reward weights: the weight at which each term's gradient on the generators has the size of the block
correlation term's at weight 10, on a freshly initialised model (iteration 0).

For each model seed s (G_z, G_Y built exactly as Trainer builds them: torch.manual_seed(s), build_models) and each
side-batch draw d, generator_rewards is called once per term on the same side batch (same generator seed), with
every other term's weight at 0:
    reference  the block correlation term, lags 1..L-1 at corr_reward_weight 10 (whatever the config's own weight)
    variance   log(V_gen / V_target)^2 at weight 1: block sums, or s-step sub-block sums when subcorr_block_steps > 0
    subcorr    the sub-block correlation term at weight 1 (only when subcorr_block_steps > 0)
Targets come from the config's own training region (mdgan.rewards.reward_targets, as the trainer uses them). The
noise-use reward stays switched on at weight 0, so the side batch is drawn in the training order. The gradient norm
is taken over all G_z and G_Y parameters together; a term's weight is ||grad reference|| / ||grad term||. The median
over seeds and draws is printed, rounded to two significant figures, as the value to put in the config.

    PYTHONPATH=src python tools/variance_weight.py configs/batch10_c2_both_paper_var.json data/cache/PE.npz
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
from pathlib import Path

import numpy as np
import torch

REFERENCE_WEIGHT = 10.0


def grad_norm(loss, params) -> float:
    grads = torch.autograd.grad(loss, params, allow_unused=True)
    return math.sqrt(sum(float(g.square().sum()) for g in grads if g is not None))


def main(argv=None) -> int:
    from mdgan.config import Config
    from mdgan.data.dataset import WindowDataset
    from mdgan.models import build_models
    from mdgan.rewards import generator_rewards, reward_targets
    from mdgan.train import load_system

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("config", help="config JSON")
    ap.add_argument("data", help="preprocessed cache .npz")
    ap.add_argument("--seeds", default="0,1,2", help="model seeds (comma list)")
    ap.add_argument("--draws", type=int, default=3, help="side batches per model seed")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args(argv)

    cfg = Config.from_dict({**Config().to_dict(), **json.loads(Path(args.config).read_text())})
    if not cfg.corr_reward_blocks:
        raise SystemExit("the config has no side chain (corr_reward_blocks = 0); there is nothing to measure on")
    ds = WindowDataset(load_system(args.data), cfg.block_len, cfg.step_skip, cfg.phase_mode, cfg.scale_inputs,
                       cfg.window_offset_max_steps, cfg.holdout_trajectories, cfg.critic_blocks,
                       region_start_steps=cfg.region_start_steps)
    off = dict(noise_use_weight=0.0, corr_reward_weight=0.0, variance_reward_weight=0.0, subcorr_weight=0.0)
    terms = {"reference": dataclasses.replace(cfg, **{**off, "corr_reward_weight": REFERENCE_WEIGHT}),
             "variance": dataclasses.replace(cfg, **{**off, "variance_reward_weight": 1.0})}
    if cfg.subcorr_block_steps:
        terms["subcorr"] = dataclasses.replace(cfg, **{**off, "subcorr_weight": 1.0})
    tg = reward_targets(terms["variance"], ds) | reward_targets(cfg, ds)
    t32 = lambda key: torch.as_tensor(tg[key], dtype=torch.float32, device=args.device) if key in tg else None  # noqa: E731
    corr_t, sub_t = t32("corr"), t32("subcorr")
    unit = f"{cfg.subcorr_block_steps}-step sub-block" if cfg.subcorr_block_steps else f"{cfg.block_len}-step block"
    print(f"config {args.config}: region {ds.region_steps} steps from {ds.region_start_steps}, {ds.n_trajectories} "
          f"trajectories; side chain {cfg.corr_reward_blocks} blocks x {cfg.reward_chains} chains")
    print(f"targets: block rho {np.round(tg['corr'], 4).tolist()}"
          + (f"; sub-block rho {np.round(tg['subcorr'], 4).tolist()} (pairs {tg['subcorr_pairs']})" if sub_t is not None else "")
          + f"; V_target {tg['variance']:.4f} sigma^2 ({unit} sums, n = {tg['variance_count']})")
    names = [k for k in terms if k != "reference"]
    print(f"{'seed':>4} {'draw':>4} {'V_gen/V_t':>9} {'|g ref|':>9} " + " ".join(f"{'|g ' + k + '|':>12} {'w_' + k:>10}" for k in names))

    weights = {k: [] for k in names}
    for seed in (int(s) for s in args.seeds.split(",")):
        torch.manual_seed(seed)                                   # as Trainer: the models' initial weights
        gz, gy, _ = build_models(cfg)
        gz, gy = gz.to(args.device).train(), gy.to(args.device).train()
        params = [*gz.parameters(), *gy.parameters()]
        for draw in range(args.draws):
            gseed = 1000 * seed + draw
            out = {k: generator_rewards(gz, gy, c, corr_t, generator=torch.Generator().manual_seed(gseed),
                                        variance_target=tg["variance"] if c.variance_reward_weight > 0 else None,
                                        subcorr_targets=sub_t) for k, c in terms.items()}
            assert all(torch.equal(out[k][1]["corr_error"], out["reference"][1]["corr_error"]) for k in out), \
                "every call must see the same side batch"
            ref = grad_norm(out["reference"][0], params)
            row = []
            for k in names:
                n = grad_norm(out[k][0], params)
                weights[k].append(ref / n)
                row.append(f"{n:>12.4g} {ref / n:>10.3g}")
            print(f"{seed:>4} {draw:>4} {float(out['variance'][1]['block_var_ratio']):>9.3f} {ref:>9.4g} " + " ".join(row))
    for k in names:
        med = float(np.median(weights[k]))
        print(f"{k}: median {med:.4g} (range {min(weights[k]):.3g} - {max(weights[k]):.3g} over {len(weights[k])} draws); "
              f"use {'variance_reward_weight' if k == 'variance' else 'subcorr_weight'} = {float(f'{med:.2g}'):g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
