"""Training-stability experiments on a synthetic Brownian system (known answer: MSD = 6 D t).

Trains MD-GAN with the Brownian-test configuration plus overrides, evaluates every
checkpoint against the exact reference, and prints, per checkpoint:

    err, D_rel                     the paper's metrics against the reference
    W, fake_step_std               training diagnostics (fake_step_std target 1)
    block var                      variance of one M-step block, per axis (target M)
    block corr sum (lags 1-10)     sum of block-to-block displacement correlations.
                                   Target 0. Long-time MSD scales by ~(1 + 2 * this),
                                   so this is the quantity that drives D off.

Usage (repo root):
    python tools/diag_brownian.py --out runs/brownian_lr1e-4 --lr 1e-4 --iterations 10000
    python tools/diag_brownian.py --out runs/brownian_lr1e-4 --resume ...      continue after an interruption
    python tools/diag_brownian.py --out runs/diag_brownian_2048 --summary-only  table for an existing run
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from mdgan.checkpoint import load_checkpoint
from mdgan.config import Config
from mdgan.data.dataset import WindowDataset
from mdgan.data.preprocess import SystemData
from mdgan.evaluate import reference_msd
from mdgan.rollout import rollout
from mdgan.train import Trainer, snapshot_name

DT_PS = 10.0
D_NM2_PS = 1.3e-10 / 1e-6


def brownian(n_traj: int, n_steps: int = 400, seed: int = 0) -> SystemData:
    """Identical to tests/test_train.py::brownian (same seed, same scale)."""
    rng = np.random.default_rng(seed)
    d = rng.normal(scale=np.sqrt(2 * D_NM2_PS * DT_PS), size=(n_traj, n_steps, 3)).astype(np.float32)
    return SystemData(name=f"brownian{seed}", displacements=d, dt_ps=DT_PS, box_nm=np.full((n_traj, 3), 5.0), meta={})


def config(args) -> Config:
    return Config(latent_dim=8, noise_dim=8, block_len=8, n_channels=3, hidden_dim=args.hidden, batch_size=64,
                  n_iterations=args.iterations, checkpoint_every=args.checkpoint_every, lr=args.lr,
                  burn_in=args.burn_in, pair_window=args.pair_window, seed=args.seed,
                  estimation_chains=2048, estimation_time_ps=4000.0, err_window_ps=(20.0, 3900.0),
                  diffusion_fit_ps=(1000.0, 2000.0))


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def block_stats(snapshot: Path) -> tuple[float, float]:
    """(mean block variance per axis, sum of block-to-block correlations over lags 1-10)."""
    ck = load_checkpoint(snapshot)
    m = ck.config.block_len
    n_blocks = int(round(ck.config.estimation_time_ps / (DT_PS * ck.config.step_skip * m)))
    d = rollout(ck.gz, ck.gy, 2048, n_blocks, ck.config.burn_in,
                generator=torch.Generator().manual_seed(0)).numpy()
    blocks = d.reshape(d.shape[0], n_blocks, m, -1).sum(axis=2)
    corr = [np.corrcoef(blocks[:, :-lag].ravel(), blocks[:, lag:].ravel())[0, 1] for lag in range(1, 11)]
    return float(blocks.var(axis=(0, 2)).mean()), float(np.sum(corr))


def summary(out: Path) -> None:
    evals = _jsonl(out / "evaluations.jsonl")
    metrics = {r["iteration"]: r for r in _jsonl(out / "metrics.jsonl")}
    if not evals:
        print("no evaluations yet")
        return
    m = load_checkpoint(out / snapshot_name(evals[0]["iteration"])).config.block_len
    print(f"{'iter':>6} {'err %':>7} {'D_rel':>7} {'W':>7} {'step std':>9} {'block var':>10} {'corr sum':>9}")
    print(f"{'target':>6} {'0':>7} {'0':>7} {'small':>7} {'1':>9} {m:>10} {'0':>9}")
    for e in evals:
        it = e["iteration"]
        r = metrics.get(it, {})
        var, csum = block_stats(out / snapshot_name(it))
        print(f"{it:6d} {e['err_percent']:7.1f} {e['D_rel_error']:+7.3f} {r.get('wasserstein', float('nan')):7.3f} "
              f"{r.get('fake_step_std', float('nan')):9.3f} {var:10.2f} {csum:+9.3f}")
    errs = np.array([e["err_percent"] for e in evals])
    later = errs[len(errs) // 2:]
    best = evals[int(errs.argmin())]
    print(f"\nbest: iteration {best['iteration']}, err {best['err_percent']:.1f} %, D_rel {best['D_rel_error']:+.3f}")
    print(f"second half of training: err range {later.min():.1f}-{later.max():.1f} %, "
          f"std {later.std():.1f} %  (small spread = stable)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--iterations", type=int, default=10_000)
    ap.add_argument("--checkpoint-every", type=int, default=1000)
    ap.add_argument("--pair-window", type=int, default=2)
    ap.add_argument("--burn-in", type=int, default=4)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--trajectories", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--summary-only", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    if not args.summary_only:
        c = config(args)
        system = brownian(args.trajectories)
        dataset = WindowDataset(system, c.block_len, c.step_skip, c.phase_mode, c.scale_inputs)
        Trainer(c, dataset, out, device=args.device, reference=reference_msd(system),
                log_every=max(1, args.checkpoint_every // 2), resume=args.resume).run()
    summary(out)


if __name__ == "__main__":
    main()