"""Diagnose finished training runs against the MD they were trained on.

Reads run directories (evaluations.jsonl, msd_XXXXXX.npy, snap_XXXXXX.pt) and the
preprocessed cache of the same system, and prints:

  1. per run: the best (lowest err), median and final checkpoint, with err and D_rel;
  2. the generated / reference MSD ratio at times from 20 ps to 78 ns for those
     checkpoints, and the median ratio over all checkpoints of all runs. A ratio that
     GROWS with time means correlated drift between blocks; a constant factor means
     a step-size (variance) error;
  3. with --blocks: for the same checkpoints, rolled out again, and for the real MD:
       block var       variance of one M-step block displacement per axis (sigma units)
       corr@lag        correlation between block displacements `lag` blocks apart
       corr sum        sum of those correlations over lags 1-30 (long-time MSD scales
                       roughly by 1 + 2 x this, relative to independent blocks)
       z persist@lag   correlation of each chain's latent with itself `lag` steps later
                       (near 1 = latents barely move = each chain decodes similar blocks)
     The MD row is the target: water in PE is not Brownian, so its correlations are
     not zero.

Usage (repository root):
    python tools/diag_runs.py runs/chtc/run_c6954730_seed* --data data/cache/PE.npz
    python tools/diag_runs.py runs/chtc/run_c6954730_seed* --data data/cache/PE.npz --blocks --device mps

Sections 1-2 read saved files only (seconds, no torch). Section 3 runs one rollout per
selected checkpoint (--chains, default 1024; about a minute each on CPU, faster on MPS).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from mdgan.data.msd import msd_time_averaged
from mdgan.data.preprocess import SystemData

TIMES_NS = (0.02, 0.1, 0.5, 1.28, 5.0, 10.0, 20.0, 40.0, 78.0)
BLOCK_LAGS = (1, 2, 5, 10, 30)
LATENT_LAGS = (1, 10, 50, 100)


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def pick_checkpoints(evals: list[dict]) -> dict[str, dict]:
    """best (lowest err), median (err closest to the run's median err), final."""
    errs = np.array([e["err_percent"] for e in evals])
    med = int(np.argmin(np.abs(errs - np.median(errs))))
    return {"best": evals[int(errs.argmin())], "median": evals[med], "final": evals[-1]}


def ratio_at(msd_gen: np.ndarray, dt_gen: float, msd_ref: np.ndarray, dt_ref: float, t_ns: float) -> float:
    k_gen = int(round(t_ns * 1000 / dt_gen))
    k_ref = int(round(t_ns * 1000 / dt_ref))
    if k_gen >= len(msd_gen) or k_ref >= len(msd_ref):
        return float("nan")
    return float(msd_gen[k_gen] / msd_ref[k_ref])


def section_summary(runs: dict[str, list[dict]]) -> None:
    print("1. Checkpoints per run")
    print(f"{'run':<28} {'best err':>9} {'(iter, D_rel)':>17} {'median err':>11} {'median D_rel':>13} "
          f"{'final err':>10} {'final D_rel':>12}")
    for name, evals in runs.items():
        err = np.array([e["err_percent"] for e in evals])
        d = np.array([e["D_rel_error"] for e in evals])
        b = int(err.argmin())
        print(f"{name:<28} {err[b]:8.1f}% ({evals[b]['iteration']:>6}, {d[b]:+.2f}) {np.median(err):10.1f}% "
              f"{np.median(d):+13.2f} {err[-1]:9.1f}% {d[-1]:+12.2f}")
    alld = np.concatenate([[e["D_rel_error"] for e in ev] for ev in runs.values()])
    print(f"all {alld.size} checkpoints: median D_rel {np.median(alld):+.2f}; "
          f"{np.mean(alld > 0) * 100:.0f}% overestimate D\n")


def section_msd(run_dirs: dict[str, Path], runs: dict[str, list[dict]], msd_ref, dt_ref) -> None:
    print("2. Generated / reference MSD at time t (1 = exact)")
    head = " ".join(f"{t:>6g}" for t in TIMES_NS)
    print(f"{'run / checkpoint':<36} t (ns): {head}")
    all_ratios = []
    for name, evals in runs.items():
        for e in evals:
            g = np.load(run_dirs[name] / f"msd_{e['iteration']:06d}.npy")
            all_ratios.append([ratio_at(g, e["dt_gen_ps"], msd_ref, dt_ref, t) for t in TIMES_NS])
        for label, e in pick_checkpoints(evals).items():
            g = np.load(run_dirs[name] / f"msd_{e['iteration']:06d}.npy")
            r = [ratio_at(g, e["dt_gen_ps"], msd_ref, dt_ref, t) for t in TIMES_NS]
            print(f"{name + ' ' + label + ' @' + str(e['iteration']):<44} " + " ".join(f"{x:6.2f}" for x in r))
    med = np.nanmedian(np.array(all_ratios), axis=0)
    print(f"{'median over all checkpoints':<44} " + " ".join(f"{x:6.2f}" for x in med) + "\n")


def _block_stats(d: np.ndarray, block_len: int) -> dict:
    """d: (chains, steps, axes) in sigma units."""
    n, t, c = d.shape
    nb = t // block_len
    blocks = d[:, :nb * block_len].reshape(n, nb, block_len, c).sum(axis=2)
    corr = {}
    for lag in range(1, 31):
        if lag < nb:
            corr[lag] = float(np.corrcoef(blocks[:, :-lag].ravel(), blocks[:, lag:].ravel())[0, 1])
    return {"var": float(blocks.var(axis=(0, 2)).mean()), "corr": corr,
            "corr_sum": float(sum(corr.values()))}


def _latent_persistence(states: np.ndarray) -> dict:
    """states: (steps, chains, n_z) after burn-in."""
    out = {}
    for lag in LATENT_LAGS:
        if lag < states.shape[0]:
            a, b = states[:-lag], states[lag:]
            out[lag] = float(np.corrcoef(a.ravel(), b.ravel())[0, 1])
    return out


def section_blocks(run_dirs, runs, system, chains: int, device: str) -> None:
    import torch

    from mdgan.checkpoint import load_checkpoint
    from mdgan.ops import mean_operation

    print(f"3. Block and latent statistics ({chains} chains per rollout; MD = the real trajectories)")
    lag_head = " ".join(f"{'@' + str(lag):>6}" for lag in BLOCK_LAGS)
    z_head = " ".join(f"{'z@' + str(lag):>6}" for lag in LATENT_LAGS)
    print(f"{'source':<44} {'blk var':>8} corr {lag_head} {'sum1-30':>8} | persist {z_head}")

    first = next(iter(runs))
    ck0 = load_checkpoint(run_dirs[first] / f"snap_{runs[first][0]['iteration']:06d}.pt")
    sigma, m = ck0.sigma_nm, ck0.config.block_len
    md = _block_stats(system.displacements.astype(np.float64) / sigma, m)
    print(f"{'MD (target)':<44} {md['var']:8.2f}      " + " ".join(f"{md['corr'].get(lag, float('nan')):+6.3f}"
          for lag in BLOCK_LAGS) + f" {md['corr_sum']:+8.3f} |")

    for name, evals in runs.items():
        for label, e in pick_checkpoints(evals).items():
            ck = load_checkpoint(run_dirs[name] / f"snap_{e['iteration']:06d}.pt", map_location="cpu")
            c = ck.config
            gz, gy = ck.gz.to(device).eval(), ck.gy.to(device).eval()
            n_blocks = c.estimation_blocks(system.dt_ps)
            g = torch.Generator().manual_seed(0)
            with torch.no_grad():
                z0 = torch.rand(chains, c.latent_dim, generator=g).to(device)
                states = gz.unroll(z0, c.burn_in + n_blocks - 1, generator=g)           # z_0 .. z_{c+n-1}
                decoded = states[c.burn_in:]                                           # z_c .. z_{c+n-1}
                blocks = [mean_operation(gy.generate(z, generator=g)) for z in decoded]
                d = torch.cat(blocks, dim=1).cpu().double().numpy()
            st = _block_stats(d, m)
            zp = _latent_persistence(decoded.cpu().double().numpy())
            tag = f"{name} {label} @{e['iteration']}"
            print(f"{tag:<44} {st['var']:8.2f}      " + " ".join(f"{st['corr'].get(lag, float('nan')):+6.3f}"
                  for lag in BLOCK_LAGS) + f" {st['corr_sum']:+8.3f} |         "
                  + " ".join(f"{zp.get(lag, float('nan')):6.3f}" for lag in LATENT_LAGS))
    print()


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("runs", nargs="+", help="run directories (evaluations.jsonl, msd_*.npy, snap_*.pt)")
    ap.add_argument("--data", required=True, help="preprocessed cache .npz of the same system")
    ap.add_argument("--blocks", action="store_true", help="section 3: rollouts for block and latent statistics")
    ap.add_argument("--chains", type=int, default=1024)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args(argv)

    run_dirs = {Path(r).name: Path(r) for r in sorted(args.runs)}
    runs = {name: _jsonl(p / "evaluations.jsonl") for name, p in run_dirs.items()}
    system = SystemData.load(args.data)
    msd_ref, _ = msd_time_averaged(system.displacements)
    dt_ref = float(system.dt_ps)

    section_summary(runs)
    section_msd(run_dirs, runs, msd_ref, dt_ref)
    if args.blocks:
        section_blocks(run_dirs, runs, system, args.chains, args.device)


if __name__ == "__main__":
    main()