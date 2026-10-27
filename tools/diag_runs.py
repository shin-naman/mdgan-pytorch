"""Diagnose finished training runs against the MD they were trained on.

Reads run directories (evaluations.jsonl, msd_XXXXXX.npy, snap_XXXXXX.pt) and the
preprocessed cache of the same system, and prints:

  1. per run: the best (lowest err), median and final checkpoint, with err and D_rel;
  2. the generated / reference MSD ratio at times from 20 ps to 78 ns for those
     checkpoints, and the median ratio over all checkpoints of all runs. A ratio that
     GROWS with time means correlated drift between blocks; a constant factor means
     a step-size (variance) error;
  3. with --blocks: for the same checkpoints, rolled out again, and for the real MD:
       sigma           the scale that run trained in (nm; 1 for scale_inputs=False runs).
                       Every row is converted to nm before any statistic, so runs trained in
                       sigma units and in nm are directly comparable
       blk var         variance of one M-step block displacement per axis (nm^2)
       var/MD          blk var relative to the MD's (unit-free)
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

    python tools/diag_runs.py runs/chtc/run_* --data data/cache/PE.npz --blocks --all --csv runs/diag_blocks.csv

--all analyses every checkpoint instead of best/median/final; --csv writes one row per analysed
checkpoint (all section-3 columns, every block lag 1-30, plus that checkpoint's err and D_rel) and
one MD reference row, for joining with the training logs (Phase 1).

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


# Section 3 table: one layout for the header and every row, so columns cannot run together.
_COLS = ([("source", 46, "s"), ("sigma", 6, ".3f"), ("blk var", 8, ".4f"), ("var/MD", 7, ".2f")]
         + [(f"ρ@{lag}", 7, "+.3f") for lag in BLOCK_LAGS] + [("Σρ1-30", 8, "+.3f")]
         + [(f"z@{lag}", 7, "+.3f") for lag in LATENT_LAGS])


def _fmt_row(values: list) -> str:
    """values in _COLS order; None prints as a dash (e.g. no latent for the MD row)."""
    cells = []
    for (_, width, spec), v in zip(_COLS, values, strict=True):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            cells.append(f"{'—':>{width}}")
        elif spec == "s":
            cells.append(f"{v:<{width}}")
        else:
            sign, rest = ("+", spec[1:]) if spec.startswith("+") else ("", spec)
            cells.append(f"{v:>{sign}{width}{rest}}")
    return "  ".join(cells)


def _fmt_header() -> str:
    return "  ".join(f"{name:<{w}}" if spec == "s" else f"{name:>{w}}" for name, w, spec in _COLS)


def _stats_row(label: str, sigma, st: dict, var_md: float, zp: dict | None) -> list:
    return ([label, sigma, st["var"], st["var"] / var_md]
            + [st["corr"].get(lag, float("nan")) for lag in BLOCK_LAGS] + [st["corr_sum"]]
            + [(zp or {}).get(lag) for lag in LATENT_LAGS])


def _csv_record(run: str, iteration, label: str, sigma, st: dict, var_md: float, zp: dict | None,
                e: dict | None) -> dict:
    rec = {"run": run, "iteration": iteration, "label": label, "sigma_nm": sigma,
           "blk_var_nm2": st["var"], "var_over_md": st["var"] / var_md,
           **{f"rho@{lag}": st["corr"].get(lag, float("nan")) for lag in range(1, 31)},
           "rho_sum_1_30": st["corr_sum"],
           **{f"z@{lag}": (zp or {}).get(lag, float("nan")) for lag in LATENT_LAGS},
           "err_percent": (e or {}).get("err_percent", float("nan")),
           "D_rel": (e or {}).get("D_rel_error", float("nan"))}
    return rec


def section_blocks(run_dirs, runs, system, chains: int, device: str, all_checkpoints: bool = False,
                   csv_path: str | None = None) -> None:
    import torch

    from mdgan.checkpoint import load_checkpoint
    from mdgan.ops import mean_operation

    print(f"3. Block and latent statistics ({chains} chains per rollout; MD = the real trajectories; "
          f"everything in nm)")
    print(_fmt_header())

    first = next(iter(runs))
    m = load_checkpoint(run_dirs[first] / f"snap_{runs[first][0]['iteration']:06d}.pt").config.block_len
    md = _block_stats(system.displacements.astype(np.float64), m)                      # raw nm
    print(_fmt_row(_stats_row("MD (target)", None, md, md["var"], None)))
    records = [_csv_record("MD", None, "reference", None, md, md["var"], None, None)]

    for name, evals in runs.items():
        picked = pick_checkpoints(evals)
        named = {e["iteration"]: label for label, e in picked.items()}
        selection = ([(named.get(e["iteration"], "checkpoint"), e) for e in evals] if all_checkpoints
                     else list(picked.items()))
        for label, e in selection:
            ck = load_checkpoint(run_dirs[name] / f"snap_{e['iteration']:06d}.pt", map_location="cpu")
            c = ck.config
            if c.block_len != m:
                raise ValueError(f"{name}: block_len {c.block_len} differs from {m}; compare like with like")
            gz, gy = ck.gz.to(device).eval(), ck.gy.to(device).eval()
            n_blocks = c.estimation_blocks(system.dt_ps)
            g = torch.Generator().manual_seed(0)
            with torch.no_grad():
                z0 = torch.rand(chains, c.latent_dim, generator=g).to(device)
                states = gz.unroll(z0, c.burn_in + n_blocks - 1, generator=g)           # z_0 .. z_{c+n-1}
                decoded = states[c.burn_in:]                                           # z_c .. z_{c+n-1}
                blocks = [mean_operation(gy.generate(z, generator=g)) for z in decoded]
                d = torch.cat(blocks, dim=1).cpu().double().numpy() * ck.sigma_nm      # model units -> nm
            st = _block_stats(d, m)
            zp = _latent_persistence(decoded.cpu().double().numpy())
            print(_fmt_row(_stats_row(f"{name} {label} @{e['iteration']}", ck.sigma_nm, st, md["var"], zp)),
                  flush=True)
            records.append(_csv_record(name, e["iteration"], label, ck.sigma_nm, st, md["var"], zp, e))
    print()
    if csv_path:
        import csv
        Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(records[0]))
            w.writeheader()
            w.writerows(records)
        print(f"wrote {len(records) - 1} checkpoint rows + 1 MD row to {csv_path}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("runs", nargs="+", help="run directories (evaluations.jsonl, msd_*.npy, snap_*.pt)")
    ap.add_argument("--data", required=True, help="preprocessed cache .npz of the same system")
    ap.add_argument("--blocks", action="store_true", help="section 3: rollouts for block and latent statistics")
    ap.add_argument("--chains", type=int, default=1024)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--all", action="store_true", help="section 3 on every checkpoint, not best/median/final")
    ap.add_argument("--csv", help="write the section-3 rows to this CSV (one per checkpoint + the MD)")
    args = ap.parse_args(argv)
    if (args.all or args.csv) and not args.blocks:
        ap.error("--all and --csv apply to section 3; add --blocks")

    run_dirs = {Path(r).name: Path(r) for r in sorted(args.runs)}
    runs = {name: _jsonl(p / "evaluations.jsonl") for name, p in run_dirs.items()}
    system = SystemData.load(args.data)
    msd_ref, _ = msd_time_averaged(system.displacements)
    dt_ref = float(system.dt_ps)

    section_summary(runs)
    section_msd(run_dirs, runs, msd_ref, dt_ref)
    if args.blocks:
        section_blocks(run_dirs, runs, system, args.chains, args.device, args.all, args.csv)


if __name__ == "__main__":
    main()