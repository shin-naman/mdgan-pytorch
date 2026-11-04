"""Step 0b and 0c: rollouts that change only how G_z's output BatchNorm normalises.

0b  bn-batch:  every snapshot rolled out exactly as evaluation does (rollout.rollout's loop and
               draw order, config.seed, config.estimation_chains, mean operation over all chains),
               except that G_z's output BatchNorm has its running buffers removed, so it normalises
               with batch statistics in eval mode (learned gamma and beta kept). All trajectory and
               latent metrics come from that one rollout; plateau needs its own pair of latent
               rollouts (definition below). For one snapshot (--stats-of) G_z's pre-normalisation
               output is recorded at depth 10 and 130 (G_z calls from z_0) against the stored
               running averages.
0c  chunked:   the same z_0 and the same noise at every step; G_z applied to all chains at once vs
               in groups of --group (64). Latent metrics and plateau in both modes. No G_Y: the mean
               operation is not involved (if decoding is added it must centre over all chains).

Metrics
  ratio@t      MSD_gen / MSD_MD at 5, 10, 20, 40, 78 ns (ensemble MSD as in evaluate; MD time-averaged)
  err, D       as in evaluate; drift78 = b * 78 / a from MSD = a t + b t^2 over 5-78 ns
               (phase0_compare.fit_ballistic); "ballistic fraction" = fraction of checkpoints with
               drift78 > 0.25
  var/MD, Σρ   block variance relative to the MD's, block correlations summed over lags 1-30
               (diag_runs._block_stats, everything in nm)
  period, cycle share   latent_cycle's spectrum analysis of the decoded latents z_c .. z_c+n-1
  env100       max |pooled latent ACF| over lags 95-105
  plateau(k)   z_0 for N = --plateau-chains chains drawn once; latent rollouts A and B from that z_0
               with independent noise streams (no G_Y, no burn-in offset; k = transitions applied).
               plateau(k) = mean_i ||z_A,i(k) - z_B,i(k)|| / mean_i ||z_A,i(k) - z_B,perm(i)(k)||.
               1: z_0 forgotten; 0: the noise does nothing. Reported at k = 1, 2, 5, 10, 20, 50, 100.
               plateau_aa(k) is the noise_checks.py check-3a variant (denominator: A vs permuted A).

arm-eval     as-trained evaluation of run directories at checkpoints --iterations (default 20k-50k):
             the mirrored rollout as evaluation does it (no stripping), block statistics on fixed
             64-step (0.64 ns) blocks for every arm with Σρ split into lags 1-3 and 4-30, latent
             metrics from a fixed 125-transition latent path (so lag 100 exists for any M), plateau at
             k = 25 and 50. Prints env100 and Σρ first, then one summary row per arm.

Usage (repository root):
    python tools/step0_rollouts.py --self-test
    python tools/step0_rollouts.py bn-batch runs/chtc/run_running_stats_c*_seed*/snap_*.pt \\
        --data data/cache/PE.npz --device mps --check-reproduction --out runs/step0b
    python tools/step0_rollouts.py chunked runs/chtc/run_scaleoff_c7166270_seed0/snap_030000.pt \\
        runs/chtc/run_scaleoff_c7166270_seed1/snap_060000.pt --device mps --out runs/step0c
    python tools/step0_rollouts.py arm-eval runs/chtc/run_{noise05,wd01,drop02,m128,bnhidden}_c*_seed* \
        --data data/cache/PE.npz --device mps --check-reproduction --out runs/batch1
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

PLATEAU_STEPS = (1, 2, 5, 10, 20, 25, 50, 100)
EVAL_BLOCK_STEPS = 64           # arm-eval: block statistics on fixed 0.64 ns blocks for every arm
LATENT_TRANSITIONS = 125        # arm-eval: latent metrics from a fixed-length path (lag 100 exists for any M)
RATIO_NS = (5, 10, 20, 40, 78)
DEPTHS = (10, 130)


def _load_tool(name: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


latent_cycle = _load_tool("latent_cycle")


# ============================================================================ BatchNorm

def strip_running_stats(gz) -> dict:
    """Remove G_z output BatchNorm's running buffers so eval mode uses batch statistics.
    Returns the removed buffers (CPU copies) for reporting; gamma and beta are kept."""
    bn = gz.norm if hasattr(gz, "norm") else gz.output_norm
    stored = {"running_mean": None if bn.running_mean is None else bn.running_mean.detach().cpu().double().clone(),
              "running_var": None if bn.running_var is None else bn.running_var.detach().cpu().double().clone()}
    bn.track_running_stats = False
    bn.running_mean = None
    bn.running_var = None
    bn.num_batches_tracked = None
    return stored


# ============================================================================ rollouts

def rollout_with_latents(gz, gy, n_chains: int, n_blocks: int, burn_in: int, *, mean_operation: bool,
                         generator, record_pre_bn_at: tuple = ()):
    """rollout.rollout's loop and draw order, also returning the decoded latents (n_blocks, N, n_z)
    and the input of G_z's BatchNorm (wherever it sits) at the requested depths (G_z calls from z_0)."""
    from mdgan.rollout import _center, _eval_mode, _initial_latents
    param = next(gz.parameters())
    out = torch.empty(n_chains, n_blocks * gy.block_len, gy.n_channels, dtype=param.dtype)
    lat = torch.empty(n_blocks, n_chains, gz.latent_dim, dtype=param.dtype)
    pre, calls = {}, [0]

    def hook(_m, inputs):
        calls[0] += 1
        if calls[0] in record_pre_bn_at:
            pre[calls[0]] = inputs[0].detach().cpu().double().clone()
    norm = gz.norm if hasattr(gz, "norm") else gz.output_norm
    handle = norm.register_forward_pre_hook(hook) if record_pre_bn_at else None
    try:
        with torch.no_grad(), _eval_mode(gz, gy):
            z = _initial_latents(gz, n_chains, generator, param.device, param.dtype)
            for _ in range(burn_in):
                z = gz(z, gz.sample_noise(n_chains, device=z.device, dtype=z.dtype, generator=generator))
            for k in range(n_blocks):
                lat[k] = z.cpu()
                y = gy.generate(z, generator=generator)
                if mean_operation:
                    y = _center(y)
                out[:, k * gy.block_len:(k + 1) * gy.block_len] = y.cpu()
                if k + 1 < n_blocks:
                    z = gz(z, gz.sample_noise(n_chains, device=z.device, dtype=z.dtype, generator=generator))
    finally:
        if handle is not None:
            handle.remove()
    return out, lat, pre


def _step(gz, z, r, group: int | None):
    if group is None or group >= z.shape[0]:
        return gz(z, r)
    return torch.cat([gz(z[i:i + group], r[i:i + group]) for i in range(0, z.shape[0], group)])


def latent_path(gz, z0, n_steps: int, generator, group: int | None = None) -> np.ndarray:
    """(n_steps + 1, N, n_z): z_0 and n_steps transitions; noise for all N drawn per step, so the
    grouped and ungrouped paths see identical noise."""
    out, z = [z0.cpu()], z0
    with torch.no_grad():
        for _ in range(n_steps):
            r = gz.sample_noise(z.shape[0], device=z.device, dtype=z.dtype, generator=generator)
            z = _step(gz, z, r, group)
            out.append(z.cpu())
    return torch.stack(out).double().numpy()


def plateau_curve(gz, z0, seeds=(11, 12), group: int | None = None, steps=PLATEAU_STEPS) -> dict:
    n = max(steps)
    a = latent_path(gz, z0, n, torch.Generator().manual_seed(seeds[0]), group)
    b = latent_path(gz, z0, n, torch.Generator().manual_seed(seeds[1]), group)
    perm = np.random.default_rng(0).permutation(a.shape[1])
    pl, pl_aa = {}, {}
    for k in steps:
        same = np.linalg.norm(a[k] - b[k], axis=1).mean()
        indep_ab = np.linalg.norm(a[k] - b[k][perm], axis=1).mean()
        indep_aa = np.linalg.norm(a[k] - a[k][perm], axis=1).mean()
        pl[k] = float(same / indep_ab) if indep_ab > 0 else float("nan")
        pl_aa[k] = float(same / indep_aa) if indep_aa > 0 else float("nan")
    return {"plateau": pl, "plateau_aa": pl_aa}


# ============================================================================ metrics

def dc_shares(z: np.ndarray) -> tuple[float, np.ndarray]:
    """Share of latent variance held by each chain's time-mean (the zero-frequency part that
    latent_cycle.spectrum removes before computing power). Pooled over dimensions, and per dimension.
    An invariant (a per-chain constant offset) puts its variance here; a process with no memory gives
    about 1/T."""
    zc = z - z.mean(axis=(0, 1), keepdims=True)
    chain_mean_var = zc.mean(axis=0).var(axis=0)                  # (D,) variance across chains of time-means
    total_var = zc.var(axis=(0, 1))                               # (D,)
    return float(chain_mean_var.sum() / total_var.sum()), chain_mean_var / total_var


def latent_metrics(z: np.ndarray) -> dict:
    """z: (T, N, n_z) latents. The pooled envelope can miss an invariant that lives in a few of the
    dimensions (it is averaged with the rest), and the spectrum removes each chain's mean, so the DC
    share and the per-dimension maximum |ACF| at lag 100 are reported alongside."""
    per, pooled = latent_cycle.autocorrelation(z)
    f, p = latent_cycle.spectrum(z)
    cyc = latent_cycle.find_cycle(f, p, pooled, z.shape[0])
    hi = min(105, len(pooled) - 2)
    dc, dc_dim = dc_shares(z)
    acf100 = np.abs(per[:, 100]) if per.shape[1] > 101 else np.full(per.shape[0], np.nan)
    return {"period_blocks": cyc["period_blocks"], "cycle_share": cyc["cycle_fraction_of_total_variance"],
            "env100": float(np.nanmax(np.abs(pooled[95:hi + 1]))) if hi >= 95 else float("nan"),
            "z@50": float(pooled[50]), "z@100": float(pooled[100]) if len(pooled) > 101 else float("nan"),
            "dc_share": dc, "dc_share_max_dim": float(dc_dim.max()), "dc_share_argmax_dim": int(dc_dim.argmax()),
            "acf100_max_dim": float(np.nanmax(acf100)), "acf100_argmax_dim": int(np.nanargmax(acf100))}


def trajectory_metrics(d_model: torch.Tensor, sigma_nm: float, cfg, ref: np.ndarray, dt_ref: float, md_block: dict,
                       block_steps: int | None = None) -> dict:
    from mdgan.data.msd import diffusion_coefficient, err, msd_ensemble
    diag_runs = _load_tool("diag_runs")
    fit_ballistic = _load_tool("phase0_compare").fit_ballistic
    dt_gen = dt_ref * cfg.step_skip
    n_steps = int(round(cfg.estimation_time_ps / dt_gen))
    msd = msd_ensemble(d_model[:, :n_steps]) * sigma_nm ** 2
    d_gen, _ = diffusion_coefficient(msd, dt_gen, *cfg.diffusion_fit_ps)
    d_md, _ = diffusion_coefficient(ref, dt_ref, *cfg.diffusion_fit_ps)
    a, b = fit_ballistic(msd, dt_gen)
    st = diag_runs._block_stats(d_model[:, :n_steps].numpy() * sigma_nm, block_steps or cfg.block_len)
    out = {"err_percent": float(err(msd, dt_gen, ref, dt_ref, *cfg.err_window_ps)), "D_ratio": float(d_gen / d_md),
           "drift78": float(b * 78.0 / a) if a > 0 else float("nan"),
           "var_over_md": st["var"] / md_block["var"], "rho_sum_1_30": st["corr_sum"], "rho@1": st["corr"].get(1),
           "rho_sum_1_3": float(sum(st["corr"].get(k, 0.0) for k in (1, 2, 3))),
           "rho_sum_4_30": float(sum(v for k, v in st["corr"].items() if k >= 4))}
    for t in RATIO_NS:
        out[f"ratio@{t}"] = float(msd[int(round(t * 1000 / dt_gen))] / ref[int(round(t * 1000 / dt_ref))])
    return out


def ballistic_share(msd: np.ndarray, dt_ps: float, t_min_ns: float = 5.0) -> float:
    """Fit MSD = a t + b t^2 on [t_min, end] with a, b >= 0 and return b T^2 / (a T + b T^2) at the end T:
    the share of the fitted MSD there that is ballistic (0 = diffusive, 1 = ballistic). Unlike b T / a,
    it stays defined when the unconstrained fit would give a <= 0, which is when the drift is strongest."""
    t = np.arange(len(msd)) * dt_ps / 1000
    m = t >= t_min_ns
    x, y = t[m], np.asarray(msd, dtype=np.float64)[m]
    (a, b), *_ = np.linalg.lstsq(np.column_stack([x, x ** 2]), y, rcond=None)
    if a < 0:                                    # two-parameter non-negative least squares, by its two faces
        a, b = 0.0, max(float((x ** 2 @ y) / (x ** 4).sum()), 0.0)
    elif b < 0:
        a, b = max(float((x @ y) / (x @ x)), 0.0), 0.0
    T = x[-1]
    total = a * T + b * T ** 2
    return float(b * T ** 2 / total) if total > 0 else float("nan")


def horizon_metrics(d_model: torch.Tensor, sigma_nm: float, dt_ps: float, ref_steps: int, block_steps: int) -> dict:
    """For a rollout longer than the reference window (ref_steps), quantities that need no MD:
    Σρ (lags 1-30, block_steps blocks) over the first and the last ref_steps; the generated MSD's local
    exponent at 3/4 of the horizon (normal diffusion = 1; persistent drift > 1); and the ballistic share at
    the end of the horizon from MSD = a t + b t^2 fitted over 5 ns to the end (ballistic_share)."""
    from mdgan.data.msd import msd_ensemble
    diag_runs = _load_tool("diag_runs")
    n = d_model.shape[1]
    d = d_model.numpy() * sigma_nm
    early = diag_runs._block_stats(d[:, :ref_steps], block_steps)["corr_sum"]
    late = diag_runs._block_stats(d[:, n - ref_steps:], block_steps)["corr_sum"]
    msd = msd_ensemble(d_model) * sigma_nm ** 2
    t_end_ns = (len(msd) - 1) * dt_ps / 1000
    k = int(0.75 * (len(msd) - 1))
    lo, hi = int(k / 1.3), min(int(k * 1.3), len(msd) - 1)
    alpha_end = float(np.log(msd[hi] / msd[lo]) / np.log(hi / lo))
    return {"horizon_ns": t_end_ns, "rho_sum_first": early, "rho_sum_last": late, "alpha_end": alpha_end,
            "ballistic_share_end": ballistic_share(msd, dt_ps)}


def pre_bn_summary(stored: dict, pre: dict) -> dict:
    """Pre-normalisation output statistics at each recorded depth vs the stored running averages."""
    out = {}
    rm, rv = stored.get("running_mean"), stored.get("running_var")
    for depth, x in sorted(pre.items()):
        mu, var = x.mean(0), x.var(0, unbiased=False)
        row = {"mean_abs_mu": float(mu.abs().mean()), "mean_var": float(var.mean())}
        if rm is not None:
            row["median_abs_mu_minus_running_over_running_sd"] = float(((mu - rm).abs() / rv.sqrt()).median())
            row["median_var_over_running_var"] = float((var / rv).median())
        out[depth] = row
    if len(pre) == 2:
        (d1, x1), (d2, x2) = sorted(pre.items())
        out[f"{d2}_vs_{d1}"] = {
            "median_abs_mu_shift_over_sd": float(((x2.mean(0) - x1.mean(0)).abs() / x1.var(0, unbiased=False).sqrt()).median()),
            "median_var_ratio": float((x2.var(0, unbiased=False) / x1.var(0, unbiased=False)).median())}
    return out


# ============================================================================ commands

def _md(data: str, cfg, block_steps: int | None = None):
    from mdgan.data.msd import msd_time_averaged
    from mdgan.data.preprocess import SystemData
    diag_runs = _load_tool("diag_runs")
    s = SystemData.load(data)
    ref, _ = msd_time_averaged(s.displacements)
    return ref, float(s.dt_ps), diag_runs._block_stats(s.displacements.astype(np.float64), block_steps or cfg.block_len)


def _evaluation_record(snapshot: Path, iteration: int) -> dict | None:
    f = snapshot.parent / "evaluations.jsonl"
    if not f.exists():
        return None
    for line in f.read_text().splitlines():
        e = json.loads(line)
        if e["iteration"] == iteration:
            return e
    return None


def cmd_bn_batch(args) -> list[dict]:
    from mdgan.checkpoint import load_checkpoint
    results, md = [], None
    for i, snap in enumerate(sorted(args.snapshots)):
        ck = load_checkpoint(snap, map_location="cpu")
        cfg = ck.config
        if md is None:
            md = _md(args.data, cfg)
        ref, dt_ref, md_block = md
        n_blocks = cfg.estimation_blocks(dt_ref)
        chains = args.chains or cfg.estimation_chains
        row = {"snapshot": snap, "run": Path(snap).parent.name, "iteration": ck.iteration}

        if args.check_reproduction and i == 0:
            gz0, gy0 = ck.gz.to(args.device), ck.gy.to(args.device)
            d0, _, _ = rollout_with_latents(gz0, gy0, chains, n_blocks, cfg.burn_in, mean_operation=cfg.mean_operation,
                                            generator=torch.Generator().manual_seed(cfg.seed))
            t0 = trajectory_metrics(d0, ck.sigma_nm, cfg, ref, dt_ref, md_block)
            rec = _evaluation_record(Path(snap), ck.iteration)
            row["reproduction"] = {"tool_err": t0["err_percent"], "tool_D_ratio": t0["D_ratio"],
                                   "logged_err": rec["err_percent"] if rec else None,
                                   "logged_D_ratio": 1 + rec["D_rel_error"] if rec else None}
            del d0

        gz, gy = ck.gz.to(args.device), ck.gy.to(args.device)
        stored = strip_running_stats(gz)
        depths = DEPTHS if (args.stats_of is not None and i == args.stats_of) else ()
        d, lat, pre = rollout_with_latents(gz, gy, chains, n_blocks, cfg.burn_in, mean_operation=cfg.mean_operation,
                                           generator=torch.Generator().manual_seed(cfg.seed), record_pre_bn_at=depths)
        row.update(trajectory_metrics(d, ck.sigma_nm, cfg, ref, dt_ref, md_block))
        del d
        row.update(latent_metrics(lat[:, : args.latent_chains].double().numpy()))
        z0 = torch.rand(args.plateau_chains, cfg.latent_dim, generator=torch.Generator().manual_seed(0)).to(args.device)
        gz.eval()
        row.update(plateau_curve(gz, z0))
        if depths:
            row["pre_bn"] = pre_bn_summary(stored, pre)
        results.append(row)
        print(_fmt_b(row), flush=True)
    return results


def _arm(run_name: str) -> str:
    import re
    m = re.fullmatch(r"run_(?:(.+?)_)?c\d+_seed\d+", run_name)
    return (m.group(1) or "baseline") if m else run_name


def cmd_arm_eval(args) -> list[dict]:
    """As-trained evaluation of each run's checkpoints at --iterations (default 20k-50k)."""
    from mdgan.checkpoint import load_checkpoint
    results, md = [], None
    snaps = [Path(r) / f"snap_{it:06d}.pt" for r in sorted(args.snapshots) for it in args.iterations]
    missing = [str(p) for p in snaps if not p.exists()]
    if missing:
        raise SystemExit("missing snapshots:\n  " + "\n  ".join(missing))
    for i, snap in enumerate(snaps):
        ck = load_checkpoint(snap, map_location="cpu")
        cfg = ck.config
        if md is None:
            md = _md(args.data, cfg, EVAL_BLOCK_STEPS)
        ref, dt_ref, md_block = md
        gz, gy = ck.gz.to(args.device), ck.gy.to(args.device)
        arm = _arm(snap.parent.name)
        if args.strip_running_stats:
            if not getattr(ck.config, "use_running_stats", False):
                raise SystemExit(f"--strip-running-stats given, but {snap.parent.name} was not trained with running stats")
            strip_running_stats(gz)
            arm += "+batchstats"
        n_blocks = args.rollout_blocks or cfg.estimation_blocks(dt_ref)
        if args.rollout_blocks:
            arm += f"+{n_blocks}blk"
        d, _, _ = rollout_with_latents(gz, gy, args.chains or cfg.estimation_chains, n_blocks,
                                       cfg.burn_in, mean_operation=cfg.mean_operation,
                                       generator=torch.Generator().manual_seed(cfg.seed))
        row = {"snapshot": str(snap), "run": snap.parent.name, "arm": arm, "iteration": ck.iteration,
               "block_len": cfg.block_len, "rollout_blocks": n_blocks}
        row.update(trajectory_metrics(d, ck.sigma_nm, cfg, ref, dt_ref, md_block, EVAL_BLOCK_STEPS))
        ref_steps = int(round(cfg.estimation_time_ps / (dt_ref * cfg.step_skip)))
        if d.shape[1] > ref_steps:
            row.update(horizon_metrics(d, ck.sigma_nm, dt_ref * cfg.step_skip, ref_steps, EVAL_BLOCK_STEPS))
        del d
        if args.check_reproduction and i == 0:
            rec = _evaluation_record(snap, ck.iteration)
            row["reproduction"] = {"tool_err": row["err_percent"], "tool_D_ratio": row["D_ratio"],
                                   "logged_err": rec["err_percent"] if rec else None,
                                   "logged_D_ratio": 1 + rec["D_rel_error"] if rec else None}
        gz.eval()
        z0 = torch.rand(args.plateau_chains, cfg.latent_dim, generator=torch.Generator().manual_seed(0)).to(args.device)
        zpath = latent_path(gz, z0, cfg.burn_in + LATENT_TRANSITIONS - 1, torch.Generator().manual_seed(cfg.seed))
        row.update(latent_metrics(zpath[cfg.burn_in:]))
        row.update(plateau_curve(gz, z0))
        results.append(row)
        print(_fmt_a(row), flush=True)
    return results


def _fmt_a(r: dict) -> str:
    it = f"{r['iteration'] // 1000}k" if r["iteration"] >= 1000 else str(r["iteration"])
    s = (f"{r['run']} @{it}{' [' + r['arm'].split('+', 1)[1] + ']' if '+' in r['arm'] else ''}: env100 {r['env100']:.3f}  "
         f"Σρ {r['rho_sum_1_30']:+.2f} "
         f"(1-3 {r['rho_sum_1_3']:+.2f} | 4-30 {r['rho_sum_4_30']:+.2f})  var/MD {r['var_over_md']:.2f}  "
         f"D {r['D_ratio']:.2f}x  err {r['err_percent']:.1f}%  ratio " + "/".join(f"{r[f'ratio@{t}']:.2f}" for t in RATIO_NS)
         + f"  drift78 {r['drift78']:+.3f}  plateau25 {r['plateau'][25]:.2f}  plateau50 {r['plateau'][50]:.2f}  "
         f"period {r['period_blocks']:.2f}  cycle {r['cycle_share'] * 100:.0f}%  "
         f"DC {r['dc_share']:.3f} (max dim {r['dc_share_max_dim']:.2f})  |ACF100| max dim {r['acf100_max_dim']:.2f}")
    if "horizon_ns" in r:
        s += (f"\n   horizon {r['horizon_ns']:.0f} ns: Σρ first 80 ns {r['rho_sum_first']:+.2f} | last 80 ns "
              f"{r['rho_sum_last']:+.2f}  alpha_end {r['alpha_end']:.2f}  ballistic share at end {r['ballistic_share_end']:.2f}")
    if "reproduction" in r:
        p = r["reproduction"]
        s += (f"\n   reproduction: tool err {p['tool_err']:.2f}% / D {p['tool_D_ratio']:.3f}x vs logged "
              + (f"{p['logged_err']:.2f}% / {p['logged_D_ratio']:.3f}x" if p["logged_err"] is not None else "n/a"))
    return s


def summarize_arms(results: list[dict]) -> None:
    arms = sorted({r["arm"] for r in results}, key=lambda a: (a != "baseline", a))
    print(f"\n{'arm':<26} {'n':>3} {'med env100':>10} {'med Σρ':>7} {'Σρ 1-3':>7} {'Σρ 4-30':>8} {'med DC':>7} "
          f"{'DC dim':>7} {'ACF100 dim':>10} {'D > MD':>7} {'med D_rel':>9} {'ballistic':>10} {'(a<=0)':>6} "
          f"{'med pl50':>8} {'med pl25':>8}  periods seen")
    for arm in arms:
        rr = [r for r in results if r["arm"] == arm]
        def col(k, rr=rr):
            return np.array([x[k] for x in rr], float)

        d78 = col("drift78")
        periods = sorted({round(x["period_blocks"], 1) for x in rr})
        print(f"{arm:<26} {len(rr):>3} {np.median(col('env100')):>10.3f} {np.median(col('rho_sum_1_30')):>+7.2f} "
              f"{np.median(col('rho_sum_1_3')):>+7.2f} {np.median(col('rho_sum_4_30')):>+8.2f} "
              f"{np.median(col('dc_share')):>7.3f} {np.median(col('dc_share_max_dim')):>7.2f} "
              f"{np.median(col('acf100_max_dim')):>10.2f} "
              f"{np.mean(col('D_ratio') > 1) * 100:>6.0f}% {np.median(col('D_ratio')) - 1:>+9.2f} "
              f"{np.mean(d78 > 0.25) * 100:>9.0f}% {int(np.sum(~np.isfinite(d78))):>6} "
              f"{np.median([x['plateau'][50] for x in rr]):>8.2f} {np.median([x['plateau'][25] for x in rr]):>8.2f}  "
              + ", ".join(f"{p:g}" for p in periods))
    print("ballistic = share with drift78 > 0.25, as in phase0_compare; (a<=0) = checkpoints whose ballistic fit has a "
          "non-positive linear term (drift78 undefined, counted as not ballistic above)")
    print("DC = share of latent variance in each chain's time-mean (no memory ~ 1/T = 0.008 for T = 125); DC dim and "
          "ACF100 dim = the largest single-dimension value (medians over checkpoints)")
    hz = [r for r in results if "horizon_ns" in r]
    if hz:
        print(f"\nlong rollouts ({hz[0]['horizon_ns']:.0f} ns): median over checkpoints per arm")
        for arm in arms:
            rr = [r for r in hz if r["arm"] == arm]
            if rr:
                med = lambda k, rr=rr: np.nanmedian([x[k] for x in rr])  # noqa: E731
                print(f"  {arm:<26} Σρ first 80 ns {med('rho_sum_first'):+.2f} | last 80 ns {med('rho_sum_last'):+.2f}  "
                      f"alpha_end {med('alpha_end'):.2f}  ballistic share at end {med('ballistic_share_end'):.2f}")


def cmd_chunked(args) -> list[dict]:
    from mdgan.checkpoint import load_checkpoint
    results = []
    for snap in args.snapshots:
        ck = load_checkpoint(snap, map_location="cpu")
        cfg = ck.config
        gz = ck.gz.to(args.device).eval()
        n_blocks = cfg.estimation_blocks(10.0)
        z0 = torch.rand(args.plateau_chains, cfg.latent_dim, generator=torch.Generator().manual_seed(0)).to(args.device)
        row = {"snapshot": snap, "run": Path(snap).parent.name, "iteration": ck.iteration, "group": args.group}
        for mode, group in (("all", None), ("grouped", args.group)):
            z = latent_path(gz, z0, cfg.burn_in + n_blocks - 1, torch.Generator().manual_seed(cfg.seed), group)
            row[mode] = {**latent_metrics(z[cfg.burn_in:]), **plateau_curve(gz, z0, group=group)}
        results.append(row)
        print(_fmt_c(row), flush=True)
    return results


def _fmt_b(r: dict) -> str:
    s = (f"{r['run']} @{r['iteration'] // 1000}k: err {r['err_percent']:.1f}%  D {r['D_ratio']:.2f}x  ratio "
         + "/".join(f"{r[f'ratio@{t}']:.2f}" for t in RATIO_NS)
         + f"  drift78 {r['drift78']:+.3f}  var/MD {r['var_over_md']:.2f}  Σρ {r['rho_sum_1_30']:+.2f}  "
         f"period {r['period_blocks']:.2f}  cycle {r['cycle_share'] * 100:.0f}%  env100 {r['env100']:.3f}  "
         f"plateau " + " ".join(f"{k}:{v:.2f}" for k, v in r["plateau"].items()))
    if "reproduction" in r:
        p = r["reproduction"]
        s += (f"\n   reproduction (unmodified model): tool err {p['tool_err']:.2f}% / D {p['tool_D_ratio']:.3f}x vs logged "
              + (f"{p['logged_err']:.2f}% / {p['logged_D_ratio']:.3f}x" if p["logged_err"] is not None else "n/a"))
    if "pre_bn" in r:
        for depth, v in r["pre_bn"].items():
            s += f"\n   pre-BN @ {depth}: " + ", ".join(f"{k} {x:.4g}" for k, x in v.items())
    return s


def _fmt_c(r: dict) -> str:
    lines = [f"{r['run']} @{r['iteration'] // 1000}k (group {r['group']}):"]
    for mode in ("all", "grouped"):
        m = r[mode]
        lines.append(f"   {mode:<8} period {m['period_blocks']:.2f}  cycle {m['cycle_share'] * 100:.0f}%  env100 {m['env100']:.3f}  "
                     "plateau " + " ".join(f"{k}:{v:.2f}" for k, v in m["plateau"].items()))
    return "\n".join(lines)


def summarize_b(results: list[dict]) -> None:
    if not results:
        return
    col = lambda k: np.array([r[k] for r in results], float)  # noqa: E731
    print(f"\n{len(results)} checkpoints: ballistic fraction (drift78 > 0.25) {np.mean(col('drift78') > 0.25) * 100:.0f}%, "
          f"median var/MD {np.median(col('var_over_md')):.2f}, median Σρ {np.median(col('rho_sum_1_30')):+.2f}, "
          f"median plateau50 {np.median([r['plateau'][50] for r in results]):.2f}, "
          f"median D {np.median(col('D_ratio')):.2f}x, median ratio@78 {np.median(col('ratio@78')):.2f}")


# ============================================================================ positive controls

class _Stub(nn.Module):
    """Minimal G_z stand-in. kind: 'identity' (z' = z), 'fresh' (z' = standardised fresh noise),
    'rowwise' (z' = 0.5 z + noise, no batch coupling), 'batchnorm' (z' = per-batch normalised z + noise)."""

    def __init__(self, kind: str, latent_dim: int = 6):
        super().__init__()
        self.kind, self.latent_dim, self.noise_dim = kind, latent_dim, latent_dim
        self.dummy = nn.Parameter(torch.zeros(1), requires_grad=False)

    def sample_noise(self, batch, *, device=None, dtype=None, generator=None):
        return torch.rand(batch, self.noise_dim, dtype=dtype, generator=generator).to(device or "cpu")

    def forward(self, z, r):
        e = (r - 0.5) * 12 ** 0.5
        if self.kind == "identity":
            return z.clone()
        if self.kind == "fresh":
            return e
        if self.kind == "rowwise":
            return 0.5 * z + e
        zc = (z - z.mean(0)) / (z.std(0, unbiased=False) + 1e-6)
        return 0.9 * zc + 0.3 * e


def self_test() -> None:
    from mdgan.config import Config
    from mdgan.models import build_models
    from mdgan.rollout import rollout

    # 1. the mirrored loop reproduces rollout.rollout exactly
    cfg = Config(latent_dim=6, noise_dim=4, block_len=8, n_channels=3, hidden_dim=32, burn_in=3)
    torch.manual_seed(0)
    gz, gy, _ = build_models(cfg)
    ref = rollout(gz, gy, 64, 6, cfg.burn_in, generator=torch.Generator().manual_seed(7))
    mine, lat, _ = rollout_with_latents(gz, gy, 64, 6, cfg.burn_in, mean_operation=True,
                                        generator=torch.Generator().manual_seed(7))
    assert torch.equal(ref, mine) and lat.shape == (6, 64, 6), "mirrored rollout differs from rollout.rollout"
    print("control 1: mirrored rollout equals rollout.rollout exactly")

    # 2. stripping the running buffers makes eval mode normalise with batch statistics
    torch.manual_seed(1)
    gz_rs, _, _ = build_models(Config(latent_dim=6, noise_dim=4, block_len=8, n_channels=3, hidden_dim=32,
                                      use_running_stats=True))
    gz_rs.train()
    for _ in range(5):
        gz_rs(torch.rand(64, 6) * 3, torch.rand(64, 4))                 # move the running stats away from 0/1
    z, r = torch.rand(64, 6), torch.rand(64, 4)
    with torch.no_grad():
        batch_stats = copy.deepcopy(gz_rs).train()(z, r)
        gz_rs.eval()
        running = gz_rs(z, r)
        stored = strip_running_stats(gz_rs)
        stripped = gz_rs(z, r)
    assert stored["running_mean"] is not None and not torch.allclose(running, batch_stats, atol=1e-3)
    torch.testing.assert_close(stripped, batch_stats, rtol=1e-5, atol=1e-6)
    print("control 2: after stripping, eval output equals batch-statistics output (and differed before)")

    # 3. plateau: identity gives 0, fresh noise gives 1
    z0 = torch.rand(1024, 6, generator=torch.Generator().manual_seed(0))
    p_id = plateau_curve(_Stub("identity"), z0)["plateau"]
    p_fr = plateau_curve(_Stub("fresh"), z0)["plateau"]
    assert max(p_id.values()) == 0.0, p_id
    assert all(abs(v - 1) < 0.05 for v in p_fr.values()), p_fr
    print(f"control 3: plateau identity {max(p_id.values()):.2f}, fresh noise "
          f"{min(p_fr.values()):.2f}-{max(p_fr.values()):.2f}")

    # 4. grouping: no effect without batch coupling, a real effect with it; same noise either way
    g = torch.Generator
    a = latent_path(_Stub("rowwise"), z0, 20, g().manual_seed(3), None)
    b = latent_path(_Stub("rowwise"), z0, 20, g().manual_seed(3), 64)
    assert np.array_equal(a, b), "grouping changed a model without batch coupling"
    a = latent_path(_Stub("batchnorm"), z0, 20, g().manual_seed(3), None)
    b = latent_path(_Stub("batchnorm"), z0, 20, g().manual_seed(3), 64)
    assert np.abs(a - b).max() > 1e-3, "grouping had no effect on a batch-coupled model"
    print("control 4: grouping leaves a row-wise model unchanged and changes a batch-normalised one")

    # 5. every batch-1 variant: mirrored rollout exact, pre-BN capture at the norm's input, plateau runs
    import dataclasses
    for kw in ({}, {"gz_additive_noise_std": 0.5}, {"gz_dropout": 0.2}, {"gz_norm_position": "hidden"}, {"block_len": 16}):
        c = dataclasses.replace(cfg, **kw)
        torch.manual_seed(2)
        gz_v, gy_v, _ = build_models(c)
        ref = rollout(gz_v, gy_v, 64, 5, c.burn_in, generator=torch.Generator().manual_seed(7))
        mine, _, pre = rollout_with_latents(gz_v, gy_v, 64, 5, c.burn_in, mean_operation=True,
                                            generator=torch.Generator().manual_seed(7), record_pre_bn_at=(2,))
        assert torch.equal(ref, mine), f"mirrored rollout differs for {kw}"
        width = c.hidden_dim if c.gz_norm_position == "hidden" else c.latent_dim
        assert pre[2].shape == (64, width), (kw, pre[2].shape)
        gz_v.eval()
        assert 0 < plateau_curve(gz_v, torch.rand(64, c.latent_dim))["plateau"][50] < 2
    print("control 5: all batch-1 variants roll out exactly as rollout.rollout; pre-BN capture has the norm's width")

    # 6. an invariant in one latent dimension: the pooled envelope misses it, DC and per-dimension ACF find it
    _control_invariant_detection()
    print("control 6: an invariant in 1 of 32 dimensions is missed by the pooled envelope and found by DC share "
          "and per-dimension |ACF100|; pure noise triggers neither")

    # 7. long-horizon metrics: Brownian chains vs the same chains with a per-chain velocity in the second half
    rng = torch.Generator().manual_seed(4)
    d = torch.randn(512, 2000, 3, generator=rng)
    flat = horizon_metrics(d, 1.0, 10.0, 1000, 10)
    v = torch.randn(512, 1, 3, generator=rng) * 0.3
    d_drift = d.clone()
    d_drift[:, 1000:] += v
    drift = horizon_metrics(d_drift, 1.0, 10.0, 1000, 10)
    assert abs(flat["alpha_end"] - 1) < 0.1 and flat["ballistic_share_end"] < 0.1, flat
    assert abs(flat["rho_sum_first"]) < 0.3 and abs(flat["rho_sum_last"]) < 0.3, flat
    assert drift["rho_sum_last"] > drift["rho_sum_first"] + 2 and drift["alpha_end"] > 1.3, drift
    assert drift["ballistic_share_end"] > 0.5, drift                     # defined (not nan) and large
    print(f"control 7: Brownian alpha_end {flat['alpha_end']:.2f}, ballistic share {flat['ballistic_share_end']:.2f}; "
          f"late drift Σρ {drift['rho_sum_first']:+.2f} -> {drift['rho_sum_last']:+.2f}, alpha_end "
          f"{drift['alpha_end']:.2f}, ballistic share {drift['ballistic_share_end']:.2f}")
    print("all positive controls passed")


def _control_invariant_detection() -> None:
    rng = np.random.default_rng(5)
    white = rng.normal(size=(125, 1024, 32))
    inv = white.copy()
    inv[:, :, 3] = rng.normal(size=(1, 1024)) + 0.3 * rng.normal(size=(125, 1024))   # per-chain constant + noise
    w, i = latent_metrics(white), latent_metrics(inv)
    assert w["dc_share_max_dim"] < 0.05 and w["acf100_max_dim"] < 0.1 and w["env100"] < 0.1, w
    assert i["env100"] < 0.1, f"the pooled envelope was expected to miss the invariant: {i['env100']}"
    assert i["dc_share_max_dim"] > 0.8 and i["dc_share_argmax_dim"] == 3, i
    assert i["acf100_max_dim"] > 0.8 and i["acf100_argmax_dim"] == 3, i


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("command", nargs="?", choices=["bn-batch", "chunked", "arm-eval"])
    ap.add_argument("snapshots", nargs="*")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--data", help="PE cache (.npz); required for bn-batch")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--chains", type=int, default=None, help="bn-batch rollout chains (default: config.estimation_chains)")
    ap.add_argument("--latent-chains", type=int, default=2048, help="chains used for latent ACF/spectrum (same rollout)")
    ap.add_argument("--plateau-chains", type=int, default=1024)
    ap.add_argument("--group", type=int, default=64, help="chunked: chains per normalisation group")
    ap.add_argument("--stats-of", type=int, default=0, help="bn-batch: index of the snapshot for pre-BN depth statistics")
    ap.add_argument("--check-reproduction", action="store_true", help="bn-batch: unmodified rollout of the first snapshot "
                                                                       "vs its evaluations.jsonl entry")
    ap.add_argument("--iterations", type=lambda t: [int(x) for x in t.split(",")], default=[20000, 30000, 40000, 50000],
                    help="arm-eval: checkpoints to evaluate in each run directory")
    ap.add_argument("--strip-running-stats", action="store_true",
                    help="arm-eval: evaluate running-stats checkpoints with batch statistics (as in 0b); arm label +batchstats")
    ap.add_argument("--rollout-blocks", type=int, default=None,
                    help="arm-eval: decode this many blocks instead of the config's (e.g. 125 for a 160 ns M=128 rollout); "
                         "MD comparisons use the first 80 ns, MD-free horizon metrics use all of it")
    ap.add_argument("--out", default="runs/step0")
    args = ap.parse_args(argv)
    if args.self_test:
        self_test()
    if args.command is None:
        return
    if args.command in ("bn-batch", "arm-eval") and not args.data:
        ap.error(f"{args.command} needs --data")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results = {"bn-batch": cmd_bn_batch, "chunked": cmd_chunked, "arm-eval": cmd_arm_eval}[args.command](args)
    if args.command == "bn-batch":
        summarize_b(results)
    if args.command == "arm-eval":
        summarize_arms(results)
    (out / f"{args.command}.json").write_text(json.dumps(results, indent=2, default=str))
    print(f"\nwrote {out / (args.command + '.json')}")


if __name__ == "__main__":
    main()