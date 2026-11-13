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

summarize    merge arm-eval JSON files (later files win for a repeated run/checkpoint) and print the
             per-arm table and verdicts again, with run lengths taken from the run directories:
                 python tools/step0_rollouts.py summarize runs/a/arm-eval.json runs/b/arm-eval.json --out runs/ab
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
# Batch-2 pass criteria, per checkpoint: all four must hold
PASS_RHO_SUM, PASS_DC_DIM, PASS_ACF_DIM, PASS_SHARE78 = 0.3, 0.15, 0.15, 0.2
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


def share78(a: float, b: float, t_ns: float = 78.0) -> float:
    """Ballistic share of the fitted MSD at 78 ns, b t^2 / (a t + b t^2), from the 5-78 ns fit
    MSD = a t + b t^2 (phase0_compare.fit_ballistic); 1 when a <= 0 (the fit puts everything in the
    quadratic term). Equals drift78 / (1 + drift78) wherever drift78 is defined, so share78 > 0.2 is
    the same rule as drift78 > 0.25."""
    if a <= 0:
        return 1.0
    return float(b * t_ns ** 2 / (a * t_ns + b * t_ns ** 2))


def checkpoint_passes(r: dict, var_range: tuple | None = None, max_err: float | None = None) -> bool:
    """The four drift criteria; with var_range = (lo, hi), also lo <= var/MD <= hi (a model that barely moves
    cannot pass). var_range None reproduces every table made before the range existed."""
    ok = (r["rho_sum_1_30"] <= PASS_RHO_SUM and r["dc_share_max_dim"] < PASS_DC_DIM
          and r["acf100_max_dim"] < PASS_ACF_DIM and r["share78"] < PASS_SHARE78)
    if var_range is not None:
        ok = ok and var_range[0] <= r["var_over_md"] <= var_range[1]
    if max_err is not None:
        ok = ok and r["err_percent"] <= max_err
    return bool(ok)


def latent_passes(r: dict) -> bool:
    """The two latent criteria on their own: no dimension holds an invariant (DC share) or a long memory
    (|ACF| at lag 100). Independent of the physics columns and of the variance and error options."""
    return bool(r["dc_share_max_dim"] < PASS_DC_DIM and r["acf100_max_dim"] < PASS_ACF_DIM)


def rule_text(var_range: tuple | None = None, max_err: float | None = None) -> str:
    return (f"pass = Σρ <= {PASS_RHO_SUM}, DC dim < {PASS_DC_DIM}, |ACF100| dim < {PASS_ACF_DIM}, share78 < {PASS_SHARE78}"
            + (f", var/MD in [{var_range[0]:g}, {var_range[1]:g}]" if var_range else ", no variance range")
            + (f", err <= {max_err:g}%" if max_err is not None else ", no error limit"))


def failed_criteria(r: dict, var_range: tuple | None = None,
                    max_err: float | None = None) -> list[tuple[str, float, float, bool]]:
    """(criterion, value, threshold, within 20 % of the threshold) for every criterion the row fails."""
    out = []
    for name, v, thr, bad in (("Σρ", r["rho_sum_1_30"], PASS_RHO_SUM, r["rho_sum_1_30"] > PASS_RHO_SUM),
                              ("DC dim", r["dc_share_max_dim"], PASS_DC_DIM, r["dc_share_max_dim"] >= PASS_DC_DIM),
                              ("ACF dim", r["acf100_max_dim"], PASS_ACF_DIM, r["acf100_max_dim"] >= PASS_ACF_DIM),
                              ("share78", r["share78"], PASS_SHARE78, r["share78"] >= PASS_SHARE78)):
        if bad:
            out.append((name, v, thr, v <= 1.2 * thr))
    if var_range is not None:
        lo, hi = var_range
        v = r["var_over_md"]
        if v < lo:
            out.append(("var/MD", v, lo, v >= 0.8 * lo))
        elif v > hi:
            out.append(("var/MD", v, hi, v <= 1.2 * hi))
    if max_err is not None and r["err_percent"] > max_err:
        out.append(("err%", r["err_percent"], max_err, r["err_percent"] <= 1.2 * max_err))
    return out


def is_borderline(r: dict, var_range: tuple | None = None, max_err: float | None = None) -> bool:
    """Fails exactly one criterion, and that one within 20 % of its threshold."""
    f = failed_criteria(r, var_range, max_err)
    return len(f) == 1 and f[0][3]


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
           "D_gen_m2s": float(d_gen),
           "drift78": float(b * 78.0 / a) if a > 0 else float("nan"),
           "share78": share78(a, b), "v2_nm2_per_ns2": float(b),
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


def _run_length(run_dir: Path) -> int:
    """How far a run actually got: its highest snapshot. config.json and the snapshots' stored configs
    keep the n_iterations the run STARTED with, which a resume (e.g. 50k -> 100k) does not rewrite."""
    its = [int(f.stem.split("_")[1]) for f in Path(run_dir).glob("snap_*.pt")]
    if its:
        return max(its)
    return int(json.loads((Path(run_dir) / "config.json").read_text())["n_iterations"])


def _run_record(run_dir: Path) -> dict:
    """From events.jsonl and metrics.jsonl: rollback count, the divergence stop (train.py records a
    'diverged' event before raising), and W by logged iteration."""
    def rows(name):
        f = Path(run_dir) / name
        return [json.loads(x) for x in f.read_text().splitlines() if x.strip()] if f.exists() else []
    events = rows("events.jsonl")
    stop = next((e for e in events if e.get("event") == "diverged"), None)
    return {"rollbacks": sum(e.get("event") == "rollback" for e in events),
            "diverged_at": stop["failed_at_iteration"] if stop else None,
            "W": {m["iteration"]: m.get("wasserstein") for m in rows("metrics.jsonl")}}


def _iterations_for(run_dir: Path, explicit: list[int] | None) -> list[int]:
    """Checkpoints to evaluate in one run: an explicit list, or by default 20-50k plus, for runs that
    reached 100k or more (highest snapshot), 70-100k (the late window their verdict is read on)."""
    if explicit is not None:
        return explicit
    if _run_record(run_dir)["diverged_at"] is not None:          # a stopped run: every checkpoint saved before the stop
        return sorted(int(f.stem.split("_")[1]) for f in Path(run_dir).glob("snap_*.pt"))
    return [20000, 30000, 40000, 50000] + ([70000, 80000, 90000, 100000] if _run_length(run_dir) >= 100_000 else [])


def draw_seeds(rollout_seed: int, config_seed: int) -> dict:
    """Every random draw of one complete evaluation. The config seed reproduces the original evaluation exactly
    (rollout and latent path on config.seed, plateau z0 on 0, plateau noise on 11 and 12); any other rollout
    seed s re-draws all of them, from seeds 10 s + k that no other rollout seed shares."""
    if rollout_seed == config_seed:
        return {"rollout": config_seed, "latent": config_seed, "z0": 0, "plateau": (11, 12)}
    b = 10 * rollout_seed
    return {"rollout": b + 1, "latent": b + 2, "z0": b + 3, "plateau": (b + 4, b + 5)}


def chain_indices(out: Path, n_store: int, n_chains: int, seed: int = 20261111) -> np.ndarray:
    """A fixed subset of chain indices, saved once as chain_indices.npy and reused for every rollout written
    to this output directory, so stored chains are comparable across seeds and checkpoints."""
    f = Path(out) / "chain_indices.npy"
    if f.exists():
        idx = np.load(f)
        if len(idx) != n_store or idx.max() >= n_chains:
            raise SystemExit(f"{f} holds {len(idx)} indices up to {idx.max()}; this evaluation asks for {n_store} "
                             f"of {n_chains} chains. Use a fresh --out or the same --store-chains and --chains.")
        return idx
    idx = np.sort(np.random.default_rng(seed).choice(n_chains, size=n_store, replace=False))
    f.parent.mkdir(parents=True, exist_ok=True)
    np.save(f, idx)
    return idx


def store_chains(out: Path, idx: np.ndarray, d_model: np.ndarray, latents: np.ndarray, sigma_nm: float,
                 block_len: int, meta: dict) -> Path:
    """For the stored chains: latents (n_blocks, k, n_z) and model-block displacement sums (k, n_blocks, C) in nm,
    chain-aligned (row j of both is chain idx[j]), float32, with meta."""
    k, nb = len(idx), latents.shape[0]
    sums = d_model[idx, : nb * block_len].reshape(k, nb, block_len, -1).sum(axis=2) * sigma_nm
    f = Path(out) / "chains" / f"{meta['run']}_it{meta['iteration']:06d}_rs{meta['rollout_seed']}.npz"
    f.parent.mkdir(parents=True, exist_ok=True)
    np.savez(f, latents=latents[:, idx].astype(np.float32), block_sums_nm=sums.astype(np.float32),
             chain_indices=idx, **{k_: np.asarray(v) for k_, v in meta.items()})
    return f


def _parse_rollout_seeds(text: str | None) -> list:
    if not text:
        return ["config"]
    return [t if t == "config" else int(t) for t in text.split(",")]


def cmd_arm_eval(args) -> list[dict]:
    """As-trained evaluation of each run's checkpoints at --iterations (default 20k-50k)."""
    from mdgan.checkpoint import load_checkpoint
    results, md = [], None
    snaps = [Path(r) / f"snap_{it:06d}.pt" for r in sorted(args.snapshots) for it in _iterations_for(Path(r), args.iterations)]
    missing = [str(p) for p in snaps if not p.exists()]
    if missing:
        raise SystemExit("missing snapshots:\n  " + "\n  ".join(missing))
    for i, snap in enumerate(snaps):
        ck = load_checkpoint(snap, map_location="cpu")
        cfg = ck.config
        if md is None:
            md = _md(args.data, cfg, args.block_steps)
            from mdgan.data.preprocess import SystemData
            args.d_analytic = SystemData.load(args.data).meta.get("D_exact_m2s")  # synthetic systems only; read per row
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
        for rs in _parse_rollout_seeds(args.rollout_seeds):
            rows = _evaluate_one(args, i, snap, ck, cfg, gz, gy, arm, n_blocks, ref, dt_ref, md_block,
                                 cfg.seed if rs == "config" else rs)
            results.extend(rows)
    return results


def _evaluate_one(args, i, snap, ck, cfg, gz, gy, arm, n_blocks, ref, dt_ref, md_block, rollout_seed) -> list[dict]:
    seeds = draw_seeds(rollout_seed, cfg.seed)
    n_chains = args.chains or cfg.estimation_chains
    d, lat, _ = rollout_with_latents(gz, gy, n_chains, n_blocks, cfg.burn_in, mean_operation=cfg.mean_operation,
                                     generator=torch.Generator().manual_seed(seeds["rollout"]))
    rec = _run_record(snap.parent)
    row = {"snapshot": str(snap), "run": snap.parent.name, "arm": arm, "iteration": ck.iteration,
           "rollout_seed": rollout_seed, "draw": "config" if rollout_seed == cfg.seed else "new",
           "rollbacks": rec["rollbacks"], "diverged_at": rec["diverged_at"], "W": rec["W"].get(ck.iteration),
           "n_iterations": cfg.n_iterations, "run_length": _run_length(snap.parent), "seed": cfg.seed,
           "block_len": cfg.block_len, "rollout_blocks": n_blocks}
    row.update(trajectory_metrics(d, ck.sigma_nm, cfg, ref, dt_ref, md_block, args.block_steps))
    row["block_steps"] = args.block_steps
    if getattr(args, "d_analytic", None):
        row["D_ratio_analytic"] = row["D_gen_m2s"] / args.d_analytic
    ref_steps = int(round(cfg.estimation_time_ps / (dt_ref * cfg.step_skip)))
    if args.rollout_blocks and d.shape[1] > ref_steps:      # only for an explicitly longer rollout
        row.update(horizon_metrics(d, ck.sigma_nm, dt_ref * cfg.step_skip, ref_steps, args.block_steps))
    if args.store_chains:
        idx = chain_indices(Path(args.out), args.store_chains, n_chains)
        row["chains_file"] = str(store_chains(Path(args.out), idx, d.numpy(), lat.numpy(), ck.sigma_nm, cfg.block_len,
                                              {"run": snap.parent.name, "iteration": ck.iteration,
                                               "rollout_seed": rollout_seed, "block_len": cfg.block_len,
                                               "sigma_nm": ck.sigma_nm, "arm": arm}))
    del d, lat
    if args.check_reproduction and i == 0 and rollout_seed == cfg.seed:
        rec = _evaluation_record(snap, ck.iteration)
        row["reproduction"] = {"tool_err": row["err_percent"], "tool_D_ratio": row["D_ratio"],
                               "logged_err": rec["err_percent"] if rec else None,
                               "logged_D_ratio": 1 + rec["D_rel_error"] if rec else None}
    gz.eval()
    z0 = torch.rand(args.plateau_chains, cfg.latent_dim,
                    generator=torch.Generator().manual_seed(seeds["z0"])).to(args.device)
    zpath = latent_path(gz, z0, cfg.burn_in + LATENT_TRANSITIONS - 1, torch.Generator().manual_seed(seeds["latent"]))
    row.update(latent_metrics(zpath[cfg.burn_in:]))
    row.update(plateau_curve(gz, z0, seeds=seeds["plateau"]))
    row["pass"] = checkpoint_passes(row, args.var_range, args.max_err)
    row["latent_pass"] = latent_passes(row)
    print(_fmt_a(row), flush=True)
    return [row]


def _fmt_a(r: dict) -> str:
    it = f"{r['iteration'] // 1000}k" if r["iteration"] >= 1000 else str(r["iteration"])
    tag = f" [{r['arm'].split('+', 1)[1]}]" if "+" in r["arm"] else ""
    s = (f"{r['run']} @{it}{tag}: {'PASS' if r['pass'] else 'fail'} (latent {'PASS' if latent_passes(r) else 'fail'})  "
         f"Σρ {r['rho_sum_1_30']:+.2f} "
         f"(1-3 {r['rho_sum_1_3']:+.2f} | 4-30 {r['rho_sum_4_30']:+.2f})  env100 {r['env100']:.3f}  "
         f"DC dim {r['dc_share_max_dim']:.2f}  |ACF100| dim {r['acf100_max_dim']:.2f}  "
         f"drift78 {r['drift78']:+.3f}  share78 {r['share78']:.2f}  v2 {r['v2_nm2_per_ns2']:+.4f}  "
         f"var/MD {r['var_over_md']:.2f}  D {r['D_ratio']:.2f}x"
         + (f" (D/D_analytic {r['D_ratio_analytic']:.2f})" if "D_ratio_analytic" in r else "")
         + f"  err {r['err_percent']:.1f}%  "
         f"period {r['period_blocks']:.2f}  cycle {r['cycle_share'] * 100:.0f}%"
         + (f"  W {r['W']:.2f}" if r.get("W") is not None else "") + f"  rollbacks {r.get('rollbacks', 0)}"
         + (f"  [run DIVERGED at {r['diverged_at']}]" if r.get("diverged_at") is not None else ""))
    if "horizon_ns" in r:
        s += (f"\n   horizon {r['horizon_ns']:.0f} ns: Σρ first 80 ns {r['rho_sum_first']:+.2f} | last 80 ns "
              f"{r['rho_sum_last']:+.2f}  alpha_end {r['alpha_end']:.2f}  ballistic share at end {r['ballistic_share_end']:.2f}")
    if "reproduction" in r:
        p = r["reproduction"]
        s += (f"\n   reproduction: tool err {p['tool_err']:.2f}% / D {p['tool_D_ratio']:.3f}x vs logged "
              + (f"{p['logged_err']:.2f}% / {p['logged_D_ratio']:.3f}x" if p["logged_err"] is not None else "n/a"))
    return s


def _late_window(n_iterations: int, step: int = 10_000) -> list[int]:
    return [n_iterations - k * step for k in (3, 2, 1, 0)]


def summarize_arms(results: list[dict], var_range: tuple | None = None, max_err: float | None = None) -> None:
    """Tables and verdicts. Every row's pass flag is recomputed under var_range, so one JSON file can be read
    with and without the variance range. Medians and verdicts use one row per checkpoint, the config-seed draw
    (or the lowest rollout seed if that one is absent); extra rollout seeds appear in the repeatability section."""
    for r in results:
        r["pass"] = checkpoint_passes(r, var_range, max_err)
        r["latent_pass"] = latent_passes(r)
    primary: dict = {}
    for r in sorted(results, key=lambda x: (x.get("draw", "config") != "config", str(x.get("rollout_seed", "")))):
        primary.setdefault((r["run"], r["arm"], r["iteration"]), r)
    results = list(primary.values())
    arms = sorted({r["arm"] for r in results}, key=lambda a: (a != "baseline", a))
    print(f"\n{'arm':<26} {'n':>3} {'Σρ':>6} {'1-3':>6} {'4-30':>6} {'env100':>6} {'DC dim':>6} {'ACF dim':>7} "
          f"{'drift78':>8} {'share78':>7} {'v2':>8} {'var/MD':>6} {'D':>5} {'err%':>6} {'D>MD':>5} {'pass':>5} {'latent':>6}  periods")
    for arm in arms:
        rr = [r for r in results if r["arm"] == arm]

        def med(k, rr=rr):
            return float(np.nanmedian([x[k] for x in rr]))

        periods = sorted({round(x["period_blocks"], 1) for x in rr})
        print(f"{arm:<26} {len(rr):>3} {med('rho_sum_1_30'):>+6.2f} {med('rho_sum_1_3'):>+6.2f} {med('rho_sum_4_30'):>+6.2f} "
              f"{med('env100'):>6.3f} {med('dc_share_max_dim'):>6.2f} {med('acf100_max_dim'):>7.2f} {med('drift78'):>+8.3f} "
              f"{med('share78'):>7.2f} {med('v2_nm2_per_ns2'):>+8.4f} {med('var_over_md'):>6.2f} {med('D_ratio'):>5.2f} "
              f"{med('err_percent'):>6.1f} {np.mean([x['D_ratio'] > 1 for x in rr]) * 100:>4.0f}% "
              f"{np.mean([x['pass'] for x in rr]) * 100:>4.0f}% {np.mean([x['latent_pass'] for x in rr]) * 100:>5.0f}%  "
              + ", ".join(f"{p:g}" for p in periods))
    rule = rule_text(var_range, max_err)
    print(f"medians over all evaluated checkpoints; {rule}")

    print(f"\nverdicts under: {rule}")
    print("(each run judged on its last four checkpoints: 20-50k for 50k runs, 70-100k for 100k runs;"
          " a seed passes with >= 3 of 4; arm pass = all seeds, promising = majority, closed = otherwise)")
    for arm in arms:
        rr = [r for r in results if r["arm"] == arm]
        seeds, lines = [], []
        for run in sorted({r["run"] for r in rr}):
            mine = {r["iteration"]: r for r in rr if r["run"] == run}
            stop = next((x["diverged_at"] for x in mine.values() if x.get("diverged_at") is not None), None)
            if stop is not None:                                   # a divergence is a result: report it, count as not passing
                n_rb = max(x.get("rollbacks", 0) for x in mine.values())
                lines.append(f"    {run:<32} DIVERGED at iteration {stop} after {n_rb} rollbacks; "
                             f"{sum(x['pass'] for x in mine.values())}/{len(mine)} of its saved checkpoints pass "
                             f"({min(mine) // 1000}-{max(mine) // 1000}k)")
                seeds.append(False)
                continue
            first = next(iter(mine.values()))
            n_it = first.get("run_length", first["n_iterations"])
            late = _late_window(n_it)
            have = [it for it in late if it in mine]
            ok = sum(mine[it]["pass"] for it in have)
            line = (f"    {run:<32} {n_it // 1000}k run: late {late[0] // 1000}-{late[-1] // 1000}k {ok}/{len(have)} pass"
                    f" (latent {sum(mine[it]['latent_pass'] for it in have)}/{len(have)})")
            if n_it > 50_000:
                early = [it for it in _late_window(50_000) if it in mine]
                if early:
                    line += f"   (20-50k: {sum(mine[it]['pass'] for it in early)}/{len(early)})"
            if len(have) < 4:
                line += "   INCOMPLETE: late checkpoints missing from this evaluation"
                seeds.append(None)
            else:
                seeds.append(ok >= 3)
            lines.append(line)
        decided = [x for x in seeds if x is not None]
        if len(decided) < len(seeds) or not decided:
            verdict = "incomplete"
        elif all(decided):
            verdict = "PASS"
        elif sum(decided) > len(decided) / 2:
            verdict = "promising"
        else:
            verdict = "closed"
        print(f"  {arm}: {verdict} ({sum(1 for x in decided if x)}/{len(seeds)} seeds pass)")
        print("\n".join(lines))


def report_repeatability(rows: list[dict], var_range: tuple | None = None, max_err: float | None = None) -> None:
    """For checkpoints evaluated under several rollout seeds: the pass flag and the deciding columns per seed."""
    by_ckpt: dict = {}
    for r in rows:
        by_ckpt.setdefault((r["run"], r["arm"], r["iteration"]), []).append(r)
    multi = {k: v for k, v in by_ckpt.items() if len({x.get("rollout_seed") for x in v}) > 1}
    if not multi:
        return
    print("\nrepeatability (one line per rollout seed; config = the original evaluation draw)")
    for (run, _arm, it), rr in sorted(multi.items()):
        flags = [checkpoint_passes(x, var_range, max_err) for x in rr]
        verdict = "all PASS" if all(flags) else "all fail" if not any(flags) else "SPLIT"
        print(f"  {run} @{it // 1000}k: {verdict}")
        for x in sorted(rr, key=lambda x: (x.get("draw") != "config", str(x.get("rollout_seed")))):
            tag = "config" if x.get("draw", "config") == "config" else str(x.get("rollout_seed"))
            print(f"    seed {tag:<6} {'PASS' if checkpoint_passes(x, var_range, max_err) else 'fail':<4} "
                  f"(latent {'PASS' if latent_passes(x) else 'fail'})  Σρ {x['rho_sum_1_30']:+.2f}  "
                  f"DC dim {x['dc_share_max_dim']:.2f}  ACF dim {x['acf100_max_dim']:.2f}  share78 {x['share78']:.2f}  "
                  f"var/MD {x['var_over_md']:.2f}  D {x['D_ratio']:.2f}x  err {x['err_percent']:.1f}%")


def report_borderline(rows: list[dict], var_range: tuple | None = None, max_err: float | None = None) -> None:
    """Checkpoints failing exactly one criterion, within 20 % of its threshold (candidates for extra rollout seeds)."""
    hits = [r for r in rows if r.get("draw", "config") == "config" and is_borderline(r, var_range, max_err)]
    if not hits:
        return
    print("\nborderline (fails one criterion, within 20 % of its threshold):")
    for r in sorted(hits, key=lambda r: (r["run"], r["iteration"])):
        name, v, thr, _ = failed_criteria(r, var_range, max_err)[0]
        print(f"  {r['run']} @{r['iteration'] // 1000}k: {name} {v:.3f} vs threshold {thr:g}")


def merge_results(paths: list[str]) -> list[dict]:
    """Rows from several arm-eval JSON files; for a repeated (run, arm, iteration) the later file wins.
    run_length is recomputed from each run directory's snapshots, so rows written before a resume
    finished are judged on the run as it now stands."""
    merged: dict = {}
    for path in paths:
        for r in json.loads(Path(path).read_text()):
            merged[(r["run"], r["arm"], r["iteration"], r.get("rollout_seed", r.get("seed")))] = r
    rows = list(merged.values())
    for r in rows:
        run_dir = Path(r["snapshot"]).parent
        if run_dir.is_dir():
            r["run_length"] = _run_length(run_dir)
        r.setdefault("draw", "config")
    return sorted(rows, key=lambda r: (r["arm"], r["run"], r["iteration"], str(r.get("rollout_seed", ""))))


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
    # 8. a resumed run (config says 50k, snapshots to 100k) is judged on 70-100k, also after merging files
    _control_resumed_run_window()
    print("control 8: a resumed run counts as 100k (highest snapshot), and merged rows are judged on 70-100k")

    # 9. a diverged run: every saved checkpoint evaluated, the stop reported, the seed counted as not passing
    _control_diverged_run()
    print("control 9: a diverged run is evaluated on all its saved checkpoints and reported as DIVERGED")

    _control_variance_range()
    print("control 10: the false passes (adam_default s0 @40k, two collapsed running_stats rows) pass without the "
          "variance range and fail with it; a genuine pass passes both ways")
    _control_draw_seeds()
    print("control 11: the config seed reproduces the original draws; new rollout seeds use disjoint ones")
    _control_chain_storage()
    print("control 12: chain indices are fixed and reused; stored latents and block sums are chain-aligned and exact")
    _control_merge_repeatability()
    print("control 13: merged rows keep every rollout seed; verdicts use the config draw; splits and borderline rows are reported")
    _control_err_and_latent()
    print("control 14: --max-err fails a row on error alone (and flags it borderline within 20 %); latent_pass is "
          "independent of the physics columns")
    print("all positive controls passed")


def _row(rho, dc, acf, share, var, **kw) -> dict:
    return {"rho_sum_1_30": rho, "dc_share_max_dim": dc, "acf100_max_dim": acf, "share78": share, "var_over_md": var, **kw}


def _control_variance_range() -> None:
    false_passes = {"adam_default s0 @40k": _row(-0.08, 0.02, 0.01, -0.03, 0.12),
                    "running_stats c7810278 s4 @20k (as trained)": _row(-0.00, 0.02, 0.02, -0.00, 0.10),
                    "running_stats c7810278 s5 @20k (as trained)": _row(-0.00, 0.02, 0.01, 0.00, 0.11)}
    genuine = _row(-0.17, 0.02, 0.02, 0.09, 0.65)                       # m128 c7675778 s2 @40k
    for name, r in false_passes.items():
        assert checkpoint_passes(r), f"{name} should pass without the range (the earlier tables)"
        assert not checkpoint_passes(r, (0.4, 1.5)), f"{name} should fail with the range"
    assert checkpoint_passes(genuine) and checkpoint_passes(genuine, (0.4, 1.5))
    assert not checkpoint_passes(_row(0.0, 0.0, 0.0, 0.0, 1.6), (0.4, 1.5))   # the upper bound too


def _control_draw_seeds() -> None:
    assert draw_seeds(3, 3) == {"rollout": 3, "latent": 3, "z0": 0, "plateau": (11, 12)}
    used = [3, 0, 11, 12]
    for s in (1001, 1002):
        d = draw_seeds(s, 3)
        mine = [d["rollout"], d["latent"], d["z0"], *d["plateau"]]
        assert len(set(mine)) == 5 and not set(mine) & set(used), (s, mine, used)
        used += mine


def _control_chain_storage() -> None:
    import tempfile
    rng = np.random.default_rng(1)
    n, nb, m, c, nz = 50, 6, 4, 3, 5
    d = rng.normal(size=(n, nb * m, c)).astype(np.float32)
    lat = rng.normal(size=(nb, n, nz)).astype(np.float32)
    with tempfile.TemporaryDirectory() as tmp:
        idx = chain_indices(Path(tmp), 8, n)
        assert np.array_equal(idx, chain_indices(Path(tmp), 8, n)) and len(set(idx.tolist())) == 8
        try:
            chain_indices(Path(tmp), 9, n)
            raise AssertionError("a different chain count should be refused")
        except SystemExit:
            pass
        f = store_chains(Path(tmp), idx, d, lat, 0.17, m, {"run": "run_x", "iteration": 40000, "rollout_seed": 1001,
                                                            "block_len": m, "sigma_nm": 0.17, "arm": "x"})
        z = np.load(f)
        assert z["latents"].shape == (nb, 8, nz) and z["block_sums_nm"].shape == (8, nb, c)
        j, k = 3, 2                                                     # chain idx[j], block k
        assert np.allclose(z["block_sums_nm"][j, k], d[idx[j], k * m:(k + 1) * m].sum(0) * 0.17, atol=1e-6)
        assert np.array_equal(z["latents"][:, j], lat[:, idx[j]]) and np.array_equal(z["chain_indices"], idx)


def _control_merge_repeatability() -> None:
    import contextlib
    import io
    import tempfile
    base = dict(rho_sum_1_3=0.0, rho_sum_4_30=0.0, env100=0.01, drift78=0.0, v2_nm2_per_ns2=0.0, D_ratio=1.0,
                err_percent=10.0, period_blocks=2.0, cycle_share=0.1, arm="m128", run="run_m128_c1_seed5",
                n_iterations=100000, rollbacks=0, diverged_at=None, seed=5)
    with tempfile.TemporaryDirectory() as tmp:
        snap = str(Path(tmp) / "run_m128_c1_seed5" / "snap_080000.pt")
        good, bad = _row(0.1, 0.02, 0.02, 0.05, 0.6), _row(0.9, 0.02, 0.02, 0.05, 0.6)
        a = [{**base, **bad, "iteration": 80000, "snapshot": snap}]                       # an old row: no rollout_seed
        b = [{**base, **good, "iteration": 80000, "snapshot": snap, "rollout_seed": 1001, "draw": "new"},
             {**base, **_row(0.33, 0.02, 0.02, 0.05, 0.6), "iteration": 90000, "snapshot": snap,
              "rollout_seed": 5, "draw": "config"}]
        fa, fb = Path(tmp) / "a.json", Path(tmp) / "b.json"
        fa.write_text(json.dumps(a))
        fb.write_text(json.dumps(b))
        rows = merge_results([str(fa), str(fb)])
        assert len(rows) == 3, rows                                   # both seeds of @80k kept
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            summarize_arms(rows)
            report_repeatability(rows)
            report_borderline(rows)
        out = buf.getvalue()
        assert "run_m128_c1_seed5 @80k: SPLIT" in out, out
        assert "run_m128_c1_seed5 @90k: Σρ 0.330 vs threshold 0.3" in out, out            # one criterion, within 20 %
        import re
        assert re.search(r"^m128\s+2\s", out, re.M), out                                  # 2 checkpoints, not 3 rows


def _control_err_and_latent() -> None:
    base = _row(0.1, 0.02, 0.02, 0.05, 0.6)
    high_err, near_err, low_err = {**base, "err_percent": 55.0}, {**base, "err_percent": 45.0}, {**base, "err_percent": 30.0}
    assert checkpoint_passes(high_err, (0.4, 1.5)) and not checkpoint_passes(high_err, (0.4, 1.5), 40.0)
    assert checkpoint_passes(low_err, (0.4, 1.5), 40.0)
    assert is_borderline(near_err, (0.4, 1.5), 40.0) and not is_borderline(high_err, (0.4, 1.5), 40.0)
    assert "err <= 40%" in rule_text((0.4, 1.5), 40.0) and "no error limit" in rule_text()
    clean_latent_bad_physics = {**_row(2.0, 0.02, 0.02, 0.9, 0.6), "err_percent": 80.0}
    invariant = {**_row(0.1, 0.50, 0.02, 0.05, 0.6), "err_percent": 20.0}
    assert latent_passes(clean_latent_bad_physics) and not checkpoint_passes(clean_latent_bad_physics)
    assert not latent_passes(invariant) and not checkpoint_passes(invariant)


def _control_diverged_run() -> None:
    import contextlib
    import io
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        r = Path(d) / "run_adam_default_c1_seed0"
        r.mkdir()
        (r / "config.json").write_text(json.dumps({"n_iterations": 50000}))
        for it in (10000, 20000, 30000):
            (r / f"snap_{it:06d}.pt").touch()
        ev = [{"iteration": 30000, "event": "rollback", "failed_at_iteration": 31417, "to_iteration": 30000, "new_seed": 1000}] * 3
        ev.append({"iteration": 30000, "event": "diverged", "failed_at_iteration": 31417, "rollbacks": 3})
        (r / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in ev))
        (r / "metrics.jsonl").write_text(json.dumps({"iteration": 30000, "wasserstein": 4.2}) + "\n")
        rec = _run_record(r)
        assert rec["rollbacks"] == 3 and rec["diverged_at"] == 31417 and rec["W"][30000] == 4.2, rec
        assert _iterations_for(r, None) == [10000, 20000, 30000]
        base = dict(rho_sum_1_3=0.0, rho_sum_4_30=0.1, env100=0.01, dc_share_max_dim=0.02, acf100_max_dim=0.03,
                    drift78=0.05, share78=0.05, v2_nm2_per_ns2=0.0, var_over_md=0.6, D_ratio=1.0, err_percent=10.0,
                    period_blocks=2.0, cycle_share=0.1, rho_sum_1_30=0.1, arm="adam_default", run=r.name,
                    n_iterations=50000, run_length=30000, rollbacks=3, diverged_at=31417, pass_=None)
        rows = [{**base, "iteration": it, "pass": True} for it in (10000, 20000, 30000)]
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            summarize_arms(rows)
        out = buf.getvalue()
        assert "DIVERGED at iteration 31417 after 3 rollbacks; 3/3 of its saved checkpoints pass (10-30k)" in out, out
        assert "adam_default: closed (0/1 seeds pass)" in out, out


def _control_resumed_run_window() -> None:
    import contextlib
    import io
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        r = Path(d) / "run_m128_c1_seed0"
        r.mkdir()
        (r / "config.json").write_text(json.dumps({"n_iterations": 50000}))
        for it in range(10000, 100001, 10000):
            (r / f"snap_{it:06d}.pt").touch()
        assert _run_length(r) == 100000, _run_length(r)
        assert _iterations_for(r, None) == [20000, 30000, 40000, 50000, 70000, 80000, 90000, 100000]
        base = dict(rho_sum_1_3=0.0, rho_sum_4_30=0.1, env100=0.01, dc_share_max_dim=0.02, acf100_max_dim=0.03,
                    drift78=0.05, share78=0.05, v2_nm2_per_ns2=0.0, var_over_md=0.6, D_ratio=1.0, err_percent=10.0,
                    period_blocks=2.0, cycle_share=0.1, arm="m128", run=r.name)
        early = [{**base, "rho_sum_1_30": 2.0, "iteration": it, "n_iterations": 50000, "snapshot": str(r / f"snap_{it:06d}.pt")}
                 for it in (20000, 30000, 40000, 50000)]                       # evaluated before the resume: fail
        late = [{**base, "rho_sum_1_30": 0.1, "iteration": it, "n_iterations": 100000, "snapshot": str(r / f"snap_{it:06d}.pt")}
                for it in (70000, 80000, 90000, 100000)]                       # evaluated after: pass
        a, b = Path(d) / "a.json", Path(d) / "b.json"
        a.write_text(json.dumps(early))
        b.write_text(json.dumps(late))
        rows = merge_results([str(a), str(b)])
        assert all(x["run_length"] == 100000 for x in rows), rows
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            summarize_arms(rows)
        out = buf.getvalue()
        assert "100k run: late 70-100k 4/4 pass" in out and "(20-50k: 0/4)" in out and "m128: PASS" in out, out


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
    ap.add_argument("command", nargs="?", choices=["bn-batch", "chunked", "arm-eval", "summarize"])
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
    ap.add_argument("--iterations", type=lambda t: [int(x) for x in t.split(",")], default=None,
                    help="arm-eval: checkpoints to evaluate in each run directory (default: 20-50k, plus 70-100k "
                         "for runs configured to 100k)")
    ap.add_argument("--strip-running-stats", action="store_true",
                    help="arm-eval: evaluate running-stats checkpoints with batch statistics (as in 0b); arm label +batchstats")
    ap.add_argument("--rollout-seeds", default=None,
                    help="arm-eval: comma list of rollout seeds, 'config' = the original evaluation draw "
                         "(e.g. config,1001,1002); default: config only")
    ap.add_argument("--var-range", type=lambda t: tuple(float(x) for x in t.split(",")), default=None,
                    help="arm-eval / summarize: also require var/MD within lo,hi to pass (e.g. 0.4,1.5); off by default")
    ap.add_argument("--block-steps", type=int, default=EVAL_BLOCK_STEPS,
                    help="arm-eval: block length for var/MD and the Σρ columns (default 64 = 0.64 ns, as every earlier "
                         "table; 8 for the miniature test bed)")
    ap.add_argument("--max-err", type=float, default=None,
                    help="arm-eval / summarize: also require err <= this many percent to pass (e.g. 40); off by default")
    ap.add_argument("--store-chains", type=int, default=0,
                    help="arm-eval: keep latents and model-block sums of this many fixed chains per rollout (in OUT/chains)")
    ap.add_argument("--rollout-blocks", type=int, default=None,
                    help="arm-eval: decode this many blocks instead of the config's (e.g. 125 for a 160 ns M=128 rollout); "
                         "MD comparisons use the first 80 ns, MD-free horizon metrics use all of it")
    ap.add_argument("--out", default="runs/step0")
    args = ap.parse_args(argv)
    if args.self_test:
        self_test()
    if args.command is None:
        return
    if args.command == "summarize":
        if not args.snapshots:
            ap.error("summarize needs one or more arm-eval JSON files")
        rows = merge_results(args.snapshots)
        summarize_arms(rows, args.var_range, args.max_err)
        report_repeatability(rows, args.var_range, args.max_err)
        report_borderline(rows, args.var_range, args.max_err)
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "arm-eval.json").write_text(json.dumps(rows, indent=2, default=str))
        print(f"\n{len(rows)} rows from {len(args.snapshots)} files; wrote {out / 'arm-eval.json'}")
        return
    if args.command in ("bn-batch", "arm-eval") and not args.data:
        ap.error(f"{args.command} needs --data")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results = {"bn-batch": cmd_bn_batch, "chunked": cmd_chunked, "arm-eval": cmd_arm_eval}[args.command](args)
    if args.command == "bn-batch":
        summarize_b(results)
    if args.command == "arm-eval":
        summarize_arms(results, args.var_range, args.max_err)
        report_repeatability(results, args.var_range, args.max_err)
        report_borderline(results, args.var_range, args.max_err)
    (out / f"{args.command}.json").write_text(json.dumps(results, indent=2, default=str))
    print(f"\nwrote {out / (args.command + '.json')}")


if __name__ == "__main__":
    main()
