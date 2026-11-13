"""Decompose fail/pass checkpoint pairs: where does the displacement correlation come from?

For each pair (two checkpoints of one run), four measurements, written as one figure and one JSON per pair:

6a  MSD decomposition on 64-step blocks. V = block variance, rho_k (k = 1..60) the pooled correlation of block sums
    k apart (exactly diag_runs._block_stats, extended), F(n) = 1 + 2 sum_{k<n} (1 - k/n) rho_k, and
    c = the OLS slope of n F(n) over n = 16..31, parallel to how D is fitted (10-20 ns). Then D_pred = V c / (2 tau)
    (tau = one block's duration; V per axis). Computed for the checkpoint's config-seed rollout (all chains,
    re-run: it reproduces the stored rollout exactly) and for the MD. Reported: V/V_MD, c/c_MD, their product
    against the measured D/D_MD, and the closure per side (D_pred / D_measured - 1), each with a bootstrap
    standard error (chains for the model, trajectories for the MD). The MD side has only 60 trajectories, so
    its closure scatters by a few percent on its own; it is the same for every checkpoint. If the product
    misses by more than 10 %: V and rho_1 in the first versus second half of the rollout (non-stationarity),
    and rho_31..60 (correlation beyond the lag-30 sum).
6b  Decoder bias with latents fixed. Each stored latent z_ik (chain i, block k) decoded 32 times with fresh noise,
    the mean operation applied across the stored chains for each decode, the net block displacements averaged:
    mu_ik. b_i = mean_k mu_ik; (ii) var_i(b_i) / var_ik(mu_ik), pooled over axes (no persistence: about
    1/n_blocks); (iii) the correlation across chains of b_i with the chain's measured net drift over the rollout.
6c  Late-start branching: from the stored latent at block 30, two continuations with independent G_z noise,
    d_same / d_indep at steps 1, 5, 10, 20, 50 after the branch, beside the z0-start plateau of arm-eval.
6d  BatchNorm numerics at block 30: pre-BN output variance per dimension over the stored chains (one transition
    from z_30 with fresh noise), its ratio to the BatchNorm eps, gamma and beta; and the same variance from a
    64-chain subset, so the batch-size independence is on record for this quantity.

Inputs: snapshots (runs/chtc/<run>/snap_*.pt), the repeatability session's stored chains
(runs/<rep>/chains/<run>_it<iteration>_rs<seed>.npz) and its arm-eval rows (labels), and the PE cache.

Usage (repository root):
    python tools/decompose_pairs.py --self-test
    python tools/decompose_pairs.py --data data/cache/PE.npz --device mps --out runs/decompose
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

BLOCK = 64
FIT_N = tuple(range(16, 32))           # n blocks over which c is fitted: 10.24-19.84 ns, the D fit window
KMAX = 60
CLOSURE_TOL = 0.10
DECODES = 32
BRANCH_AT = 30
LATE_STEPS = (1, 5, 10, 20, 50)
VAR_RANGE, MAX_ERR = (0.4, 1.5), 40.0
NM2_PER_PS_TO_M2_PER_S = 1e-6
PAIRS = (("run_m128_c7675778_seed0", 70000, 80000, "rep_m128s0"),
         ("run_m128_c7810278_seed5", 80000, 90000, "rep_m128s5"),
         ("run_wd01_c7675778_seed2", 40000, 50000, "rep_wd01s2"))
NEW_SEEDS = (1001, 1002)


def _load_tool(name: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ============================================================================ 6a: block decomposition (numpy)

def block_sums(d: np.ndarray, block: int = BLOCK) -> np.ndarray:
    """(chains, steps, axes) -> (chains, n_blocks, axes), float64."""
    n, t, c = d.shape
    nb = t // block
    return d[:, :nb * block].reshape(n, nb, block, c).sum(axis=2, dtype=np.float64)


class BlockMoments:
    """Per-chain sufficient statistics of the block sums, so V and every rho_k (exactly as
    diag_runs._block_stats: V = across-chain-and-axis variance per block, averaged over blocks; rho_k = Pearson
    correlation of all block pairs k apart, pooled over chains and axes) can be recomputed for any chain weights,
    which makes the bootstrap cheap."""

    def __init__(self, b: np.ndarray, kmax: int = KMAX):
        n, nb, c = b.shape
        self.n, self.nb, self.c, self.b = n, nb, c, b
        self.s1, self.s2 = b.sum(axis=2), (b ** 2).sum(axis=2)                     # (n, nb)
        self.lags = [k for k in range(1, kmax + 1) if k < nb]
        self.lag = {}
        for k in self.lags:
            x, y = b[:, :-k], b[:, k:]
            self.lag[k] = np.stack([x.sum(axis=(1, 2)), y.sum(axis=(1, 2)), (x * x).sum(axis=(1, 2)),
                                    (y * y).sum(axis=(1, 2)), (x * y).sum(axis=(1, 2))], axis=1)   # (n, 5)

    def stats(self, w: np.ndarray | None = None, blocks: slice | None = None) -> dict:
        w = np.ones(self.n) if w is None else w
        tot = w.sum()
        s1, s2 = (self.s1, self.s2) if blocks is None else (self.s1[:, blocks], self.s2[:, blocks])
        m1, m2 = (w @ s1) / (tot * self.c), (w @ s2) / (tot * self.c)
        rho = {}
        for k in self.lags:
            sx, sy, sxx, syy, sxy = w @ self.lag[k]
            m = tot * (self.nb - k) * self.c
            cov = sxy / m - (sx / m) * (sy / m)
            rho[k] = float(cov / np.sqrt((sxx / m - (sx / m) ** 2) * (syy / m - (sy / m) ** 2)))
        return {"V": float((m2 - m1 ** 2).mean()), "rho": rho}

    def rho1_range(self, lo: int, hi: int) -> float:
        """rho_1 among blocks lo..hi-1 only (the half-split diagnostic)."""
        if hi - lo < 2:
            return float("nan")
        x = self.b[:, lo:hi]
        return float(np.corrcoef(x[:, :-1].ravel(), x[:, 1:].ravel())[0, 1])


def corr_factor(rho: dict, n: int) -> float:
    return 1.0 + 2.0 * sum((1 - k / n) * rho.get(k, 0.0) for k in range(1, n))


def corr_slope(rho: dict, ns=FIT_N) -> float:
    """c = OLS slope of n F(n) over n = 16..31."""
    x = np.array(ns, float)
    return float(np.polyfit(x, np.array([n * corr_factor(rho, n) for n in ns]), 1)[0])


def d_pred_m2s(V: float, c: float, tau_ps: float) -> float:
    """MSD_3D(n blocks) = 3 n V F(n) (V per axis), t = n tau, D = slope / 6 = V c / (2 tau)."""
    return V * c / (2.0 * tau_ps) * NM2_PER_PS_TO_M2_PER_S


def ols_d_m2s(curve_window: np.ndarray, t_ps: np.ndarray) -> float:
    """D from an MSD over the fit window, as mdgan.data.msd.diffusion_coefficient (OLS with intercept, / 6)."""
    return float(np.polyfit(t_ps, curve_window, 1)[0] / 6.0 * NM2_PER_PS_TO_M2_PER_S)


def fit_window(dt_ps: float, t_min_ps: float = 10_000.0, t_max_ps: float = 20_000.0) -> np.ndarray:
    """MSD indices t (time t dt) inside the inclusive fit window, as diffusion_coefficient selects them."""
    t = np.arange(int(round(t_max_ps / dt_ps)) + 1)
    return t[(t * dt_ps >= t_min_ps) & (t * dt_ps <= t_max_ps)]


def from_origin_sq(d: np.ndarray, idx: np.ndarray, chunk: int = 512) -> np.ndarray:
    """Per-chain |r(t)|^2 at MSD indices idx (r(0) = 0, r(t) = sum of the first t steps): (chains, len(idx)).
    Its mean over chains is msd_ensemble at those indices."""
    out = np.empty((d.shape[0], len(idx)))
    for a in range(0, d.shape[0], chunk):
        r = np.cumsum(d[a:a + chunk, :idx.max()].astype(np.float64), axis=1)
        out[a:a + chunk] = (r[:, idx - 1] ** 2).sum(axis=2)
    return out


class Side:
    """One side of the decomposition (a model rollout or the MD): block moments plus the per-chain (or
    per-trajectory) MSD curves over the fit window, from which D is measured as the evaluation measures it."""

    def __init__(self, d_nm: np.ndarray, dt_ps: float, curves_window: np.ndarray, kind: str):
        self.kind, self.dt, self.tau = kind, dt_ps, BLOCK * dt_ps
        self.mom = BlockMoments(block_sums(d_nm))
        self.idx = fit_window(dt_ps)
        self.t_ps = self.idx * dt_ps
        self.curves = curves_window                                                # (chains, len(idx))

    def numbers(self, w: np.ndarray | None = None) -> dict:
        st = self.mom.stats(w)
        w_ = np.ones(self.mom.n) if w is None else w
        d_meas = ols_d_m2s((w_ @ self.curves) / w_.sum(), self.t_ps)
        c = corr_slope(st["rho"])
        dp = d_pred_m2s(st["V"], c, self.tau)
        return {"V": st["V"], "c": c, "D_pred": dp, "D_measured": d_meas, "closure": dp / d_meas - 1, "rho": st["rho"]}


def model_side(d_nm: np.ndarray, dt_ps: float) -> Side:
    idx = fit_window(dt_ps)
    return Side(d_nm, dt_ps, from_origin_sq(d_nm, idx), "model")


def md_side(d_nm: np.ndarray, dt_ps: float) -> Side:
    from mdgan.data.msd import msd_time_averaged
    _, per = msd_time_averaged(d_nm)
    return Side(d_nm, dt_ps, per[:, fit_window(dt_ps)], "md")


def decompose(model: Side, md: Side, n_boot: int = 200, seed: int = 0) -> dict:
    """Ratios, closures and bootstrap standard errors; on a product miss beyond 10 %, the diagnostics."""
    m, r = model.numbers(), md.numbers()
    out = {"V_ratio": m["V"] / r["V"], "c_ratio": m["c"] / r["c"], "product": (m["V"] / r["V"]) * (m["c"] / r["c"]),
           "D_ratio": m["D_measured"] / r["D_measured"],
           "closure_model": m["closure"], "closure_md": r["closure"]}
    out["closure_product"] = out["product"] / out["D_ratio"] - 1
    rng = np.random.default_rng(seed)
    cm, cr, cp = [], [], []
    for _ in range(n_boot):
        wm = np.bincount(rng.integers(0, model.mom.n, model.mom.n), minlength=model.mom.n).astype(float)
        wr = np.bincount(rng.integers(0, md.mom.n, md.mom.n), minlength=md.mom.n).astype(float)
        a, b = model.numbers(wm)["closure"], md.numbers(wr)["closure"]
        cm.append(a)
        cr.append(b)
        cp.append((1 + a) / (1 + b) - 1)
    out.update({"se_closure_model": float(np.std(cm)), "se_closure_md": float(np.std(cr)),
                "se_closure_product": float(np.std(cp)), "n_boot": n_boot,
                "closes": bool(abs(out["closure_product"]) <= CLOSURE_TOL),
                "model": {k: v for k, v in m.items() if k != "rho"}, "md": {k: v for k, v in r.items() if k != "rho"},
                "rho_model": m["rho"], "rho_md": r["rho"]})
    if not out["closes"]:
        half = model.mom.nb // 2
        h1, h2 = model.mom.stats(blocks=slice(0, half)), model.mom.stats(blocks=slice(half, None))
        out["diagnostics"] = {
            "V_first_half": h1["V"], "V_second_half": h2["V"],
            "rho1_first_half": model.mom.rho1_range(0, half), "rho1_second_half": model.mom.rho1_range(half, model.mom.nb),
            "rho_31_60_model": {k: v for k, v in m["rho"].items() if 31 <= k <= 60},
            "rho_31_60_md": {k: v for k, v in r["rho"].items() if 31 <= k <= 60}}
    return out


# ============================================================================ 6b: decoder bias (torch)

def expected_block_displacement(gy, latents: torch.Tensor, sigma_nm: float, n_decodes: int, seed: int,
                                mean_operation: bool = True, chunk_blocks: int = 8) -> np.ndarray:
    """mu_ik (chains, blocks, axes) in nm: each latent decoded n_decodes times with fresh (r', u), the mean
    operation applied across the chains for each block of each decode, net block displacements averaged.
    Decoded chunk_blocks blocks (all chains) at a time: equivalent, since G_Y is row-wise and the mean
    operation acts within a block, and it keeps G_Y's 2048-wide activations to a few hundred MB."""
    nb, k, nz = latents.shape
    g = torch.Generator().manual_seed(seed)
    acc = torch.zeros(nb, k, gy.n_channels, dtype=torch.float64)
    was = gy.training
    gy.eval()
    try:
        with torch.no_grad():
            for _ in range(n_decodes):
                for a in range(0, nb, chunk_blocks):
                    zc = latents[a:a + chunk_blocks]
                    nc = zc.shape[0]
                    y = gy.generate(zc.reshape(nc * k, nz), generator=g).reshape(nc, k, gy.block_len, gy.n_channels)
                    if mean_operation:
                        y = y - y.mean(dim=1, keepdim=True)
                    acc[a:a + nc] += y.sum(dim=2).cpu().double()
    finally:
        gy.train(was)
    return (acc / n_decodes * sigma_nm).permute(1, 0, 2).numpy()


def bias_numbers(mu: np.ndarray, drift_nm: np.ndarray) -> dict:
    """(ii) var_i(b_i) / var_ik(mu_ik), pooled over axes; (iii) correlation of b_i with measured drift across
    chains (each axis centred, then pooled)."""
    b = mu.mean(axis=1)                                                         # (chains, axes)
    ratio = float(b.var(axis=0).sum() / mu.reshape(-1, mu.shape[2]).var(axis=0).sum())
    bc, dc = (b - b.mean(axis=0)).ravel(), (drift_nm - drift_nm.mean(axis=0)).ravel()
    corr = float(bc @ dc / np.sqrt((bc @ bc) * (dc @ dc)))
    return {"b_var_ratio": ratio, "no_persistence_reference": 1.0 / mu.shape[1], "corr_b_drift": corr,
            "b_rms_nm": float(np.sqrt((b ** 2).sum(axis=1).mean())), "drift_rms_nm": float(np.sqrt((drift_nm ** 2).sum(axis=1).mean()))}


# ============================================================================ 6c, 6d (torch)

def plateau_late(s0, gz, z_branch: torch.Tensor, seeds: tuple[int, int]) -> dict:
    return {int(k): v for k, v in s0.plateau_curve(gz, z_branch, seeds=seeds, steps=LATE_STEPS)["plateau"].items()}


def bn_numerics(gz, z_branch: torch.Tensor, seed: int) -> dict:
    bn = gz.norm
    was = gz.training
    gz.eval()
    try:
        with torch.no_grad():
            r = gz.sample_noise(z_branch.shape[0], device=z_branch.device, dtype=z_branch.dtype,
                                generator=torch.Generator().manual_seed(seed))
            pre = gz._trunk(z_branch, r).cpu().double()
    finally:
        gz.train(was)
    var = pre.var(dim=0, unbiased=False)
    var64 = pre[:64].var(dim=0, unbiased=False)
    ratio = (var / bn.eps).numpy()
    rel64 = (var64 / var - 1).abs().numpy()
    return {"eps": float(bn.eps), "pre_bn_var": var.tolist(), "var_over_eps": ratio.tolist(),
            "var_over_eps_min": float(ratio.min()), "var_over_eps_median": float(np.median(ratio)),
            "gamma": bn.weight.detach().cpu().double().tolist(), "beta": bn.bias.detach().cpu().double().tolist(),
            "batch": int(z_branch.shape[0]), "pre_bn_var_64_chains": var64.tolist(),
            "var_64_vs_full_rel_diff_median": float(np.median(rel64)), "var_64_vs_full_rel_diff_max": float(rel64.max())}


# ============================================================================ per pair

def _chains_file(rep_dir: Path, run: str, it: int, seed: int) -> Path:
    return rep_dir / "chains" / f"{run}_it{it:06d}_rs{seed}.npz"


def _labels(rows: list[dict], s0, run: str, it: int) -> dict:
    hit = [r for r in rows if r["run"] == run and r["iteration"] == it and r.get("draw", "config") == "config"]
    if not hit:
        return {"label": "unknown (no config-draw row in the labels file)"}
    r = hit[0]
    ok = s0.checkpoint_passes(r, VAR_RANGE, MAX_ERR)
    return {"label": "PASS" if ok else "fail", "latent": "PASS" if s0.latent_passes(r) else "fail",
            "rule": s0.rule_text(VAR_RANGE, MAX_ERR), "arm_eval_D_ratio": r["D_ratio"],
            "z0_plateau": {int(k): v for k, v in r["plateau"].items()}, "failed": [f[0] for f in s0.failed_criteria(r, VAR_RANGE, MAX_ERR)]}


def analyse_checkpoint(args, s0, md: Side, run: str, it: int, rep_dir: Path, label_rows: list[dict]) -> dict:
    from mdgan.checkpoint import load_checkpoint
    ck = load_checkpoint(Path(args.runs_dir) / run / f"snap_{it:06d}.pt", map_location="cpu")
    cfg = ck.config
    gz, gy = ck.gz.to(args.device), ck.gy.to(args.device)
    out = {"iteration": it, **_labels(label_rows, s0, run, it)}
    print(f"{run} @{it // 1000}k: {out['label']}", flush=True)

    # 6a: the config-seed rollout re-run, all chains
    dt = md.dt * cfg.step_skip
    n_steps = int(round(cfg.estimation_time_ps / dt))
    d, _, _ = s0.rollout_with_latents(gz, gy, cfg.estimation_chains, cfg.estimation_blocks(md.dt), cfg.burn_in,
                                      mean_operation=cfg.mean_operation,
                                      generator=torch.Generator().manual_seed(s0.draw_seeds(cfg.seed, cfg.seed)["rollout"]))
    d_nm = d[:, :n_steps].numpy().astype(np.float32) * np.float32(ck.sigma_nm)
    del d
    dec = decompose(model_side(d_nm, dt), md, n_boot=args.boot)
    del d_nm
    if "arm_eval_D_ratio" in out:
        dec["D_ratio_matches_arm_eval"] = bool(abs(dec["D_ratio"] / out["arm_eval_D_ratio"] - 1) < 1e-4)
    out["6a"] = dec
    print(f"   6a: V/V_MD {dec['V_ratio']:.3f}  c/c_MD {dec['c_ratio']:.3f}  product {dec['product']:.3f}  "
          f"D/D_MD {dec['D_ratio']:.3f}  closure {dec['closure_product']:+.3f} +- {dec['se_closure_product']:.3f} "
          f"(model {dec['closure_model']:+.3f} +- {dec['se_closure_model']:.3f}, MD {dec['closure_md']:+.3f} "
          f"+- {dec['se_closure_md']:.3f})" + ("" if dec["closes"] else "  DOES NOT CLOSE"), flush=True)
    if not dec["closes"]:
        dg = dec["diagnostics"]
        print(f"       halves: V {dg['V_first_half']:.4f} -> {dg['V_second_half']:.4f}, rho1 {dg['rho1_first_half']:+.3f} -> "
              f"{dg['rho1_second_half']:+.3f}; rho_31..60 model sum {sum(dg['rho_31_60_model'].values()):+.3f}, "
              f"MD sum {sum(dg['rho_31_60_md'].values()):+.3f}", flush=True)

    # 6b, 6c (all three stored rollout seeds), 6d (config chains)
    out["6b"], out["6c"] = {}, {}
    for seed in (cfg.seed, *NEW_SEEDS):
        f = _chains_file(rep_dir, run, it, seed)
        if not f.exists():
            out["6b"][str(seed)] = out["6c"][str(seed)] = f"missing {f}"
            continue
        z = np.load(f)
        lat = torch.from_numpy(z["latents"]).to(args.device)
        drift = z["block_sums_nm"].astype(np.float64).sum(axis=1)
        tag = "config" if seed == cfg.seed else str(seed)
        dseed, late = (17, (13, 14)) if seed == cfg.seed else (10 * seed + 8, (10 * seed + 6, 10 * seed + 7))
        mu = expected_block_displacement(gy, lat, ck.sigma_nm, args.decodes, dseed, cfg.mean_operation)
        out["6b"][tag] = bias_numbers(mu, drift)
        out["6c"][tag] = plateau_late(s0, gz, lat[BRANCH_AT], late)
        if seed == cfg.seed:
            out["_scatter"] = {"b": mu.mean(axis=1).ravel().tolist(), "drift": drift.ravel().tolist()}
            out["6d"] = bn_numerics(gz, lat[BRANCH_AT], 15)
        print(f"   6b [{tag}]: b var ratio {out['6b'][tag]['b_var_ratio']:.3f} (no persistence {out['6b'][tag]['no_persistence_reference']:.3f}), "
              f"corr(b, drift) {out['6b'][tag]['corr_b_drift']:+.3f} | 6c late plateau "
              + " ".join(f"{k}:{v:.2f}" for k, v in out["6c"][tag].items()), flush=True)
    if "6d" in out:
        b = out["6d"]
        print(f"   6d: var/eps min {b['var_over_eps_min']:.3g}, median {b['var_over_eps_median']:.3g}; gamma "
              f"[{min(b['gamma']):.3f}, {max(b['gamma']):.3f}]; 64-chain variance within "
              f"{b['var_64_vs_full_rel_diff_max'] * 100:.0f} % of the {b['batch']}-chain value", flush=True)
    gz.cpu()
    gy.cpu()
    return out


def figure(pair: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    run, cks = pair["run"], pair["checkpoints"]
    colors = ("tab:blue", "tab:red")
    fig, ax = plt.subplots(2, 2, figsize=(13, 9.5))
    fig.suptitle(f"{run}: " + "  vs  ".join(f"@{c['iteration'] // 1000}k {c['label']}" for c in cks)
                 + f"\n{cks[0].get('rule', '')}", fontsize=10)
    # (a)
    a = ax[0, 0]
    names = ["V/V_MD", "c/c_MD", "product", "D/D_MD"]
    for j, c in enumerate(cks):
        dec = c["6a"]
        vals = [dec["V_ratio"], dec["c_ratio"], dec["product"], dec["D_ratio"]]
        a.bar(np.arange(4) + (j - 0.5) * 0.38, vals, 0.36, color=colors[j],
              label=f"@{c['iteration'] // 1000}k {c['label']}: closure {dec['closure_product']:+.1%} ± {dec['se_closure_product']:.1%}")
    a.axhline(1, color="k", lw=0.6)
    a.set_xticks(range(4), names)
    a.set_title("6a  block decomposition against MD")
    a.legend(fontsize=8)
    # (b)
    b = ax[0, 1]
    for j, c in enumerate(cks):
        if "_scatter" in c:
            b.scatter(c["_scatter"]["drift"], c["_scatter"]["b"], s=3, alpha=0.4, color=colors[j],
                      label=f"@{c['iteration'] // 1000}k: var ratio {c['6b']['config']['b_var_ratio']:.3f}, "
                            f"corr {c['6b']['config']['corr_b_drift']:+.2f}")
    b.set_xlabel("measured net drift over the rollout (nm, per axis)")
    b.set_ylabel("decoder bias b_i (nm per block, per axis)")
    b.set_title("6b  decoder's expected displacement vs measured drift")
    b.legend(fontsize=8, markerscale=4)
    # (c)
    cax = ax[1, 0]
    for j, c in enumerate(cks):
        if "config" in c.get("6c", {}) and isinstance(c["6c"]["config"], dict):
            late = c["6c"]["config"]
            cax.plot(list(late), list(late.values()), "o-", color=colors[j], label=f"@{c['iteration'] // 1000}k late (block 30)")
        if "z0_plateau" in c:
            z0 = c["z0_plateau"]
            cax.plot(list(z0), list(z0.values()), "s--", color=colors[j], alpha=0.6, label=f"@{c['iteration'] // 1000}k z0 start")
    cax.axhline(1, color="k", lw=0.6)
    cax.set_xscale("log")
    cax.set_xlabel("steps after the branch")
    cax.set_ylabel("d_same / d_indep")
    cax.set_title("6c  late-start vs z0-start branching")
    cax.legend(fontsize=8)
    # (d)
    dax = ax[1, 1]
    twin = dax.twinx()
    for j, c in enumerate(cks):
        if "6d" in c:
            dims = np.arange(len(c["6d"]["var_over_eps"]))
            dax.semilogy(dims, c["6d"]["var_over_eps"], "o", color=colors[j], ms=4,
                         label=f"@{c['iteration'] // 1000}k var/eps (min {c['6d']['var_over_eps_min']:.3g})")
            twin.plot(dims, c["6d"]["gamma"], "x", color=colors[j], ms=5, label=f"@{c['iteration'] // 1000}k gamma")
    dax.axhline(10, color="k", lw=0.6, ls=":")
    dax.set_xlabel("latent dimension")
    dax.set_ylabel("pre-BN variance / eps (log)")
    twin.set_ylabel("gamma (x)")
    dax.set_title("6d  BatchNorm numerics at block 30")
    h1, l1 = dax.get_legend_handles_labels()
    h2, l2 = twin.get_legend_handles_labels()
    dax.legend(h1 + h2, l1 + l2, fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--data", help="PE cache (data/cache/PE.npz)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--runs-dir", default="runs/chtc")
    ap.add_argument("--rep-root", default="runs", help="where the repeatability output directories are")
    ap.add_argument("--labels", default="runs/rep_all/arm-eval.json")
    ap.add_argument("--pairs", nargs="*", help="run:it_a:it_b:rep_dir (default: the three agreed pairs)")
    ap.add_argument("--decodes", type=int, default=DECODES)
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--out", default="runs/decompose")
    args = ap.parse_args(argv)
    if args.self_test:
        self_test()
        return
    if not args.data:
        ap.error("--data is required")
    s0 = _load_tool("step0_rollouts")
    from mdgan.data.preprocess import SystemData
    sysd = SystemData.load(args.data)
    md = md_side(sysd.displacements.astype(np.float32), float(sysd.dt_ps))
    labels = json.loads(Path(args.labels).read_text()) if Path(args.labels).exists() else []
    if not labels:
        print(f"note: no labels file at {args.labels}; checkpoints will be labelled unknown")
    pairs = [(p.split(":")[0], int(p.split(":")[1]), int(p.split(":")[2]), p.split(":")[3]) for p in args.pairs] if args.pairs else PAIRS
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for run, it_a, it_b, rep in pairs:
        cks = [analyse_checkpoint(args, s0, md, run, it, Path(args.rep_root) / rep, labels) for it in (it_a, it_b)]
        pair = {"run": run, "checkpoints": cks}
        figure(pair, out / f"{run}.png")
        for c in cks:
            c.pop("_scatter", None)
        (out / f"{run}.json").write_text(json.dumps(pair, indent=1, default=float))
        print(f"wrote {out / run}.png and .json\n", flush=True)


# ============================================================================ self-test

def _ar1(n: int, t: int, phi: float, rng, c: int = 3) -> np.ndarray:
    x = np.empty((n, t, c), np.float32)
    s = np.sqrt(1 - phi ** 2)
    x[:, 0] = rng.normal(size=(n, c))
    for i in range(1, t):
        x[:, i] = phi * x[:, i - 1] + s * rng.normal(size=(n, c))
    return x


def _ar1_block_moments(phi: float, m: int = BLOCK, kmax: int = KMAX) -> tuple[float, dict]:
    """Exact V and rho_k of 64-step block sums of a stationary unit-variance AR(1)."""
    V = m + 2 * sum((m - j) * phi ** j for j in range(1, m))
    return V, {k: sum((m - abs(dl)) * phi ** (k * m + dl) for dl in range(-(m - 1), m)) / V for k in range(1, kmax + 1)}


def _control_exact_closure() -> None:
    """Exact moments: the decomposition's D (from V, rho_k, c) equals the exact MSD slope over 10-20 ns."""
    for phi in (0.0, 0.9, 0.99, -0.5):
        V, rho = _ar1_block_moments(phi)
        t = fit_window(10.0)
        msd = np.array([3 * (tt + 2 * sum((tt - j) * phi ** j for j in range(1, tt))) for tt in t])   # exact, 3 axes
        d_exact = ols_d_m2s(msd, t * 10.0)
        d_dec = d_pred_m2s(V, corr_slope(rho), BLOCK * 10.0)
        assert abs(d_dec / d_exact - 1) < 0.01, (phi, d_dec / d_exact - 1)


def _control_sampled_closure() -> None:
    """Sampled: an AR(1) model (4096 chains) and white MD (60 trajectories) close within twice their bootstrap
    SE, and the tool's D paths equal mdgan.data.msd's."""
    from mdgan.data.msd import diffusion_coefficient, msd_ensemble, msd_time_averaged
    rng = np.random.default_rng(7)
    model_d, md_d = _ar1(4096, 8000, 0.99, rng), _ar1(60, 8000, 0.0, rng)
    model, md = model_side(model_d, 10.0), md_side(md_d, 10.0)
    dg, _ = diffusion_coefficient(msd_ensemble(model_d), 10.0, 10000.0, 20000.0)
    ref, _ = msd_time_averaged(md_d)
    dm, _ = diffusion_coefficient(ref, 10.0, 10000.0, 20000.0)
    assert abs(model.numbers()["D_measured"] / dg - 1) < 1e-9 and abs(md.numbers()["D_measured"] / dm - 1) < 1e-9
    dec = decompose(model, md, n_boot=100)
    for side in ("model", "md", "product"):
        assert abs(dec[f"closure_{side}"]) < 2 * dec[f"se_closure_{side}"] + 1e-12, (side, dec[f"closure_{side}"], dec[f"se_closure_{side}"])
    st = _load_tool("diag_runs")._block_stats(model_d.astype(np.float64)[:256], BLOCK)   # arm-eval's definitions
    mine = BlockMoments(block_sums(model_d[:256])).stats()
    assert abs(mine["V"] / st["var"] - 1) < 1e-9 and all(abs(mine["rho"][k] - st["corr"][k]) < 1e-9 for k in st["corr"])


def _control_decoder_bias() -> None:
    """A decoder adding a fixed per-chain offset to every block: b-variance ratio near 1 and correlation with
    drift near 1. A decoder ignoring the latent: ratio near 1/n_blocks and no correlation."""
    class Offset(torch.nn.Module):
        def __init__(self, uses_z: bool):
            super().__init__()
            self.block_len, self.n_channels, self.uses_z = 8, 3, uses_z

        def generate(self, z, generator=None):
            noise = 0.05 * torch.randn(z.shape[0], self.block_len, self.n_channels, generator=generator)
            off = z[:, :3, None].transpose(1, 2) / self.block_len if self.uses_z else 0.0
            return noise + off

    rng = torch.Generator().manual_seed(3)
    nb, k = 40, 512
    offsets = torch.randn(1, k, 6, generator=rng)
    lat = offsets.expand(nb, k, 6).contiguous()                               # each chain's latent constant over blocks
    for uses_z in (True, False):
        gy = Offset(uses_z)
        sums = torch.stack([gy.generate(lat[i], generator=rng).sum(dim=1) for i in range(nb)], dim=1)   # one rollout
        sums = sums - sums.mean(dim=0, keepdim=True)
        mu = expected_block_displacement(gy, lat, 1.0, 32, 5)
        res = bias_numbers(mu, sums.sum(dim=1).double().numpy())
        if uses_z:
            assert res["b_var_ratio"] > 0.9 and res["corr_b_drift"] > 0.95, res
        else:
            assert res["b_var_ratio"] < 3.0 / nb and abs(res["corr_b_drift"]) < 0.15, res


def self_test() -> None:
    _control_exact_closure()
    print("control 1: exact moments: the decomposition's D equals the exact MSD slope within 1 % (AR(1), phi = 0, 0.9, 0.99, -0.5)")
    _control_sampled_closure()
    print("control 2: sampled AR(1) (4096 chains) and white MD (60 trajectories) close within 2 bootstrap SE on each side "
          "and for the product; the tool's D equals mdgan.data.msd's")
    _control_decoder_bias()
    print("control 3: a per-chain offset decoder gives b-variance ratio > 0.9 and corr(b, drift) > 0.95; "
          "a latent-blind decoder gives ratio < 3/n_blocks and no correlation")
    print("all positive controls passed")


if __name__ == "__main__":
    main()