"""Figures for MD-GAN runs: architecture, generated vs MD, and comparison with the paper.

Everything is computed from raw inputs, with the same code for every source:
  - the MD: the preprocessed trajectories (per-trajectory MSDs give the 95 % band and the
    half-split noise floor);
  - our runs: every checkpoint's msd_XXXXXX.npy and evaluations.jsonl (+ metrics.jsonl);
  - the authors: their released generated MSDs (gen_data/MSD_gen/PE/mean_operation/
    01ss_msd_bs0064-nw032-ite300000_scale_addex{1..5}/{Z}ln_skip10.npy, 799 values on
    t = 100 (k+1) ps) and dc_result_mdgan.txt. Their err and D are RECOMPUTED with our
    metric code, never read off the paper's figures;
  - two baselines that use only the training data (first 1.28 ns of each trajectory):
    linear and power-law extrapolation of its MSD.

"Best" checkpoint = lowest err against the MD (the papers' protocol: it uses the answer).
"Blind" = the median over all checkpoints of a run (what you get without the answer).
Both are shown, for us and for the authors.

Usage (repository root):
    python tools/make_figures.py \\
        --runs runs/chtc/run_c6954730_seed* \\
        --data data/cache/PE.npz \\
        --authors data/raw/wat_in_polymer_mdgan/gen_data/MSD_gen/PE/mean_operation \\
        --authors-dc data/raw/wat_in_polymer_mdgan/dc_result/dc_result_mdgan.txt \\
        --out runs/figures
Writes fig1..fig8 as PDF and PNG plus summary.json (every number plotted).
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
from matplotlib.ticker import FixedLocator, NullFormatter, ScalarFormatter  # noqa: E402

from mdgan.config import Config  # noqa: E402
from mdgan.data.msd import diffusion_coefficient, err, msd_time_averaged  # noqa: E402
from mdgan.data.preprocess import SystemData  # noqa: E402

TRAIN_NS = 1.28
C_MD, C_OURS, C_AUTH, C_BASE = "black", "#1f77b4", "#d95f02", "0.45"
AUTH_DT_PS = 100.0                       # authors' grid: t = 100 (k + 1) ps, k = 0..798
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "legend.fontsize": 7.5, "figure.dpi": 150,
                     "savefig.bbox": "tight", "axes.spines.top": False, "axes.spines.right": False})


# ============================================================================ data

def load_md(path: str, cfg: Config):
    system = SystemData.load(path)
    mean, per = msd_time_averaged(system.displacements)
    dt = float(system.dt_ps)
    n = per.shape[0]
    band = 1.96 * per.std(axis=0, ddof=1) / np.sqrt(n)
    d_md, _ = diffusion_coefficient(mean, dt, *cfg.diffusion_fit_ps)
    return {"system": system, "msd": mean, "per": per, "band": band, "dt": dt, "D": d_md}


def load_ours(run_dirs: list[str]) -> dict:
    runs = {}
    for r in sorted(run_dirs):
        p = Path(r)
        evals = [json.loads(x) for x in (p / "evaluations.jsonl").read_text().splitlines() if x.strip()]
        metrics = [json.loads(x) for x in (p / "metrics.jsonl").read_text().splitlines() if x.strip()] \
            if (p / "metrics.jsonl").exists() else []
        curves = {e["iteration"]: np.load(p / f"msd_{e['iteration']:06d}.npy") for e in evals}
        runs[p.name] = {"evals": evals, "metrics": metrics, "curves": curves, "dt": evals[0]["dt_gen_ps"]}
    return runs


def load_authors(root: str | None, md: dict, cfg: Config) -> dict:
    """Their 5 nw032 runs, every checkpoint, with err and D recomputed by our code."""
    if not root:
        return {}
    runs = {}
    for d in sorted(Path(root).glob("01ss_msd_bs0064-nw032-ite300000_scale_addex*")):
        evals, curves = [], {}
        for f in Path(d).glob("*ln_skip10.npy"):
            it = int(re.match(r"(\d+)ln", f.name).group(1))
            g = np.concatenate([[0.0], np.load(f).astype(np.float64)])        # prepend t = 0
            curves[it] = g
        for it in sorted(curves):
            g = curves[it]
            e = err(g, AUTH_DT_PS, md["msd"], md["dt"], *cfg.err_window_ps)
            dg, _ = diffusion_coefficient(g, AUTH_DT_PS, *cfg.diffusion_fit_ps)
            evals.append({"iteration": it, "err_percent": e, "D_gen_m2s": dg, "D_rel_error": dg / md["D"] - 1})
        runs["authors run " + d.name.rsplit("addex", 1)[1]] = {"evals": evals, "curves": curves, "dt": AUTH_DT_PS}
    return runs


def baselines(md: dict, cfg: Config) -> dict:
    """Extrapolations that use ONLY the training region (first 1.28 ns of every trajectory)."""
    n_train = int(round(TRAIN_NS * 1000 / md["dt"]))
    train = md["system"].displacements[:, :n_train]
    m_train, _ = msd_time_averaged(train)
    t_train = np.arange(n_train + 1) * md["dt"]
    t_full = np.arange(len(md["msd"])) * md["dt"]
    half = t_train >= t_train[-1] / 2
    slope, icpt = np.polyfit(t_train[half], m_train[half], 1)                          # linear on 0.64-1.28 ns
    fit = (t_train >= 100) & (t_train <= t_train[-1])
    alpha, logc = np.polyfit(np.log(t_train[fit]), np.log(m_train[fit]), 1)          # power law on 0.1-1.28 ns
    # Inside the training window a baseline simply reports the training data (it is known);
    # beyond it, the extrapolation. Only the extrapolated part is a prediction.
    lin = np.concatenate([m_train, icpt + slope * t_full[n_train + 1:]])
    pw = np.concatenate([m_train, np.exp(logc) * t_full[n_train + 1:] ** alpha])
    out = {}
    for name, curve in (("linear extrapolation", lin), (f"power-law extrapolation (α={alpha:.2f})", pw)):
        e = err(curve, md["dt"], md["msd"], md["dt"], *cfg.err_window_ps)
        d, _ = diffusion_coefficient(curve, md["dt"], *cfg.diffusion_fit_ps)
        out[name] = {"curve": curve, "err_percent": e, "D_rel_error": d / md["D"] - 1}
    return out


def noise_floor(md: dict, cfg: Config, n_splits: int = 200, seed: int = 0) -> dict:
    """How well a PERFECT model could match the 60-trajectory reference.

    Split the trajectories into two random halves and compare them. Each half has twice the
    variance of the full reference, so the half-vs-half difference has 2x the standard
    deviation of (full reference - truth). Both err and |D_rel| scale linearly with that
    deviation, so the floor is the half-split value / 2."""
    rng = np.random.default_rng(seed)
    per, n = md["per"], md["per"].shape[0]
    errs, drels = [], []
    for _ in range(n_splits):
        idx = rng.permutation(n)
        a, b = per[idx[: n // 2]].mean(0), per[idx[n // 2:]].mean(0)
        errs.append(err(a, md["dt"], b, md["dt"], *cfg.err_window_ps))
        da, _ = diffusion_coefficient(a, md["dt"], *cfg.diffusion_fit_ps)
        db, _ = diffusion_coefficient(b, md["dt"], *cfg.diffusion_fit_ps)
        drels.append(da / db - 1)
    errs, drels = np.array(errs) / 2, np.abs(np.array(drels)) / 2
    return {"err_median": float(np.median(errs)), "err_p95": float(np.percentile(errs, 95)),
            "D_rel_abs_p95": float(np.percentile(drels, 95))}


def best_and_blind(run: dict) -> dict:
    ev = run["evals"]
    errs = np.array([e["err_percent"] for e in ev])
    drel = np.array([e["D_rel_error"] for e in ev])
    b = int(errs.argmin())
    return {"best_iteration": ev[b]["iteration"], "best_err": float(errs[b]), "best_D_rel": float(drel[b]),
            "blind_err": float(np.median(errs)), "blind_D_rel": float(np.median(drel))}


def plain_log_ticks(ax, axis: str, ticks) -> None:
    """Log axis with plain-number labels at the given values."""
    a = ax.yaxis if axis == "y" else ax.xaxis
    a.set_major_locator(FixedLocator(ticks))
    a.set_major_formatter(ScalarFormatter())
    a.set_minor_formatter(NullFormatter())


def common_times(t_min: float, t_max: float, n: int = 120, step_ns: float = 0.1) -> np.ndarray:
    """Log-spaced times rounded to multiples of 100 ps: on our 10 ps grid AND the authors' 100 ps grid,
    so every ratio compares curves at exactly the same time."""
    t = np.round(np.logspace(np.log10(t_min), np.log10(t_max), n) / step_ns) * step_ns
    return np.unique(t[t >= step_ns])


def ratio_curve(g: np.ndarray, dt_g: float, md: dict, t_ns: np.ndarray) -> np.ndarray:
    kg = np.round(t_ns * 1000 / dt_g).astype(int)
    km = np.round(t_ns * 1000 / md["dt"]).astype(int)
    ok = (kg < len(g)) & (km < len(md["msd"])) & (kg > 0)
    out = np.full(t_ns.shape, np.nan)
    out[ok] = g[kg[ok]] / md["msd"][km[ok]]
    return out


def local_exponent(curve: np.ndarray, dt: float, t_ns: np.ndarray, factor: float = 1.6) -> np.ndarray:
    """d ln MSD / d ln t over a log window [t / sqrt(f), t * sqrt(f)], only where the curve has
    data at both ends (no extrapolation outside a curve's own grid)."""
    t = np.arange(len(curve)) * dt / 1000
    m = (t > 0) & (curve > 0)
    lt, lm = np.log(t[m]), np.log(curve[m])
    lo, hi = np.log(t_ns / np.sqrt(factor)), np.log(t_ns * np.sqrt(factor))
    out = (np.interp(hi, lt, lm) - np.interp(lo, lt, lm)) / np.log(factor)
    out[(lo < lt[0] + np.log(1.5)) | (hi > lt[-1])] = np.nan
    return out


def save(fig, out: Path, name: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}")
    plt.close(fig)


# ===================================================================== figures 1-2

def param_counts(c: Config) -> dict:
    h, nz, nr, m, ch = c.hidden_dim, c.latent_dim, c.noise_dim, c.block_len, c.n_channels
    gz = (nz + nr) * h + h + 2 * (h * h + h) + h * nz + nz + 2 * nz
    gy = ch * ch + ch + (nr + nz + m * ch) * h + h + 2 * (h * h + h) + h * m * ch + m * ch
    dg = ch * h * 2 * m + h + 3 * (h * h + h) + h + 1
    try:                                                     # cross-check against the real modules
        import torch  # noqa: F401

        from mdgan.models import build_models
        real = [sum(p.numel() for p in net.parameters()) for net in build_models(c)]
        assert real == [gz, gy, dg], (real, [gz, gy, dg])
    except ImportError:
        pass
    return {"G_z": gz, "G_Y": gy, "D_G": dg}


def _box(ax, x, y, w, h, text, color="#e8eef7", fs=7.5, edge="0.3"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=color, ec=edge, lw=0.8))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)


def _arrow(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=8, lw=0.8, color="0.25"))


def fig_architecture(c: Config, out: Path) -> dict:
    counts = param_counts(c)
    h, nz, nr, m, ch = c.hidden_dim, c.latent_dim, c.noise_dim, c.block_len, c.n_channels
    fig, axes = plt.subplots(3, 1, figsize=(9.6, 5.4))
    rows = [
        (f"G_z  latent transition  ({counts['G_z'] / 1e6:.2f} M parameters)", "#e8eef7", [
            f"z ({nz})\n+ r ({nr})\nr ~ U[0,1)", f"Linear\n{nz + nr}→{h}\nLReLU", f"Linear\n{h}→{h}\nLReLU",
            f"Linear\n{h}→{h}\nLReLU", f"Linear\n{h}→{nz}", f"BatchNorm\n({nz}, batch\nstatistics)", f"z′ ({nz})"]),
        (f"G_Y  trajectory decoder  ({counts['G_Y'] / 1e6:.2f} M parameters)", "#eef7e8", [
            f"noise u ({m}×{ch})\nConv1d k=1\nflatten → {m * ch}", f"concat\nr′ {nr} + z {nz}\n+ u′ {m * ch} = {nr + nz + m * ch}",
            f"Linear\n{nr + nz + m * ch}→{h}\nLReLU", f"Residual\n2 × Linear {h}\n+ skip", "LReLU",
            f"Linear\n{h}→{m * ch}", f"Y block\n({m}×{ch})"]),
        (f"D_G  critic  ({counts['D_G'] / 1e6:.2f} M parameters)", "#f7eee8", [
            f"window\n({2 * m}×{ch})", f"Conv1d k={2 * m}\n{ch}→{h}\nLReLU", f"Linear\n{h}→{h}\nLReLU",
            f"Linear\n{h}→{h}\nLReLU", f"Linear\n{h}→{h}\nLReLU", f"Linear\n{h}→1", "score\n(unbounded)"]),
    ]
    for ax, (title, color, labels) in zip(axes, rows, strict=True):
        ax.set_xlim(0, 9.75)
        ax.set_ylim(0.05, 1.15)
        ax.axis("off")
        ax.set_title(title, loc="left", fontsize=9.5)
        w, gap = 1.17, 0.23
        for i, lab in enumerate(labels):
            x = 0.05 + i * (w + gap)
            _box(ax, x, 0.15, w, 0.95, lab, color=color if 0 < i < len(labels) - 1 else "white", fs=7)
            if i:
                _arrow(ax, x - gap + 0.01, 0.62, x - 0.01, 0.62)
    fig.subplots_adjust(left=0.01, right=0.99, hspace=0.35)
    fig.text(0.01, -0.01, f"Kawada et al. 2022 ESI Fig. S2, as implemented in src/mdgan/models; parameter counts "
             f"computed from the configuration.\nn_z={nz}, n_r={nr}, M={m}, C={ch}, width {h}, LeakyReLU slope "
             f"{c.negative_slope}; total {sum(counts.values()) / 1e6:.1f} M parameters.", fontsize=7, color="0.3")
    save(fig, out, "fig1_architecture")
    return counts


def fig_dataflow(c: Config, out: Path) -> None:
    n_blocks = c.estimation_blocks(10.0)
    fig, axes = plt.subplots(2, 1, figsize=(9.6, 4.6))
    ax = axes[0]
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 2.6)
    ax.axis("off")
    ax.set_title(f"Training: one fake window per sample (WGAN-GP, λ={c.gp_lambda:g}, "
                 f"{c.n_critic} critic steps per generator step)", loc="left", fontsize=9.5)
    top = [(0.0, 1.25, "z₀ ~ U[0,1)", "white"),
           (1.65, 1.95, f"G_z × (j+1)\nburn-in c={c.burn_in}\nj ∈ [{c.burn_in}, {c.burn_in + c.pair_window - 1}]", "#e8eef7"),
           (4.0, 1.5, "latent pair\n(z_j, z_j+1)", "white"),
           (5.9, 1.5, "G_Y on each\n(+ noise r′, u)", "#eef7e8"),
           (7.8, 1.6, f"fake window\nY_j | Y_j+1\n({2 * c.block_len}×{c.n_channels})", "white"),
           (9.8, 1.6, "mean\noperation", "white")]
    for x, w, text, col in top:
        _box(ax, x, 1.45, w, 0.95, text, col)
    for (x0, w0, *_), (x1, *_) in zip(top[:-1], top[1:], strict=True):
        _arrow(ax, x0 + w0, 1.92, x1, 1.92)
    _box(ax, 9.8, 0.1, 1.6, 0.85, "D_G\nscore", "#f7eee8")
    _arrow(ax, 10.6, 1.45, 10.6, 0.95)
    _box(ax, 4.0, 0.1, 4.9, 0.85, f"real window: first {TRAIN_NS} ns of an MD trajectory\n"
                                  f"({2 * c.block_len} steps × {c.n_channels}, divided by σ)", "white")
    _arrow(ax, 8.9, 0.52, 9.8, 0.52)

    ax = axes[1]
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 2.6)
    ax.axis("off")
    ax.set_title(f"Estimation: {c.estimation_chains} chains × {n_blocks} blocks "
                 f"= {c.estimation_time_ps / 1000:g} ns", loc="left", fontsize=9.5)
    row = [(0.0, 1.25, "z₀ ~ U[0,1)", "white"),
           (1.65, 1.5, f"G_z × {c.burn_in}\n(burn-in)", "#e8eef7"),
           (3.55, 2.6, f"z_c → z_c+1 → … → z_c+{n_blocks - 1}\n(G_z, fresh noise each step)", "#e8eef7"),
           (6.55, 1.75, "G_Y per latent\nY_0 … Y_" + str(n_blocks - 1), "#eef7e8"),
           (8.7, 2.7, f"concatenate {n_blocks}×{c.block_len} steps\n× σ  →  trajectories", "white")]
    for x, w, text, col in row:
        _box(ax, x, 1.45, w, 0.95, text, col)
    for (x0, w0, *_), (x1, *_) in zip(row[:-1], row[1:], strict=True):
        _arrow(ax, x0 + w0, 1.92, x1, 1.92)
    _box(ax, 6.55, 0.1, 4.85, 0.85, "ensemble MSD → err vs MD (0.2–78 ns),\nD from the 10–20 ns slope", "white")
    _arrow(ax, 10.05, 1.45, 10.05, 0.95)
    ax.text(0.0, 0.52, f"Training decodes latents only up to step {c.burn_in + c.pair_window};\n"
                       f"estimation decodes up to step {c.burn_in + n_blocks - 1}.", fontsize=8,
            color="#b22222", va="center")
    save(fig, out, "fig2_dataflow")


# ===================================================================== figures 3-8

def fig_msd(md, ours, authors, base, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    t = np.arange(len(md["msd"])) * md["dt"] / 1000
    ax.fill_between(t[1:], (md["msd"] - md["band"])[1:], (md["msd"] + md["band"])[1:], color="0.8", lw=0)
    ax.plot(t[1:], md["msd"][1:], color=C_MD, lw=1.6, label="MD (60 trajectories, 95 % band)")
    ax.axvspan(t[1], TRAIN_NS, color="#fff3c4", zorder=0, label=f"training window (≤ {TRAIN_NS} ns)")
    for i, r in enumerate(ours.values()):
        it = best_and_blind(r)["best_iteration"]
        g = r["curves"][it]
        tg = np.arange(len(g)) * r["dt"] / 1000
        ax.plot(tg[1:], g[1:], color=C_OURS, lw=0.9, alpha=0.8, label="ours, best checkpoint per seed" if i == 0 else None)
    for i, r in enumerate(authors.values()):
        it = best_and_blind(r)["best_iteration"]
        g = r["curves"][it]
        tg = np.arange(len(g)) * r["dt"] / 1000
        ax.plot(tg[1:], g[1:], color=C_AUTH, lw=0.9, ls="--", alpha=0.85,
                label="authors, best checkpoint per run" if i == 0 else None)
    beyond = t >= TRAIN_NS
    for (name, b), ls in zip(base.items(), (":", "-."), strict=True):
        ax.plot(t[beyond], b["curve"][beyond], color=C_BASE, lw=1.0, ls=ls, label=f"baseline: {name}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.01, 80)
    ax.set_xlabel("t (ns)")
    ax.set_ylabel("MSD (nm²)")
    ax.set_title("Water in PE: MSD, generated vs MD")
    ax.legend(loc="upper left", frameon=True, framealpha=0.95, edgecolor="none")
    save(fig, out, "fig3_msd")


def fig_ratio(md, ours, authors, base, out: Path) -> dict:
    t_ns = common_times(0.1, 78)
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4), sharey=True)
    summary = {}
    for ax, label, runs, color in ((axes[0], "ours", ours, C_OURS), (axes[1], "authors", authors, C_AUTH)):
        if not runs:
            ax.set_visible(False)
            continue
        allr = np.array([ratio_curve(g, r["dt"], md, t_ns) for r in runs.values() for g in r["curves"].values()])
        q25, q50, q75 = np.nanpercentile(allr, [25, 50, 75], axis=0)
        ax.fill_between(t_ns, q25, q75, color=color, alpha=0.18, lw=0, label="all checkpoints: IQR")
        ax.plot(t_ns, q50, color=color, lw=1.6, label="all checkpoints: median (blind)")
        for i, r in enumerate(runs.values()):
            g = r["curves"][best_and_blind(r)["best_iteration"]]
            ax.plot(t_ns, ratio_curve(g, r["dt"], md, t_ns), color=color, lw=0.7, ls="--", alpha=0.8,
                    label="best checkpoint per run" if i == 0 else None)
        beyond = t_ns >= TRAIN_NS
        for (name, b), ls in zip(base.items(), (":", "-."), strict=True):
            ax.plot(t_ns[beyond], ratio_curve(b["curve"], md["dt"], md, t_ns)[beyond], color=C_BASE, lw=0.9, ls=ls,
                    label=name if label == "ours" else None)
        ax.axhline(1, color=C_MD, lw=0.8)
        ax.axvspan(0.1, TRAIN_NS, color="#fff3c4", zorder=0)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("t (ns)")
        ax.set_title(f"{label}: generated / MD")
        ax.legend(frameon=False, loc="upper left")
        summary[label] = {f"{t:g} ns": float(v) for t, v in zip(t_ns[::20], q50[::20], strict=True)}
    axes[0].set_ylabel("MSD ratio (1 = exact)")
    lo, hi = axes[0].get_ylim()
    plain_log_ticks(axes[0], "y", [v for v in (0.3, 0.5, 0.7, 1, 1.5, 2, 3, 5, 10, 20) if lo <= v <= hi])
    save(fig, out, "fig4_ratio")
    return summary


def fig_alpha(md, ours, authors, out: Path) -> None:
    t_ns = np.logspace(np.log10(0.02), np.log10(60), 110)
    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    ax.plot(t_ns, local_exponent(md["msd"], md["dt"], t_ns), color=C_MD, lw=1.6, label="MD")
    for runs, color, ls, lab in ((ours, C_OURS, "-", "ours, best per seed"), (authors, C_AUTH, "--", "authors, best per run")):
        for i, r in enumerate(runs.values()):
            g = r["curves"][best_and_blind(r)["best_iteration"]]
            ax.plot(t_ns, local_exponent(g, r["dt"], t_ns), color=color, lw=0.8, ls=ls, alpha=0.8,
                    label=lab if i == 0 else None)
    ax.axhline(1, color="0.5", lw=0.7, ls=":")
    ax.axvspan(0.02, TRAIN_NS, color="#fff3c4", zorder=0)
    ax.set_xscale("log")
    ax.set_ylim(0.3, 1.6)
    ax.set_xlabel("t (ns)")
    ax.set_ylabel("local exponent  d ln MSD / d ln t")
    ax.set_title("Anomalous → normal diffusion (1 = Einstein regime)")
    ax.legend(frameon=False, loc="lower right")
    save(fig, out, "fig5_local_exponent")


def fig_traces(ours, authors, out: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 5.2), sharex=True, sharey="row")
    for col, (label, runs) in enumerate((("ours", ours), ("authors", authors))):
        if not runs:
            axes[0, col].set_visible(False)
            axes[1, col].set_visible(False)
            continue
        shades = plt.cm.Blues(np.linspace(0.45, 0.95, len(runs))) if label == "ours" \
            else plt.cm.Oranges(np.linspace(0.45, 0.95, len(runs)))
        for r, s in zip(runs.values(), shades, strict=True):
            it = np.array([e["iteration"] for e in r["evals"]]) / 1000
            e = np.array([x["err_percent"] for x in r["evals"]])
            d = np.array([x["D_rel_error"] for x in r["evals"]])
            b = int(e.argmin())
            axes[0, col].plot(it, e, color=s, lw=0.9)
            axes[0, col].plot(it[b], e[b], "o", color=s, ms=4)
            axes[1, col].plot(it, 1 + d, color=s, lw=0.9)
            axes[1, col].plot(it[b], 1 + d[b], "o", color=s, ms=4)
        axes[0, col].set_yscale("log")
        axes[0, col].set_title(f"{label}: every checkpoint (● = best)")
        axes[1, col].set_yscale("log")
        axes[1, col].axhline(1, color=C_MD, lw=0.8)
        axes[1, col].set_xlabel("iteration (×1000)")
    axes[0, 0].set_ylabel("err (%)")
    axes[1, 0].set_ylabel("D_gen / D_MD")
    plain_log_ticks(axes[0, 0], "y", [3, 10, 30, 100, 300, 1000, 3000])
    plain_log_ticks(axes[1, 0], "y", [0.3, 0.5, 1, 2, 5, 10, 30, 100])
    save(fig, out, "fig6_checkpoint_traces")


def fig_summary(ours, authors, base, floor, md, authors_dc, out: Path) -> dict:
    groups = {}
    if ours:
        groups["ours"] = [best_and_blind(r) for r in ours.values()]
    if authors:
        groups["authors"] = [best_and_blind(r) for r in authors.values()]
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.9))
    cols = {"ours": C_OURS, "authors": C_AUTH}
    xt, xl = [], []
    for gi, (g, vals) in enumerate(groups.items()):
        for ki, (key, marker) in enumerate((("best", "o"), ("blind", "s"))):
            x = gi * 2.6 + ki
            xt.append(x)
            xl.append(f"{g}\n{key}")
            e = [v[f"{key}_err"] for v in vals]
            d = [1 + v[f"{key}_D_rel"] for v in vals]
            jit = np.linspace(-0.15, 0.15, len(e))
            axes[0].scatter(x + jit, e, color=cols[g], marker=marker, s=22, zorder=3)
            axes[1].scatter(x + jit, d, color=cols[g], marker=marker, s=22, zorder=3)
            axes[0].hlines(np.median(e), x - 0.3, x + 0.3, color=cols[g], lw=1.5)
            axes[1].hlines(np.median(d), x - 0.3, x + 0.3, color=cols[g], lw=1.5)
    xb = len(groups) * 2.6
    for i, (name, b) in enumerate(base.items()):
        axes[0].scatter([xb + i], [b["err_percent"]], color=C_BASE, marker="^", s=30)
        axes[1].scatter([xb + i], [1 + b["D_rel_error"]], color=C_BASE, marker="^", s=30)
        xt.append(xb + i)
        xl.append("baseline\n" + name.split(" ")[0])
    axes[0].axhspan(floor["err_median"], floor["err_p95"], color="0.88", lw=0, zorder=0,
                    label=f"perfect model vs MD reference: median {floor['err_median']:.1f} %, "
                          f"95th pct {floor['err_p95']:.1f} %")
    axes[1].axhspan(1 - floor["D_rel_abs_p95"], 1 + floor["D_rel_abs_p95"], color="0.85", lw=0, zorder=0,
                    label=f"MD reference uncertainty ±{floor['D_rel_abs_p95'] * 100:.0f} %")
    axes[1].axhline(1, color=C_MD, lw=0.8)
    if authors_dc:
        axes[1].axhline(authors_dc / md["D"], color=C_AUTH, lw=0.8, ls=":",
                        label=f"authors' published D ({authors_dc:.3g} m²/s)")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("err (%)")
    axes[1].set_ylabel("D_gen / D_MD")
    axes[1].set_yscale("log")
    for a in axes:
        a.set_xticks(xt)
        a.set_xticklabels(xl, fontsize=7)
        a.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.17), fontsize=7)
    lo, hi = axes[0].get_ylim()
    plain_log_ticks(axes[0], "y", [v for v in (2, 3, 5, 10, 20, 30, 50, 100, 200, 500, 1000) if lo <= v <= hi])
    lo, hi = axes[1].get_ylim()
    plain_log_ticks(axes[1], "y", [v for v in (0.3, 0.5, 0.7, 1, 1.5, 2, 3, 5, 10) if lo <= v <= hi])
    axes[0].set_title("err per run (best: lowest err vs MD; blind: median)")
    axes[1].set_title(f"D per run (D_MD = {md['D']:.3g} m²/s)")
    save(fig, out, "fig7_summary")
    return groups


def fig_metrics(ours, out: Path) -> None:
    if not any(r["metrics"] for r in ours.values()):
        return
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.2), sharex=True)
    shades = plt.cm.Blues(np.linspace(0.45, 0.95, len(ours)))
    for (name, r), s in zip(ours.items(), shades, strict=True):
        it = np.array([m["iteration"] for m in r["metrics"]]) / 1000
        axes[0].plot(it, [m["fake_step_std"] for m in r["metrics"]], color=s, lw=0.8, label=name[-5:])
        axes[1].plot(it, [m["wasserstein"] for m in r["metrics"]], color=s, lw=0.8)
    axes[0].axhline(1, color=C_MD, lw=0.8, ls=":")
    axes[0].set_ylabel("generated step std (σ units; MD = 1)")
    axes[1].set_ylabel("W (critic's Wasserstein estimate)")
    for a in axes:
        a.set_xlabel("iteration (×1000)")
    axes[0].legend(frameon=False, fontsize=7)
    save(fig, out, "fig8_training_metrics")


# ============================================================================ main

def _commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs", nargs="*", default=[], help="our run directories")
    ap.add_argument("--data", required=True, help="preprocessed cache .npz of the system")
    ap.add_argument("--authors", help="the authors' MSD_gen/PE/mean_operation directory")
    ap.add_argument("--authors-dc", help="the authors' dc_result_mdgan.txt")
    ap.add_argument("--config", help="config JSON the runs used (default: the runs' own config.json, else Config())")
    ap.add_argument("--out", default="runs/figures")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.config:
        cfg = Config.from_dict(json.loads(Path(args.config).read_text()))
    elif args.runs and (Path(sorted(args.runs)[0]) / "config.json").exists():
        cfg = Config.from_dict(json.loads((Path(sorted(args.runs)[0]) / "config.json").read_text()))
    else:
        cfg = Config()
    md = load_md(args.data, cfg)
    ours = load_ours(args.runs)
    authors = load_authors(args.authors, md, cfg)
    authors_dc = None
    if args.authors_dc:
        for line in Path(args.authors_dc).read_text().splitlines():
            parts = line.split()
            if parts and parts[0] == md["system"].name:
                authors_dc = float(parts[1])
    base = baselines(md, cfg)
    floor = noise_floor(md, cfg)

    counts = fig_architecture(cfg, out)
    fig_dataflow(cfg, out)
    fig_msd(md, ours, authors, base, out)
    ratios = fig_ratio(md, ours, authors, base, out)
    fig_alpha(md, ours, authors, out)
    fig_traces(ours, authors, out)
    groups = fig_summary(ours, authors, base, floor, md, authors_dc, out)
    fig_metrics(ours, out)

    summary = {
        "inputs": {"runs": sorted(args.runs), "data": args.data, "authors": args.authors, "commit": _commit()},
        "parameters": counts, "D_MD_m2s": md["D"], "authors_published_D_m2s": authors_dc,
        "noise_floor": floor,
        "baselines": {k: {"err_percent": v["err_percent"], "D_rel_error": v["D_rel_error"]} for k, v in base.items()},
        "runs": {g: {name: v for name, v in zip((ours if g == "ours" else authors), vals, strict=True)}
                 for g, vals in groups.items()},
        "median_ratio_over_checkpoints": ratios,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {len(list(out.glob('*.pdf')))} figures and summary.json to {out}")
    for g, vals in groups.items():
        print(f"{g:8s} best err {np.median([v['best_err'] for v in vals]):5.1f} % (median over runs), "
              f"blind err {np.median([v['blind_err'] for v in vals]):6.1f} %, "
              f"best D_rel {np.median([v['best_D_rel'] for v in vals]):+.2f}, "
              f"blind D_rel {np.median([v['blind_D_rel'] for v in vals]):+.2f}")
    for name, b in base.items():
        print(f"baseline {name}: err {b['err_percent']:.1f} %, D_rel {b['D_rel_error']:+.2f}")
    print(f"MD reference noise floor (perfect model): err median {floor['err_median']:.1f} %, 95th pct {floor['err_p95']:.1f} %; "
          f"|D_rel| 95th pct {floor['D_rel_abs_p95']:.2f}")


if __name__ == "__main__":
    main()