"""Phase 1: which internal variable does the D error track? (H1: critic strength, W; H2: latent persistence)

Joins, per checkpoint and for every run:
  metrics.jsonl      mean of the logged rows in the `window` iterations up to the checkpoint
                     (W, gradient penalty, grad_norm, generator loss, fake step std)
  evaluations.jsonl  err, D_rel
  diag CSV           from tools/diag_runs.py --blocks --all --csv (var/MD, rho@1..30, sum, z@lags)
  dataset.json       sigma_nm and window_std_nm, to put step std in units of the MD step

Normalisation, so runs trained in sigma units and in nm can be pooled:
  W_rel         W / median W of the same run
  step_std_rel  fake step std * sigma_nm / window_std_nm   (1 = the MD step size)

Statistics (Spearman throughout, so monotone transforms such as log D do not matter):
  pooled rho with a bootstrap CI that resamples checkpoints WITHIN each run (stratified), and the
  same correlation per run (consistency across seeds is one of the H1/H2 predictions);
  partial correlations on ranks: D_rel~z@50 | W_rel, D_rel~W_rel | z@50, D_rel~z@50 | sum (mediation);
  lead/lag: per-run Spearman of W_rel(t) with D_rel(t + k), k = -3..+3 checkpoints (k > 0: W leads).

Writes joined.csv, stats.json, scatter_matrix.(png|pdf), timeseries.(png|pdf).

Usage (repository root):
    python tools/diag_logs.py runs/chtc/run_c6954730_seed* runs/chtc/run_scaleoff_c7166270_seed* \\
        --diag-csv runs/diag_blocks_all.csv --out runs/phase1
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

TARGET = "D_ratio"                                   # D_gen / D_MD = 1 + D_rel (Spearman-equivalent to D_rel)
PREDICTORS = ["z@10", "z@50", "rho_sum_1_30", "W_rel", "step_std_rel"]
EXTRA = ["grad_norm", "var_over_md", "rho@1", "err_percent"]
LAGS = range(-3, 4)
N_BOOT = 2000


# ============================================================================ join

def _jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def _num(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def join(run_dirs: list[str], diag_csv: str, window: int = 100) -> list[dict]:
    diag = {}
    with open(diag_csv) as fh:
        for r in csv.DictReader(fh):
            if r["run"] != "MD":
                diag[(r["run"], int(float(r["iteration"])))] = r
    rows = []
    for d in sorted(run_dirs):
        d = Path(d)
        ds = json.loads((d / "dataset.json").read_text())
        sigma, wstd = float(ds["sigma_nm"]), float(ds["window_std_nm"])
        metrics = _jsonl(d / "metrics.jsonl")
        arm = "scale_off" if ds.get("scale_inputs") is False else "baseline"
        run_rows = []
        for e in _jsonl(d / "evaluations.jsonl"):
            it = e["iteration"]
            win = [m for m in metrics if it - window < m["iteration"] <= it]
            if not win or (d.name, it) not in diag:
                continue
            mean = {k: float(np.mean([m[k] for m in win])) for k in
                    ("wasserstein", "gradient_penalty", "grad_norm", "generator_loss", "fake_step_std")}
            g = diag[(d.name, it)]
            run_rows.append({
                "run": d.name, "arm": arm, "iteration": it, "checkpoint": it // 10_000,
                "D_rel": e["D_rel_error"], TARGET: 1.0 + e["D_rel_error"], "err_percent": e["err_percent"],
                "W": mean["wasserstein"], "grad_norm": mean["grad_norm"], "gradient_penalty": mean["gradient_penalty"],
                "step_std_rel": mean["fake_step_std"] * sigma / wstd,
                "var_over_md": _num(g["var_over_md"]), "rho@1": _num(g["rho@1"]),
                "rho_sum_1_30": _num(g["rho_sum_1_30"]), "z@10": _num(g["z@10"]), "z@50": _num(g["z@50"]),
            })
        if run_rows:
            w_med = float(np.median([r["W"] for r in run_rows]))
            for r in run_rows:
                r["W_rel"] = r["W"] / w_med
            rows += run_rows
    return rows


# ============================================================================ statistics

def _rank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x))
    r[order] = np.arange(len(x))
    xs = x[order]                                    # average ranks for ties
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and xs[j + 1] == xs[i]:
            j += 1
        if j > i:
            r[order[i:j + 1]] = (i + j) / 2
        i = j + 1
    return r


def spearman(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 4 or np.ptp(x[ok]) == 0 or np.ptp(y[ok]) == 0:
        return float("nan")                          # too few points, or a constant column
    return float(np.corrcoef(_rank(x[ok]), _rank(y[ok]))[0, 1])


def partial_spearman(x, y, z) -> float:
    """Spearman correlation of x and y after removing the linear effect of rank(z) from both ranks."""
    x, y, z = (np.asarray(v, float) for v in (x, y, z))
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if ok.sum() < 5 or min(np.ptp(x[ok]), np.ptp(y[ok]), np.ptp(z[ok])) == 0:
        return float("nan")
    rx, ry, rz = _rank(x[ok]), _rank(y[ok]), _rank(z[ok])
    a = np.column_stack([np.ones_like(rz), rz])
    res = lambda v: v - a @ np.linalg.lstsq(a, v, rcond=None)[0]  # noqa: E731
    return float(np.corrcoef(res(rx), res(ry))[0, 1])


def stratified_boot(rows: list[dict], stat, n: int = N_BOOT, seed: int = 0) -> tuple[float, float, float]:
    """Point estimate and 95 % CI, resampling checkpoints within each run."""
    rng = np.random.default_rng(seed)
    by_run = {}
    for i, r in enumerate(rows):
        by_run.setdefault(r["run"], []).append(i)
    groups = [np.array(v) for v in by_run.values()]
    point = stat(np.arange(len(rows)))
    draws = []
    for _ in range(n):
        idx = np.concatenate([g[rng.integers(0, len(g), len(g))] for g in groups])
        draws.append(stat(idx))
    draws = np.array(draws, float)
    if not np.isfinite(draws).any():
        return point, float("nan"), float("nan")
    lo, hi = np.nanpercentile(draws, [2.5, 97.5])
    return point, float(lo), float(hi)


def col(rows, key, idx=None) -> np.ndarray:
    v = np.array([r[key] for r in rows], float)
    return v if idx is None else v[idx]


def lead_lag(rows: list[dict]) -> dict:
    """Per run: Spearman of W_rel(t) and D_ratio(t + k); averaged over runs (k > 0: W leads)."""
    out = {}
    for k in LAGS:
        vals = []
        for run in sorted({r["run"] for r in rows}):
            rr = sorted((r for r in rows if r["run"] == run), key=lambda r: r["iteration"])
            w, d = col(rr, "W_rel"), col(rr, TARGET)
            if k > 0:
                w, d = w[:-k], d[k:]
            elif k < 0:
                w, d = w[-k:], d[:k]
            vals.append(spearman(w, d))
        out[k] = {"mean": float(np.nanmean(vals)), "per_run": vals}
    return out


def analyse(rows: list[dict]) -> dict:
    stats = {"n_checkpoints": len(rows), "runs": sorted({r["run"] for r in rows}), "spearman": {}, "partial": {}}
    for p in PREDICTORS + EXTRA:
        if p == "err_percent":
            continue
        pt, lo, hi = stratified_boot(rows, lambda i, p=p: spearman(col(rows, p, i), col(rows, TARGET, i)))
        per_run = {run: spearman(col([r for r in rows if r["run"] == run], p),
                                 col([r for r in rows if r["run"] == run], TARGET))
                   for run in stats["runs"]}
        stats["spearman"][p] = {"rho": pt, "ci95": [lo, hi], "per_run": per_run}
    for name, x, z in (("D~z@50 | W_rel", "z@50", "W_rel"), ("D~W_rel | z@50", "W_rel", "z@50"),
                       ("D~z@50 | rho_sum_1_30", "z@50", "rho_sum_1_30"),
                       ("D~rho_sum_1_30 | z@50", "rho_sum_1_30", "z@50")):
        pt, lo, hi = stratified_boot(rows, lambda i, x=x, z=z: partial_spearman(col(rows, x, i), col(rows, TARGET, i),
                                                                               col(rows, z, i)))
        stats["partial"][name] = {"rho": pt, "ci95": [lo, hi]}
    stats["lead_lag_W_vs_D"] = lead_lag(rows)
    return stats


# ============================================================================ figures

def _colors(rows):
    runs = sorted({r["run"] for r in rows})
    base = [r for r in runs if "scaleoff" not in r]
    off = [r for r in runs if "scaleoff" in r]
    cmap = {r: plt.cm.Blues(0.45 + 0.5 * i / max(1, len(base) - 1)) for i, r in enumerate(base)}
    cmap.update({r: plt.cm.Oranges(0.5 + 0.4 * i / max(1, len(off) - 1)) for i, r in enumerate(off)})
    return runs, cmap


def scatter_matrix(rows, out: Path) -> None:
    keys = [TARGET, "err_percent", "W_rel", "grad_norm", "step_std_rel", "var_over_md", "rho_sum_1_30", "z@10", "z@50"]
    labels = {TARGET: "log10 D_gen/D_MD", "err_percent": "log10 err %"}

    def values(rr, k):
        v = col(rr, k)
        return np.log10(v) if k in labels else v

    runs, cmap = _colors(rows)
    n = len(keys)
    fig, axes = plt.subplots(n, n, figsize=(13, 13))
    for i, ky in enumerate(keys):
        for j, kx in enumerate(keys):
            ax = axes[i, j]
            if i == j:
                v = values(rows, kx)
                ax.hist(v[np.isfinite(v)], bins=20, color="0.6")
            else:
                for run in runs:
                    rr = [r for r in rows if r["run"] == run]
                    ax.scatter(values(rr, kx), values(rr, ky), s=6, color=cmap[run],
                               marker="s" if "scaleoff" in run else "o")
                rho = spearman(col(rows, kx), col(rows, ky))
                ax.text(0.03, 0.88, f"ρ={rho:+.2f}" if np.isfinite(rho) else "ρ=—", transform=ax.transAxes,
                        fontsize=7, bbox={"fc": "white", "ec": "none", "alpha": 0.7, "pad": 0.5})
            ax.tick_params(labelsize=6)
            if i < n - 1:
                ax.set_xticklabels([])
            if j > 0:
                ax.set_yticklabels([])
            if i == n - 1:
                ax.set_xlabel(labels.get(kx, kx), fontsize=7.5)
            if j == 0:
                ax.set_ylabel(labels.get(ky, ky), fontsize=7.5)
    handles = [plt.Line2D([], [], marker="s" if "scaleoff" in r else "o", ls="", color=cmap[r], label=r) for r in runs]
    fig.suptitle("Phase 1: every checkpoint (circles: baseline, sigma units; squares: scale_inputs=False); "
                 "ρ = Spearman", y=0.995, fontsize=10)
    fig.legend(handles=handles, loc="upper center", ncol=4, fontsize=7, frameon=False, bbox_to_anchor=(0.5, 0.975))
    fig.subplots_adjust(top=0.93, hspace=0.08, wspace=0.08)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"scatter_matrix.{ext}", bbox_inches="tight", dpi=130)
    plt.close(fig)


def timeseries(rows, out: Path) -> None:
    panels = [(TARGET, "D_gen / D_MD", True), ("z@50", "z@50", False), ("rho_sum_1_30", "Σρ (lags 1-30)", False),
              ("W_rel", "W / run median", False), ("step_std_rel", "step std / MD step", False)]
    runs, cmap = _colors(rows)
    fig, axes = plt.subplots(len(panels), 1, figsize=(8, 10), sharex=True)
    for ax, (k, label, logy) in zip(axes, panels, strict=True):
        for run in runs:
            rr = sorted((r for r in rows if r["run"] == run), key=lambda r: r["iteration"])
            ax.plot(col(rr, "iteration") / 1000, col(rr, k), color=cmap[run], lw=1.0,
                    marker="s" if "scaleoff" in run else "o", ms=2.5, label=run if k == TARGET else None)
        if logy:
            ax.set_yscale("log")
            ax.axhline(1, color="black", lw=0.8)
        if k in ("step_std_rel", "W_rel"):
            ax.axhline(1, color="black", lw=0.6, ls=":")
        if k in ("z@50", "rho_sum_1_30"):
            ax.axhline(0, color="black", lw=0.6, ls=":")
        ax.set_ylabel(label, fontsize=8)
        ax.tick_params(labelsize=7)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=4, fontsize=7, frameon=False,
               bbox_to_anchor=(0.5, 1.02))
    axes[-1].set_xlabel("iteration (×1000)")
    for ext in ("png", "pdf"):
        fig.savefig(out / f"timeseries.{ext}", bbox_inches="tight", dpi=130)
    plt.close(fig)


# ============================================================================ report

def report(stats: dict) -> None:
    print(f"{stats['n_checkpoints']} checkpoints from {len(stats['runs'])} runs\n")
    print("Spearman with D_gen/D_MD: pooled [95 % CI, stratified bootstrap]   per run")
    for p, s in stats["spearman"].items():
        per = "  ".join(f"{v:+.2f}" for v in s["per_run"].values())
        print(f"  {p:<13} {s['rho']:+.2f} [{s['ci95'][0]:+.2f}, {s['ci95'][1]:+.2f}]   {per}")
    print("\nPartial correlations (ranks)")
    for name, s in stats["partial"].items():
        print(f"  {name:<24} {s['rho']:+.2f} [{s['ci95'][0]:+.2f}, {s['ci95'][1]:+.2f}]")
    print("\nLead/lag: Spearman(W_rel(t), D(t+k)), mean over runs (k > 0 means W leads D)")
    print("  " + "  ".join(f"k={k:+d}: {v['mean']:+.2f}" for k, v in stats["lead_lag_W_vs_D"].items()))
    sp = stats["spearman"]
    print("\nPredictions (written before computing; diagnostic plan, Phase 1):")
    print(f"  H1: rho(D, W_rel) strongly positive        observed {sp['W_rel']['rho']:+.2f}")
    print(f"  H2: rho(D, z@50) strongly positive         observed {sp['z@50']['rho']:+.2f}")
    print(f"  H2: rho(D, sum) strongly positive          observed {sp['rho_sum_1_30']['rho']:+.2f}")
    print(f"  H1: rho(D, step std) negative              observed {sp['step_std_rel']['rho']:+.2f}")
    print(f"  H1: grad_norm drifts with D                observed rho(D, grad_norm) {sp['grad_norm']['rho']:+.2f}")
    ll = stats["lead_lag_W_vs_D"]
    best_k = max(ll, key=lambda k: ll[k]["mean"])
    print(f"  H1: W leads D by >= 1 checkpoint           strongest lag k={best_k:+d} ({ll[best_k]['mean']:+.2f})")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("runs", nargs="+", help="run directories")
    ap.add_argument("--diag-csv", required=True, help="CSV from tools/diag_runs.py --blocks --all --csv")
    ap.add_argument("--window", type=int, default=100, help="metric iterations averaged before each checkpoint")
    ap.add_argument("--out", default="runs/phase1")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = join(args.runs, args.diag_csv, args.window)
    if not rows:
        raise SystemExit("no checkpoints joined: check that the diag CSV run names match the run directories")
    with open(out / "joined.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    stats = analyse(rows)
    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    scatter_matrix(rows, out)
    timeseries(rows, out)
    report(stats)
    print(f"\nwrote joined.csv, stats.json, scatter_matrix.(png|pdf), timeseries.(png|pdf) to {out}")


if __name__ == "__main__":
    main()