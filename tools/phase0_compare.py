"""Phase 0: is the long-time overestimate our bug or a property of the method?

Applies the same analysis to every checkpoint curve from two sources:
  - ours:    run directories (msd_XXXXXX.npy on the 10 ps grid, from evaluate)
  - authors: gen_data/MSD_gen/PE/mean_operation/01ss_msd_bs0064-nw032-ite300000_scale_addex{1..5}/
             {Z}ln_skip10.npy (799 values on t = 100 (k+1) ps; t = 0 prepended here)
against one MD reference (time-averaged MSD of the preprocessed trajectories).

Per curve:
  err        2023 eqs. 6-7, 0.2-78 ns (mdgan.data.msd.err)
  D_rel      D_gen / D_MD - 1, D from the 10-20 ns OLS slope (mdgan.data.msd.diffusion_coefficient)
  ratio@t    MSD_gen / MSD_MD at t = 5, 10, 20, 40, 78 ns (multiples of 100 ps: on both grids)
  fit        MSD = 6 D_fit t + v^2 t^2 over 5-78 ns (least squares, no intercept). A positive v^2 is
             upward bending (persistent drift); drift78 = v^2 t / (6 D_fit) at t = 78 ns is the
             quadratic term relative to the linear one there. The MD itself is fitted the same way,
             as the reference value of drift78.

Usage (repository root):
    python tools/phase0_compare.py \\
        --ours runs/chtc/run_c6954730_seed* \\
        --authors data/raw/wat_in_polymer_mdgan/gen_data/MSD_gen/PE/mean_operation \\
        --data data/cache/PE.npz --out runs/phase0
Writes curves.csv (one row per checkpoint), summary.json, and two figures (PDF + PNG).
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import FixedLocator, NullFormatter, ScalarFormatter  # noqa: E402

from mdgan.config import Config  # noqa: E402
from mdgan.data.msd import diffusion_coefficient, err, msd_time_averaged  # noqa: E402
from mdgan.data.preprocess import SystemData  # noqa: E402

RATIO_NS = (5, 10, 20, 40, 78)
FIT_NS = (5.0, 78.0)
AUTH_DT_PS = 100.0
COLORS = {"ours": "#1f77b4", "authors": "#d95f02"}


def fit_ballistic(curve: np.ndarray, dt_ps: float) -> tuple[float, float]:
    """Least squares MSD = a t + b t^2 on FIT_NS (t in ns). Returns (a = 6 D_fit, b = v^2) in nm^2/ns, nm^2/ns^2."""
    t = np.arange(len(curve)) * dt_ps / 1000
    m = (t >= FIT_NS[0]) & (t <= FIT_NS[1])
    x = np.column_stack([t[m], t[m] ** 2])
    (a, b), *_ = np.linalg.lstsq(x, curve[m], rcond=None)
    return float(a), float(b)


def analyse(curve, dt_ps, ref, dt_ref, d_md, cfg) -> dict:
    a, b = fit_ballistic(curve, dt_ps)
    d, _ = diffusion_coefficient(curve, dt_ps, *cfg.diffusion_fit_ps)
    row = {"err": err(curve, dt_ps, ref, dt_ref, *cfg.err_window_ps), "D_rel": d / d_md - 1,
           "v2_nm2_per_ns2": b, "drift78": b * 78.0 / a if a > 0 else float("nan")}
    for t in RATIO_NS:
        row[f"ratio@{t}ns"] = float(curve[int(round(t * 1000 / dt_ps))] / ref[int(round(t * 1000 / dt_ref))])
    return row


def load_ours(dirs) -> list[dict]:
    rows = []
    for d in sorted(dirs):
        d = Path(d)
        for line in (d / "evaluations.jsonl").read_text().splitlines():
            if line.strip():
                e = json.loads(line)
                rows.append({"source": "ours", "run": d.name, "iteration": e["iteration"],
                             "curve": np.load(d / f"msd_{e['iteration']:06d}.npy"), "dt": e["dt_gen_ps"]})
    return rows


def load_authors(root) -> list[dict]:
    rows = []
    for d in sorted(Path(root).glob("01ss_msd_bs0064-nw032-ite300000_scale_addex*")):
        for f in d.glob("*ln_skip10.npy"):
            rows.append({"source": "authors", "run": "run " + d.name.rsplit("addex", 1)[1],
                         "iteration": int(re.match(r"(\d+)ln", f.name).group(1)),
                         "curve": np.concatenate([[0.0], np.load(f).astype(np.float64)]), "dt": AUTH_DT_PS})
    return sorted(rows, key=lambda r: (r["run"], r["iteration"]))


def summarize(rows: list[dict]) -> dict:
    out = {}
    for src in ("ours", "authors"):
        r = [x for x in rows if x["source"] == src]
        if not r:
            continue
        def col(k, r=r):
            return np.array([x[k] for x in r])

        drel, drift = col("D_rel"), col("drift78")
        out[src] = {
            "n_curves": len(r), "n_runs": len({x["run"] for x in r}),
            "fraction_D_rel_positive": float(np.mean(drel > 0)),
            "median_D_rel": float(np.median(drel)), "median_err": float(np.median(col("err"))),
            "fraction_D_rel_over_+100%": float(np.mean(drel > 1.0)),
            **{f"median_ratio@{t}ns": float(np.median(col(f"ratio@{t}ns"))) for t in RATIO_NS},
            "median_drift78": float(np.median(drift)),
            "fraction_drift78_over_0.25": float(np.mean(drift > 0.25)),
        }
    return out


def _log_ticks(ax, ticks):
    lo, hi = ax.get_ylim()
    ax.yaxis.set_major_locator(FixedLocator([t for t in ticks if lo <= t <= hi]))
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())


def plot_traces(rows, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    for src, marker in (("ours", "o"), ("authors", "s")):
        runs = sorted({x["run"] for x in rows if x["source"] == src})
        for i, run in enumerate(runs):
            r = sorted((x for x in rows if x["source"] == src and x["run"] == run), key=lambda x: x["iteration"])
            k = np.arange(1, len(r) + 1)
            ax.plot(k, [1 + x["D_rel"] for x in r], color=COLORS[src], lw=0.8, alpha=0.75, marker=marker, ms=2.5,
                    label=f"{src} ({len(runs)} runs)" if i == 0 else None)
    ax.axhline(1, color="black", lw=0.9)
    ax.set_yscale("log")
    ax.set_xlabel("checkpoint index (× 10k iterations)")
    ax.set_ylabel("D_gen / D_MD")
    ax.set_title("D at every checkpoint, every run")
    _log_ticks(ax, [0.3, 0.5, 0.7, 1, 1.5, 2, 3, 5, 10, 20, 50, 100])
    ax.legend(frameon=False)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"phase0_D_vs_checkpoint.{ext}", bbox_inches="tight", dpi=150)
    plt.close(fig)


def plot_hist(rows, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    vals = {src: np.array([1 + x["D_rel"] for x in rows if x["source"] == src]) for src in ("ours", "authors")}
    allv = np.concatenate([v for v in vals.values() if v.size])
    bins = np.logspace(np.log10(max(allv.min(), 0.05) * 0.9), np.log10(allv.max() * 1.1), 40)
    for src, v in vals.items():
        if v.size:
            ax.hist(v, bins=bins, color=COLORS[src], alpha=0.55,
                    label=f"{src}: median {np.median(v) - 1:+.2f}, {np.mean(v > 1) * 100:.0f} % above MD")
    ax.axvline(1, color="black", lw=0.9)
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator([0.3, 0.5, 1, 2, 5, 10, 30, 100]))
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("D_gen / D_MD")
    ax.set_ylabel("checkpoints")
    ax.set_title("Distribution of D over all checkpoints")
    ax.legend(frameon=False, fontsize=8)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"phase0_D_histogram.{ext}", bbox_inches="tight", dpi=150)
    plt.close(fig)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ours", nargs="*", default=[], help="our run directories")
    ap.add_argument("--authors", required=True, help="the authors' MSD_gen/PE/mean_operation directory")
    ap.add_argument("--data", required=True, help="preprocessed cache .npz of the system")
    ap.add_argument("--out", default="runs/phase0")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cfg = Config()
    system = SystemData.load(args.data)
    ref, _ = msd_time_averaged(system.displacements)
    dt_ref = float(system.dt_ps)
    d_md, _ = diffusion_coefficient(ref, dt_ref, *cfg.diffusion_fit_ps)
    a_md, b_md = fit_ballistic(ref, dt_ref)

    rows = load_ours(args.ours) + load_authors(args.authors)
    for r in rows:
        r.update(analyse(r.pop("curve"), r.pop("dt"), ref, dt_ref, d_md, cfg))

    keys = ["source", "run", "iteration", "err", "D_rel", *[f"ratio@{t}ns" for t in RATIO_NS],
            "v2_nm2_per_ns2", "drift78"]
    with open(out / "curves.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows({k: r[k] for k in keys} for r in rows)
    summary = {"D_MD_m2s": d_md, "MD_fit": {"v2_nm2_per_ns2": b_md, "drift78": b_md * 78.0 / a_md},
               **summarize(rows)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    plot_traces(rows, out)
    plot_hist(rows, out)

    print(f"MD reference: D = {d_md:.4e} m^2/s; its own 5-78 ns fit gives drift78 = {b_md * 78.0 / a_md:+.3f}")
    head = f"{'':10s} {'curves':>6} {'D_rel>0':>8} {'med D_rel':>10} {'>+100%':>7} {'med err':>8}  " + \
        "  ".join(f"r@{t}ns" for t in RATIO_NS) + f"  {'drift78':>8} {'>0.25':>6}"
    print(head)
    for src in ("ours", "authors"):
        if src in summary:
            s = summary[src]
            print(f"{src:10s} {s['n_curves']:>6} {s['fraction_D_rel_positive'] * 100:7.0f}% {s['median_D_rel']:+10.2f} "
                  f"{s['fraction_D_rel_over_+100%'] * 100:6.0f}% {s['median_err']:7.1f}%  "
                  + "  ".join(f"{s[f'median_ratio@{t}ns']:6.2f}" for t in RATIO_NS)
                  + f"  {s['median_drift78']:+8.3f} {s['fraction_drift78_over_0.25'] * 100:5.0f}%")
    print(f"wrote {out}/curves.csv, summary.json, phase0_D_vs_checkpoint.(pdf|png), phase0_D_histogram.(pdf|png)")


if __name__ == "__main__":
    main()