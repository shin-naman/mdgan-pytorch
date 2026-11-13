"""Matched-grid rescoring: every reproduction checkpoint and every author checkpoint under one definition.

Definition (the authors' archive convention, as recovered by the audit):
    grid        100 ps: a curve is MSD at t_k = 100 k ps, k = 0..799, with MSD(0) = 0. Reproduction curves
                (msd_XXXXXX.npy, 10 ps) give every tenth point from 100 ps to 79.9 ns; author curves (799 values)
                are assigned to 100 ps .. 79.9 ns; the MD reference (msd.out, 10 ps) likewise.
    reference   the authors' msd.out (column 1, mean MSD), for every row. Our cache's MD agrees with it to the
                level printed in the CSV header: a convention, not a correction.
    err         weighted mean of 100 |MSD_gen - MSD_MD| / MSD_MD with weights 1/k, k = 2..780 (0.2-78 ns).
    D           OLS slope (with intercept) over 10 < t <= 20 ns (k = 101..200), / 6, in m^2/s; D/D_MD.
    curvature   MSD = a t + b t^2 fitted (no intercept) over 5-78 ns (k = 50..780); flagged when b 78 / a > 0.25,
                or a <= 0 (the fit puts everything in the quadratic term; no author fit has a <= 0).

Blocks: 20-50k for every run; 10-100k for runs that reached 100k; 10-300k for runs that reached 300k. The
authors appear in every block. Columns: runs, checkpoints, median err, mean and median of the per-run best
err, median final err (each run's last checkpoint in the block), median D/D_MD, curvature-flag frequency.
running_stats rows are the training-time evaluation, i.e. the running-average generator, and are labelled
so and left out of the ranking note.

Usage (repository root):
    python tools/rescore_grid.py --self-test
    python tools/rescore_grid.py runs/chtc/run_* --authors <.../gen_data/MSD_gen/PE/mean_operation> \\
        --md <.../msd.out> --cache data/cache/PE.npz --out runs/rescore
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np

DT_GRID = 100.0
N_GRID = 800                                   # k = 0..799 (t = 0..79.9 ns)
ERR_K = np.arange(2, 781)                      # 0.2-78 ns
D_K = np.arange(101, 201)                      # 10 < t <= 20 ns
CURV_K = np.arange(50, 781)                    # 5-78 ns
NM2_PER_PS_TO_M2_PER_S = 1e-6
BLOCKS = (("20-50k", (20000, 30000, 40000, 50000)),
          ("10-100k", tuple(range(10000, 100001, 10000))),
          ("10-300k", tuple(range(10000, 300001, 10000))))
AS_TRAINED = {"running_stats": "running_stats (as trained: running averages)"}
EXCLUDE_ARMS = {"sizing"}


# ============================================================================ the definition

def native_to_grid(msd_native: np.ndarray) -> np.ndarray:
    """10 ps curve (index = t / 10 ps) -> 100 ps grid: MSD(0) = 0, then every tenth point from 100 ps to 79.9 ns."""
    if len(msd_native) < 10 * (N_GRID - 1) + 1:
        raise ValueError(f"native curve has {len(msd_native)} points; need {10 * (N_GRID - 1) + 1}")
    return np.concatenate([[0.0], np.asarray(msd_native, float)[10:10 * N_GRID:10]])


def author_to_grid(msd_author: np.ndarray) -> np.ndarray:
    """799 values on 100 ps .. 79.9 ns -> prepend MSD(0) = 0."""
    if len(msd_author) != N_GRID - 1:
        raise ValueError(f"author curve has {len(msd_author)} values; expected {N_GRID - 1}")
    return np.concatenate([[0.0], np.asarray(msd_author, float)])


def err_percent(gen: np.ndarray, ref: np.ndarray) -> float:
    w = 1.0 / ERR_K
    return float((w * 100.0 * np.abs(gen[ERR_K] - ref[ERR_K]) / ref[ERR_K]).sum() / w.sum())


def diffusion_m2s(curve: np.ndarray) -> float:
    t = D_K * DT_GRID
    return float(np.polyfit(t, curve[D_K], 1)[0] / 6.0 * NM2_PER_PS_TO_M2_PER_S)


def curvature_flag(curve: np.ndarray) -> bool:
    t = CURV_K * DT_GRID / 1000.0
    (a, b), *_ = np.linalg.lstsq(np.column_stack([t, t * t]), curve[CURV_K], rcond=None)
    return bool(a <= 0 or b * 78.0 / a > 0.25)


def score(curve: np.ndarray, ref: np.ndarray, d_md: float) -> dict:
    return {"err": err_percent(curve, ref), "D_ratio": diffusion_m2s(curve) / d_md, "curved": curvature_flag(curve)}


def load_md_out(path: Path) -> np.ndarray:
    """msd.out: time [ps] in column 0, mean MSD in column 1, 10 ps grid -> the 100 ps grid."""
    rows = np.loadtxt(path, comments="#", usecols=(0, 1))
    if not np.allclose(rows[:11, 0], np.arange(11) * 10.0):
        raise ValueError(f"{path}: expected a 10 ps grid starting at 0")
    return native_to_grid(rows[:, 1])


# ============================================================================ collecting curves

def _arm(run_name: str) -> str:
    m = re.fullmatch(r"run_(?:(.+?)_)?c\d+_seed\d+", run_name)
    return (m.group(1) or "baseline") if m else run_name


def reproduction_curves(run_dirs: list[Path]) -> dict:
    """{arm: {run: {iteration: grid curve}}} from msd_XXXXXX.npy, plus each run's native err / D from
    evaluations.jsonl for the ranking comparison."""
    out, native = {}, {}
    for r in sorted(run_dirs):
        arm = _arm(r.name)
        if arm in EXCLUDE_ARMS or not r.is_dir():
            continue
        curves = {int(f.stem.split("_")[1]): native_to_grid(np.load(f)) for f in sorted(r.glob("msd_*.npy"))}
        if not curves:
            continue
        out.setdefault(arm, {})[r.name] = curves
        ev = r / "evaluations.jsonl"
        if ev.exists():
            rows = [json.loads(x) for x in ev.read_text().splitlines() if x.strip()]
            native.setdefault(arm, {})[r.name] = {e["iteration"]: e["err_percent"] for e in rows}
    return {"curves": out, "native_err": native}


def author_curves(root: Path, variant: str = "nw032") -> dict:
    """{run: {iteration: grid curve}} for the authors' mean-operation runs of one latent size."""
    runs = sorted(d for d in root.iterdir() if d.is_dir() and f"-{variant}-" in d.name)
    if not runs:
        raise SystemExit(f"no '{variant}' runs under {root}")
    out = {}
    for d in runs:
        out[d.name] = {int(f.name.split("ln_")[0]): author_to_grid(np.load(f)) for f in d.glob("*ln_skip10.npy")}
    return out


# ============================================================================ the table

def block_rows(name: str, runs: dict, its: tuple, ref: np.ndarray, d_md: float) -> dict | None:
    """Aggregate over runs that have every checkpoint of the block."""
    complete = {run: [c[i] for i in its] for run, c in runs.items() if all(i in c for i in its)}
    if not complete:
        return None
    sc = {run: [score(c, ref, d_md) for c in cs] for run, cs in complete.items()}
    flat = [s for ss in sc.values() for s in ss]
    best = [min(s["err"] for s in ss) for ss in sc.values()]
    final = [ss[-1]["err"] for ss in sc.values()]
    return {"arm": name, "runs": len(sc), "checkpoints": len(flat),
            "median_err": float(np.median([s["err"] for s in flat])),
            "mean_best_err": float(np.mean(best)), "median_best_err": float(np.median(best)),
            "median_final_err": float(np.median(final)),
            "median_D_ratio": float(np.median([s["D_ratio"] for s in flat])),
            "curvature_flag_frequency": float(np.mean([s["curved"] for s in flat]))}


def native_median(native: dict, its: tuple, runs: dict) -> float | None:
    vals = [native[run][i] for run in runs if run in native and all(i in native[run] for i in its) for i in its]
    return float(np.median(vals)) if vals else None


def build(args) -> None:
    ref = load_md_out(Path(args.md))
    d_md = diffusion_m2s(ref)
    rep = reproduction_curves([Path(p) for p in args.runs])
    auth = author_curves(Path(args.authors), args.variant)
    rows10 = np.loadtxt(args.md, comments="#", usecols=(0, 1))
    w10 = (rows10[:, 0] > 10_000.0) & (rows10[:, 0] <= 20_000.0)
    d_md10 = float(np.polyfit(rows10[w10, 0], rows10[w10, 1], 1)[0] / 6.0 * NM2_PER_PS_TO_M2_PER_S)
    header = [f"# matched-grid rescoring: 100 ps grid, reference {Path(args.md).name} (D_MD {d_md:.6e} m^2/s on this grid, "
              f"10 < t <= 20 ns), err weights 1/k k=2..780, curvature flag b*78/a > 0.25 or a <= 0 on 5-78 ns",
              f"# the published D_MD ({d_md10:.8e}) is the same reference on its native 10 ps grid; on the 100 ps grid used "
              f"for every row here it is {(d_md / d_md10 - 1) * 100:+.3f} %"]
    if args.cache:
        from mdgan.data.msd import msd_time_averaged
        from mdgan.data.preprocess import SystemData
        ours = native_to_grid(msd_time_averaged(SystemData.load(args.cache).displacements)[0])
        header.append(f"# our cache's MD on the same grid and window: D {diffusion_m2s(ours):.6e} m^2/s, "
                      f"{(diffusion_m2s(ours) / d_md - 1) * 100:+.4f} % vs the reference; mean |MSD difference| over k=2..780 "
                      f"{np.mean(np.abs(ours[ERR_K] / ref[ERR_K] - 1)) * 100:.4f} % (a convention, not a correction)")
    header.append("# best-per-run err: mean and median of each run's minimum over the block; runs = run count")
    rows, notes = [], []
    for block, its in BLOCKS:
        block_out = []
        r = block_rows(f"authors ({args.variant}, mean operation)", auth, its, ref, d_md)
        if r:
            block_out.append(r)
        for arm, runs in sorted(rep["curves"].items(), key=lambda kv: (kv[0] != "baseline", kv[0])):
            r = block_rows(AS_TRAINED.get(arm, arm), runs, its, ref, d_md)
            if r:
                r["_arm"] = arm
                block_out.append(r)
        for r in block_out:
            rows.append({"block": block, **{k: v for k, v in r.items() if not k.startswith("_")}})
        # ranking note: matched vs native median err, reproduction arms only (running_stats excluded)
        ranked = [r for r in block_out if r.get("_arm") and r["_arm"] not in AS_TRAINED]
        nat = {r["_arm"]: native_median(rep["native_err"].get(r["_arm"], {}), its, rep["curves"][r["_arm"]]) for r in ranked}
        ranked = [r for r in ranked if nat[r["_arm"]] is not None]
        order_m = [r["_arm"] for r in sorted(ranked, key=lambda r: r["median_err"])]
        order_n = [a for a, _ in sorted(((r["_arm"], nat[r["_arm"]]) for r in ranked), key=lambda x: x[1])]
        notes.append(f"{block}: ranking by median err " + ("unchanged" if order_m == order_n else
                     f"CHANGED: native {order_n} -> matched {order_m}") + f" ({len(order_m)} arms)")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cols = ["block", "arm", "runs", "checkpoints", "median_err", "mean_best_err", "median_best_err",
            "median_final_err", "median_D_ratio", "curvature_flag_frequency"]
    with open(out / "matched_grid.csv", "w", newline="") as f:
        f.write("\n".join(header) + "\n")
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})
    (out / "ranking_note.txt").write_text("\n".join(notes) + "\n")
    print("\n".join(header))
    print(f"\n{'block':<8} {'arm':<46} {'runs':>4} {'ckpts':>5} {'med err':>8} {'best mean':>9} {'best med':>8} "
          f"{'final':>7} {'D/D_MD':>7} {'curved':>7}")
    for r in rows:
        print(f"{r['block']:<8} {r['arm']:<46} {r['runs']:>4} {r['checkpoints']:>5} {r['median_err']:>8.2f} "
              f"{r['mean_best_err']:>9.2f} {r['median_best_err']:>8.2f} {r['median_final_err']:>7.2f} "
              f"{r['median_D_ratio']:>7.3f} {r['curvature_flag_frequency'] * 100:>6.0f}%")
    print("\n" + "\n".join(notes))
    print(f"\nwrote {out / 'matched_grid.csv'} and {out / 'ranking_note.txt'}")


# ============================================================================ self-test

def self_test(authors: str | None, md: str | None) -> None:
    # 1. the grid: a native curve c t lands on the grid exactly, with D = c / 6
    native = np.arange(8001) * 10.0 * 0.0042
    g = native_to_grid(native)
    assert len(g) == N_GRID and g[0] == 0 and np.isclose(g[1], 100.0 * 0.0042) and np.isclose(g[-1], 79900.0 * 0.0042)
    assert np.isclose(diffusion_m2s(g), 0.0042 / 6 * 1e-6, rtol=1e-12)
    assert np.array_equal(author_to_grid(g[1:]), g)
    print("control 1: native and author curves land on the same 100 ps grid; a linear MSD gives D = slope / 6 exactly")
    # 2. err: a curve 3 % above the reference everywhere scores 3 %
    ref = 0.01 + np.arange(N_GRID) * 0.0042 * 100
    assert np.isclose(err_percent(ref * 1.03, ref), 3.0, rtol=1e-12)
    # 3. endpoints: only 10 < t <= 20 ns enters D
    assert D_K[0] * DT_GRID == 10100.0 and D_K[-1] * DT_GRID == 20000.0
    spiky = g.copy()
    spiky[100] += 1e6                                                     # t = 10 ns exactly: excluded
    assert np.isclose(diffusion_m2s(spiky), diffusion_m2s(g), rtol=1e-12)
    print("control 2: err weights 1/k over 0.2-78 ns; the D window excludes t = 10 ns and includes 20 ns")
    # 4. curvature flag on both sides of 0.25, and a <= 0
    t = np.arange(N_GRID) * DT_GRID / 1000.0
    assert not curvature_flag(1.0 * t + 0.2 / 78.0 * t ** 2) and curvature_flag(1.0 * t + 0.3 / 78.0 * t ** 2)
    assert curvature_flag(-0.1 * t + 0.01 * t ** 2)
    print("control 3: the curvature flag switches at b 78 / a = 0.25 and flags a <= 0")
    # 5. the audit's numbers on the authors' archive (if given): err does not depend on the D endpoints
    if authors and md:
        refg = load_md_out(Path(md))
        dm = diffusion_m2s(refg)
        auth = author_curves(Path(authors))
        sc = {run: {it: score(c, refg, dm) for it, c in cs.items()} for run, cs in auth.items()}
        errs = [s["err"] for ss in sc.values() for s in ss.values()]
        best = [min(s["err"] for s in ss.values()) for ss in sc.values()]
        assert len(errs) == 150, len(errs)
        assert abs(np.median(errs) - 16.25) < 0.01, np.median(errs)
        assert np.allclose(sorted(best), sorted([7.55, 5.64, 5.98, 4.66, 6.63]), atol=0.006), best
        print(f"control 4: the authors' 150 curves reproduce the audit: median err {np.median(errs):.2f} %, per-run best "
              + " ".join(f"{b:.2f}" for b in best))
    else:
        print("control 4: skipped (pass --authors and --md to check the audit's numbers)")
    print("all positive controls passed")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="*", help="reproduction run directories")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--authors", help="the archive's gen_data/MSD_gen/PE/mean_operation directory")
    ap.add_argument("--variant", default="nw032", help="latent size of the author runs (ours: 32)")
    ap.add_argument("--md", help="the authors' msd.out")
    ap.add_argument("--cache", help="our MD cache (PE.npz), for the agreement line in the header")
    ap.add_argument("--out", default="runs/rescore")
    args = ap.parse_args(argv)
    if args.self_test:
        self_test(args.authors, args.md)
        return
    if not (args.runs and args.authors and args.md):
        ap.error("runs, --authors and --md are required")
    build(args)


if __name__ == "__main__":
    main()
