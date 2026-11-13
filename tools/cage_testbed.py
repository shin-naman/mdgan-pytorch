"""Miniature test bed (plan step 8): the cage-walk dataset, and the verdicts read from arm-eval rows.

data      writes the synthetic "MD" (60 trajectories x 8000 steps at 10 ps, PE-shaped, so every evaluation
          window applies unchanged) of CageWalk.miniature() as a SystemData .npz, with the exact D in its meta;
          and the 8a deliverable: the time-averaged MSD of a large simulation against the analytic curve, with
          the parameters (figure + JSON).
verdict   reads arm-eval JSON (evaluated with --block-steps 8 and --data the synthetic .npz) and applies the
          failure definition per seed: at one or more checkpoints from 2k on, (a period <= 10 blocks holding
          >= 20 % of latent variance, or DC share max dim > 0.3) and Σρ > +0.5 (8-step blocks). An arm reproduces
          the failure when >= 3 of its seeds do (8b). For a sweep (8c), each arm (one latent_noise_share value)
          passes when no seed meets the definition and the median D/D_analytic over checkpoints 7-10k is within
          20 % of 1.

Usage (repository root):
    python tools/cage_testbed.py --self-test
    python tools/cage_testbed.py data --out data/cache/cage_mini.npz --figure-dir runs/testbed
    python tools/cage_testbed.py verdict runs/testbed_8b/arm-eval.json [more.json ...]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

FROM_ITERATION = 2000
PERIOD_MAX, CYCLE_MIN, DC_MAX, RHO_MIN = 10.0, 0.20, 0.30, 0.5
LATE = (7000, 8000, 9000, 10000)
D_TOL = 0.20


# ============================================================================ data and the 8a figure

def write_data(out: Path, figure_dir: Path, n_traj: int = 60, n_steps: int = 8000, seed: int = 2026) -> dict:
    from mdgan.data.msd import msd_time_averaged
    from mdgan.data.synthetic import CageWalk
    w = CageWalk.miniature()
    system = w.simulate(n_traj, n_steps, seed=seed, name="cage_mini")
    out.parent.mkdir(parents=True, exist_ok=True)
    system.save(out)
    big = w.simulate(1024, 4000, seed=seed + 1)
    msd, _ = msd_time_averaged(big.displacements)
    lag = np.arange(160, 481)                                        # blocks 20-60 of M = 8
    slope = float(np.polyfit(lag, msd[lag], 1)[0])
    own, _ = msd_time_averaged(system.displacements)
    params = {"cage_rms_nm_per_axis": w.cage_rms_nm, "tau_steps": w.tau_steps, "hop_length_nm": w.hop_length_nm,
              "hop_rate_per_step": w.hop_rate_per_step, "crossover_steps": w.crossover_steps,
              "step_rms_nm_per_axis": w.step_rms_nm(), "hop_over_step_rms": w.hop_length_nm / w.step_rms_nm(),
              "local_exponent_block1_block2": [float(a) for a in w.local_exponent(np.array([8, 16]))],
              "D_exact_m2s": w.diffusion_coefficient_m2s(), "dt_ps": w.dt_ps,
              "slope_blocks_20_60_over_kappa_h2_minus_1": slope / (w.hop_rate_per_step * w.hop_length_nm ** 2) - 1,
              "dataset": {"path": str(out), "trajectories": n_traj, "steps": n_steps, "seed": seed,
                          "own_slope_blocks_20_60_over_kappa_h2_minus_1":
                              float(np.polyfit(lag, own[lag], 1)[0]) / (w.hop_rate_per_step * w.hop_length_nm ** 2) - 1}}
    figure_dir.mkdir(parents=True, exist_ok=True)
    (figure_dir / "cage_mini_params.json").write_text(json.dumps(params, indent=2))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = np.arange(1, 4001)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.loglog(t, msd[1:], color="tab:blue", label="simulated (1024 x 4000), time-averaged")
    ax.loglog(t, w.msd(t), "k--", lw=1, label="analytic 6s²(1-e^{-t/τ}) + κh²t")
    ax.loglog(t, w.hop_rate_per_step * w.hop_length_nm ** 2 * t, color="gray", lw=0.8, label="κh²t (long-time line)")
    ax.axvspan(1, 16, color="tab:green", alpha=0.08, label="critic window (16 steps)")
    ax.axvline(w.crossover_steps, color="tab:red", lw=0.8, label=f"t* = {w.crossover_steps:.0f} steps (block 6)")
    ax.axvspan(160, 480, color="tab:orange", alpha=0.08, label=f"blocks 20-60: slope {params['slope_blocks_20_60_over_kappa_h2_minus_1']:+.1%}")
    ax.set_xlabel("lag (steps of 10 ps; M = 8 per block)")
    ax.set_ylabel("MSD (units², 3D)")
    ax.set_title(f"cage walk, miniature: s = 1/axis, τ = 2, h = {w.hop_length_nm:.2f} (7 step rms), "
                 f"κ = {w.hop_rate_per_step:.5f}/step", fontsize=9)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(figure_dir / "cage_mini_msd.png", dpi=130)
    plt.close(fig)
    return params


# ============================================================================ verdicts

def failure_at(r: dict) -> bool:
    latent = (r["period_blocks"] <= PERIOD_MAX and r["cycle_share"] >= CYCLE_MIN) or r["dc_share_max_dim"] > DC_MAX
    return bool(latent and r["rho_sum_1_30"] > RHO_MIN)


def verdicts(rows: list[dict]) -> dict:
    """{arm: {"seeds": {run: {...}}, "reproduced": n, "share": a or None, "median_D_analytic_late": x, ...}}"""
    out = {}
    for arm in sorted({r["arm"] for r in rows}):
        rr = [r for r in rows if r["arm"] == arm and r.get("draw", "config") == "config"]
        seeds = {}
        for run in sorted({r["run"] for r in rr}):
            mine = sorted((r for r in rr if r["run"] == run), key=lambda r: r["iteration"])
            hits = [r["iteration"] for r in mine if r["iteration"] >= FROM_ITERATION and failure_at(r)]
            late = [r.get("D_ratio_analytic") for r in mine if r["iteration"] in LATE and r.get("D_ratio_analytic") is not None]
            seeds[run] = {"failure_at": hits, "reproduces": bool(hits),
                          "median_D_analytic_late": float(np.median(late)) if late else None}
        late_all = [s["median_D_analytic_late"] for s in seeds.values() if s["median_D_analytic_late"] is not None]
        n_rep = sum(s["reproduces"] for s in seeds.values())
        med_d = float(np.median(late_all)) if late_all else None
        out[arm] = {"seeds": seeds, "reproduced": n_rep, "n_seeds": len(seeds),
                    "failure_reproduced": n_rep >= 3,
                    "share": rr[0].get("latent_noise_share") if rr else None,
                    "median_D_analytic_late": med_d,
                    "passes_8c": bool(n_rep == 0 and med_d is not None and abs(med_d - 1) <= D_TOL),
                    "median_dc_dim": float(np.median([r["dc_share_max_dim"] for r in rr if r["iteration"] >= FROM_ITERATION])),
                    "median_rho": float(np.median([r["rho_sum_1_30"] for r in rr if r["iteration"] >= FROM_ITERATION]))}
    return out


def print_verdicts(rows: list[dict], v: dict) -> None:
    for arm, a in v.items():
        print(f"\n== {arm}")
        print(f"   {'run':<28} {'it':>6} {'period':>6} {'cycle':>6} {'DC dim':>6} {'ACF dim':>7} {'Σρ(8)':>6} {'share78':>7} "
              f"{'var/MD':>6} {'D/D_an':>6}  failure")
        for run in a["seeds"]:
            for r in sorted((r for r in rows if r["run"] == run and r.get("draw", "config") == "config"), key=lambda r: r["iteration"]):
                da = r.get("D_ratio_analytic")
                print(f"   {run:<28} {r['iteration']:>6} {r['period_blocks']:>6.2f} {r['cycle_share'] * 100:>5.0f}% "
                      f"{r['dc_share_max_dim']:>6.2f} {r['acf100_max_dim']:>7.2f} {r['rho_sum_1_30']:>+6.2f} {r['share78']:>7.2f} "
                      f"{r['var_over_md']:>6.2f} {('%6.2f' % da) if da is not None else '     -'}  "
                      + ("YES" if r["iteration"] >= FROM_ITERATION and failure_at(r) else ""))
        print(f"   failure definition met by {a['reproduced']}/{a['n_seeds']} seeds -> "
              + ("REPRODUCED (>= 3 seeds)" if a["failure_reproduced"] else "not reproduced"))
    print(f"\nsweep (8c: pass = no seed meets the definition and median D/D_analytic at 7-10k within {D_TOL:.0%})")
    print(f"   {'arm':<16} {'share a':>7} {'seeds failing':>13} {'med DC dim':>10} {'med Σρ':>7} {'med D/D_an (7-10k)':>18}  pass")
    for arm, a in sorted(v.items(), key=lambda kv: (kv[1]["share"] or 0.0)):
        md = a["median_D_analytic_late"]
        print(f"   {arm:<16} {a['share'] if a['share'] is not None else 0.0:>7.2f} {a['reproduced']:>6}/{a['n_seeds']:<6} "
              f"{a['median_dc_dim']:>10.2f} {a['median_rho']:>+7.2f} {('%18.2f' % md) if md is not None else '                 -'}  "
              + ("PASS" if a["passes_8c"] else "-"))


def attach_share(rows: list[dict]) -> list[dict]:
    """latent_noise_share from each run's config.json (arm-eval rows do not carry it)."""
    cache = {}
    for r in rows:
        run_dir = Path(r["snapshot"]).parent
        if run_dir not in cache:
            f = run_dir / "config.json"
            cache[run_dir] = json.loads(f.read_text()).get("latent_noise_share", 0.0) if f.exists() else None
        r["latent_noise_share"] = cache[run_dir]
    return rows


# ============================================================================ self-test

def self_test() -> None:
    import tempfile
    base = dict(acf100_max_dim=0.05, share78=0.1, var_over_md=0.6, draw="config", snapshot="x/snap.pt")

    def row(run, it, period, cycle, dc, rho, d=1.0, arm="mini", share=0.0):
        return {**base, "arm": arm, "run": run, "iteration": it, "period_blocks": period, "cycle_share": cycle,
                "dc_share_max_dim": dc, "rho_sum_1_30": rho, "D_ratio_analytic": d, "latent_noise_share": share}
    assert failure_at(row("r", 3000, 5.0, 0.25, 0.1, 0.6)) and failure_at(row("r", 3000, 30.0, 0.05, 0.4, 0.6))
    assert not failure_at(row("r", 3000, 5.0, 0.25, 0.1, 0.4)) and not failure_at(row("r", 3000, 12.0, 0.25, 0.1, 0.9))
    rows = []
    for s in range(5):                        # 8b arm: seeds 0-2 fail at 4k only; seed 3 fails only at 1k (ignored)
        for it in range(1000, 10001, 1000):
            bad = (s < 3 and it == 4000) or (s == 3 and it == 1000)
            rows.append(row(f"run_mini_c1_seed{s}", it, 4.0 if bad else 30.0, 0.3, 0.1, 0.9 if bad else 0.1, d=2.0))
    for s in range(5):                        # 8c arm at a = 0.5: nothing fails, D on target
        for it in range(1000, 10001, 1000):
            rows.append(row(f"run_share050_c2_seed{s}", it, 30.0, 0.05, 0.05, 0.05, d=1.1, arm="share050", share=0.5))
    for s in range(5):                        # 8c arm at a = 0.1: nothing fails but D off by 40 %
        for it in range(1000, 10001, 1000):
            rows.append(row(f"run_share010_c3_seed{s}", it, 30.0, 0.05, 0.05, 0.05, d=1.4, arm="share010", share=0.1))
    v = verdicts(rows)
    assert v["mini"]["reproduced"] == 3 and v["mini"]["failure_reproduced"] and not v["mini"]["passes_8c"]
    assert v["share050"]["reproduced"] == 0 and v["share050"]["passes_8c"]
    assert not v["share010"]["passes_8c"]
    print("control 1: the failure definition (latent signature and Σρ, from 2k on) and both verdicts on planted rows")
    with tempfile.TemporaryDirectory() as tmp:
        p = write_data(Path(tmp) / "c.npz", Path(tmp) / "fig", n_traj=8, n_steps=600, seed=1)
        from mdgan.data.preprocess import SystemData
        s = SystemData.load(Path(tmp) / "c.npz")
        assert s.displacements.shape == (8, 600, 3) and s.meta["D_exact_m2s"] == p["D_exact_m2s"]
        assert (Path(tmp) / "fig" / "cage_mini_msd.png").exists() and abs(p["crossover_steps"] - 48) < 1e-9
        assert abs(p["slope_blocks_20_60_over_kappa_h2_minus_1"]) < 0.05
    print("control 2: the dataset round-trips with its exact D; the 8a figure is written; the slope check holds")
    print("all positive controls passed")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", nargs="?", choices=["data", "verdict"])
    ap.add_argument("files", nargs="*", help="verdict: arm-eval JSON files")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--out", default="data/cache/cage_mini.npz")
    ap.add_argument("--figure-dir", default="runs/testbed")
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args(argv)
    if args.self_test:
        self_test()
        return
    if args.command == "data":
        p = write_data(Path(args.out), Path(args.figure_dir), seed=args.seed)
        print(json.dumps(p, indent=2))
        print(f"wrote {args.out}, {args.figure_dir}/cage_mini_msd.png and cage_mini_params.json")
    elif args.command == "verdict":
        rows = attach_share([r for f in args.files for r in json.loads(Path(f).read_text())])
        v = verdicts(rows)
        print_verdicts(rows, v)
        out = Path(args.files[0]).with_name("testbed_verdict.json")
        out.write_text(json.dumps(v, indent=1))
        print(f"\nwrote {out}")
    else:
        ap.error("give a command (data, verdict) or --self-test")


if __name__ == "__main__":
    main()
