"""Find the short cycle in G_z's latent dynamics: its period and which dimensions carry it.

For each snapshot: roll the latent forward exactly as estimation does (z_0 ~ U[0,1), burn-in,
then one latent per decoded block), and analyse the latents z_c .. z_c+n-1 that become blocks:

  autocorrelation   per dimension and pooled, at every lag 1 .. n-1 (same pooled definition as
                    diag_runs.py's z@lag columns)
  spectrum          per-dimension power spectrum along the rollout, averaged over chains
  period            the spectrum's peak (excluding the slowest frequencies), in blocks and ns,
                    cross-checked against the first autocorrelation peak
  carriers          per dimension: the share of its variance at the cycle frequency (band of
                    +/- 1 bin around the peak) and its share of the total cycle power; also the
                    share of its variance at periods longer than 25 blocks (the slow part)

Writes per snapshot: latent_cycle_<tag>.png/.pdf (pooled ACF, per-dimension ACF heatmap, a few
chains' carrier components over the rollout) and latent_cycle_<tag>.csv (one row per dimension).

Usage (repository root):
    python tools/latent_cycle.py \\
        runs/chtc/run_scaleoff_c7166270_seed0/snap_030000.pt \\
        runs/chtc/run_scaleoff_c7166270_seed1/snap_060000.pt \\
        --device mps --out runs/latent_cycle
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SLOW_PERIOD_BLOCKS = 25          # periods longer than this count as "slow" (persistent) power
N_FFT = 1024                     # zero-padded FFT length: fine frequency grid for the peak


# ============================================================================ rollout (torch)

def latent_rollout(snapshot: str, chains: int, device: str, seed: int = 0, dt_ps: float = 10.0):
    """Latents of the decoded blocks, (n_blocks, chains, n_z), and the block duration in ns."""
    import torch

    from mdgan.checkpoint import load_checkpoint
    ck = load_checkpoint(snapshot, map_location="cpu")
    c = ck.config
    gz = ck.gz.to(device).eval()
    n_blocks = c.estimation_blocks(dt_ps)
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        z0 = torch.rand(chains, c.latent_dim, generator=g).to(device)
        states = gz.unroll(z0, c.burn_in + n_blocks - 1, generator=g)          # z_0 .. z_{c+n-1}
    z = states[c.burn_in:].cpu().double().numpy()                             # z_c .. z_{c+n-1}
    return z, c.block_len * dt_ps * c.step_skip / 1000.0, ck.iteration


# ============================================================================ analysis (numpy)

def autocorrelation(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """z: (T, B, D). Returns (per-dimension ACF (D, T), pooled ACF (T,)); index = lag, lag 0 = 1."""
    t_len, _, d = z.shape
    zc = z - z.mean(axis=(0, 1), keepdims=True)
    per = np.full((d, t_len), np.nan)
    pooled = np.full(t_len, np.nan)
    per[:, 0], pooled[0] = 1.0, 1.0
    for lag in range(1, t_len - 1):
        a, b = zc[:-lag], zc[lag:]
        num = (a * b).sum(axis=(0, 1))
        den = np.sqrt((a * a).sum(axis=(0, 1)) * (b * b).sum(axis=(0, 1)))
        per[:, lag] = num / den
        pooled[lag] = np.corrcoef(a.ravel(), b.ravel())[0, 1]
    return per, pooled


def spectrum(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-dimension power spectrum along the rollout, averaged over chains.
    Returns (freq in cycles per block (F,), power (D, F)); each chain's series is mean-removed."""
    zc = z - z.mean(axis=0, keepdims=True)
    f = np.fft.rfftfreq(N_FFT)
    p = (np.abs(np.fft.rfft(zc, n=N_FFT, axis=0)) ** 2).mean(axis=1).T          # (D, F)
    return f, p


def find_cycle(f: np.ndarray, p: np.ndarray, pooled_acf: np.ndarray, n_blocks: int) -> dict:
    """Period from the summed spectrum's peak above the slow band; carriers per dimension."""
    total = p.sum(axis=0)
    fast = f > 1.0 / SLOW_PERIOD_BLOCKS
    k = int(np.argmax(np.where(fast, total, -np.inf)))
    bin_width = 1.0 / n_blocks                                                # true resolution
    band = np.abs(f - f[k]) <= bin_width
    slow = (f > 0) & ~fast
    var = p[:, f > 0].sum(axis=1)
    share_cycle = p[:, band].sum(axis=1) / var
    share_slow = p[:, slow].sum(axis=1) / var
    cycle_power = p[:, band].sum(axis=1)
    acf_peak = None
    for lag in range(2, len(pooled_acf) - 2):                                 # first local max after a dip
        if np.isfinite(pooled_acf[lag - 1:lag + 2]).all() and pooled_acf[lag] > pooled_acf[lag - 1] \
                and pooled_acf[lag] >= pooled_acf[lag + 1] and pooled_acf[1:lag].min() < pooled_acf[lag]:
            acf_peak = lag
            break
    return {"freq": float(f[k]), "period_blocks": float(1.0 / f[k]),
            "cycle_fraction_of_total_variance": float(total[band].sum() / total[f > 0].sum()),
            "slow_fraction_of_total_variance": float(total[slow].sum() / total[f > 0].sum()),
            "acf_first_peak_lag": acf_peak,
            "share_cycle": share_cycle, "share_slow": share_slow,
            "share_of_cycle_power": cycle_power / cycle_power.sum()}


def carriers(cyc: dict, min_share: float = 0.25) -> list[int]:
    """Dimensions whose variance is at least min_share at the cycle, strongest first."""
    order = np.argsort(-cyc["share_of_cycle_power"])
    return [int(d) for d in order if cyc["share_cycle"][d] >= min_share]


# ============================================================================ output

def plot(tag: str, z, per, pooled, cyc, block_ns: float, out: Path, n_chains: int = 3) -> None:
    lags = np.arange(per.shape[1])
    fig = plt.figure(figsize=(10, 7.2))
    gs = fig.add_gridspec(3, n_chains, height_ratios=[1, 1.2, 1.3], hspace=0.6, wspace=0.3)

    ax = fig.add_subplot(gs[0, :])
    ax.plot(lags[1:], pooled[1:], color="black", lw=1.0, marker=".", ms=3)
    ax.axhline(0, color="0.6", lw=0.7)
    ax.set_xlim(0, lags[-1])
    ax.set_xlabel("lag (blocks)")
    ax.set_ylabel("pooled ACF")
    ax.set_title(f"{tag}: period {cyc['period_blocks']:.2f} blocks ({cyc['period_blocks'] * block_ns:.2f} ns); "
                 f"cycle {cyc['cycle_fraction_of_total_variance'] * 100:.0f} % and slow "
                 f"{cyc['slow_fraction_of_total_variance'] * 100:.0f} % of latent variance", fontsize=9.5)

    ax = fig.add_subplot(gs[1, :])
    im = ax.imshow(per[:, 1:], aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest",
                   extent=(0.5, per.shape[1] - 0.5, per.shape[0] - 0.5, -0.5))
    ax.set_xlabel("lag (blocks)")
    ax.set_ylabel("z dimension")
    fig.colorbar(im, ax=ax, pad=0.01, label="ACF")

    dims = carriers(cyc)[:4] or [int(np.argmax(cyc["share_of_cycle_power"]))]
    slow = [int(d) for d in np.argsort(-cyc["share_slow"]) if cyc["share_slow"][d] >= 0.5][:1]
    t = np.arange(z.shape[0])
    for chain in range(n_chains):
        ax = fig.add_subplot(gs[2, chain])
        for k, d in enumerate(dims):
            ax.plot(t, z[:, chain, d], lw=0.8, color=plt.cm.tab10(k), label=f"z[{d}] (cycle)")
        for d in slow:
            ax.plot(t, z[:, chain, d], lw=1.2, color="black", label=f"z[{d}] (slow)")
        ax.set_xlim(0, 40)
        ax.set_xlabel("block (first 40)")
        ax.set_title(f"chain {chain}", fontsize=9)
        if chain == 0:
            ax.set_ylabel("z component")
            ax.legend(frameon=False, fontsize=6.5, loc="upper left")
    for ext in ("png", "pdf"):
        fig.savefig(out / f"latent_cycle_{tag}.{ext}", bbox_inches="tight", dpi=150)
    plt.close(fig)


def write_csv(tag: str, per, cyc, out: Path) -> None:
    with open(out / f"latent_cycle_{tag}.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["dim", "share_cycle", "share_slow", "share_of_cycle_power",
                    *[f"acf@{lag}" for lag in range(1, per.shape[1] - 1)]])
        for d in range(per.shape[0]):
            w.writerow([d, f"{cyc['share_cycle'][d]:.4f}", f"{cyc['share_slow'][d]:.4f}",
                        f"{cyc['share_of_cycle_power'][d]:.4f}",
                        *[f"{per[d, lag]:.4f}" for lag in range(1, per.shape[1] - 1)]])


def report(tag: str, z, block_ns: float, out: Path) -> dict:
    per, pooled = autocorrelation(z)
    f, p = spectrum(z)
    cyc = find_cycle(f, p, pooled, z.shape[0])
    plot(tag, z, per, pooled, cyc, block_ns, out)
    write_csv(tag, per, cyc, out)
    car = carriers(cyc)
    held = float(cyc["share_of_cycle_power"][car].sum()) if car else 0.0
    print(f"\n=== {tag}")
    print(f"period {cyc['period_blocks']:.2f} blocks = {cyc['period_blocks'] * block_ns:.2f} ns "
          f"(first ACF peak at lag {cyc['acf_first_peak_lag']}); the cycle is "
          f"{cyc['cycle_fraction_of_total_variance'] * 100:.0f} % of all latent variance, the slow part "
          f"(> {SLOW_PERIOD_BLOCKS} blocks) {cyc['slow_fraction_of_total_variance'] * 100:.0f} %")
    print("pooled ACF at lags 1-6: " + "  ".join(f"{pooled[k]:+.3f}" for k in range(1, 7))
          + f"   z@50 {pooled[50]:+.3f}  z@100 {pooled[100]:+.3f}")
    print(f"carriers (>= 25 % of their own variance at the cycle): {len(car)} of {z.shape[2]} dimensions, "
          f"holding {held * 100:.0f} % of the cycle power")
    slow_dims = [int(d) for d in np.argsort(-cyc["share_slow"]) if cyc["share_slow"][d] >= 0.5]
    print(f"slow dimensions (>= 50 % of their variance at periods > {SLOW_PERIOD_BLOCKS} blocks): "
          f"{slow_dims if slow_dims else 'none'}   (white noise gives about 8 %)")
    print(f"{'dim':>4} {'at cycle':>9} {'slow':>6} {'of cycle power':>15}")
    for d in np.argsort(-cyc["share_of_cycle_power"])[:8]:
        print(f"{d:>4} {cyc['share_cycle'][d] * 100:8.0f}% {cyc['share_slow'][d] * 100:5.0f}% "
              f"{cyc['share_of_cycle_power'][d] * 100:14.1f}%")
    return cyc


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("snapshots", nargs="+", help="snap_XXXXXX.pt files")
    ap.add_argument("--chains", type=int, default=1024)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="runs/latent_cycle")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for snap in args.snapshots:
        p = Path(snap)
        z, block_ns, it = latent_rollout(snap, args.chains, args.device)
        report(f"{p.parent.name}_{it // 1000}k", z, block_ns, out)
    print(f"\nwrote figures and per-dimension CSVs to {out}")


if __name__ == "__main__":
    main()