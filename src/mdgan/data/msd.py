"""Mean square displacement, diffusion coefficient, and the paper's err metric.

Everything here takes DISPLACEMENT arrays, (N, T, 3) in nm, one row per
trajectory, as produced by preprocess.py or by the generator. Positions are
their cumulative sum from the origin.

TWO MSD ESTIMATORS, TWO USES
----------------------------
  msd_time_averaged  Reference curve from MD. For each trajectory, average
                     |r(t0+tau) - r(t0)|^2 over every origin t0, then average
                     over trajectories. This is what the authors' msd.out
                     contains (reproduced to 5e-6), and it is the only sensible
                     estimator when each trajectory is a single molecule.
  msd_ensemble       Curve from generated trajectories. Average |r(t) - r(0)|^2
                     over the 8192 chains at each t, as in 2023 eq. 5. This is
                     the paper's reported quantity for MD-GAN output.
The two are equal in expectation for a stationary process; they are kept as
separately named functions so no result silently swaps one for the other.

TIME GRIDS
----------
A displacement array with step dt_ps gives MSD values at t = 0, dt, 2dt, ...
(index k <-> t = k*dt_ps). The MD reference is on the 10 ps grid; a generated
curve is on the n_s*10 ps grid. err() looks the reference up at the generated
times, which must therefore fall on the reference grid.

Units: nm, ps internally; D is returned in m^2/s.
"""

import numpy as np

NM2_PER_PS_TO_M2_PER_S = 1e-18 / 1e-12


def positions_from_displacements(d: np.ndarray) -> np.ndarray:
    """(N, T, 3) displacements -> (N, T+1, 3) positions starting at the origin. float64."""
    d = np.asarray(d, dtype=np.float64)
    if d.ndim != 3 or d.shape[2] != 3:
        raise ValueError(f"displacements must be (N, T, 3); got {d.shape}")
    pos = np.zeros((d.shape[0], d.shape[1] + 1, 3))
    np.cumsum(d, axis=1, out=pos[:, 1:])
    return pos


# --- reference estimator ------------------------------------------------------------

def _msd_time_averaged_one(pos: np.ndarray) -> np.ndarray:
    """All-origins time-averaged MSD of one (T, 3) trajectory via FFT.

    MSD(tau) = (1/(T-tau)) * sum_t0 |r(t0+tau) - r(t0)|^2
             = (1/(T-tau)) * [ sum_t0 (|r(t0+tau)|^2 + |r(t0)|^2) ] - 2*C(tau)/(T-tau)
    where C(tau) = sum_t0 r(t0).r(t0+tau) is the autocorrelation, computed with
    an FFT of length 2T (zero-padded so it is linear, not circular). The first
    bracket is accumulated by a running sum. Exact (not approximate), and the
    origin count T-tau is applied per lag. O(T log T) instead of O(T^2).
    """
    T = pos.shape[0]
    sq = (pos ** 2).sum(axis=1)                       # |r(t)|^2, (T,)
    autocorr = np.zeros(T)
    for axis in range(3):
        f = np.fft.rfft(pos[:, axis], 2 * T)
        autocorr += np.fft.irfft(f * np.conj(f), 2 * T)[:T]
    sq_padded = np.append(sq, 0.0)
    running = 2.0 * sq.sum()
    pair_sum = np.empty(T)
    for tau in range(T):                              # sum_t0 (|r(t0+tau)|^2 + |r(t0)|^2)
        running -= sq_padded[tau - 1] + sq_padded[T - tau]
        pair_sum[tau] = running
    counts = np.arange(T, 0, -1)                      # origins available at each lag
    return pair_sum / counts - 2.0 * autocorr / counts


def msd_time_averaged(d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Reference MSD from MD displacements (N, T, 3).

    Returns (mean over trajectories (T+1,), per-trajectory curves (N, T+1)),
    both with index k <-> lag k*dt. Value at lag 0 is exactly 0.
    """
    pos = positions_from_displacements(d)
    per = np.stack([_msd_time_averaged_one(p) for p in pos])
    per[:, 0] = 0.0
    return per.mean(axis=0), per


# --- generated-trajectory estimator ----------------------------------------------------

def msd_ensemble(d: np.ndarray) -> np.ndarray:
    """Ensemble MSD from origin for generated displacements (N, T, 3): (T+1,)."""
    pos = positions_from_displacements(d)
    return (pos ** 2).sum(axis=2).mean(axis=0)


# --- diffusion coefficient ----------------------------------------------------------------

def diffusion_coefficient(msd: np.ndarray, dt_ps: float,
                          t_min_ps: float = 10_000.0, t_max_ps: float = 20_000.0) -> tuple[float, float]:
    """D = (1/6) d MSD/dt from an ordinary least-squares line (with intercept)
    over [t_min, t_max] (2023 eq. 8; the paper fits 10-20 ns).

    Returns (D in m^2/s, slope in nm^2/ns). The slope is reported too because
    it is the unit-free check: PE reference slope is 0.7896 nm^2/ns.
    """
    msd = np.asarray(msd, dtype=np.float64)
    t = np.arange(len(msd)) * dt_ps
    m = (t >= t_min_ps) & (t <= t_max_ps)
    if m.sum() < 2:
        raise ValueError(f"fewer than two MSD points in [{t_min_ps}, {t_max_ps}] ps at dt={dt_ps}")
    slope_nm2_per_ps, _ = np.polyfit(t[m], msd[m], 1)
    return slope_nm2_per_ps / 6.0 * NM2_PER_PS_TO_M2_PER_S, slope_nm2_per_ps * 1e3


# --- err metric ---------------------------------------------------------------------------------

def err(msd_gen: np.ndarray, dt_gen_ps: float, msd_ref: np.ndarray, dt_ref_ps: float,
        t1_ps: float = 200.0, t2_ps: float = 78_000.0) -> float:
    """Log-time-averaged absolute percentage error of a generated MSD (2023 eqs. 6-7).

    f(t) = |MSD_gen(t) - MSD_ref(t)| / MSD_ref(t) * 100
    err  = int f(t) d(log t) / int d(log t) over [t1, t2]

    Discretized as in the 2022 ESI eq. S4 on the GENERATED grid: with
    t_k = k*dt_gen, err = sum_k f(t_k)/k / sum_k 1/k over k with t1 <= t_k <= t2
    (endpoints inclusive, k >= 1 always since t1 > 0). The reference is looked
    up at exactly those times, so dt_gen must be a multiple of dt_ref.
    Returns percent.
    """
    msd_gen = np.asarray(msd_gen, dtype=np.float64)
    msd_ref = np.asarray(msd_ref, dtype=np.float64)
    if t1_ps <= 0 or t2_ps <= t1_ps:
        raise ValueError("need 0 < t1 < t2")
    ratio = dt_gen_ps / dt_ref_ps
    if abs(ratio - round(ratio)) > 1e-9:
        raise ValueError(f"dt_gen={dt_gen_ps} is not a multiple of dt_ref={dt_ref_ps}")
    ratio = int(round(ratio))

    k = np.arange(len(msd_gen))
    t = k * dt_gen_ps
    m = (t >= t1_ps - 1e-9) & (t <= t2_ps + 1e-9) & (k >= 1)
    if not m.any():
        raise ValueError("no generated MSD points inside [t1, t2]")
    ref_idx = k[m] * ratio
    if ref_idx[-1] >= len(msd_ref):
        raise ValueError("reference MSD is shorter than the requested window")
    f = np.abs(msd_gen[m] - msd_ref[ref_idx]) / msd_ref[ref_idx] * 100.0
    w = 1.0 / k[m]
    return float((f * w).sum() / w.sum())