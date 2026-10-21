"""Unit tests for mdgan.data.msd (synthetic), plus opt-in real-data checks.

Real-data tests use the session `pe` fixture (real_data; skips unless MDGAN_PE_DIR is set).

Run: python -m pytest tests/data/test_msd.py -v
"""

import numpy as np
import pytest

from mdgan.data.msd import (
    diffusion_coefficient, err, msd_ensemble, msd_time_averaged, positions_from_displacements,
)


def brownian(rng, n, T, D_m2s=1.3e-10, dt_ps=10.0):
    """Isotropic Brownian displacements with a known D: per-axis variance 2*D*dt."""
    D_nm2_ps = D_m2s / (1e-18 / 1e-12)
    return rng.normal(scale=np.sqrt(2 * D_nm2_ps * dt_ps), size=(n, T, 3))


def msd_direct(pos):
    """O(T^2) reference implementation of the all-origins time average."""
    T = len(pos); out = np.zeros(T)
    for tau in range(1, T):
        out[tau] = ((pos[tau:] - pos[:-tau]) ** 2).sum(1).mean()
    return out


# --- positions ----------------------------------------------------------------------------

def test_positions_start_at_origin_and_cumsum():
    d = np.random.default_rng(0).normal(size=(2, 5, 3))
    pos = positions_from_displacements(d)
    assert pos.shape == (2, 6, 3) and pos.dtype == np.float64
    np.testing.assert_array_equal(pos[:, 0], 0.0)
    np.testing.assert_allclose(pos[:, -1], d.sum(1))


@pytest.mark.parametrize("shape", [(5, 3), (2, 5, 2), (5,)])
def test_positions_rejects_bad_shape(shape):
    with pytest.raises(ValueError):
        positions_from_displacements(np.zeros(shape))


# --- time-averaged MSD ------------------------------------------------------------------------

def test_fft_time_average_equals_direct_at_every_lag():
    rng = np.random.default_rng(1)
    d = brownian(rng, 3, 300)
    mean, per = msd_time_averaged(d)
    pos = positions_from_displacements(d)
    for i in range(3):
        np.testing.assert_allclose(per[i], msd_direct(pos[i]), rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(mean, per.mean(0))
    assert mean[0] == 0.0 and per.shape == (3, 301)


def test_time_average_uses_correct_origin_count_at_last_lag():
    """At the last lag there is exactly one origin: MSD = |r(T) - r(0)|^2."""
    d = brownian(np.random.default_rng(2), 1, 50)
    _, per = msd_time_averaged(d)
    pos = positions_from_displacements(d)[0]
    assert per[0, -1] == pytest.approx(((pos[-1] - pos[0]) ** 2).sum())


def test_time_average_of_straight_line_is_quadratic():
    v = np.array([0.1, -0.2, 0.05])
    d = np.tile(v, (1, 40, 1))
    _, per = msd_time_averaged(d)
    np.testing.assert_allclose(per[0], (np.arange(41) ** 2) * (v ** 2).sum(), rtol=1e-10)


def test_time_average_is_translation_invariant():
    d = brownian(np.random.default_rng(3), 2, 100)
    a, _ = msd_time_averaged(d)
    pos = positions_from_displacements(d) + np.array([5.0, -3.0, 2.0])
    b = np.stack([msd_direct(p) for p in pos]).mean(0)
    np.testing.assert_allclose(a, b, rtol=1e-9, atol=1e-10)


# --- ensemble MSD -----------------------------------------------------------------------------------

def test_ensemble_msd_definition():
    d = brownian(np.random.default_rng(4), 10, 20)
    pos = positions_from_displacements(d)
    expected = ((pos - pos[:, :1]) ** 2).sum(2).mean(0)
    np.testing.assert_allclose(msd_ensemble(d), expected)
    assert msd_ensemble(d)[0] == 0.0


@pytest.mark.statistical
def test_ensemble_and_time_average_agree_for_brownian_motion():
    """Both estimators match 6 D t on Brownian motion.

    Tolerance. Per chain, |r(t)|^2 is a sum of 3 axes, each s * chi^2_1 with
    s = 2 D t, so mean 3s and variance 3 * 2 s^2 = 6 s^2: relative std sqrt(2/3).
    Over N independent chains the ensemble MSD has relative standard error
    sqrt(2 / (3N)). The time average uses many correlated origins, so its error
    at every lag is AT MOST that of a single origin (equality at the last lag,
    where only one origin exists); the same bound covers it. rtol = 5 standard
    errors: a correct implementation essentially never exceeds it under any seed,
    while the bugs this test exists for (wrong factor 2/3/6, wrong dt, missing
    axis) are errors of 30 % or more.
    """
    N, T = 2000, 400
    d = brownian(np.random.default_rng(5), N, T)
    ens = msd_ensemble(d)
    ta, _ = msd_time_averaged(d)
    theory = 6 * (1.3e-10 / 1e-6) * np.arange(T + 1) * 10.0
    rtol = 5 * np.sqrt(2 / (3 * N))                       # = 0.091
    np.testing.assert_allclose(ens[1:], theory[1:], rtol=rtol)
    np.testing.assert_allclose(ta[1:], theory[1:], rtol=rtol)


# --- diffusion coefficient ---------------------------------------------------------------------------

@pytest.mark.statistical
def test_diffusion_coefficient_recovers_known_D():
    """OLS over 10-20 ns of an ensemble MSD recovers the input D.

    Tolerance. The slope is set by the MSD increment between the window ends.
    With r2 = r1 + Delta (independent, per-axis variances s each),
    |r2|^2 - |r1|^2 = 2 r1.Delta + |Delta|^2 has mean 3s and variance
    4 * 3 s^2 + 6 s^2 = 18 s^2 per chain: relative std sqrt(2). Over N chains the
    relative standard error of D is ~sqrt(2 / N); OLS over the window is no worse
    than the two-endpoint difference. rtol = 5 standard errors (0.11 for N = 4000);
    a factor-of-2/3/6 or unit bug is far outside it.
    """
    N = 4000
    d = brownian(np.random.default_rng(6), N, 2500, D_m2s=1.3e-10)
    D, slope = diffusion_coefficient(msd_ensemble(d), 10.0)
    rtol = 5 * np.sqrt(2 / N)
    assert D == pytest.approx(1.3e-10, rel=rtol)
    assert slope == pytest.approx(0.78, rel=rtol)      # 6 D in nm^2/ns


def test_diffusion_coefficient_exact_on_a_line():
    """MSD = 6 D t + c  ->  the fit returns D exactly, with intercept ignored."""
    t = np.arange(3001) * 10.0
    D_nm2_ps = 1.3e-10 / 1e-6
    msd = 6 * D_nm2_ps * t + 0.37
    D, slope = diffusion_coefficient(msd, 10.0)
    assert D == pytest.approx(1.3e-10, rel=1e-12)
    assert slope == pytest.approx(6 * D_nm2_ps * 1e3, rel=1e-12)


def test_diffusion_coefficient_window_and_errors():
    msd = np.arange(100, dtype=float)
    with pytest.raises(ValueError):
        diffusion_coefficient(msd, 10.0, t_min_ps=5000, t_max_ps=5005)


# --- err -------------------------------------------------------------------------------------------------

def test_err_zero_for_identical_curves():
    ref = np.linspace(0, 100, 8001)
    assert err(ref, 10.0, ref, 10.0) == 0.0


def test_err_constant_relative_offset():
    ref = np.linspace(0, 100, 8001) + 1.0
    assert err(ref * 1.1, 10.0, ref, 10.0) == pytest.approx(10.0, rel=1e-12)


def test_err_harmonic_weights_hand_computed():
    """Generated on a 100 ps grid, reference on 10 ps; window chosen so k = 2, 3, 4."""
    ref = np.ones(8001) * 2.0
    gen = np.ones(50) * 2.0
    gen[2], gen[3], gen[4] = 2.2, 2.4, 2.6             # f = 10, 20, 30 %
    e = err(gen, 100.0, ref, 10.0, t1_ps=200, t2_ps=400)
    expected = (10 / 2 + 20 / 3 + 30 / 4) / (1 / 2 + 1 / 3 + 1 / 4)
    assert e == pytest.approx(expected, rel=1e-12)


def test_err_endpoints_inclusive_and_k0_excluded():
    ref = np.ones(8001)
    gen = np.ones(8001); gen[0] = 1e9              # k = 0 must never be touched
    gen[20] = 2.0; gen[7800] = 2.0                 # exactly t1 and t2 at 10 ps
    e = err(gen, 10.0, ref, 10.0)
    w = 1 / np.arange(20, 7801); expected = (100 / 20 + 100 / 7800) / w.sum()
    assert e == pytest.approx(expected, rel=1e-9)


def test_err_rejects_off_grid_and_bad_windows():
    ref = np.ones(8001); gen = np.ones(100)
    with pytest.raises(ValueError, match="multiple"):
        err(gen, 15.0, ref, 10.0)
    with pytest.raises(ValueError, match="t1"):
        err(gen, 10.0, ref, 10.0, t1_ps=0.0)
    with pytest.raises(ValueError, match="shorter"):
        err(np.ones(8001), 10.0, np.ones(100), 10.0)


# --- real data (opt-in) -----------------------------------------------------------------------------------

@pytest.mark.real_data
def test_real_pe_reference_msd_reproduces_msd_out(pe):
    data, ref = pe
    mean, per = msd_time_averaged(data.displacements)
    np.testing.assert_allclose(per[:, 1:], ref[1:, 4:].T, rtol=1e-5)      # 60 per-trajectory columns
    np.testing.assert_allclose(mean[1:], ref[1:, 1], rtol=1e-4)           # mean column (5 sig. figs.)
    np.testing.assert_allclose(per[:, 1:].std(0, ddof=1), ref[1:, 3], rtol=1e-3)


@pytest.mark.real_data
def test_real_pe_diffusion_coefficient(pe):
    data, _ = pe
    mean, _ = msd_time_averaged(data.displacements)
    D, slope = diffusion_coefficient(mean, data.dt_ps)
    assert D == pytest.approx(1.31593866e-10, rel=1e-4)      # dc_result_md.txt
    assert slope == pytest.approx(0.7896, rel=1e-3)           # nm^2/ns