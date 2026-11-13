"""Tests for mdgan.data.synthetic: the simulated cage walk against its exact answers."""

import numpy as np
import pytest

from mdgan.data.msd import diffusion_coefficient, msd_time_averaged
from mdgan.data.synthetic import CageWalk


@pytest.fixture(scope="module")
def walk():
    return CageWalk()


@pytest.fixture(scope="module")
def sim(walk):
    return walk.simulate(n_traj=2048, n_steps=4000, seed=1)      # D estimate scatter ~1.5 %, well inside 5 %


def test_simulated_slope_matches_exact_D_within_5_percent(walk, sim):
    """Plan step 2a: the long-time slope (fitted well past the crossover) within 5 % of kappa h^2 / 6."""
    msd, _ = msd_time_averaged(sim.displacements)
    d_fit, _ = diffusion_coefficient(msd, sim.dt_ps, 5000.0, 15000.0)       # 500-1500 steps
    assert d_fit == pytest.approx(walk.diffusion_coefficient_m2s(), rel=0.05)


def test_simulated_msd_matches_the_exact_curve(walk, sim):
    msd, _ = msd_time_averaged(sim.displacements)
    for t in (1, 2, 5, 8, 16, 32, 64, 256, 1000):
        assert msd[t] == pytest.approx(walk.msd(t), rel=0.05), t


def test_default_shape_matches_the_targets(walk):
    """Subdiffusive inside the 16-step window, normal (exponent >= 0.9) by block 8 (64 steps)."""
    a = walk.local_exponent(np.array([8, 16, 64, 128]))
    assert 0.5 < a[0] < 0.7 and 0.6 < a[1] < 0.8 and a[2] >= 0.9 and a[3] > 0.93


def test_hops_are_rare_large_single_steps(walk, sim):
    d = sim.displacements.reshape(-1, 3).astype(np.float64)
    sigma = d.std()                                                          # per-axis step rms
    length = np.linalg.norm(d, axis=1)
    hop_frac = np.mean(length > 0.5 * walk.hop_length_nm)
    assert hop_frac == pytest.approx(walk.hop_rate_per_step, rel=0.1)       # one hop ~ one big step
    assert walk.hop_length_nm / sigma > 6.5                                 # 7-sigma hops, as in PE
    assert sigma == pytest.approx(np.sqrt(2 * walk.cage_rms_nm ** 2 * (1 - np.exp(-1 / walk.tau_steps))
                                          + walk.hop_rate_per_step * walk.hop_length_nm ** 2 / 3), rel=0.03)


def test_exact_D_formula_and_units(walk):
    k, h, dt = walk.hop_rate_per_step, walk.hop_length_nm, walk.dt_ps
    assert walk.diffusion_coefficient_m2s() == pytest.approx(k * h * h / 6 / dt * 1e-6)
    slope = (walk.msd(10_000.0) - walk.msd(5_000.0)) / 5_000.0              # nm^2 per step, far past tau
    assert slope / 6 / dt * 1e-6 == pytest.approx(walk.diffusion_coefficient_m2s(), rel=1e-9)


def test_reproducible_and_shaped(walk):
    a, b = walk.simulate(4, 50, seed=3), walk.simulate(4, 50, seed=3)
    assert a.displacements.shape == (4, 50, 3) and a.displacements.dtype == np.float32
    np.testing.assert_array_equal(a.displacements, b.displacements)
    assert a.meta["D_exact_m2s"] == walk.diffusion_coefficient_m2s()


@pytest.mark.parametrize("bad", ["dt_ps", "tau_steps", "cage_rms_nm", "hop_length_nm", "hop_rate_per_step"])
def test_invalid_parameters_rejected(bad):
    with pytest.raises(ValueError, match=bad):
        CageWalk(**{bad: 0.0})


# =============================================================================
# The miniature test bed (plan step 8a)

@pytest.fixture(scope="module")
def mini():
    return CageWalk.miniature()


def test_miniature_parameters_follow_the_rule(mini):
    assert mini.cage_rms_nm == 1.0 and mini.tau_steps == 2.0
    assert mini.crossover_steps == pytest.approx(48.0, rel=1e-12)                 # block 6 of M = 8: inside 4-8
    assert mini.hop_length_nm == pytest.approx(7 * mini.step_rms_nm(), rel=1e-12)  # 7-sigma hops
    assert 32 <= mini.crossover_steps <= 64


def test_miniature_is_subdiffusive_inside_the_critic_window(mini):
    """Local exponent below 0.7 at blocks 1-2 (8 and 16 steps)."""
    a = mini.local_exponent(np.array([8, 16]))
    assert (a < 0.7).all() and 0.15 < a[0] < 0.3, a


def test_miniature_slope_over_blocks_20_to_60_matches_kappa_h2(mini):
    """Time-averaged MSD slope over blocks 20-60 (160-480 steps) within 5 % of the analytic kappa h^2."""
    sim = mini.simulate(n_traj=1024, n_steps=4000, seed=11)
    msd, _ = msd_time_averaged(sim.displacements)
    lag = np.arange(160, 481)
    slope = np.polyfit(lag, msd[lag], 1)[0]
    assert slope == pytest.approx(mini.hop_rate_per_step * mini.hop_length_nm ** 2, rel=0.05)
    assert sim.meta["D_exact_m2s"] == mini.diffusion_coefficient_m2s()


def test_miniature_simulated_step_rms_and_hops(mini):
    sim = mini.simulate(n_traj=256, n_steps=4000, seed=12)
    d = sim.displacements.reshape(-1, 3).astype(np.float64)
    assert d.std() == pytest.approx(mini.step_rms_nm(), rel=0.03)
    # the miniature's cage rattling is large (per-axis rms ~0.9): at half the hop length (3.6 sigma of a 3D step)
    # ~0.5 % of plain steps would count as hops; at three quarters (5.4 sigma) none do, and every hop still does
    big = np.linalg.norm(d, axis=1) > 0.75 * mini.hop_length_nm
    assert big.mean() == pytest.approx(mini.hop_rate_per_step, rel=0.15)

