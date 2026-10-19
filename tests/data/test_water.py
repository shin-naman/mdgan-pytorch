"""Tests for mdgan.data.water (pure numpy; no trajectory files needed).

Strategy: build an UNWRAPPED ground truth, wrap it the way GROMACS would, and
require the module to recover the ground truth exactly.

Run: python -m pytest tests/data/test_water.py -v
"""

import numpy as np
import pytest

from mdgan.data.water import (
    GEOMETRY_TOL_NM, HH_DIST_NM, MAX_STEP_FRACTION, OH_BOND_NM,
    minimum_image, step_displacements, unwrapped_displacements, water_com,
)
from synthetic_water import unwrapped_trajectory, wrap


# --- minimum_image ---------------------------------------------------------------

def test_minimum_image_maps_into_half_open_interval():
    rng = np.random.default_rng(0)
    box = np.array([4.0, 5.0, 6.0])
    v = rng.uniform(-30, 30, size=(1000, 3))
    m = minimum_image(v, box)
    assert (m > -box / 2 - 1e-12).all() and (m <= box / 2 + 1e-12).all()
    # differs from v by an integer number of boxes, exactly
    k = (v - m) / box
    np.testing.assert_allclose(k, np.round(k), atol=1e-9)


def test_minimum_image_is_identity_inside_and_does_not_modify_input():
    box = np.array([4.0, 4.0, 4.0])
    v = np.array([[0.1, -1.9, 1.99]])
    orig = v.copy()
    np.testing.assert_array_equal(minimum_image(v, box), v)
    np.testing.assert_array_equal(v, orig)


def test_minimum_image_broadcasts_per_frame_box():
    v = np.array([[3.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    box = np.array([[4.0, 4.0, 4.0], [10.0, 10.0, 10.0]])
    m = minimum_image(v, box)
    np.testing.assert_allclose(m, [[-1.0, 0, 0], [3.0, 0, 0]])


@pytest.mark.parametrize("bad_box", [np.array([4.0, 0.0, 4.0]), np.array([4.0, -1.0, 4.0]), np.array([4.0, 4.0])])
def test_minimum_image_rejects_bad_box(bad_box):
    with pytest.raises(ValueError):
        minimum_image(np.zeros((2, 3)), bad_box)


# --- water_com -------------------------------------------------------------------

def test_water_com_matches_unwrapped_truth_after_wrapping():
    rng = np.random.default_rng(1)
    box = np.array([4.75, 4.75, 4.75])
    xyz, com_true, _ = unwrapped_trajectory(rng, 500, box)
    com = water_com(wrap(xyz, box), box)
    # COM is returned wrapped; compare modulo the box
    np.testing.assert_allclose(minimum_image(com - com_true, box), 0.0, atol=1e-10)


def test_water_com_uses_topology_masses():
    """Hand-computed COM with 16.0/1.008/1.008 for a molecule straddling a boundary."""
    box = np.array([4.0, 4.0, 4.0])
    half = np.arcsin(HH_DIST_NM / 2 / OH_BOND_NM)
    o = np.array([3.98, 1.0, 1.0])
    h1 = o + OH_BOND_NM * np.array([np.sin(half), np.cos(half), 0.0])      # x = 4.0557 -> wraps to 0.0557
    h2 = o + OH_BOND_NM * np.array([-np.sin(half), np.cos(half), 0.0])
    xyz = wrap(np.stack([o, h1, h2])[None], box)
    expected = (16.0 * o + 1.008 * h1 + 1.008 * h2) / (16.0 + 2 * 1.008)
    got = water_com(xyz, box)[0]
    np.testing.assert_allclose(minimum_image(got - expected, box), 0.0, atol=1e-12)


def test_water_com_rejects_wrong_atom_order():
    rng = np.random.default_rng(2)
    box = np.array([4.75, 4.75, 4.75])
    xyz, _, _ = unwrapped_trajectory(rng, 20, box)
    with pytest.raises(ValueError, match="O-H|H-H"):
        water_com(xyz[:, [1, 0, 2]], box)      # H, O, H


def test_water_com_rejects_non_water_geometry():
    box = np.array([4.0, 4.0, 4.0])
    xyz = np.zeros((5, 3, 3)); xyz[:, 1, 0] = 0.3; xyz[:, 2, 1] = 0.3
    with pytest.raises(ValueError):
        water_com(xyz, box)


def test_water_com_geometry_tolerance_is_tight():
    """A bond stretched by more than the tolerance must be rejected."""
    rng = np.random.default_rng(3)
    box = np.array([4.75, 4.75, 4.75])
    xyz, _, _ = unwrapped_trajectory(rng, 10, box)
    xyz[:, 1] += (xyz[:, 1] - xyz[:, 0]) * (2 * GEOMETRY_TOL_NM / OH_BOND_NM)   # stretch by 2*tol
    with pytest.raises(ValueError):
        water_com(xyz, box)


@pytest.mark.parametrize("shape", [(10, 3), (10, 4, 3), (10, 3, 2), (3, 3)])
def test_water_com_rejects_bad_shapes(shape):
    with pytest.raises(ValueError):
        water_com(np.zeros(shape), np.ones(3))


def test_water_com_rejects_nan():
    rng = np.random.default_rng(4)
    xyz, _, _ = unwrapped_trajectory(rng, 5, np.ones(3) * 4)
    xyz[2, 0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        water_com(xyz, np.ones(3) * 4)


# --- step_displacements --------------------------------------------------------------

def test_step_displacements_recover_unwrapped_steps_exactly():
    rng = np.random.default_rng(5)
    box = np.array([4.75, 4.75, 4.75])
    xyz, com_true, steps = unwrapped_trajectory(rng, 2000, box, step_std=0.15)
    d = unwrapped_displacements(wrap(xyz, box), box)
    assert d.shape == (1999, 3)
    np.testing.assert_allclose(d, steps, atol=1e-10)


def test_large_but_legal_hops_survive():
    """Regression for the real PE data: steps above L/4 are real cavity hops and
    must pass through unchanged. Only steps beyond MAX_STEP_FRACTION*L are errors."""
    rng = np.random.default_rng(6)
    box = np.array([4.75, 4.75, 4.75])
    hop = 0.40 * box[0]                                   # > L/4, < 0.45 L
    xyz, _, steps = unwrapped_trajectory(rng, 400, box, step_std=0.05, hop_every=50, hop_size=hop)
    d = unwrapped_displacements(wrap(xyz, box), box)
    np.testing.assert_allclose(d, steps, atol=1e-10)
    assert (np.abs(d[:, 0]) > box[0] / 4).sum() == len(steps[50::50])


def test_step_beyond_fraction_raises():
    box = np.array([4.0, 4.0, 4.0])
    com = np.array([[0.0, 0, 0], [0.5 * MAX_STEP_FRACTION * 4.0 + 1.0, 0, 0]])  # 1.9 nm of 4.0 -> 0.475 > 0.45
    with pytest.raises(ValueError, match="box length"):
        step_displacements(com, box)


def test_step_exactly_at_fraction_is_allowed():
    box = np.array([4.0, 4.0, 4.0])
    com = np.array([[0.0, 0, 0], [MAX_STEP_FRACTION * 4.0, 0, 0]])
    d = step_displacements(com, box)
    np.testing.assert_allclose(d, [[MAX_STEP_FRACTION * 4.0, 0, 0]])


def test_step_displacements_use_per_frame_box_of_earlier_frame():
    com = np.array([[3.9, 0, 0], [0.1, 0, 0], [0.3, 0, 0]])
    box = np.array([[4.0, 4, 4], [4.0, 4, 4], [8.0, 8, 8]])    # third box irrelevant to the two steps
    d = step_displacements(com, box)
    np.testing.assert_allclose(d, [[0.2, 0, 0], [0.2, 0, 0]])


def test_step_displacements_rejects_single_frame_and_bad_shape():
    with pytest.raises(ValueError):
        step_displacements(np.zeros((1, 3)), np.ones(3))
    with pytest.raises(ValueError):
        step_displacements(np.zeros((5, 2)), np.ones(3))


def test_outputs_are_float64_and_inputs_untouched():
    rng = np.random.default_rng(7)
    box = np.array([4.75, 4.75, 4.75])
    xyz, _, _ = unwrapped_trajectory(rng, 50, box)
    w = wrap(xyz, box).astype(np.float32)
    orig = w.copy()
    d = unwrapped_displacements(w, box)
    assert d.dtype == np.float64
    np.testing.assert_array_equal(w, orig)


def test_cumsum_of_displacements_reproduces_unwrapped_com():
    rng = np.random.default_rng(8)
    box = np.array([4.75, 4.75, 4.75])
    xyz, com_true, _ = unwrapped_trajectory(rng, 300, box, step_std=0.2)
    d = unwrapped_displacements(wrap(xyz, box), box)
    pos = com_true[0] + np.concatenate([np.zeros((1, 3)), np.cumsum(d, axis=0)])
    np.testing.assert_allclose(pos, com_true, atol=1e-9)