"""Tests for mdgan.data.water (pure numpy; no trajectory files needed).

Strategy: build an UNWRAPPED ground truth, wrap it, and require the module to
recover the ground truth exactly. Merges the former test_water.py and
test_water_wrapping.py; the wrapping scenarios are the backbone.

Run: python -m pytest tests/test_water.py -v
"""

import numpy as np
import pytest

from mdgan.data.water import (
    GEOMETRY_TOL_NM, HH_DIST_NM, MAX_STEP_FRACTION, OH_BOND_NM,
    minimum_image, step_displacements, unwrapped_displacements, water_com,
)
from synthetic_water import rigid_water, unwrapped_trajectory, wrap

BOX = np.array([4.75, 4.75, 4.75])
RECT = np.array([3.2, 5.1, 7.9])           # non-cubic orthorhombic


def _place(rng, n_frames, box, step_std=0.15, **kw):
    """Unwrapped trajectory that definitely crosses boundaries many times."""
    xyz, com, steps = unwrapped_trajectory(rng, n_frames, box, step_std=step_std, **kw)
    return xyz, steps


def _molecule_at(o, rng=None):
    """One rigid water with O at `o`, fixed orientation."""
    mol = rigid_water(np.random.default_rng(123), 1)[0]    # (3, 3), O at origin
    return mol + o


# =============================================================================
class TestMinimumImage:
    """The fold v - L*round(v/L) on its own."""

    def test_minimum_image_maps_into_half_open_interval(self):
        rng = np.random.default_rng(0)
        box = np.array([4.0, 5.0, 6.0])
        v = rng.uniform(-30, 30, size=(1000, 3))
        m = minimum_image(v, box)
        assert (m > -box / 2 - 1e-12).all() and (m <= box / 2 + 1e-12).all()
        # differs from v by an integer number of boxes, exactly
        k = (v - m) / box
        np.testing.assert_allclose(k, np.round(k), atol=1e-9)

    def test_minimum_image_is_identity_inside_and_does_not_modify_input(self):
        box = np.array([4.0, 4.0, 4.0])
        v = np.array([[0.1, -1.9, 1.99]])
        orig = v.copy()
        np.testing.assert_array_equal(minimum_image(v, box), v)
        np.testing.assert_array_equal(v, orig)

    def test_minimum_image_broadcasts_per_frame_box(self):
        v = np.array([[3.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
        box = np.array([[4.0, 4.0, 4.0], [10.0, 10.0, 10.0]])
        m = minimum_image(v, box)
        np.testing.assert_allclose(m, [[-1.0, 0, 0], [3.0, 0, 0]])

    @pytest.mark.parametrize("bad_box", [np.array([4.0, 0.0, 4.0]), np.array([4.0, -1.0, 4.0]), np.array([4.0, 4.0])])
    def test_minimum_image_rejects_bad_box(self, bad_box):
        with pytest.raises(ValueError):
            minimum_image(np.zeros((2, 3)), bad_box)


# =============================================================================
class TestWaterCom:
    """Making the molecule whole and the geometry/mass contract."""

    def test_water_com_matches_unwrapped_truth_after_wrapping(self):
        rng = np.random.default_rng(1)
        box = np.array([4.75, 4.75, 4.75])
        xyz, com_true, _ = unwrapped_trajectory(rng, 500, box)
        com = water_com(wrap(xyz, box), box)
        # COM is returned wrapped; compare modulo the box
        np.testing.assert_allclose(minimum_image(com - com_true, box), 0.0, atol=1e-10)

    def test_water_com_uses_topology_masses(self):
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

    def test_water_com_rejects_wrong_atom_order(self):
        rng = np.random.default_rng(2)
        box = np.array([4.75, 4.75, 4.75])
        xyz, _, _ = unwrapped_trajectory(rng, 20, box)
        with pytest.raises(ValueError, match="O-H|H-H"):
            water_com(xyz[:, [1, 0, 2]], box)      # H, O, H

    def test_water_com_rejects_non_water_geometry(self):
        box = np.array([4.0, 4.0, 4.0])
        xyz = np.zeros((5, 3, 3)); xyz[:, 1, 0] = 0.3; xyz[:, 2, 1] = 0.3
        with pytest.raises(ValueError):
            water_com(xyz, box)

    def test_water_com_geometry_tolerance_is_tight(self):
        """A bond stretched by more than the tolerance must be rejected."""
        rng = np.random.default_rng(3)
        box = np.array([4.75, 4.75, 4.75])
        xyz, _, _ = unwrapped_trajectory(rng, 10, box)
        xyz[:, 1] += (xyz[:, 1] - xyz[:, 0]) * (2 * GEOMETRY_TOL_NM / OH_BOND_NM)   # stretch by 2*tol
        with pytest.raises(ValueError):
            water_com(xyz, box)

    @pytest.mark.parametrize("shape", [(10, 3), (10, 4, 3), (10, 3, 2), (3, 3)])
    def test_water_com_rejects_bad_shapes(self, shape):
        with pytest.raises(ValueError):
            water_com(np.zeros(shape), np.ones(3))

    def test_water_com_rejects_nan(self):
        rng = np.random.default_rng(4)
        xyz, _, _ = unwrapped_trajectory(rng, 5, np.ones(3) * 4)
        xyz[2, 0, 0] = np.nan
        with pytest.raises(ValueError, match="non-finite"):
            water_com(xyz, np.ones(3) * 4)


# =============================================================================
class TestStepDisplacements:
    """Removing frame-to-frame jumps, and the 0.45 L guard."""

    def test_step_displacements_recover_unwrapped_steps_exactly(self):
        rng = np.random.default_rng(5)
        box = np.array([4.75, 4.75, 4.75])
        xyz, com_true, steps = unwrapped_trajectory(rng, 2000, box, step_std=0.15)
        d = unwrapped_displacements(wrap(xyz, box), box)
        assert d.shape == (1999, 3)
        np.testing.assert_allclose(d, steps, atol=1e-10)

    def test_large_but_legal_hops_survive(self):
        """Regression for the real PE data: steps above L/4 are real cavity hops and
        must pass through unchanged. Only steps beyond MAX_STEP_FRACTION*L are errors."""
        rng = np.random.default_rng(6)
        box = np.array([4.75, 4.75, 4.75])
        hop = 0.40 * box[0]                                   # > L/4, < 0.45 L
        xyz, _, steps = unwrapped_trajectory(rng, 400, box, step_std=0.05, hop_every=50, hop_size=hop)
        d = unwrapped_displacements(wrap(xyz, box), box)
        np.testing.assert_allclose(d, steps, atol=1e-10)
        assert (np.abs(d[:, 0]) > box[0] / 4).sum() == len(steps[50::50])

    def test_step_beyond_fraction_raises(self):
        box = np.array([4.0, 4.0, 4.0])
        com = np.array([[0.0, 0, 0], [0.5 * MAX_STEP_FRACTION * 4.0 + 1.0, 0, 0]])  # 1.9 nm of 4.0 -> 0.475 > 0.45
        with pytest.raises(ValueError, match="box length"):
            step_displacements(com, box)

    def test_step_exactly_at_fraction_is_allowed(self):
        box = np.array([4.0, 4.0, 4.0])
        com = np.array([[0.0, 0, 0], [MAX_STEP_FRACTION * 4.0, 0, 0]])
        d = step_displacements(com, box)
        np.testing.assert_allclose(d, [[MAX_STEP_FRACTION * 4.0, 0, 0]])

    def test_step_displacements_use_per_frame_box_of_earlier_frame(self):
        com = np.array([[3.9, 0, 0], [0.1, 0, 0], [0.3, 0, 0]])
        box = np.array([[4.0, 4, 4], [4.0, 4, 4], [8.0, 8, 8]])    # third box irrelevant to the two steps
        d = step_displacements(com, box)
        np.testing.assert_allclose(d, [[0.2, 0, 0], [0.2, 0, 0]])

    def test_step_displacements_rejects_single_frame_and_bad_shape(self):
        with pytest.raises(ValueError):
            step_displacements(np.zeros((1, 3)), np.ones(3))
        with pytest.raises(ValueError):
            step_displacements(np.zeros((5, 2)), np.ones(3))

    def test_outputs_are_float64_and_inputs_untouched(self):
        rng = np.random.default_rng(7)
        box = np.array([4.75, 4.75, 4.75])
        xyz, _, _ = unwrapped_trajectory(rng, 50, box)
        w = wrap(xyz, box).astype(np.float32)
        orig = w.copy()
        d = unwrapped_displacements(w, box)
        assert d.dtype == np.float64
        np.testing.assert_array_equal(w, orig)

    def test_cumsum_of_displacements_reproduces_unwrapped_com(self):
        rng = np.random.default_rng(8)
        box = np.array([4.75, 4.75, 4.75])
        xyz, com_true, _ = unwrapped_trajectory(rng, 300, box, step_std=0.2)
        d = unwrapped_displacements(wrap(xyz, box), box)
        pos = com_true[0] + np.concatenate([np.zeros((1, 3)), np.cumsum(d, axis=0)])
        np.testing.assert_allclose(pos, com_true, atol=1e-9)


# =============================================================================
class TestWrappingInvariant:
    """The general invariant: any integer image shift of any atom in any frame
    leaves the output unchanged. Every wrapping convention is a special case."""

    @pytest.mark.parametrize("box", [BOX, RECT])
    @pytest.mark.parametrize("seed", range(5))
    def test_arbitrary_independent_images_per_atom_per_frame(self, box, seed):
        rng = np.random.default_rng(seed)
        xyz, steps = _place(rng, 600, box)
        shifts = rng.integers(-3, 4, size=xyz.shape)           # independent per atom, frame, axis
        d_ref = unwrapped_displacements(xyz, box)
        d_shift = unwrapped_displacements(xyz + shifts * box, box)
        np.testing.assert_allclose(d_shift, d_ref, atol=1e-9)
        np.testing.assert_allclose(d_ref, steps, atol=1e-9)
        assert (shifts != 0).any()


# =============================================================================
class TestSplitMolecules:
    """One molecule written across a face or a corner within a frame."""

    @pytest.mark.parametrize("axis", [0, 1, 2])
    @pytest.mark.parametrize("which", ["H1", "H2", "both_H", "O_only"])
    def test_molecule_split_across_one_face(self, axis, which):
        """H wrapped while O isn't, or O wrapped while the H's aren't."""
        box = BOX.copy()
        o = np.array([2.0, 2.0, 2.0])
        mol = _molecule_at(o)
        # Put the O just inside the face on `axis`, then push the chosen atoms across.
        mol[:, axis] += box[axis] - 0.02 - o[axis]
        truth = water_com(mol[None], box)[0]
        shifted = mol.copy()
        idx = {"H1": [1], "H2": [2], "both_H": [1, 2], "O_only": [0]}[which]
        shifted[idx, axis] += box[axis] if which != "O_only" else -box[axis]
        got = water_com(shifted[None], box)[0]
        np.testing.assert_allclose(minimum_image(got - truth, box), 0.0, atol=1e-12)

    def test_molecule_split_across_a_corner_all_three_axes(self):
        box = BOX.copy()
        mol = _molecule_at(np.array([0.01, 0.01, 0.01]))      # O at the corner
        truth = water_com(mol[None], box)[0]
        shifted = mol.copy()
        shifted[1] += np.array([box[0], 0, -box[2]])           # H1 in two other images
        shifted[2] += np.array([0, -box[1], box[2]])
        shifted[0] += np.array([-box[0], -box[1], -box[2]])    # O in the far image
        got = water_com(shifted[None], box)[0]
        np.testing.assert_allclose(minimum_image(got - truth, box), 0.0, atol=1e-12)


# =============================================================================
class TestFrameCrossings:
    """Consecutive frames written in different images."""

    @pytest.mark.parametrize("axis", [0, 1, 2])
    @pytest.mark.parametrize("direction", [+1, -1])
    def test_com_crosses_boundary_between_frames(self, axis, direction):
        """Frame t just inside one face, frame t+1 just past it (so it wraps to the
        other side). The displacement must be the small physical step."""
        box = BOX.copy()
        step = 0.03
        o0 = np.array([2.0, 2.0, 2.0])
        o0[axis] = (box[axis] - 0.01) if direction > 0 else 0.01
        o1 = o0.copy(); o1[axis] += direction * step
        m0, m1 = _molecule_at(o0), _molecule_at(o1)
        xyz = wrap(np.stack([m0, m1]), box)                    # frame 1 lands on the opposite side
        assert abs(xyz[0, 0, axis] - xyz[1, 0, axis]) > box[axis] / 2   # confirm it wrapped
        d = unwrapped_displacements(xyz, box)
        expected = np.zeros(3); expected[axis] = direction * step
        np.testing.assert_allclose(d[0], expected, atol=1e-12)

    def test_frames_in_different_images_entirely(self):
        """Frame t in image (0,0,0), frame t+1 in image (2,-1,3): still a small step."""
        box = BOX.copy()
        m0 = _molecule_at(np.array([1.0, 1.0, 1.0]))
        m1 = _molecule_at(np.array([1.05, 0.98, 1.02])) + np.array([2, -1, 3]) * box
        d = unwrapped_displacements(np.stack([m0, m1]), box)
        np.testing.assert_allclose(d[0], [0.05, -0.02, 0.02], atol=1e-12)

    def test_hop_near_guard_across_boundary(self):
        """A legal hop (0.44 L) that also crosses a periodic face in the same step."""
        box = BOX.copy()
        hop = 0.44 * box[0]
        o0 = np.array([box[0] - 0.5, 2.0, 2.0])
        o1 = o0 + np.array([hop, 0, 0])                        # physically past the face
        xyz = wrap(np.stack([_molecule_at(o0), _molecule_at(o1)]), box)
        d = unwrapped_displacements(xyz, box)
        np.testing.assert_allclose(d[0], [hop, 0, 0], atol=1e-12)
        assert hop < MAX_STEP_FRACTION * box[0]


# =============================================================================
class TestFileConventions:
    """Wrapping schemes seen in real files."""

    def test_whole_molecule_wrapped_as_a_unit_like_pbc_mol(self):
        """GROMACS -pbc mol keeps the molecule whole and wraps by COM; atoms may then
        sit slightly outside [0, L), as in the real PE files (-0.08 .. 4.84 nm)."""
        rng = np.random.default_rng(9)
        xyz, steps = _place(rng, 500, BOX)
        com = (xyz * np.array([16.0, 1.008, 1.008])[None, :, None]).sum(1) / (16.0 + 2.016)
        shift = np.floor(com / BOX)[:, None, :] * BOX           # same shift for all 3 atoms
        d = unwrapped_displacements(xyz - shift, BOX)
        np.testing.assert_allclose(d, steps, atol=1e-9)
        assert ((xyz - shift) < 0).any() or ((xyz - shift) > BOX).any()

    def test_per_atom_wrap_splits_molecules_and_is_still_exact(self):
        """Count that the per-atom wrap really produced split molecules in this test."""
        rng = np.random.default_rng(10)
        xyz, steps = _place(rng, 2000, BOX)
        w = wrap(xyz, BOX)
        split = (np.abs(w[:, 1:, :] - w[:, :1, :]) > BOX / 2).any(axis=(1, 2)).sum()
        assert split > 0, "test did not exercise split molecules"
        d = unwrapped_displacements(w, BOX)
        np.testing.assert_allclose(d, steps, atol=1e-9)

    def test_wrap_into_centered_box_convention(self):
        """Some tools wrap into [-L/2, L/2) instead of [0, L)."""
        rng = np.random.default_rng(11)
        xyz, steps = _place(rng, 500, BOX)
        w = xyz - BOX * np.round(xyz / BOX)
        d = unwrapped_displacements(w, BOX)
        np.testing.assert_allclose(d, steps, atol=1e-9)

    def test_wrapping_with_per_frame_box_array(self):
        """Per-frame box (constant values) must behave identically to a (3,) box."""
        rng = np.random.default_rng(12)
        xyz, steps = _place(rng, 300, RECT)
        w = wrap(xyz, RECT)
        d_vec = unwrapped_displacements(w, RECT)
        d_frame = unwrapped_displacements(w, np.tile(RECT, (300, 1)))
        np.testing.assert_array_equal(d_vec, d_frame)
        np.testing.assert_allclose(d_vec, steps, atol=1e-9)


# =============================================================================
class TestDocumentedLimit:
    """What minimum image cannot do, and how the guard catches its symptom.
    Referenced from MAX_STEP_FRACTION in src/mdgan/data/water.py."""

    def test_true_step_beyond_half_box_is_misread_and_caught_only_if_over_guard(self):
        """Documents the limit of minimum image: a real step of 0.6 L is folded to
        -0.4 L. Below the 0.45 L guard that fold would be silent; this is why the
        guard exists and why it is a heuristic. A fold to 0.46 L must raise."""
        box = BOX.copy()
        o0 = np.array([1.0, 2.0, 2.0])
        o1 = o0 + np.array([0.54 * box[0], 0, 0])             # folds to -0.46 L -> must raise
        with pytest.raises(ValueError, match="box length"):
            unwrapped_displacements(np.stack([_molecule_at(o0), _molecule_at(o1)]), box)