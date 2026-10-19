"""Exhaustive periodic-image tests for mdgan.data.water.

The invariant: the unwrapped COM displacements are a function of the physical
trajectory only. Adding ANY integer number of box lengths to ANY atom in ANY
frame, independently, must leave the output bit-for-bit unchanged. Every
wrapping scheme GROMACS can produce (per atom, per molecule, per residue,
compact, cluster) is a special case of such shifts.

Run: python -m pytest tests/data/test_water_wrapping.py -v
"""

import numpy as np
import pytest

from mdgan.data.water import MAX_STEP_FRACTION, minimum_image, unwrapped_displacements, water_com
from synthetic_water import rigid_water, unwrapped_trajectory, wrap

BOX = np.array([4.75, 4.75, 4.75])
RECT = np.array([3.2, 5.1, 7.9])           # non-cubic orthorhombic


def _place(rng, n_frames, box, step_std=0.15, **kw):
    """Unwrapped trajectory that definitely crosses boundaries many times."""
    xyz, com, steps = unwrapped_trajectory(rng, n_frames, box, step_std=step_std, **kw)
    return xyz, steps


# --- 1. The general invariant --------------------------------------------------------

@pytest.mark.parametrize("box", [BOX, RECT])
@pytest.mark.parametrize("seed", range(5))
def test_arbitrary_independent_images_per_atom_per_frame(box, seed):
    rng = np.random.default_rng(seed)
    xyz, steps = _place(rng, 600, box)
    shifts = rng.integers(-3, 4, size=xyz.shape)           # independent per atom, frame, axis
    d_ref = unwrapped_displacements(xyz, box)
    d_shift = unwrapped_displacements(xyz + shifts * box, box)
    np.testing.assert_allclose(d_shift, d_ref, atol=1e-9)
    np.testing.assert_allclose(d_ref, steps, atol=1e-9)
    assert (shifts != 0).any()


# --- 2. Enumerated single-boundary cases (static geometry, one frame) ---------------

def _molecule_at(o, rng=None):
    """One rigid water with O at `o`, fixed orientation."""
    mol = rigid_water(np.random.default_rng(123), 1)[0]    # (3, 3), O at origin
    return mol + o


@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("which", ["H1", "H2", "both_H", "O_only"])
def test_molecule_split_across_one_face(axis, which):
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


def test_molecule_split_across_a_corner_all_three_axes():
    box = BOX.copy()
    mol = _molecule_at(np.array([0.01, 0.01, 0.01]))      # O at the corner
    truth = water_com(mol[None], box)[0]
    shifted = mol.copy()
    shifted[1] += np.array([box[0], 0, -box[2]])           # H1 in two other images
    shifted[2] += np.array([0, -box[1], box[2]])
    shifted[0] += np.array([-box[0], -box[1], -box[2]])    # O in the far image
    got = water_com(shifted[None], box)[0]
    np.testing.assert_allclose(minimum_image(got - truth, box), 0.0, atol=1e-12)


# --- 3. Consecutive frames wrapped differently -------------------------------------

@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("direction", [+1, -1])
def test_com_crosses_boundary_between_frames(axis, direction):
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


def test_frames_in_different_images_entirely():
    """Frame t in image (0,0,0), frame t+1 in image (2,-1,3): still a small step."""
    box = BOX.copy()
    m0 = _molecule_at(np.array([1.0, 1.0, 1.0]))
    m1 = _molecule_at(np.array([1.05, 0.98, 1.02])) + np.array([2, -1, 3]) * box
    d = unwrapped_displacements(np.stack([m0, m1]), box)
    np.testing.assert_allclose(d[0], [0.05, -0.02, 0.02], atol=1e-12)


def test_hop_near_guard_across_boundary():
    """A legal hop (0.44 L) that also crosses a periodic face in the same step."""
    box = BOX.copy()
    hop = 0.44 * box[0]
    o0 = np.array([box[0] - 0.5, 2.0, 2.0])
    o1 = o0 + np.array([hop, 0, 0])                        # physically past the face
    xyz = wrap(np.stack([_molecule_at(o0), _molecule_at(o1)]), box)
    d = unwrapped_displacements(xyz, box)
    np.testing.assert_allclose(d[0], [hop, 0, 0], atol=1e-12)
    assert hop < MAX_STEP_FRACTION * box[0]


# --- 4. Wrapping conventions seen in real files -----------------------------------

def test_whole_molecule_wrapped_as_a_unit_like_pbc_mol():
    """GROMACS -pbc mol keeps the molecule whole and wraps by COM; atoms may then
    sit slightly outside [0, L), as in the real PE files (-0.08 .. 4.84 nm)."""
    rng = np.random.default_rng(9)
    xyz, steps = _place(rng, 500, BOX)
    com = (xyz * np.array([16.0, 1.008, 1.008])[None, :, None]).sum(1) / (16.0 + 2.016)
    shift = np.floor(com / BOX)[:, None, :] * BOX           # same shift for all 3 atoms
    d = unwrapped_displacements(xyz - shift, BOX)
    np.testing.assert_allclose(d, steps, atol=1e-9)
    assert ((xyz - shift) < 0).any() or ((xyz - shift) > BOX).any()


def test_per_atom_wrap_splits_molecules_and_is_still_exact():
    """Count that the per-atom wrap really produced split molecules in this test."""
    rng = np.random.default_rng(10)
    xyz, steps = _place(rng, 2000, BOX)
    w = wrap(xyz, BOX)
    split = (np.abs(w[:, 1:, :] - w[:, :1, :]) > BOX / 2).any(axis=(1, 2)).sum()
    assert split > 0, "test did not exercise split molecules"
    d = unwrapped_displacements(w, BOX)
    np.testing.assert_allclose(d, steps, atol=1e-9)


def test_wrap_into_centered_box_convention():
    """Some tools wrap into [-L/2, L/2) instead of [0, L)."""
    rng = np.random.default_rng(11)
    xyz, steps = _place(rng, 500, BOX)
    w = xyz - BOX * np.round(xyz / BOX)
    d = unwrapped_displacements(w, BOX)
    np.testing.assert_allclose(d, steps, atol=1e-9)


def test_wrapping_with_per_frame_box_array():
    """Per-frame box (constant values) must behave identically to a (3,) box."""
    rng = np.random.default_rng(12)
    xyz, steps = _place(rng, 300, RECT)
    w = wrap(xyz, RECT)
    d_vec = unwrapped_displacements(w, RECT)
    d_frame = unwrapped_displacements(w, np.tile(RECT, (300, 1)))
    np.testing.assert_array_equal(d_vec, d_frame)
    np.testing.assert_allclose(d_vec, steps, atol=1e-9)


# --- 5. What must NOT be silently accepted ---------------------------------------

def test_true_step_beyond_half_box_is_misread_and_caught_only_if_over_guard():
    """Documents the limit of minimum image: a real step of 0.6 L is folded to
    -0.4 L. Below the 0.45 L guard that fold would be silent; this is why the
    guard exists and why it is a heuristic. A fold to 0.46 L must raise."""
    box = BOX.copy()
    o0 = np.array([1.0, 2.0, 2.0])
    o1 = o0 + np.array([0.54 * box[0], 0, 0])             # folds to -0.46 L -> must raise
    with pytest.raises(ValueError, match="box length"):
        unwrapped_displacements(np.stack([_molecule_at(o0), _molecule_at(o1)]), box)