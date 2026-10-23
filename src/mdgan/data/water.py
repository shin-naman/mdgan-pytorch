"""Unwrap a wrapped single-water trajectory into physical COM displacements.

WHAT THE INPUT LOOKS LIKE
-------------------------
GROMACS writes .xtc coordinates *wrapped*: whenever an atom leaves the box it is
written back on the opposite face, so every coordinate stays within roughly
[0, L). Two artefacts follow from this, and both must be undone before any
diffusion quantity can be computed:

  (a) SPLIT MOLECULE. The O and the two H's of one water molecule can be written
      on opposite faces of the box in the same frame (the O at x = 0.01 nm, an H
      at x = 4.74 nm). A naive center of mass would then sit in the middle of the
      box, where no atom is.

  (b) JUMP BETWEEN FRAMES. When the molecule's COM crosses a face between frame t
      and t+1, its written position jumps by about one box length. Diffed
      naively, that step looks like a 4.7 nm displacement in 10 ps.

WHAT THIS MODULE DOES
---------------------
Both artefacts are removed with the minimum-image convention: a vector whose
component exceeds half a box length in magnitude is shifted by whole box
lengths until it lies in (-L/2, L/2]. Applied to intra-molecular vectors it
fixes (a); applied to frame-to-frame COM differences it fixes (b). The result is
the trajectory the molecule actually followed: per-step displacements whose
cumulative sum runs to infinity, never wrapped. This is the same operation
MDAnalysis' `unwrap` + `NoJump` performs frame by frame; here it is done for
all frames at once with numpy (checked once against MDAnalysis: agreement ~2e-6 nm;
see docs/decisions.md, "Unwrapping").

WHEN MINIMUM IMAGE IS VALID
---------------------------
It is exact as long as the true vector is shorter than L/2 in every component.
For intra-molecular vectors that is always true (O-H = 0.096 nm). For 10 ps COM
steps it is true for water in any of these polymers (largest observed step in PE
is 1.84 nm against L/2 = 2.38 nm). A step that is truly longer than L/2 would be
folded to the wrong image silently; the MAX_STEP_FRACTION guard catches the
symptom of that (a folded step near L/2) and refuses the trajectory.

CONSTANTS come from the dataset's water.top (TIP3P, SETTLE constraints). Every
system in the 2023 dataset contains exactly one such water, so this module is
water-specific by design. Units: nm throughout.
"""

import numpy as np

# --- Water model, from water.top ----------------------------------------------------
WATER_ATOM_ORDER = ("O", "H", "H")              # atom order the .gro/.xtc must follow
WATER_MASSES = np.array([16.0, 1.008, 1.008])   # water.top masses (15.999 is indistinguishable)
OH_BOND_NM = 0.09572                            # SETTLE constraint doh
HH_DIST_NM = 0.15139                            # SETTLE constraint dhh
GEOMETRY_TOL_NM = 0.01                          # how far a rigid bond may deviate before we reject

# --- Quality-control guard on step size -----------------------------------------------
# Heuristic, NOT a physical invariant. Minimum image becomes ambiguous at 0.5 L.
# 0.45 L leaves a margin while admitting the real cavity-to-cavity hops seen in
# PE (max 1.84 nm, i.e. 0.39 L). Validity on other polymers is to be confirmed
# when their data are processed; PE is the fastest-diffusing system in the set.
# The silent-fold limit is pinned by tests/test_water.py::TestDocumentedLimit.
MAX_STEP_FRACTION = 0.45


def minimum_image(v: np.ndarray, box: np.ndarray) -> np.ndarray:
    """Shift each component of `v` by whole box lengths into (-L/2, L/2].

    v   : (..., 3) vectors (differences of positions, never positions themselves)
    box : (3,) box lengths, or (T, 3) / (T, 1, 3) to use a different box per frame

    The formula v - L*round(v/L) subtracts the integer number of box lengths
    nearest to v/L. For |v| < L/2 that integer is 0 and v is returned unchanged;
    for a wrapped vector such as v = -4.65 nm in a 4.75 nm box it returns +0.10 nm.
    Only valid for orthorhombic (rectangular) boxes; xtc.py rejects others.
    Returns a new array; the input is not modified.
    """
    v = np.asarray(v, dtype=np.float64)
    box = np.asarray(box, dtype=np.float64)
    if box.shape[-1] != 3 or v.shape[-1] != 3:
        raise ValueError(f"last axis must be 3; got v{v.shape} box{box.shape}")
    if not (box > 0).all():
        raise ValueError("box lengths must be positive")
    return v - box * np.round(v / box)


def water_com(xyz: np.ndarray, box: np.ndarray) -> np.ndarray:
    """Center of mass of one water molecule per frame, with the molecule made whole.

    xyz : (T, 3 atoms, 3) wrapped coordinates in (O, H, H) order, nm
    box : (3,) or (T, 3) box lengths, nm
    Returns (T, 3) COM positions. These are still *wrapped* (each lies near the
    wrapped O of its frame); jumps between frames are removed in
    step_displacements, not here.

    Steps:
      1. Validate shapes and finiteness.
      2. MAKE WHOLE: express each H relative to the O and minimum-image that
         vector. If an H was written on the far face of the box (artefact (a)),
         this moves it back next to its O. The O itself is left where it was
         written; it only serves as the anchor.
      3. GEOMETRY CHECK: after step 2 the O-H and H-H distances must equal the
         SETTLE constraints. Anything else means wrong atom order, a non-water
         file, or coordinates in the wrong unit (Angstrom would read as 0.957).
      4. Mass-weighted COM of the whole molecule.
    """
    # 1. Validate
    xyz = np.asarray(xyz, dtype=np.float64)
    if xyz.ndim != 3 or xyz.shape[1:] != (3, 3):
        raise ValueError(f"xyz must be (T, 3 atoms, 3); got {xyz.shape}")
    if not np.isfinite(xyz).all():
        raise ValueError("xyz contains non-finite values")
    box = np.asarray(box, dtype=np.float64)
    box_t = box.reshape(-1, 1, 3) if box.ndim == 2 else box.reshape(1, 1, 3)  # broadcast over atoms

    # 2. Make whole: H positions relative to O, folded to the nearest image of O
    o = xyz[:, 0:1]                                   # (T, 1, 3) oxygen of each frame
    h_rel = minimum_image(xyz[:, 1:] - o, box_t)      # (T, 2, 3) O->H vectors, now < L/2

    # 3. Geometry check
    _check_rigid_water(h_rel)

    # 4. COM over the whole molecule (O + the two re-attached H's)
    atoms = np.concatenate([o, o + h_rel], axis=1)    # (T, 3, 3), molecule whole
    return (atoms * WATER_MASSES[None, :, None]).sum(axis=1) / WATER_MASSES.sum()


def _check_rigid_water(h_rel: np.ndarray) -> None:
    """Reject anything that is not a rigid TIP3P water in (O, H, H) order."""
    oh = np.linalg.norm(h_rel, axis=-1)                       # (T, 2) O-H lengths
    hh = np.linalg.norm(h_rel[:, 0] - h_rel[:, 1], axis=-1)   # (T,)   H-H length
    oh_dev = np.abs(oh - OH_BOND_NM).max()
    hh_dev = np.abs(hh - HH_DIST_NM).max()
    if oh_dev > GEOMETRY_TOL_NM:
        raise ValueError(
            f"O-H distance deviates by up to {oh_dev:.4f} nm from {OH_BOND_NM} nm; "
            "expected rigid water in (O, H, H) order with coordinates in nm"
        )
    if hh_dev > GEOMETRY_TOL_NM:
        raise ValueError(
            f"H-H distance deviates by up to {hh_dev:.4f} nm from {HH_DIST_NM} nm; "
            "expected rigid water in (O, H, H) order with coordinates in nm"
        )


def step_displacements(com: np.ndarray, box: np.ndarray) -> np.ndarray:
    """Physical COM displacement at every step, with box crossings removed.

    com : (T, 3) wrapped COM positions from water_com, nm
    box : (3,) or (T, 3); with a per-frame box, the box of the earlier frame of
          each step is used (a step t->t+1 is judged in frame t's box)
    Returns (T-1, 3) float64 displacements d[t] = r(t+1) - r(t), unwrapped.
    np.cumsum(d) + r(0) is the trajectory the molecule actually followed.

    Steps:
      1. Validate.
      2. Raw frame-to-frame difference of the wrapped COM.
      3. REMOVE JUMPS: minimum-image the difference. A crossing of a face shows
         up as a difference of about +/- L; the fold turns it into the small
         physical step (artefact (b)).
      4. GUARD: refuse the trajectory if any folded component is longer than
         MAX_STEP_FRACTION * L. Nothing physical moves that far in one sampling
         interval here, so such a value means the fold in step 3 was ambiguous
         (true step near or beyond L/2) and the result cannot be trusted.
    """
    # 1. Validate
    com = np.asarray(com, dtype=np.float64)
    if com.ndim != 2 or com.shape[1] != 3:
        raise ValueError(f"com must be (T, 3); got {com.shape}")
    if com.shape[0] < 2:
        raise ValueError("need at least two frames for a displacement")
    box = np.asarray(box, dtype=np.float64)
    step_box = box[:-1] if box.ndim == 2 else box     # box of the earlier frame per step

    # 2.+3. Difference, then fold box crossings away
    d = minimum_image(np.diff(com, axis=0), step_box)

    # 4. Guard against ambiguous folds
    frac = np.abs(d) / step_box
    if (frac > MAX_STEP_FRACTION).any():
        t, axis = np.unravel_index(frac.argmax(), frac.shape)
        raise ValueError(
            f"step {t} axis {axis} spans {frac.max():.3f} of the box length "
            f"(limit {MAX_STEP_FRACTION}); minimum-image unwrapping is not trustworthy here"
        )
    return d


def unwrapped_displacements(xyz: np.ndarray, box: np.ndarray) -> np.ndarray:
    """Wrapped (O, H, H) coordinates (T, 3, 3) -> unwrapped COM displacements (T-1, 3)."""
    return step_displacements(water_com(xyz, box), box)