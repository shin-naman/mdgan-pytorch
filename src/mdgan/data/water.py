"""Water-molecule geometry under periodic boundary conditions.

Pure numpy. Converts wrapped (O, H, H) coordinates into unwrapped per-step
center-of-mass displacements, which is the subsystem MD-GAN models.

Constants come from the dataset's water.top (TIP3P with SETTLE constraints).
Every system in the 2023 dataset contains exactly one water molecule, so this
module is water-specific by design; a different penetrant would need its own.
"""

import numpy as np

WATER_ATOM_ORDER = ("O", "H", "H")
WATER_MASSES = np.array([16.0, 1.008, 1.008])   # water.top; 15.999 is numerically indistinguishable
OH_BOND_NM = 0.09572                            # SETTLE doh
HH_DIST_NM = 0.15139                            # SETTLE dhh
GEOMETRY_TOL_NM = 0.01                          # tolerance on the two constrained distances

# Quality-control heuristic, NOT a physical invariant. Minimum image is ambiguous
# at 0.5 L; 0.45 L leaves a margin while admitting the real cavity-to-cavity hops
# seen in PE (max observed 1.84 nm against L/2 = 2.38 nm). Validity on the other
# polymers is to be confirmed when their data are processed.
MAX_STEP_FRACTION = 0.45


def minimum_image(v: np.ndarray, box: np.ndarray) -> np.ndarray:
    """Map each component of v into (-L/2, L/2] for an orthorhombic box.

    v: (..., 3). box: (3,) or broadcastable to v's leading dims, e.g. (T, 1, 3)
    or (T, 3). Returns a new array; v is not modified.
    """
    v = np.asarray(v, dtype=np.float64)
    box = np.asarray(box, dtype=np.float64)
    if box.shape[-1] != 3 or v.shape[-1] != 3:
        raise ValueError(f"last axis must be 3; got v{v.shape} box{box.shape}")
    if not (box > 0).all():
        raise ValueError("box lengths must be positive")
    return v - box * np.round(v / box)


def water_com(xyz: np.ndarray, box: np.ndarray) -> np.ndarray:
    """Mass-weighted COM of one water molecule per frame, with the molecule made
    whole first: each H is minimum-imaged relative to the O.

    xyz: (T, 3 atoms, 3) in (O, H, H) order. box: (3,) or (T, 3). Returns (T, 3).
    The returned COM is still wrapped (it lives near the wrapped O); unwrapping
    happens in step_displacements.
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    if xyz.ndim != 3 or xyz.shape[1:] != (3, 3):
        raise ValueError(f"xyz must be (T, 3 atoms, 3); got {xyz.shape}")
    if not np.isfinite(xyz).all():
        raise ValueError("xyz contains non-finite values")
    box = np.asarray(box, dtype=np.float64)
    box_t = box.reshape(-1, 1, 3) if box.ndim == 2 else box.reshape(1, 1, 3)

    o = xyz[:, 0:1]                                # (T, 1, 3)
    rel = minimum_image(xyz[:, 1:] - o, box_t)     # (T, 2, 3) H positions relative to O
    _check_geometry(rel)

    atoms = np.concatenate([o, o + rel], axis=1)   # (T, 3, 3), molecule whole
    return (atoms * WATER_MASSES[None, :, None]).sum(axis=1) / WATER_MASSES.sum()


def _check_geometry(rel: np.ndarray) -> None:
    """Reject anything that is not a rigid TIP3P water in (O, H, H) order."""
    oh = np.linalg.norm(rel, axis=-1)                            # (T, 2)
    hh = np.linalg.norm(rel[:, 0] - rel[:, 1], axis=-1)          # (T,)
    if np.abs(oh - OH_BOND_NM).max() > GEOMETRY_TOL_NM:
        raise ValueError(
            f"O-H distance off by up to {np.abs(oh - OH_BOND_NM).max():.4f} nm; "
            "expected rigid water in (O, H, H) order"
        )
    if np.abs(hh - HH_DIST_NM).max() > GEOMETRY_TOL_NM:
        raise ValueError(
            f"H-H distance off by up to {np.abs(hh - HH_DIST_NM).max():.4f} nm; "
            "expected rigid water in (O, H, H) order"
        )


def step_displacements(com: np.ndarray, box: np.ndarray) -> np.ndarray:
    """Unwrapped per-step displacements of a wrapped COM trajectory.

    com: (T, 3). box: (3,) or (T, 3) (per-frame; the box of the *earlier* frame
    of each step is used). Returns (T-1, 3) float64.
    Raises if any component exceeds MAX_STEP_FRACTION * L (see the constant).
    """
    com = np.asarray(com, dtype=np.float64)
    if com.ndim != 2 or com.shape[1] != 3:
        raise ValueError(f"com must be (T, 3); got {com.shape}")
    if com.shape[0] < 2:
        raise ValueError("need at least two frames for a displacement")
    box = np.asarray(box, dtype=np.float64)
    step_box = box[:-1] if box.ndim == 2 else box
    d = minimum_image(np.diff(com, axis=0), step_box)
    limit = MAX_STEP_FRACTION * step_box
    if (np.abs(d) > limit).any():
        worst = np.abs(d / step_box).max()
        raise ValueError(
            f"a step reaches {worst:.3f} of the box length (limit {MAX_STEP_FRACTION}); "
            "minimum-image unwrapping is not trustworthy for this trajectory"
        )
    return d


def unwrapped_displacements(xyz: np.ndarray, box: np.ndarray) -> np.ndarray:
    """xyz (T, 3, 3) wrapped (O, H, H) coordinates -> (T-1, 3) COM displacements."""
    return step_displacements(water_com(xyz, box), box)