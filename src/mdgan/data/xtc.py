"""Read GROMACS .xtc trajectories without a topology.

Uses mdtraj's low-level XTCTrajectoryFile, which returns coordinates, times,
step numbers and box vectors directly. mdtraj is an optional dependency
(install extra ``data``); it is imported lazily so the models never need it.

Units follow the XTC format: nm and ps.

The minimum-image unwrapping in water.py is only correct under three
assumptions about the file. Each is checked here so a violation fails loudly
at read time instead of producing silently wrong displacements:
 
  1. ORTHORHOMBIC BOX (rectangular, axes aligned with x, y, z). The fold
     v - L*round(v/L) is applied per axis and is wrong for a tilted (triclinic)
     box. Checked: all off-diagonal box-vector components are ~0.
  2. UNIFORM TIME SPACING. Every displacement must span the same interval
     (10 ps here) for the step-skip and MSD arithmetic to be meaningful.
     Checked: time stamps strictly increasing with a constant difference.
  3. SANE DATA. At least two frames (one displacement), all values finite,
     positive box lengths.
"""

from dataclasses import dataclass
from pathlib import Path
 
import numpy as np
 
ORTHORHOMBIC_TOL_NM = 1e-6   # off-diagonal box components above this = tilted box
TIME_TOL_PS = 1e-3           # allowed jitter in the sampling interval (float32 time stamps)
 
 
@dataclass(frozen=True)
class RawTrajectory:
    """One trajectory exactly as stored in the file, plus derived facts.
 
    xyz     : (T, A, 3) float64 coordinates, nm, wrapped as written
    time_ps : (T,) float64 time stamps
    box_nm  : (T, 3) float64 box lengths per frame (the diagonal of the box matrix)
    source  : path the data came from, for provenance
    Frozen: the arrays are the record of what the file contained; nothing
    downstream may modify them in place.
    """
 
    xyz: np.ndarray
    time_ps: np.ndarray
    box_nm: np.ndarray
    source: str
 
    @property
    def n_frames(self) -> int:
        return self.xyz.shape[0]
 
    @property
    def n_atoms(self) -> int:
        return self.xyz.shape[1]
 
    @property
    def dt_ps(self) -> float:
        """Sampling interval. Uniformity is guaranteed by read_xtc, so one difference suffices."""
        return float(self.time_ps[1] - self.time_ps[0])
 
    @property
    def box_is_constant(self) -> bool:
        """True for an NVE/NVT run; False means the box changed during the trajectory."""
        return bool(np.allclose(self.box_nm, self.box_nm[0], rtol=0, atol=ORTHORHOMBIC_TOL_NM))
 
 
def read_xtc(path) -> RawTrajectory:
    """Decode one .xtc file and run the checks described in the module docstring.
 
    mdtraj is imported here, not at module level, so that the models and the
    rest of the package never depend on it (optional extra ``data``).
    Raises FileNotFoundError or ValueError; never returns a partially valid object.
    """
    try:
        from mdtraj.formats import XTCTrajectoryFile
    except ImportError as e:  # pragma: no cover
        raise ImportError("reading .xtc requires mdtraj: pip install 'mdgan-pytorch[data]'") from e
 
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
 
    # One call decodes every frame. `step` (MD step numbers) is not needed:
    # time_ps already encodes it, and nothing downstream uses step counts.
    with XTCTrajectoryFile(str(path)) as f:
        xyz, time, _step, box = f.read()
 
    return _validate(np.asarray(xyz), np.asarray(time), np.asarray(box), str(path))
 
 
def _validate(xyz: np.ndarray, time: np.ndarray, box: np.ndarray, source: str) -> RawTrajectory:
    """Apply checks 1-3 from the module docstring and build the RawTrajectory."""
 
    # --- 3. Sane coordinates -------------------------------------------------------
    if xyz.ndim != 3 or xyz.shape[2] != 3:
        raise ValueError(f"{source}: expected coordinates (T, A, 3); got {xyz.shape}")
    if xyz.shape[0] < 2:
        raise ValueError(f"{source}: need at least two frames; got {xyz.shape[0]}")
    if not np.isfinite(xyz).all():
        raise ValueError(f"{source}: non-finite coordinates")
 
    # --- 2. Uniform, increasing time ------------------------------------------------
    time = time.astype(np.float64)
    if time.shape != (xyz.shape[0],) or not np.isfinite(time).all():
        raise ValueError(f"{source}: bad time array {time.shape}")
    dts = np.diff(time)
    if (dts <= 0).any():
        raise ValueError(f"{source}: time stamps are not strictly increasing")
    if np.abs(dts - dts[0]).max() > TIME_TOL_PS:
        raise ValueError(
            f"{source}: non-uniform sampling interval (min {dts.min():.4f}, max {dts.max():.4f} ps)"
        )
 
    # --- 1. Orthorhombic box --------------------------------------------------------
    # box is (T, 3, 3): row i is box vector i. Orthorhombic means the matrix is
    # diagonal; the diagonal entries are then the box lengths Lx, Ly, Lz.
    box = box.astype(np.float64)
    if box.shape != (xyz.shape[0], 3, 3):
        raise ValueError(f"{source}: expected box vectors (T, 3, 3); got {box.shape}")
    diag = np.einsum("tii->ti", box)                         # (T, 3) box lengths
    off_diagonal = box - diag[:, :, None] * np.eye(3)        # zero out the diagonal
    if np.abs(off_diagonal).max() > ORTHORHOMBIC_TOL_NM:
        raise ValueError(
            f"{source}: box is not orthorhombic (off-diagonal component up to "
            f"{np.abs(off_diagonal).max():.3e} nm); minimum image per axis would be wrong"
        )
    if not (diag > 0).all():
        raise ValueError(f"{source}: box has a non-positive length")
 
    return RawTrajectory(xyz=xyz.astype(np.float64), time_ps=time, box_nm=diag, source=source)