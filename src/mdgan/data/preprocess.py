"""Turn one system's trajectory directory into cached, unwrapped COM displacements.

INPUT LAYOUT (the Zenodo archive, one folder per polymer system)
----------------------------------------------------------------
    MD_data/<system>/
        msd.out                 reference MSD written by the authors (not read here)
        trajectory/traj001.xtc  one water molecule in the polymer, wrapped coordinates
        trajectory/traj002.xtc
        ...

OUTPUT
------
A SystemData object, optionally cached as one .npz per system:

    displacements : (N, T-1, 3) float32   unwrapped COM step displacements, nm
    dt_ps         : float                 sampling interval (10 ps in this dataset)
    box_nm        : (N, 3) float64        box lengths per trajectory
    meta          : provenance (files, md5, frame count, constants used)

The displacement array is the ONLY thing the rest of the pipeline consumes.
dataset.py slices the first 2M*n_s steps of it for training; msd.py uses all
of it for the reference curve. Positions never leave this module.

WHAT IS CHECKED HERE
--------------------
Per trajectory (delegated): xtc.py validates the file; water.py validates the
molecule and unwraps. This module adds the checks that only make sense across
trajectories or at the system level:
  - exactly 3 atoms (the water-only contract of this dataset)
  - the box is constant within each trajectory (NVE); a varying box is refused
    because minimum image across a changing box is not well defined and nothing
    downstream could use it anyway
  - every trajectory in the system has the same frame count, dt and atom count,
    so they can be stacked into one array
  - the file list is natural-sorted (traj001 < traj002 < ... < traj100), ignoring
    anything that is not traj*.xtc (macOS ._ sidecars, notes, etc.)

WHY float32 IN THE CACHE
------------------------
.xtc stores coordinates to 0.001 nm, so displacements carry ~3 significant
digits of physical precision; float32 (7 digits) loses nothing. The cache for
60 x 8000 steps is 5.8 MB. Consumers that accumulate (cumsum, MSD) cast to
float64 first; the reference MSD reproduces msd.out to 5e-6 from this cache.

Units: nm and ps throughout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from mdgan.data.water import (
    GEOMETRY_TOL_NM, HH_DIST_NM, MAX_STEP_FRACTION, OH_BOND_NM, WATER_MASSES,
    unwrapped_displacements,
)
from mdgan.data.xtc import read_xtc

PREPROCESS_VERSION = 1                 # bump when anything here changes what gets cached
TRAJECTORY_SUBDIR = "trajectory"
TRAJECTORY_GLOB = "traj*.xtc"
WATER_ATOM_COUNT = 3


@dataclass
class SystemData:
    """All trajectories of one system as a stacked displacement array plus provenance."""

    name: str
    displacements: np.ndarray      # (N, T-1, 3) float32, nm
    dt_ps: float
    box_nm: np.ndarray             # (N, 3) float64
    meta: dict = field(default_factory=dict)

    @property
    def n_trajectories(self) -> int:
        return self.displacements.shape[0]

    @property
    def n_steps(self) -> int:
        """Displacement steps per trajectory (= frames - 1)."""
        return self.displacements.shape[1]

    def save(self, path) -> None:
        """Write one .npz. meta is stored as a JSON string so it survives np.savez."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, name=self.name, displacements=self.displacements, dt_ps=self.dt_ps,
                 box_nm=self.box_nm, meta=json.dumps(self.meta))

    @classmethod
    def load(cls, path) -> "SystemData":
        with np.load(path) as z:
            return cls(name=str(z["name"]), displacements=z["displacements"],
                       dt_ps=float(z["dt_ps"]), box_nm=z["box_nm"],
                       meta=json.loads(str(z["meta"])))


# --- discovery --------------------------------------------------------------------------

def _natural_key(path: Path):
    """Sort key so that traj2 < traj10 (split the name into text and integer runs)."""
    return [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", path.name)]


def discover_trajectories(system_dir) -> list[Path]:
    """List <system_dir>/trajectory/traj*.xtc in natural order; nothing else."""
    traj_dir = Path(system_dir) / TRAJECTORY_SUBDIR
    if not traj_dir.is_dir():
        raise FileNotFoundError(f"no '{TRAJECTORY_SUBDIR}/' directory under {system_dir}")
    files = [p for p in traj_dir.glob(TRAJECTORY_GLOB)
             if p.is_file() and not p.name.startswith(".")]       # skip ._traj001.xtc sidecars
    if not files:
        raise FileNotFoundError(f"no {TRAJECTORY_GLOB} files in {traj_dir}")
    return sorted(files, key=_natural_key)


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --- per trajectory ------------------------------------------------------------------------

def process_trajectory(path) -> tuple[np.ndarray, float, np.ndarray, int]:
    """One .xtc -> (displacements (T-1, 3) float64, dt_ps, box (3,), n_frames).

    Steps: read + validate file (xtc.py); require 3 atoms and a constant box;
    make whole, COM, unwrap (water.py).
    """
    raw = read_xtc(path)
    if raw.n_atoms != WATER_ATOM_COUNT:
        raise ValueError(f"{path}: expected {WATER_ATOM_COUNT} atoms (one water); got {raw.n_atoms}")
    if not raw.box_is_constant:
        raise ValueError(f"{path}: box changes during the trajectory; expected NVE (constant box)")
    box = raw.box_nm[0]
    d = unwrapped_displacements(raw.xyz, box)
    return d, raw.dt_ps, box, raw.n_frames


# --- per system -----------------------------------------------------------------------------

def process_system(system_dir, name: str | None = None, cache=None, force: bool = False) -> SystemData:
    """Process every trajectory of a system, or load the cache if it is current.

    system_dir : MD_data/<system>
    name       : label stored in the result; defaults to the directory name
    cache      : optional .npz path. If it exists, it is loaded instead of
                 reprocessing, unless `force` is set or the md5 list of the
                 source files no longer matches the cache's provenance.
    """
    system_dir = Path(system_dir)
    name = name or system_dir.name
    files = discover_trajectories(system_dir)
    md5s = [_md5(p) for p in files]

    if cache is not None and Path(cache).is_file() and not force:
        cached = SystemData.load(cache)
        if cached.meta.get("md5") == md5s and cached.meta.get("preprocess_version") == PREPROCESS_VERSION:
            return cached
        # Otherwise fall through and reprocess: the sources or the code changed.

    disp, boxes, dts, frames = [], [], [], []
    for p in files:
        d, dt, box, n_frames = process_trajectory(p)
        disp.append(d.astype(np.float32)); boxes.append(box); dts.append(dt); frames.append(n_frames)

    # Cross-trajectory consistency: one array needs one shape, one dt, one geometry.
    if len(set(frames)) != 1:
        raise ValueError(f"{name}: trajectories have different frame counts {sorted(set(frames))}")
    if np.ptp(dts) > 1e-6:
        raise ValueError(f"{name}: trajectories have different sampling intervals {sorted(set(dts))}")

    data = SystemData(
        name=name,
        displacements=np.stack(disp),                 # (N, T-1, 3) float32
        dt_ps=float(dts[0]),
        box_nm=np.stack(boxes),                       # (N, 3)
        meta={
            "preprocess_version": PREPROCESS_VERSION,
            "system_dir": str(system_dir),
            "files": [p.name for p in files],
            "md5": md5s,
            "n_trajectories": len(files),
            "n_frames": frames[0],
            "n_steps": frames[0] - 1,
            "dt_ps": float(dts[0]),
            "box_constant_across_trajectories": bool(np.allclose(boxes, boxes[0])),
            "water_masses": WATER_MASSES.tolist(),
            "oh_bond_nm": OH_BOND_NM, "hh_dist_nm": HH_DIST_NM, "geometry_tol_nm": GEOMETRY_TOL_NM,
            "max_step_fraction": MAX_STEP_FRACTION,
            "max_step_nm": float(np.abs(np.stack(disp)).max()),
            "units": {"length": "nm", "time": "ps"},
        },
    )
    if cache is not None:
        data.save(cache)
    return data


# --- command line -----------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Preprocess one polymer system into a displacement cache.")
    ap.add_argument("system_dir", help="MD_data/<system>")
    ap.add_argument("--out", required=True, help="cache .npz to write")
    ap.add_argument("--force", action="store_true", help="reprocess even if the cache is current")
    args = ap.parse_args(argv)
    data = process_system(args.system_dir, cache=args.out, force=args.force)
    m = data.meta
    print(f"{data.name}: {data.n_trajectories} trajectories x {data.n_steps} steps, "
          f"dt = {data.dt_ps} ps, max step = {m['max_step_nm']:.3f} nm -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())