"""Synthetic data helpers shared by the tests (not a test module).

- rigid_water / unwrapped_trajectory / wrap: rigid TIP3P water on a random-walk
  COM with a known unwrapped ground truth.
- write_xtc: write a small .xtc with mdtraj (imported lazily).
"""

import numpy as np

from mdgan.data.water import HH_DIST_NM, OH_BOND_NM, WATER_MASSES


def rigid_water(rng, n_frames):
    """(T, 3, 3) unwrapped rigid TIP3P water with random orientation per frame,
    O at the origin of each frame's molecular frame."""
    half = np.arcsin(HH_DIST_NM / 2 / OH_BOND_NM)        # half the H-O-H angle
    h1 = OH_BOND_NM * np.array([np.sin(half), np.cos(half), 0.0])
    h2 = OH_BOND_NM * np.array([-np.sin(half), np.cos(half), 0.0])
    out = np.zeros((n_frames, 3, 3))
    for t in range(n_frames):
        q = rng.normal(size=4); q /= np.linalg.norm(q)
        a, b, c, d = q
        R = np.array([[a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c)],
                      [2*(b*c+a*d), a*a-b*b+c*c-d*d, 2*(c*d-a*b)],
                      [2*(b*d-a*c), 2*(c*d+a*b), a*a-b*b-c*c+d*d]])
        out[t, 1] = R @ h1
        out[t, 2] = R @ h2
    return out


def unwrapped_trajectory(rng, n_frames, box, step_std=0.1, hop_every=None, hop_size=None):
    """Unwrapped (T, 3, 3) coordinates: water COM random walk + rigid geometry."""
    steps = rng.normal(scale=step_std, size=(n_frames - 1, 3))
    if hop_every:
        steps[hop_every::hop_every, 0] += hop_size
    start = np.array([0.3, 0.2, 0.1]) * box
    com = np.concatenate([[start], start + np.cumsum(steps, axis=0)])
    mol = rigid_water(rng, n_frames)
    mol_com = (mol * WATER_MASSES[None, :, None]).sum(1) / WATER_MASSES.sum()   # (T, 3)
    return mol - mol_com[:, None] + com[:, None], com, steps


def wrap(xyz, box):
    """Wrap into [0, L) per atom, like a trajectory written with wrapped coordinates."""
    return xyz - box * np.floor(xyz / box)


def write_xtc(path, xyz, *, box, time=None, dt=10.0):
    """Write coordinates to an .xtc.

    box: (3,) box lengths, tiled over frames, or a full (T, 3, 3) box-vector array.
    time: (T,) time stamps; defaults to arange(T) * dt.
    """
    from mdtraj.formats import XTCTrajectoryFile
    xyz = np.asarray(xyz)
    T = xyz.shape[0]
    box = np.asarray(box)
    boxes = np.tile(np.diag(box), (T, 1, 1)) if box.shape == (3,) else box
    time = np.arange(T) * dt if time is None else np.asarray(time)
    with XTCTrajectoryFile(str(path), "w") as f:
        f.write(xyz.astype(np.float32), time=time.astype(np.float32),
                step=np.arange(T), box=boxes.astype(np.float32))