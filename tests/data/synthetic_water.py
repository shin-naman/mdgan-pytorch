"""Synthetic rigid-water trajectories with known unwrapped ground truth.

Shared by the water tests. Not a test module (no test_ prefix).
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