"""Training windows for MD-GAN from a system's displacement array.

WHAT THE MODEL IS TRAINED ON (2023 paper, Method)
-------------------------------------------------
The input MD data per trajectory is the first 2M * n_s * dt of it: 1.28 ns for
PE (M = 64, n_s = 1, dt = 10 ps). After Step Skip (keep every n_s-th frame,
i.e. sum every n_s consecutive fine displacements) that is exactly one window
of 2M coarse displacements per trajectory. For PE that gives 60 windows; the
training batch of 64 is drawn from them with replacement. The remaining ~79 ns
of each trajectory is the reference region used only by msd.py; this class is
built from a slice and never holds it.

STEP SKIP AND PHASES
--------------------
A coarse displacement is the sum of n_s consecutive fine ones:
    d_coarse[j] = d[p + j*n_s] + ... + d[p + (j+1)*n_s - 1]
where p is the phase (which fine step the coarse grid starts on). The paper's
input length 2M*n_s fits exactly one phase (p = 0). phase_mode="all" extends
the region by n_s - 1 fine steps so every p in 0..n_s-1 yields a window; those
windows are time-shifted copies of the same MD and are NOT independent
samples. For n_s = 1 the two modes are identical. Default: "single".

SCALING
-------
Windows are divided by one scalar sigma, the pooled std of the training
windows over trajectories, steps and axes (x, y, z share a scale, which keeps
the system isotropic). sigma is computed from the training windows only; the
reference region never enters. This normalization is OUR choice (not in the
paper): WGAN-GP's gradient penalty is in input units, and water steps here are
~0.17 nm. sigma is reported in `metadata` so the trainer stores it and the
estimator multiplies it back before any physical quantity is computed.

Units: windows in sigma units (dimensionless); dt_ps and sigma_nm in metadata.
"""

from __future__ import annotations

import numpy as np
import torch

from mdgan._util import positive_int
from mdgan.data.preprocess import SystemData

PHASE_MODES = ("single", "all")


def coarsen(displacements: np.ndarray, step_skip: int, phase: int = 0) -> np.ndarray:
    """Sum every `step_skip` consecutive fine displacements, starting at `phase`.

    displacements : (N, T, 3). Returns (N, T', 3) with T' = (T - phase) // step_skip.
    Equivalent to differencing the positions sampled every step_skip frames from
    frame `phase`. step_skip = 1, phase = 0 is the identity.
    """
    d = np.asarray(displacements)
    if d.ndim != 3:
        raise ValueError(f"displacements must be (N, T, 3); got {d.shape}")
    positive_int("step_skip", step_skip)
    if not isinstance(phase, int) or isinstance(phase, bool) or not 0 <= phase < step_skip:
        raise ValueError("phase must be an integer in [0, step_skip)")
    n_coarse = (d.shape[1] - phase) // step_skip
    if n_coarse < 1:
        raise ValueError("too few steps for one coarse displacement")
    block = d[:, phase: phase + n_coarse * step_skip]            # (N, n_coarse*step_skip, 3)
    return block.reshape(d.shape[0], n_coarse, step_skip, 3).sum(axis=2)


class WindowDataset:
    """The training set: scaled 2M-step coarse displacement windows and sigma."""

    def __init__(self, system: SystemData, block_len: int = 64, step_skip: int = 1,
                 phase_mode: str = "single"):
        positive_int("block_len", block_len)
        positive_int("step_skip", step_skip)
        if phase_mode not in PHASE_MODES:
            raise ValueError(f"phase_mode must be one of {PHASE_MODES}")

        self.system_name = system.name
        self.block_len = block_len                  # M
        self.step_skip = step_skip                  # n_s
        self.phase_mode = phase_mode
        self.window_len = 2 * block_len             # 2M coarse steps per window
        self.dt_ps = system.dt_ps * step_skip       # coarse sampling interval
        self.n_channels = system.displacements.shape[2]

        # --- 1. Training region: the first 2M*n_s fine steps (+ n_s-1 for "all") ---
        phases = range(step_skip) if phase_mode == "all" else range(1)
        self.region_steps = self.window_len * step_skip + (step_skip - 1 if phase_mode == "all" else 0)
        if self.region_steps > system.n_steps:
            raise ValueError(
                f"training region needs {self.region_steps} fine steps but trajectories have {system.n_steps}"
            )
        region = np.array(system.displacements[:, :self.region_steps], dtype=np.float64)   # copy

        # --- 2. One window per trajectory per phase ---
        windows = [coarsen(region, step_skip, p)[:, :self.window_len] for p in phases]   # each (N, 2M, C)
        windows = np.concatenate(windows, axis=0)                                         # (W, 2M, C)
        assert windows.shape[1] == self.window_len

        # --- 3. Scale by one scalar sigma computed from these windows only ---
        self.sigma_nm = float(windows.std())
        if not np.isfinite(self.sigma_nm) or self.sigma_nm <= 0:
            raise ValueError("training windows have zero or non-finite spread")
        self._windows = torch.from_numpy((windows / self.sigma_nm).astype(np.float32))    # (W, 2M, C)

        self.n_windows = int(self._windows.shape[0])
        self.n_trajectories = system.n_trajectories
        self.source_md5 = list(system.meta.get("md5", []))

    # --- access ----------------------------------------------------------------------------

    @property
    def windows(self) -> torch.Tensor:
        """All training windows, (W, 2M, C) float32 in sigma units. A copy; the set is immutable."""
        return self._windows.clone()

    def sample(self, batch: int, generator: torch.Generator | None = None,
               device=None) -> torch.Tensor:
        """`batch` windows drawn WITH replacement (W may be < batch): (B, 2M, C) float32."""
        positive_int("batch", batch)
        idx = torch.randint(0, self.n_windows, (batch,), generator=generator)
        out = self._windows[idx]
        return out.to(device) if device is not None else out

    @property
    def metadata(self) -> dict:
        """What a checkpoint must record to reproduce this training set."""
        return {
            "system": self.system_name,
            "block_len": self.block_len,
            "step_skip": self.step_skip,
            "phase_mode": self.phase_mode,
            "region_steps": self.region_steps,
            "window_len": self.window_len,
            "n_windows": self.n_windows,
            "n_trajectories": self.n_trajectories,
            "dt_ps": self.dt_ps,
            "sigma_nm": self.sigma_nm,
            "source_md5": self.source_md5,
        }

    def __len__(self) -> int:
        return self.n_windows

    def __repr__(self) -> str:
        return (f"WindowDataset({self.system_name}: {self.n_windows} windows x {self.window_len} steps, "
                f"n_s={self.step_skip}, dt={self.dt_ps} ps, sigma={self.sigma_nm:.4f} nm)")