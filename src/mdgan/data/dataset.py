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

scale_inputs=False is the unscaled ablation: windows stay in nm and sigma_nm = 1.
window_std_nm (the raw spread) is recorded either way, so a resumed run can check it
is training on the same data.

LONGER CRITIC WINDOWS AND CORRELATION TARGETS (not in the papers)
----------------------------------------------------------------
critic_blocks = L > 2 makes sample() return windows of L*M steps for a critic over L blocks.
They are drawn from the same region of (2M + K) steps that window offsets define, so L*M may
not exceed it, and the region, sigma and the split are the same as for a 2-block run with the
same K. block_sum_correlations gives the correlations of M-step block sums in that region, the
target of the generator's correlation reward (mdgan.rewards); block_sum_variance gives their variance on
the same block grids, the target of the variance reward.

REGION START (not in the papers)
--------------------------------
region_start_steps = s > 0 moves the whole training region s coarse steps (s * n_s fine steps) into
every trajectory: the windows, the offsets, sigma, the position RMS ratio and the reward targets all come
from steps s .. s + 2M + K instead of 0 .. 2M + K. s = 0 is the papers' region and leaves everything,
metadata included, exactly as before the option existed.

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
                 phase_mode: str = "single", scale_inputs: bool = True, window_offset_max_steps: int = 0,
                 holdout_trajectories: tuple[int, ...] = (), critic_blocks: int = 2, region_start_steps: int = 0):
        positive_int("block_len", block_len)
        positive_int("step_skip", step_skip)
        if phase_mode not in PHASE_MODES:
            raise ValueError(f"phase_mode must be one of {PHASE_MODES}")
        if int(window_offset_max_steps) != window_offset_max_steps or window_offset_max_steps < 0:
            raise ValueError("window_offset_max_steps must be a non-negative integer")
        # Held-out trajectories (Config.holdout_trajectories): never enter training, sigma, the windows or the
        # offsets; the reference MSD (built by the caller from the full system) still uses every trajectory.
        holdout = tuple(int(i) for i in holdout_trajectories)
        if len(set(holdout)) != len(holdout) or any(i < 0 or i >= system.n_trajectories for i in holdout):
            raise ValueError(f"holdout_trajectories must be distinct indices in [0, {system.n_trajectories})")
        if len(holdout) >= system.n_trajectories:
            raise ValueError("holdout_trajectories leaves no trajectory to train on")
        self.holdout_trajectories = tuple(sorted(holdout))
        keep = [i for i in range(system.n_trajectories) if i not in set(holdout)]

        self.system_name = system.name
        self.block_len = block_len                  # M
        self.step_skip = step_skip                  # n_s
        self.phase_mode = phase_mode
        self.window_len = 2 * block_len             # 2M coarse steps per window
        self.dt_ps = system.dt_ps * step_skip       # coarse sampling interval
        self.n_channels = system.displacements.shape[2]

        # Window offsets (Config.window_offset_max_steps; 0 = the paper's fixed windows). With a
        # maximum K > 0, every draw starts its window at a fresh offset o ~ U{0..K} coarse steps into
        # the trajectory, from the training generator; the region grows to (2M + K) coarse steps.
        # sigma is still computed from the offset-0 windows only, so it does not depend on K.
        self.window_offset_max_steps = int(window_offset_max_steps)
        span = self.window_len + self.window_offset_max_steps        # coarse steps a draw may touch
        # Critic windows of L blocks (Config.critic_blocks; 2 = the papers'). A draw is then L*M steps starting at
        # an offset in [0, 2M + K - L*M], inside the same region; sigma and the offset-0 windows do not depend on L.
        positive_int("critic_blocks", critic_blocks)
        if critic_blocks < 2 or critic_blocks * block_len > span:
            raise ValueError(f"critic_blocks = {critic_blocks} needs 2 <= critic_blocks and "
                             f"window_offset_max_steps >= {(critic_blocks - 2) * block_len}")
        self.critic_blocks = critic_blocks
        self.sample_len = critic_blocks * block_len                  # steps in one sampled window (2M by default)

        # --- 1. Training region: (2M + K)*n_s fine steps (+ n_s-1 for "all") from region_start_steps*n_s on ---
        if isinstance(region_start_steps, bool) or int(region_start_steps) != region_start_steps or region_start_steps < 0:
            raise ValueError("region_start_steps must be a non-negative integer")
        self.region_start_steps = int(region_start_steps)
        first = self.region_start_steps * step_skip                              # in fine steps
        phases = range(step_skip) if phase_mode == "all" else range(1)
        self.region_steps = span * step_skip + (step_skip - 1 if phase_mode == "all" else 0)
        if first + self.region_steps > system.n_steps:
            raise ValueError(
                f"training region needs fine steps {first} .. {first + self.region_steps} but trajectories have "
                f"{system.n_steps}"
            )
        region = np.array(system.displacements[keep, first:first + self.region_steps], dtype=np.float64)   # a copy

        # --- 2. One window per trajectory per phase (the offset-0 windows) ---
        series = np.concatenate([coarsen(region, step_skip, p)[:, :span] for p in phases], axis=0)   # (W, 2M+K, C)
        windows = series[:, :self.window_len]                                                         # (W, 2M, C)
        assert windows.shape[1] == self.window_len

        # --- 3. Scale by one scalar sigma computed from these windows only ---
        self.window_std_nm = float(windows.std())
        if not np.isfinite(self.window_std_nm) or self.window_std_nm <= 0:
            raise ValueError("training windows have zero or non-finite spread")
        # RMS of the cumulative positions over RMS of the increments, pooled over windows, steps and axes
        # (scale-free, so sigma cancels). The constant the "positions_scaled" critic input divides by;
        # written to dataset.json and checked against Config.critic_position_scale, but not part of
        # metadata, which resume compares for equality with checkpoints that predate it.
        w64 = windows.astype(np.float64)
        self.position_rms_ratio = float(np.sqrt((np.cumsum(w64, axis=1) ** 2).mean()) / np.sqrt((w64 ** 2).mean()))
        # sigma is the scale actually applied: the spread, or 1 for the unscaled ablation.
        self.scale_inputs = bool(scale_inputs)
        self.sigma_nm = self.window_std_nm if self.scale_inputs else 1.0
        self._windows = torch.from_numpy((windows / self.sigma_nm).astype(np.float32))    # (W, 2M, C)
        self._series = (torch.from_numpy((series / self.sigma_nm).astype(np.float32))      # (W, 2M+K, C)
                        if self.window_offset_max_steps > 0 else None)

        self.n_windows = int(self._windows.shape[0])
        self.n_trajectories = len(keep)                      # trajectories trained on
        self.source_md5 = list(system.meta.get("md5", []))
        # a synthetic system's exact long-time D (mdgan.data.synthetic), carried to dataset.json; None for MD
        d_exact = system.meta.get("D_exact_m2s")
        self.analytic_D_m2s = float(d_exact) if d_exact is not None else None

    # --- access ----------------------------------------------------------------------------

    @property
    def windows(self) -> torch.Tensor:
        """All training windows, (W, 2M, C) float32 in sigma units. A copy; the set is immutable."""
        return self._windows.clone()

    def sample(self, batch: int, generator: torch.Generator | None = None,
               device=None) -> torch.Tensor:
        """`batch` windows drawn WITH replacement (W may be < batch): (B, 2M, C) float32.
        With window offsets, each window then starts at its own offset ~ U{0..K}, drawn after the
        indices from the same generator; with K = 0 nothing extra is drawn (the paper's sampling).
        With critic_blocks = L the windows are (B, L*M, C) and the offsets ~ U{0..K - (L - 2)M}."""
        positive_int("batch", batch)
        idx = torch.randint(0, self.n_windows, (batch,), generator=generator)
        if self._series is None:
            out = self._windows[idx]
        else:
            last = self.window_len + self.window_offset_max_steps - self.sample_len     # K at critic_blocks = 2
            off = torch.randint(0, last + 1, (batch,), generator=generator)
            pos = off[:, None] + torch.arange(self.sample_len)[None, :]                     # (B, L*M)
            out = self._series[idx[:, None], pos]
        return out.to(device) if device is not None else out

    def _block_sum_grids(self, n_blocks: int):
        """The (W, nb, C) M-step block sums of the training region for each block grid o = 0 .. M-1 that still
        holds n_blocks blocks: the grids block_sum_correlations and block_sum_variance pool over."""
        positive_int("n_blocks", n_blocks)
        series = (self._windows if self._series is None else self._series).numpy().astype(np.float64)
        w, span, c = series.shape
        if n_blocks < 2 or n_blocks * self.block_len > span:
            raise ValueError(f"block-sum statistics over {n_blocks} blocks need 2 <= n_blocks and "
                             f"window_offset_max_steps >= {(n_blocks - 2) * self.block_len}")
        for o in range(self.block_len):
            nb = (span - o) // self.block_len
            if nb < n_blocks:                                   # this grid position no longer holds n_blocks blocks
                break
            yield series[:, o:o + nb * self.block_len].reshape(w, nb, self.block_len, c).sum(axis=2)

    def block_sum_correlations(self, n_blocks: int) -> np.ndarray:
        """(n_blocks - 1,) correlations of the training region's M-step block sums at lags 1 .. n_blocks - 1.

        Pooled over trajectories, blocks and axes (means removed per lag), for the block grid started at each of
        the M positions o = 0 .. M-1 of the region, then averaged over o. Scale-free. Training trajectories only,
        the same (2M + K) steps the windows come from; needs n_blocks * M of them."""
        rho = []
        for sums in self._block_sum_grids(n_blocks):
            row = []
            for k in range(1, n_blocks):
                x, y = sums[:, :-k].ravel(), sums[:, k:].ravel()
                x, y = x - x.mean(), y - y.mean()
                row.append((x * y).mean() / np.sqrt((x * x).mean() * (y * y).mean()))
            rho.append(row)
        return np.mean(rho, axis=0)

    def block_sum_variance(self, n_blocks: int) -> float:
        """Variance of the training region's M-step block sums, in the units the model sees (sigma units, or nm
        without scaling): pooled over trajectories, blocks and axes (one mean removed), on the same block grids
        and trajectories as block_sum_correlations(n_blocks), then averaged over the grids. The target of the
        generator's variance reward (mdgan.rewards, Config.variance_reward_weight)."""
        return float(np.mean([((x - x.mean()) ** 2).mean() for x in (s.ravel() for s in self._block_sum_grids(n_blocks))]))

    def block_sum_count(self, n_blocks: int) -> int:
        """How many M-step block sums (trajectories x blocks x axes, over every grid used) block_sum_variance pools."""
        return int(sum(x.size for x in self._block_sum_grids(n_blocks)))

    def block_pair_counts(self, n_blocks: int) -> list[int]:
        """Pairs behind each lag k = 1 .. n_blocks - 1 of block_sum_correlations: trajectories x (blocks - k) x axes,
        summed over the block grids it averages."""
        grids = [s.shape for s in self._block_sum_grids(n_blocks)]
        return [int(sum(w * (nb - k) * c for w, nb, c in grids)) for k in range(1, n_blocks)]

    def _subblock_sums(self, steps: int) -> np.ndarray:
        """(W, nb, C): each training trajectory's region cut into consecutive `steps`-step sub-blocks from its first
        step (one grid), each summed; a remainder shorter than `steps` is dropped."""
        positive_int("steps", steps)
        series = (self._windows if self._series is None else self._series).numpy().astype(np.float64)
        w, span, c = series.shape
        nb = span // steps
        if nb < 2:
            raise ValueError(f"the region ({span} steps) holds fewer than two {steps}-step sub-blocks")
        return series[:, :nb * steps].reshape(w, nb, steps, c).sum(axis=2)

    def subblock_sum_correlations(self, steps: int, lags: int) -> np.ndarray:
        """(lags,) correlations of `steps`-step sub-block sums k = 1 .. lags apart: pairs taken within each
        trajectory's region, pooled over all pairs, trajectories and axes, each side's mean removed over the pool.
        The target of the sub-block correlation reward (Config.subcorr_block_steps)."""
        sums = self._subblock_sums(steps)
        if not 1 <= lags < sums.shape[1]:
            raise ValueError(f"lags must be in [1, {sums.shape[1] - 1}] for {sums.shape[1]} sub-blocks")
        rho = []
        for k in range(1, lags + 1):
            x, y = sums[:, :-k].ravel(), sums[:, k:].ravel()
            x, y = x - x.mean(), y - y.mean()
            rho.append((x * y).mean() / np.sqrt((x * x).mean() * (y * y).mean()))
        return np.array(rho)

    def subblock_pair_counts(self, steps: int, lags: int) -> list[int]:
        """Pairs behind each lag of subblock_sum_correlations: trajectories x (sub-blocks - k) x axes."""
        w, nb, c = self._subblock_sums(steps).shape
        return [w * (nb - k) * c for k in range(1, lags + 1)]

    def subblock_sum_variance(self, steps: int) -> tuple[float, int]:
        """(variance, count) of the `steps`-step sub-block sums, pooled over trajectories, sub-blocks and axes with
        one mean removed: the variance reward's target when the sub-block reward is on."""
        x = self._subblock_sums(steps).ravel()
        return float(((x - x.mean()) ** 2).mean()), int(x.size)

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
            "scale_inputs": self.scale_inputs,
            "window_std_nm": self.window_std_nm,
            "sigma_nm": self.sigma_nm,
            "source_md5": self.source_md5,
            # present only when > 0: runs without offsets keep their metadata exactly (resume compares
            # it for equality), and a run with offsets cannot be resumed with a different maximum
            **({"window_offset_max_steps": self.window_offset_max_steps} if self.window_offset_max_steps else {}),
            # likewise present only when non-empty: the split is in dataset.json and checked on resume
            **({"holdout_trajectories": list(self.holdout_trajectories)} if self.holdout_trajectories else {}),
            **({"critic_blocks": self.critic_blocks} if self.critic_blocks != 2 else {}),
            # likewise only when > 0: where the region starts is checked on resume, and s = 0 leaves metadata unchanged
            **({"region_start_steps": self.region_start_steps} if self.region_start_steps else {}),
        }

    def __len__(self) -> int:
        return self.n_windows

    def __repr__(self) -> str:
        return (f"WindowDataset({self.system_name}: {self.n_windows} windows x {self.window_len} steps, "
                f"n_s={self.step_skip}, dt={self.dt_ps} ps, sigma={self.sigma_nm:.4f} nm)")