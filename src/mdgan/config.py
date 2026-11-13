"""One configuration object for a run: model, data, training, estimation.

Every default here is the baseline recorded in docs/decisions.md (paper values
where the paper gives them, our documented choices otherwise). The trainer, the
checkpoint and the dataset all read the same Config, and a checkpoint stores it,
so a run is fully described by one JSON-serializable object.

Not in Config, on purpose:
  - sigma: derived from the training windows (WindowDataset), recorded per run.
  - device: a property of where the run executes, not of the experiment.
  - paths: belong to the caller.

Pure Python (no torch import), so it can be loaded anywhere.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields

from mdgan._util import positive_int

PHASE_MODES = ("single", "all")
INIT_SCHEMES = ("pytorch", "chainer")    # see mdgan.models.init
GZ_NORM_POSITIONS = ("output", "hidden")  # see mdgan.models.latent_transition
CRITIC_INPUTS = ("increments", "positions", "positions_scaled")  # see mdgan.train.critic_view


@dataclass(frozen=True)
class Config:
    # --- model (2022 ESI Fig. S2; n_z from 2023) ------------------------------------
    latent_dim: int = 32            # n_z
    noise_dim: int = 32             # n_r, shared by G_z (r) and G_Y (r')
    block_len: int = 64             # M
    n_channels: int = 3             # C: 3 single-particle, 9 three-particle
    hidden_dim: int = 2048
    negative_slope: float = 0.2
    use_running_stats: bool = False  # G_z BatchNorm: batch statistics in train and eval
    init: str = "pytorch"           # weight initialisation scheme (mdgan.models.init)
    # Batch-1 reproduction probes on G_z (paper behaviour at these defaults; mdgan.models.latent_transition):
    gz_additive_noise_std: float = 0.0  # > 0: input z + N(0, s^2) instead of [z, r]
    gz_weight_decay: float = 0.0        # > 0: AdamW decoupled decay on G_z parameters only
    gz_weight_decay_on_norm: bool = True  # False: no decay on G_z's BatchNorm gamma/beta (biases still decayed)
    gz_dropout: float = 0.0             # > 0: dropout after G_z's three hidden activations (training)
    gz_norm_position: str = "output"    # "hidden": Affine(H->H) -> BatchNorm(H) -> Affine(H->n_z)
    latent_noise_share: float = 0.0     # a > 0: z' <- sqrt(1 - a^2) z' + a eps after the BatchNorm (plan step 8c)

    # --- data ----------------------------------------------------------------------------
    step_skip: int = 1              # n_s (PE 1, E-VDF 3, PVDF 5, PVAC/E-VAC 7, PAAM/E-AAM 20)
    phase_mode: str = "single"
    scale_inputs: bool = True       # divide by sigma; False is the unscaled ablation
    window_offset_max_steps: int = 0  # > 0: each training window starts at a fresh offset in [0, K] steps
    holdout_trajectories: tuple[int, ...] = ()  # trajectory indices never trained on (reference MSD still uses all)
    critic_input: str = "increments"  # "positions": the critic sees cumsum(window) over all 2M steps
    critic_position_scale: float | None = None  # "positions_scaled": cumsum(window) / this (the training set's RMS ratio)

    # --- training (WGAN-GP) ---------------------------------------------------------------
    batch_size: int = 64
    n_iterations: int = 300_000     # generator updates
    n_critic: int = 5               # critic updates per generator update
    lr: float = 1e-4
    betas: tuple[float, float] = (0.0, 0.9)
    gp_lambda: float = 10.0
    burn_in: int = 7                # c: latent transitions before the first decode
    pair_window: int = 4            # K: the decoded pair (z_j, z_j+1) has j in [c, c+K-1]
    mean_operation: bool = True     # 2023 eq. 9 on the generated window
    seed: int = 0

    # --- checkpoints and estimation ------------------------------------------------------
    checkpoint_every: int = 10_000
    full_checkpoint_iterations: tuple[int, ...] = ()  # keep a copy of the full state (critic, optimisers, RNG) here
    estimation_chains: int = 8192
    estimation_time_ps: float = 80_000.0
    err_window_ps: tuple[float, float] = (200.0, 78_000.0)
    diffusion_fit_ps: tuple[float, float] = (10_000.0, 20_000.0)

    def __post_init__(self):
        for name in ("latent_dim", "noise_dim", "block_len", "n_channels", "hidden_dim",
                     "step_skip", "batch_size", "n_iterations", "n_critic", "pair_window",
                     "checkpoint_every", "estimation_chains"):
            positive_int(name, getattr(self, name))
        if not isinstance(self.burn_in, int) or isinstance(self.burn_in, bool) or self.burn_in < 0:
            raise ValueError("burn_in must be a non-negative integer")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if self.phase_mode not in PHASE_MODES:
            raise ValueError(f"phase_mode must be one of {PHASE_MODES}")
        if self.init not in INIT_SCHEMES:
            raise ValueError(f"init must be one of {INIT_SCHEMES}")
        if int(self.window_offset_max_steps) != self.window_offset_max_steps or self.window_offset_max_steps < 0:
            raise ValueError("window_offset_max_steps must be a non-negative integer")
        if not self.gz_additive_noise_std >= 0.0:
            raise ValueError("gz_additive_noise_std must be >= 0")
        if not self.gz_weight_decay >= 0.0:
            raise ValueError("gz_weight_decay must be >= 0")
        if not isinstance(self.gz_weight_decay_on_norm, bool):
            raise ValueError("gz_weight_decay_on_norm must be true or false")
        if not 0.0 <= self.gz_dropout < 1.0:
            raise ValueError("gz_dropout must be in [0, 1)")
        if any(int(i) != i or i < 0 for i in self.holdout_trajectories) or \
                len(set(self.holdout_trajectories)) != len(self.holdout_trajectories):
            raise ValueError("holdout_trajectories must be distinct non-negative integers")
        if any(int(i) != i or i <= 0 or i % self.checkpoint_every for i in self.full_checkpoint_iterations):
            raise ValueError("full_checkpoint_iterations must be positive multiples of checkpoint_every")
        if self.critic_input not in CRITIC_INPUTS:
            raise ValueError(f"critic_input must be one of {CRITIC_INPUTS}")
        if self.critic_input == "positions_scaled":
            if not (isinstance(self.critic_position_scale, (int, float)) and self.critic_position_scale > 0):
                raise ValueError("critic_input 'positions_scaled' needs critic_position_scale > 0")
        elif self.critic_position_scale is not None:
            raise ValueError("critic_position_scale is only used with critic_input 'positions_scaled'")
        if not 0.0 <= self.latent_noise_share < 1.0:
            raise ValueError("latent_noise_share must be in [0, 1)")
        if self.gz_norm_position not in GZ_NORM_POSITIONS:
            raise ValueError(f"gz_norm_position must be one of {GZ_NORM_POSITIONS}")
        if not self.lr > 0:
            raise ValueError("lr must be positive")
        if len(self.betas) != 2 or not all(0.0 <= b < 1.0 for b in self.betas):
            raise ValueError("betas must be two values in [0, 1)")
        if self.gp_lambda < 0:
            raise ValueError("gp_lambda must be non-negative")
        if not self.negative_slope >= 0:
            raise ValueError("negative_slope must be non-negative")
        for name in ("err_window_ps", "diffusion_fit_ps"):
            lo, hi = getattr(self, name)
            if not 0 < lo < hi:
                raise ValueError(f"{name} must satisfy 0 < start < end")
        if self.estimation_time_ps <= 0:
            raise ValueError("estimation_time_ps must be positive")
        for name in ("err_window_ps", "diffusion_fit_ps"):
            if getattr(self, name)[1] > self.estimation_time_ps:
                raise ValueError(f"{name} ends after estimation_time_ps ({self.estimation_time_ps} ps)")

    # --- derived -------------------------------------------------------------------------

    @property
    def window_len(self) -> int:
        """2M: the critic's input length in coarse steps."""
        return 2 * self.block_len

    def estimation_blocks(self, dt_fine_ps: float) -> int:
        """Number of M-step blocks needed to cover estimation_time_ps at this n_s.

        PE: 80 ns / (10 ps * n_s=1) = 8000 coarse steps = 125 blocks of 64.
        Rounded up, so the rollout always reaches estimation_time_ps.
        """
        coarse_dt = dt_fine_ps * self.step_skip
        steps = self.estimation_time_ps / coarse_dt
        return int(-(-round(steps, 9) // self.block_len))     # ceil without float drift

    # --- serialization --------------------------------------------------------------------

    def to_dict(self) -> dict:
        """JSON-serializable (tuples become lists)."""
        return {k: list(v) if isinstance(v, tuple) else v for k, v in asdict(self).items()}

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        """Inverse of to_dict. Unknown keys are an error, so a stale or foreign
        checkpoint cannot be loaded with silently ignored settings."""
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown Config keys: {sorted(unknown)}")
        tuple_fields = {"betas", "err_window_ps", "diffusion_fit_ps", "full_checkpoint_iterations", "holdout_trajectories"}
        return cls(**{k: tuple(v) if k in tuple_fields else v for k, v in d.items()})