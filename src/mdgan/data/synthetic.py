"""Synthetic diffusion processes with known answers, for testing the model where the truth is exact.

Cage walk (the PE-like control, diagnostic plan step 2a)
    x(t) = c(t) + y(t), in 3D, one step = dt_ps.
    y: Ornstein-Uhlenbeck rattling in a harmonic cage, relaxation time tau_steps, stationary rms
       cage_rms_nm per axis (exact discretisation, no time-step error).
    c: the cage centre; it jumps by a fixed length hop_length_nm in a uniformly random direction,
       at Poisson rate hop_rate_per_step. The particle jumps with its cage, so a hop is one large
       step (h / sigma_step ~ 7 at the default parameters, like PE's cavity hops).
    c and y are independent, so the mean squared displacement is exact at every lag:
        MSD(t) = 6 s^2 (1 - exp(-t / tau)) + kappa h^2 t          (t in steps, nm^2)
    and the long-time diffusion coefficient is exactly D = kappa h^2 / 6 (the cage term is bounded).
    Defaults: subdiffusive inside a 16-step (2-block, M = 8) window (local exponent ~0.6 at 8 steps,
    ~0.73 at 16), normal (exponent >= 0.9) from ~55 steps (block ~7), step rms ~0.155 nm.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from mdgan.data.preprocess import SystemData

NM2_PER_PS_TO_M2_PER_S = 1e-6


@dataclass(frozen=True)
class CageWalk:
    dt_ps: float = 10.0
    tau_steps: float = 2.0
    cage_rms_nm: float = 0.15
    hop_length_nm: float = 1.1
    hop_rate_per_step: float = 6 * 0.15 ** 2 / (1.1 ** 2 * 6.0)        # crossover scale t* = 6 steps

    def __post_init__(self):
        for name in ("dt_ps", "tau_steps", "cage_rms_nm", "hop_length_nm", "hop_rate_per_step"):
            if not getattr(self, name) > 0:
                raise ValueError(f"{name} must be > 0")

    # --- exact answers -------------------------------------------------------------------------
    def msd(self, t_steps) -> np.ndarray:
        """Exact MSD (nm^2) at lag t (steps; array or scalar)."""
        t = np.asarray(t_steps, dtype=np.float64)
        s, h, k, tau = self.cage_rms_nm, self.hop_length_nm, self.hop_rate_per_step, self.tau_steps
        return 6 * s ** 2 * (1 - np.exp(-t / tau)) + k * h ** 2 * t

    def diffusion_coefficient_m2s(self) -> float:
        """Exact long-time D = kappa h^2 / 6, in m^2/s."""
        return self.hop_rate_per_step * self.hop_length_nm ** 2 / 6 / self.dt_ps * NM2_PER_PS_TO_M2_PER_S

    def local_exponent(self, t_steps, factor: float = 1.05) -> np.ndarray:
        """d ln MSD / d ln t of the exact MSD."""
        t = np.asarray(t_steps, dtype=np.float64)
        return np.log(self.msd(t * factor) / self.msd(t / factor)) / np.log(factor ** 2)

    # --- simulation ----------------------------------------------------------------------------
    def simulate(self, n_traj: int, n_steps: int, seed: int = 0, name: str = "cage_walk") -> SystemData:
        """n_traj independent trajectories of n_steps displacements, each starting from the stationary
        state (y ~ N(0, s^2) per axis). Returns a SystemData (float32 displacements in nm)."""
        if n_traj < 1 or n_steps < 1:
            raise ValueError("n_traj and n_steps must be >= 1")
        rng = np.random.default_rng(seed)
        s, h, k = self.cage_rms_nm, self.hop_length_nm, self.hop_rate_per_step
        decay = np.exp(-1.0 / self.tau_steps)
        kick = s * np.sqrt(1 - decay ** 2)
        y = rng.normal(scale=s, size=(n_traj, 3))
        d = np.empty((n_traj, n_steps, 3), dtype=np.float64)
        for t in range(n_steps):
            y_new = y * decay + kick * rng.normal(size=(n_traj, 3))
            n_hops = rng.poisson(k, size=n_traj)
            jump = np.zeros((n_traj, 3))
            one = np.nonzero(n_hops == 1)[0]                        # the common case, vectorised
            if one.size:
                u = rng.normal(size=(one.size, 3))
                jump[one] = h * u / np.linalg.norm(u, axis=1, keepdims=True)
            for i in np.nonzero(n_hops > 1)[0]:                      # rare: several hops in one step
                u = rng.normal(size=(n_hops[i], 3))
                jump[i] = (h * u / np.linalg.norm(u, axis=1, keepdims=True)).sum(axis=0)
            d[:, t] = (y_new - y) + jump
            y = y_new
        meta = {"process": "cage_walk", **asdict(self), "seed": seed,
                "D_exact_m2s": self.diffusion_coefficient_m2s()}
        return SystemData(name=name, displacements=d.astype(np.float32), dt_ps=float(self.dt_ps),
                          box_nm=np.full((n_traj, 3), np.inf), meta=meta)
