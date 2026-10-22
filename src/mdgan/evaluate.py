"""From a trained model to the paper's numbers: err and D against MD.

    rollout (sigma units) -> trim to estimation_time_ps -> ensemble MSD -> x sigma^2
    -> err vs the MD reference MSD (2023 eqs. 6-7) and D from the 10-20 ns slope (eq. 8)

Grids: the reference is on the fine MD grid (dt, 10 ps); the generated curve is on
the coarse grid dt * n_s. err looks the reference up at the generated times.

Units: rollout returns displacements in sigma units. MSD is quadratic, so the
physical MSD is the sigma-unit MSD times sigma^2: one multiplication of the final
curve instead of a copy of the (8192, 8000, 3) displacement array.

The MD reference is computed once (reference_msd) and passed in, so evaluating
many checkpoints does not recompute it. Evaluation randomness goes through an
explicit generator created on the models' device from `seed`.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass

import numpy as np
import torch

from mdgan.config import Config
from mdgan.data.msd import diffusion_coefficient, err, msd_ensemble, msd_time_averaged
from mdgan.rollout import rollout


@dataclass
class EvaluationResult:
    iteration: int
    err_percent: float
    D_gen_m2s: float
    D_ref_m2s: float
    D_rel_error: float           # D_gen / D_ref - 1
    slope_gen_nm2_per_ns: float
    slope_ref_nm2_per_ns: float
    sigma_nm: float
    n_chains: int
    n_steps: int                 # generated coarse steps (= estimation_time_ps / dt_gen)
    dt_gen_ps: float
    msd_gen_nm2: np.ndarray      # (n_steps + 1,), physical units

    def to_dict(self) -> dict:
        """JSON-serializable summary (the MSD curve is left out; save it separately)."""
        d = asdict(self)
        d.pop("msd_gen_nm2")
        return d


def reference_msd(system) -> tuple[np.ndarray, float]:
    """MD reference: mean over trajectories of the all-origins time-averaged MSD.
    Returns (msd_nm2 on the fine grid, dt_ps)."""
    mean, _ = msd_time_averaged(system.displacements)
    return mean, float(system.dt_ps)


def generated_msd(gz, gy, config: Config, sigma_nm: float, dt_fine_ps: float, *,
                  n_chains: int | None = None,
                  generator: torch.Generator | None = None) -> tuple[np.ndarray, float, int]:
    """Ensemble MSD of generated trajectories in nm^2 on the coarse grid.
    Returns (msd (n_steps + 1,), dt_gen_ps, n_chains)."""
    n_chains = config.estimation_chains if n_chains is None else n_chains
    dt_gen = dt_fine_ps * config.step_skip
    n_steps = int(round(config.estimation_time_ps / dt_gen))
    n_blocks = config.estimation_blocks(dt_fine_ps)
    d = rollout(gz, gy, n_chains, n_blocks, config.burn_in,
                mean_operation=config.mean_operation, generator=generator)
    msd = msd_ensemble(d[:, :n_steps]) * sigma_nm ** 2
    return msd, dt_gen, n_chains


def evaluate(gz, gy, config: Config, sigma_nm: float, msd_ref: np.ndarray, dt_ref_ps: float, *,
             iteration: int = 0, n_chains: int | None = None,
             generator: torch.Generator | None = None) -> EvaluationResult:
    """err and D of the generated ensemble against a precomputed MD reference."""
    t1, t2 = config.err_window_ps
    f1, f2 = config.diffusion_fit_ps
    ref_end = (len(msd_ref) - 1) * dt_ref_ps
    if max(t2, f2) > ref_end + 1e-9:
        raise ValueError(f"reference MSD reaches {ref_end} ps; evaluation needs {max(t2, f2)} ps")
    msd_gen, dt_gen, n_chains = generated_msd(gz, gy, config, sigma_nm, dt_ref_ps,
                                              n_chains=n_chains, generator=generator)
    d_gen, s_gen = diffusion_coefficient(msd_gen, dt_gen, f1, f2)
    d_ref, s_ref = diffusion_coefficient(msd_ref, dt_ref_ps, f1, f2)
    return EvaluationResult(
        iteration=iteration,
        err_percent=err(msd_gen, dt_gen, msd_ref, dt_ref_ps, t1, t2),
        D_gen_m2s=d_gen, D_ref_m2s=d_ref, D_rel_error=d_gen / d_ref - 1.0,
        slope_gen_nm2_per_ns=s_gen, slope_ref_nm2_per_ns=s_ref,
        sigma_nm=float(sigma_nm), n_chains=n_chains, n_steps=len(msd_gen) - 1, dt_gen_ps=dt_gen,
        msd_gen_nm2=msd_gen,
    )


def evaluate_checkpoint(path, system, *, msd_ref: np.ndarray | None = None, n_chains: int | None = None,
                        seed: int | None = None, map_location="cpu") -> EvaluationResult:
    """Load a checkpoint and evaluate it against `system` (a preprocess.SystemData)."""
    from mdgan.checkpoint import load_checkpoint
    ck = load_checkpoint(path, map_location=map_location)
    if msd_ref is None:
        msd_ref, dt_ref = reference_msd(system)
    else:
        dt_ref = float(system.dt_ps)
    device = next(ck.gz.parameters()).device
    generator = torch.Generator(device=device).manual_seed(ck.config.seed if seed is None else seed)
    return evaluate(ck.gz, ck.gy, ck.config, ck.sigma_nm, msd_ref, dt_ref,
                    iteration=ck.iteration, n_chains=n_chains, generator=generator)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Evaluate an MD-GAN checkpoint against MD.")
    ap.add_argument("checkpoint")
    ap.add_argument("system_dir", help="MD_data/<system>")
    ap.add_argument("--cache", help="preprocess cache .npz (created if missing)")
    ap.add_argument("--chains", type=int, help="override estimation_chains")
    ap.add_argument("--seed", type=int)
    args = ap.parse_args(argv)
    from mdgan.data.preprocess import process_system
    system = process_system(args.system_dir, cache=args.cache)
    result = evaluate_checkpoint(args.checkpoint, system, n_chains=args.chains, seed=args.seed)
    print(json.dumps(result.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())