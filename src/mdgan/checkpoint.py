"""Saving and resuming MD-GAN runs.

One file per checkpoint holding everything needed to resume or evaluate:

    format_version   int, bumped on any incompatible change
    config           Config.to_dict(): the models are REBUILT from this on load
    models           {"G_z" | "G_Y" | "D_G": {"class": <class name>, "state_dict": ...}}
    optimizers       {name: optimizer.state_dict()}       (optional; for resuming)
    iteration        generator updates completed
    sigma_nm         the scale the training windows were divided by (1.0 if unscaled)
    rng              {"torch": global CPU state, "generator": explicit generator state | None}
    dataset          WindowDataset.metadata (optional; provenance)
    extra            anything JSON-like (optional; e.g. last losses)

Loading is STRICT, by design (decisions.md "Checkpoint format and loading"):
  - the models are built from the stored Config, never from the caller's;
  - each stored class name must match the module it is loaded into, checked
    before any tensor is touched;
  - load_state_dict(strict=True) only. There is no non-strict path. The three
    modules share key names (input_layer.*, hidden1.*, output_layer.*), so a
    non-strict cross-load would silently copy same-shaped tensors.
Files are read with torch.load(weights_only=True): no arbitrary unpickling.
Writes are atomic: a temporary file is renamed over the target, so a crash or a
failed save never leaves a half-written checkpoint in place of a good one.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import torch

from mdgan.config import Config
from mdgan.models import Critic, LatentTransitionGenerator, TrajectoryGenerator, build_models

FORMAT_VERSION = 1
MODULE_KEYS = ("G_z", "G_Y", "D_G")
_EXPECTED_CLASS = {"G_z": LatentTransitionGenerator, "G_Y": TrajectoryGenerator, "D_G": Critic}


@dataclass
class Checkpoint:
    config: Config
    gz: LatentTransitionGenerator
    gy: TrajectoryGenerator
    dg: Critic
    iteration: int
    sigma_nm: float
    optimizer_states: dict = field(default_factory=dict)
    generator_state: torch.Tensor | None = None
    torch_rng_state: torch.Tensor | None = None
    dataset: dict | None = None
    extra: dict = field(default_factory=dict)

    @property
    def models(self):
        return self.gz, self.gy, self.dg


def _check_models_match_config(config: Config, models) -> None:
    gz, gy, dg = models
    for key, module in zip(MODULE_KEYS, models, strict=True):
        if type(module) is not _EXPECTED_CLASS[key]:
            raise ValueError(f"{key} must be a {_EXPECTED_CLASS[key].__name__}; got {type(module).__name__}")
    checks = {
        "latent_dim": (gz.latent_dim, gy.latent_dim), "noise_dim": (gz.noise_dim, gy.noise_dim),
        "hidden_dim": (gz.hidden_dim, gy.hidden_dim, dg.hidden_dim),
        "block_len": (gy.block_len, dg.block_len), "n_channels": (gy.n_channels, dg.n_channels),
    }
    for name, values in checks.items():
        if any(v != getattr(config, name) for v in values):
            raise ValueError(f"models do not match the config: {name} = {values}, config has {getattr(config, name)}")


def save_checkpoint(path, *, config: Config, models, iteration: int, sigma_nm: float,
                    optimizers: dict | None = None, generator: torch.Generator | None = None,
                    dataset: dict | None = None, extra: dict | None = None) -> None:
    """Write a checkpoint atomically. `models` is (G_z, G_Y, D_G) and must match `config`."""
    _check_models_match_config(config, models)
    if not isinstance(iteration, int) or isinstance(iteration, bool) or iteration < 0:
        raise ValueError("iteration must be a non-negative integer")
    if not sigma_nm > 0:
        raise ValueError("sigma_nm must be positive")
    payload = {
        "format_version": FORMAT_VERSION,
        "config": config.to_dict(),
        "models": {key: {"class": type(m).__name__, "state_dict": m.state_dict()}
                   for key, m in zip(MODULE_KEYS, models, strict=True)},
        "optimizers": {name: opt.state_dict() for name, opt in (optimizers or {}).items()},
        "iteration": iteration,
        "sigma_nm": float(sigma_nm),
        "rng": {"torch": torch.get_rng_state(),
                "generator": generator.get_state() if generator is not None else None},
        "dataset": dataset,
        "extra": extra or {},
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        torch.save(payload, tmp)
        os.replace(tmp, path)                 # atomic on POSIX and Windows
    finally:
        if tmp.exists():
            tmp.unlink()


def load_checkpoint(path, map_location="cpu") -> Checkpoint:
    """Rebuild the models from the stored Config and load them strictly."""
    payload = torch.load(Path(path), map_location=map_location, weights_only=True)
    version = payload.get("format_version")
    if version != FORMAT_VERSION:
        raise ValueError(f"checkpoint format_version {version}; this code reads {FORMAT_VERSION}")
    config = Config.from_dict(payload["config"])
    models = build_models(config)
    stored = payload["models"]
    if set(stored) != set(MODULE_KEYS):
        raise ValueError(f"checkpoint modules {sorted(stored)}; expected {list(MODULE_KEYS)}")
    for key, module in zip(MODULE_KEYS, models, strict=True):
        if stored[key]["class"] != type(module).__name__:
            raise ValueError(f"checkpoint entry {key} holds a {stored[key]['class']}; "
                             f"expected {type(module).__name__}")
        module.load_state_dict(stored[key]["state_dict"], strict=True)
    gz, gy, dg = models
    rng = payload.get("rng", {})
    return Checkpoint(config=config, gz=gz, gy=gy, dg=dg,
                      iteration=payload["iteration"], sigma_nm=payload["sigma_nm"],
                      optimizer_states=payload.get("optimizers", {}),
                      generator_state=rng.get("generator"), torch_rng_state=rng.get("torch"),
                      dataset=payload.get("dataset"), extra=payload.get("extra", {}))