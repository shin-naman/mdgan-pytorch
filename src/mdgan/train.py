"""Training MD-GAN (WGAN-GP; decisions.md "Training design").

One ITERATION = one generator update:
  1. n_critic critic steps. Each: 64 real windows, one fake window built under
     no_grad (critic_loss detaches it anyway, so its graph would be wasted),
     critic_loss, Adam step on D_G.
  2. one generator step: a fresh fake window WITH graph, generator_loss through a
     frozen critic (no parameter gradients for D_G), Adam step on G_z + G_Y.
  3. every checkpoint_every iterations (and at the end): latest.pt (full resume
     state, overwritten), snap_XXXXXX.pt (generators only), and, if an MD
     reference was given, an evaluation appended to evaluations.jsonl.

Randomness: ONE CPU generator (seeded from config.seed) drives data sampling and
model noise; noise is moved to the device (rollout._draw_device), so the stream is
the same on CPU and MPS. Evaluation uses its own generator with a fixed seed at
every checkpoint, so turning evaluation on or off does not change training, and
every checkpoint is evaluated on identical noise. Bit-exact reproducibility is a
CPU guarantee; GPU/MPS kernels are not deterministic.

Divergence: after each iteration both losses are checked once for finiteness (one
device sync). On a non-finite loss the run rolls back to latest.pt and re-seeds the
generator deterministically (seed + 1000 * number of rollbacks so far); the history
is stored in the checkpoint and in events.jsonl. More than 3 consecutive rollbacks
to the same checkpoint stop the run with DivergenceError. latest.pt is written at
iteration 0, so there is always something to roll back to.

Run directory:
    config.json, dataset.json, provenance.json   written at start
    latest.pt, snap_XXXXXX.pt                     checkpoints
    metrics.jsonl                                 running means every log_every iterations
    evaluations.jsonl, msd_XXXXXX.npy             if a reference is given
    events.jsonl                                  start / resume / rollback / checkpoint
On resume or rollback, the .jsonl logs are truncated to the restored iteration, so
iteration numbers in them are strictly increasing.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import platform
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

from mdgan.checkpoint import load_checkpoint, save_checkpoint, save_snapshot
from mdgan.config import Config
from mdgan.data.dataset import WindowDataset
from mdgan.evaluate import evaluate
from mdgan.losses import critic_loss, generator_loss
from mdgan.models import build_models
from mdgan.rollout import make_fake_window

LATEST = "latest.pt"
MAX_CONSECUTIVE_ROLLBACKS = 3
ROLLBACK_SEED_STRIDE = 1000
METRIC_NAMES = ("wasserstein", "gradient_penalty", "grad_norm", "generator_loss", "fake_step_std")


class DivergenceError(RuntimeError):
    """Training produced non-finite losses after repeated rollbacks to the same checkpoint."""


def snapshot_name(iteration: int) -> str:
    return f"snap_{iteration:06d}.pt"


@contextmanager
def _frozen(module):
    """No parameter gradients for `module` inside the block; restored even on error."""
    flags = [p.requires_grad for p in module.parameters()]
    try:
        for p in module.parameters():
            p.requires_grad_(False)
        yield
    finally:
        for p, f in zip(module.parameters(), flags, strict=True):
            p.requires_grad_(f)


def _append_jsonl(path: Path, record: dict) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def _truncate_jsonl(path: Path, iteration: int) -> None:
    """Keep only records with record["iteration"] <= iteration (and drop a torn last line)."""
    if not path.exists():
        return
    kept = []
    for line in path.read_text().splitlines():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("iteration", -1) <= iteration:
            kept.append(line)
    path.write_text("".join(k + "\n" for k in kept))


def _git_state() -> dict:
    root = Path(__file__).resolve().parents[2]
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True,
                                timeout=10, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True,
                                    timeout=10, check=True).stdout.strip())
        return {"git_commit": commit, "git_dirty": dirty}
    except (OSError, subprocess.SubprocessError):
        return {"git_commit": None, "git_dirty": None}


def _check_dataset_matches_config(config: Config, dataset: WindowDataset) -> None:
    pairs = {"block_len": dataset.block_len, "step_skip": dataset.step_skip, "phase_mode": dataset.phase_mode,
             "scale_inputs": dataset.scale_inputs, "n_channels": dataset.n_channels}
    bad = {k: (v, getattr(config, k)) for k, v in pairs.items() if v != getattr(config, k)}
    if bad:
        raise ValueError(f"dataset does not match the config (dataset, config): {bad}")


class Trainer:
    """Owns models, optimizers, the generator and the run directory.

    New run:  Trainer(config, dataset, out_dir)          refuses an existing non-empty out_dir
    Resume:   Trainer(config, dataset, out_dir, resume=True)
              config must equal the checkpoint's except n_iterations (train longer);
              dataset.metadata must equal the checkpoint's (same files, same windows, same sigma).
    reference: (msd_ref, dt_ref_ps) from evaluate.reference_msd, or None to skip evaluation.
    """

    def __init__(self, config: Config, dataset: WindowDataset, out_dir, *, device="cpu",
                 reference: tuple[np.ndarray, float] | None = None, log_every: int = 100,
                 resume: bool = False):
        _check_dataset_matches_config(config, dataset)
        self.config, self.dataset, self.reference = config, dataset, reference
        self.out = Path(out_dir)
        self.device = torch.device(device)
        self.log_every = log_every
        self.gen = torch.Generator()
        self.rollbacks: list[dict] = []
        self._rollbacks_since_checkpoint = 0

        torch.manual_seed(config.seed)
        self.gz, self.gy, self.dg = (m.to(self.device) for m in build_models(config))
        self.opt_g = torch.optim.Adam([*self.gz.parameters(), *self.gy.parameters()], lr=config.lr, betas=config.betas)
        self.opt_d = torch.optim.Adam(self.dg.parameters(), lr=config.lr, betas=config.betas)

        if resume:
            self._resume()
        else:
            self._start()
        self._reset_metrics()

    # --- run directory -----------------------------------------------------------------

    def _start(self) -> None:
        if self.out.exists() and any(self.out.iterdir()):
            raise FileExistsError(f"{self.out} exists and is not empty; pass resume=True to continue it")
        self.out.mkdir(parents=True, exist_ok=True)
        self.gen.manual_seed(self.config.seed)
        self.iteration = 0
        (self.out / "config.json").write_text(json.dumps(self.config.to_dict(), indent=2))
        (self.out / "dataset.json").write_text(json.dumps(self.dataset.metadata, indent=2))
        (self.out / "provenance.json").write_text(json.dumps({
            **_git_state(), "torch": torch.__version__, "python": platform.python_version(),
            "device": str(self.device), "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }, indent=2))
        self._event("start")
        self._save_latest()                                   # something to roll back to

    def _resume(self) -> None:
        path = self.out / LATEST
        if not path.exists():
            raise FileNotFoundError(f"cannot resume: {path} does not exist")
        ck = load_checkpoint(path, map_location="cpu")                # check first, then copy
        if ck.kind != "full":
            raise ValueError(f"{path} is a {ck.kind!r} checkpoint; resuming needs a full one")
        if dataclasses.replace(ck.config, n_iterations=self.config.n_iterations) != self.config:
            diff = {f.name: (getattr(ck.config, f.name), getattr(self.config, f.name))
                    for f in dataclasses.fields(Config) if getattr(ck.config, f.name) != getattr(self.config, f.name)}
            raise ValueError(f"config differs from the checkpoint (only n_iterations may change): {diff}")
        if ck.dataset != self.dataset.metadata:
            raise ValueError("dataset differs from the one this run was trained on (metadata mismatch)")
        self._copy_from(ck)
        torch.set_rng_state(ck.torch_rng_state)
        self.rollbacks = list(ck.extra.get("rollbacks", []))
        self.iteration = ck.iteration
        for name in ("metrics.jsonl", "evaluations.jsonl", "events.jsonl"):
            _truncate_jsonl(self.out / name, self.iteration)
        self._event("resume")

    def _copy_from(self, ck) -> None:
        """Copy a full checkpoint into the existing modules, optimizers and generator (strict)."""
        for mine, theirs in zip((self.gz, self.gy, self.dg), ck.models, strict=True):
            mine.load_state_dict(theirs.state_dict(), strict=True)
        self.opt_g.load_state_dict(ck.optimizer_states["generator"])
        self.opt_d.load_state_dict(ck.optimizer_states["critic"])
        self.gen.set_state(ck.generator_state)

    def _event(self, kind: str, **fields) -> None:
        _append_jsonl(self.out / "events.jsonl", {"iteration": self.iteration, "event": kind,
                                                  "time": time.time(), **fields})

    # --- one iteration -------------------------------------------------------------------

    def _fake(self):
        c = self.config
        return make_fake_window(self.gz, self.gy, c.batch_size, c.burn_in, c.pair_window,
                                mean_operation=c.mean_operation, generator=self.gen)

    def _critic_step(self) -> dict:
        real = self.dataset.sample(self.config.batch_size, generator=self.gen, device=self.device)
        with torch.no_grad():
            fake = self._fake()
        loss, info = critic_loss(self.dg, real, fake, self.config.gp_lambda, generator=self.gen)
        self.opt_d.zero_grad(set_to_none=True)
        loss.backward()
        self.opt_d.step()
        return {"loss": loss.detach(), **info}

    def _generator_step(self) -> dict:
        with _frozen(self.dg):
            fake = self._fake()
            loss = generator_loss(self.dg, fake)
            self.opt_g.zero_grad(set_to_none=True)
            loss.backward()
            self.opt_g.step()
        return {"loss": loss.detach(), "fake_step_std": fake.detach().std(dim=0).mean()}

    def _iteration(self) -> bool:
        """One generator update. Returns False if a loss was non-finite."""
        for m in (self.gz, self.gy, self.dg):
            m.train()
        crit = [self._critic_step() for _ in range(self.config.n_critic)]
        gen = self._generator_step()
        if not bool(torch.isfinite(crit[-1]["loss"] + gen["loss"])):      # the one sync per iteration
            return False
        mean = lambda key: torch.stack([c[key] for c in crit]).mean()     # noqa: E731
        self._acc += torch.stack([mean("wasserstein"), mean("gradient_penalty"), mean("grad_norm"),
                                  gen["loss"], gen["fake_step_std"]]).to(self._acc)
        self._acc_n += 1
        return True

    # --- metrics, checkpoints, rollback ------------------------------------------------------

    def _reset_metrics(self) -> None:
        self._acc = torch.zeros(len(METRIC_NAMES), device=self.device)       # float32: MPS has no float64
        self._acc_n = 0
        self._t0 = time.perf_counter()

    def _write_metrics(self) -> None:
        if self._acc_n == 0:
            return
        values = (self._acc / self._acc_n).tolist()
        dt = time.perf_counter() - self._t0
        _append_jsonl(self.out / "metrics.jsonl", {
            "iteration": self.iteration, **dict(zip(METRIC_NAMES, values, strict=True)),
            "iterations_per_second": self._acc_n / dt if dt > 0 else None, "rollbacks": len(self.rollbacks)})
        self._reset_metrics()

    def _save_latest(self) -> None:
        save_checkpoint(self.out / LATEST, config=self.config, models=(self.gz, self.gy, self.dg),
                        iteration=self.iteration, sigma_nm=self.dataset.sigma_nm,
                        optimizers={"generator": self.opt_g, "critic": self.opt_d}, generator=self.gen,
                        dataset=self.dataset.metadata, extra={"rollbacks": self.rollbacks})

    def _checkpoint(self) -> None:
        self._save_latest()
        save_snapshot(self.out / snapshot_name(self.iteration), config=self.config, gz=self.gz, gy=self.gy,
                      iteration=self.iteration, sigma_nm=self.dataset.sigma_nm, dataset=self.dataset.metadata)
        self._rollbacks_since_checkpoint = 0
        self._event("checkpoint")
        if self.reference is not None:
            msd_ref, dt_ref = self.reference
            result = evaluate(self.gz, self.gy, self.config, self.dataset.sigma_nm, msd_ref, dt_ref,
                              iteration=self.iteration,
                              generator=torch.Generator().manual_seed(self.config.seed))   # never self.gen
            _append_jsonl(self.out / "evaluations.jsonl", result.to_dict())
            np.save(self.out / f"msd_{self.iteration:06d}.npy", result.msd_gen_nm2)

    def _rollback(self) -> None:
        self._rollbacks_since_checkpoint += 1
        if self._rollbacks_since_checkpoint > MAX_CONSECUTIVE_ROLLBACKS:
            raise DivergenceError(f"non-finite losses after {MAX_CONSECUTIVE_ROLLBACKS} consecutive rollbacks "
                                  f"to the checkpoint at iteration {self.iteration}")
        failed_at = self.iteration + 1
        ck = load_checkpoint(self.out / LATEST, map_location="cpu")
        self._copy_from(ck)
        self.iteration = ck.iteration
        seed = self.config.seed + ROLLBACK_SEED_STRIDE * (len(self.rollbacks) + 1)
        self.gen.manual_seed(seed)
        record = {"failed_at_iteration": failed_at, "to_iteration": self.iteration, "new_seed": seed}
        self.rollbacks.append(record)
        for name in ("metrics.jsonl", "evaluations.jsonl"):
            _truncate_jsonl(self.out / name, self.iteration)
        self._event("rollback", **record)
        self._reset_metrics()

    # --- driver --------------------------------------------------------------------------------

    def run(self) -> None:
        c = self.config
        while self.iteration < c.n_iterations:
            if not self._iteration():
                self._rollback()
                continue
            self.iteration += 1
            if self.iteration % self.log_every == 0:
                self._write_metrics()
            if self.iteration % c.checkpoint_every == 0 or self.iteration == c.n_iterations:
                self._write_metrics()
                self._checkpoint()


# --- command line ------------------------------------------------------------------------------

def _parse_value(text: str):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def build_config(config_file: str | None, overrides: list[str]) -> Config:
    """Config() <- JSON file <- key=value overrides, validated by Config.from_dict."""
    d = Config().to_dict()
    if config_file:
        d.update(json.loads(Path(config_file).read_text()))
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"--set expects key=value, got {item!r}")
        key, value = item.split("=", 1)
        d[key.strip()] = _parse_value(value.strip())
    return Config.from_dict(d)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Train MD-GAN on one polymer system.")
    ap.add_argument("system_dir", help="MD_data/<system>")
    ap.add_argument("--out", required=True, help="run directory (must not exist unless --resume)")
    ap.add_argument("--cache", help="preprocess cache .npz (created if missing)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--config", help="JSON file of Config overrides")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="Config override (repeatable)")
    ap.add_argument("--resume", action="store_true", help="continue the run in --out from latest.pt")
    ap.add_argument("--no-eval", action="store_true", help="skip evaluation at checkpoints")
    ap.add_argument("--log-every", type=int, default=100)
    args = ap.parse_args(argv)

    from mdgan.data.preprocess import process_system
    from mdgan.evaluate import reference_msd

    if args.resume:
        ck_config = load_checkpoint(Path(args.out) / LATEST).config
        override = build_config(args.config, args.set).to_dict()
        changed = {k for k, v in override.items() if v != Config().to_dict()[k]}
        if changed - {"n_iterations"}:
            raise SystemExit(f"--resume only allows overriding n_iterations; got {sorted(changed)}")
        config = dataclasses.replace(ck_config, n_iterations=override["n_iterations"]) if changed else ck_config
    else:
        config = build_config(args.config, args.set)

    system = process_system(args.system_dir, cache=args.cache)
    dataset = WindowDataset(system, config.block_len, config.step_skip, config.phase_mode, config.scale_inputs)
    reference = None if args.no_eval else reference_msd(system)
    trainer = Trainer(config, dataset, args.out, device=args.device, reference=reference,
                      log_every=args.log_every, resume=args.resume)
    trainer.run()
    print(json.dumps({"out": str(trainer.out), "iteration": trainer.iteration, "rollbacks": len(trainer.rollbacks)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())