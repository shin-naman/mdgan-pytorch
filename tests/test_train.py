"""Tests for mdgan.train.

Tiny models and a short synthetic Brownian system keep every run to a fraction of a
second, so the schedule, resume, rollback and logging guarantees are tested on real
runs rather than on mocks. Bit-exactness claims are CPU claims (docs/testing.md).

Run: python -m pytest tests/test_train.py -v
     python -m pytest -m learning tests/test_train.py -v          (does it learn?)
"""

import copy
import json

import numpy as np
import pytest
import torch

from mdgan.checkpoint import load_checkpoint
from mdgan.config import Config
from mdgan.data.dataset import WindowDataset
from mdgan.data.preprocess import SystemData
from mdgan.evaluate import reference_msd
from mdgan.train import (
    LATEST, MAX_CONSECUTIVE_ROLLBACKS, ROLLBACK_SEED_STRIDE, DivergenceError, Trainer, _frozen, build_config,
    main, snapshot_name,
)

DT = 10.0
D_NM2_PS = 1.3e-10 / 1e-6
TINY = dict(latent_dim=7, noise_dim=3, block_len=8, n_channels=3, hidden_dim=24, batch_size=8, n_critic=2,
            burn_in=2, pair_window=2, checkpoint_every=3, n_iterations=6, estimation_chains=16,
            estimation_time_ps=1000.0, err_window_ps=(20.0, 900.0), diffusion_fit_ps=(250.0, 500.0))


def tiny(**kw) -> Config:
    return Config(**{**TINY, **kw})


def brownian(seed=0, n_traj=6, n_steps=200):
    rng = np.random.default_rng(seed)
    d = rng.normal(scale=np.sqrt(2 * D_NM2_PS * DT), size=(n_traj, n_steps, 3)).astype(np.float32)
    return SystemData(name=f"brownian{seed}", displacements=d, dt_ps=DT, box_nm=np.full((n_traj, 3), 5.0), meta={})


def windows(system, c: Config) -> WindowDataset:
    return WindowDataset(system, c.block_len, c.step_skip, c.phase_mode, c.scale_inputs)


def trainer(path, c=None, system=None, **kw) -> Trainer:
    c = c or tiny()
    return Trainer(c, windows(system or brownian(), c), path, log_every=kw.pop("log_every", 1), **kw)


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def weights(path):
    ck = load_checkpoint(path)
    return {f"{k}.{n}": t for k, m in zip(("gz", "gy", "dg"), ck.models, strict=True) for n, t in m.state_dict().items()}


def assert_same_weights(a, b):
    assert a.keys() == b.keys()
    for k in a:
        torch.testing.assert_close(a[k], b[k], rtol=0, atol=0, msg=k)


# =============================================================================
class TestSteps:
    def test_critic_step_changes_only_the_critic(self, tmp_path):
        t = trainer(tmp_path / "run")
        g_before = copy.deepcopy([t.gz.state_dict(), t.gy.state_dict()])
        d_before = copy.deepcopy(t.dg.state_dict())
        t._critic_step()
        for before, m in zip(g_before, (t.gz, t.gy), strict=True):
            for k, v in m.state_dict().items():
                assert torch.equal(v, before[k]), k
        assert t.opt_g.state_dict()["state"] == {}                       # generators' Adam untouched
        assert any(not torch.equal(v, d_before[k]) for k, v in t.dg.state_dict().items())

    def test_generator_step_changes_only_the_generators(self, tmp_path):
        t = trainer(tmp_path / "run")
        t._critic_step()                                                 # critic now has grads and Adam state
        d_params = copy.deepcopy(t.dg.state_dict())
        d_grads = [p.grad.clone() for p in t.dg.parameters()]
        d_opt = copy.deepcopy(t.opt_d.state_dict())
        g_before = copy.deepcopy(t.gy.state_dict())
        t._generator_step()
        for k, v in t.dg.state_dict().items():
            assert torch.equal(v, d_params[k]), k
        for p, g in zip(t.dg.parameters(), d_grads, strict=True):
            assert torch.equal(p.grad, g)                                # frozen: no gradient accumulated
        for pid, st in t.opt_d.state_dict()["state"].items():
            assert torch.equal(st["step"], d_opt["state"][pid]["step"])
        assert all(p.requires_grad for p in t.dg.parameters())
        assert any(not torch.equal(v, g_before[k]) for k, v in t.gy.state_dict().items())

    def test_frozen_restores_gradients_even_on_error(self, tmp_path):
        t = trainer(tmp_path / "run")
        with pytest.raises(RuntimeError, match="boom"):
            with _frozen(t.dg):
                assert not any(p.requires_grad for p in t.dg.parameters())
                raise RuntimeError("boom")
        assert all(p.requires_grad for p in t.dg.parameters())

    def test_n_critic_critic_updates_per_generator_update(self, tmp_path):
        t = trainer(tmp_path / "run", tiny(n_critic=3, n_iterations=2))
        t.run()
        d_steps = {int(s["step"]) for s in t.opt_d.state_dict()["state"].values()}
        g_steps = {int(s["step"]) for s in t.opt_g.state_dict()["state"].values()}
        assert d_steps == {6} and g_steps == {2}


# =============================================================================
class TestRun:
    def test_files_schedule_and_logs(self, tmp_path):
        system = brownian()
        out = tmp_path / "run"
        t = trainer(out, system=system, reference=reference_msd(system))
        t.run()
        for name in ("config.json", "dataset.json", "provenance.json", LATEST, snapshot_name(3), snapshot_name(6),
                     "msd_000003.npy", "msd_000006.npy"):
            assert (out / name).exists(), name
        assert Config.from_dict(json.loads((out / "config.json").read_text())) == tiny()
        assert "git_commit" in json.loads((out / "provenance.json").read_text())
        assert [r["iteration"] for r in jsonl(out / "metrics.jsonl")] == [1, 2, 3, 4, 5, 6]
        m = jsonl(out / "metrics.jsonl")[-1]
        assert {"wasserstein", "gradient_penalty", "grad_norm", "generator_loss", "fake_step_std",
                "iterations_per_second", "rollbacks"} <= set(m)
        assert all(np.isfinite(m[k]) for k in ("wasserstein", "gradient_penalty", "grad_norm", "fake_step_std"))
        evals = jsonl(out / "evaluations.jsonl")
        assert [e["iteration"] for e in evals] == [3, 6] and all("err_percent" in e for e in evals)
        assert load_checkpoint(out / LATEST).iteration == 6
        snap = load_checkpoint(out / snapshot_name(3))
        assert snap.kind == "generators" and snap.iteration == 3
        assert [e["event"] for e in jsonl(out / "events.jsonl")] == ["start", "checkpoint", "checkpoint"]

    def test_latest_exists_at_iteration_zero(self, tmp_path):
        trainer(tmp_path / "run")
        assert load_checkpoint(tmp_path / "run" / LATEST).iteration == 0

    def test_final_checkpoint_when_not_a_multiple(self, tmp_path):
        out = tmp_path / "run"
        trainer(out, tiny(n_iterations=4)).run()
        assert (out / snapshot_name(3)).exists() and (out / snapshot_name(4)).exists()

    def test_refuses_a_non_empty_run_directory(self, tmp_path):
        out = tmp_path / "run"; out.mkdir(); (out / "notes.txt").write_text("x")
        with pytest.raises(FileExistsError):
            trainer(out)
        trainer(tmp_path / "empty")                                     # an empty existing dir is fine
        (tmp_path / "empty2").mkdir()
        trainer(tmp_path / "empty2")

    def test_dataset_must_match_the_config(self, tmp_path):
        c = tiny()
        ds = WindowDataset(brownian(), 4, 1)                             # block_len 4 vs config 8
        with pytest.raises(ValueError, match="does not match"):
            Trainer(c, ds, tmp_path / "run")


# =============================================================================
class TestResume:
    def test_crash_and_resume_is_bit_exact_with_clean_logs(self, tmp_path):
        straight = tmp_path / "straight"
        trainer(straight).run()

        crashed = tmp_path / "crashed"
        t = trainer(crashed)
        original = t._generator_step

        def crash_during_update_5():
            if t.iteration == 4:
                raise RuntimeError("simulated crash")
            return original()

        t._generator_step = crash_during_update_5
        with pytest.raises(RuntimeError, match="simulated crash"):
            t.run()
        assert [r["iteration"] for r in jsonl(crashed / "metrics.jsonl")] == [1, 2, 3, 4]
        trainer(crashed, resume=True).run()                              # restarts from latest.pt at 3

        assert_same_weights(weights(straight / LATEST), weights(crashed / LATEST))
        assert [r["iteration"] for r in jsonl(crashed / "metrics.jsonl")] == [1, 2, 3, 4, 5, 6]
        assert "resume" in [e["event"] for e in jsonl(crashed / "events.jsonl")]

    def test_resume_may_extend_n_iterations(self, tmp_path):
        straight = tmp_path / "straight"
        trainer(straight).run()                                          # 6 iterations
        short = tmp_path / "short"
        trainer(short, tiny(n_iterations=3)).run()
        trainer(short, tiny(n_iterations=6), resume=True).run()
        assert_same_weights(weights(straight / LATEST), weights(short / LATEST))

    def test_resume_refuses_other_config_changes(self, tmp_path):
        out = tmp_path / "run"
        trainer(out, tiny(n_iterations=3)).run()
        with pytest.raises(ValueError, match="only n_iterations"):
            trainer(out, tiny(lr=1e-3), resume=True)

    def test_resume_refuses_a_different_dataset(self, tmp_path):
        out = tmp_path / "run"
        trainer(out, tiny(n_iterations=3)).run()
        with pytest.raises(ValueError, match="dataset differs"):
            trainer(out, system=brownian(seed=9), resume=True)

    def test_resume_needs_latest_and_a_full_checkpoint(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            trainer(tmp_path / "nothing", resume=True)
        out = tmp_path / "run"
        trainer(out, tiny(n_iterations=3)).run()
        (out / snapshot_name(3)).replace(out / LATEST)
        with pytest.raises(ValueError, match="full"):
            trainer(out, resume=True)


# =============================================================================
class TestEvaluationIsolation:
    def test_training_is_identical_with_and_without_evaluation(self, tmp_path):
        system = brownian()
        trainer(tmp_path / "with", system=system, reference=reference_msd(system)).run()
        trainer(tmp_path / "without", system=system).run()
        assert_same_weights(weights(tmp_path / "with" / LATEST), weights(tmp_path / "without" / LATEST))

    def test_every_checkpoint_is_evaluated_on_the_same_noise(self, tmp_path, monkeypatch):
        """The evaluation generator is freshly seeded at every checkpoint: same starting state."""
        import mdgan.train as train_module
        states = []
        original = train_module.evaluate

        def spy(*a, generator, **kw):
            states.append(generator.get_state().clone())
            return original(*a, generator=generator, **kw)

        monkeypatch.setattr(train_module, "evaluate", spy)
        system = brownian()
        trainer(tmp_path / "run", system=system, reference=reference_msd(system)).run()
        assert len(states) == 2 and torch.equal(states[0], states[1])


# =============================================================================
class TestRollback:
    def _inject_nan(self, monkeypatch, t, when):
        """Make critic_loss non-finite during the updates selected by when(iteration, n_calls_so_far)."""
        import mdgan.train as train_module
        original = train_module.critic_loss
        calls = {}

        def poisoned(*a, **kw):
            loss, info = original(*a, **kw)
            calls[t.iteration] = calls.get(t.iteration, 0) + 1
            if when(t.iteration, calls[t.iteration]):
                return loss * float("nan"), info
            return loss, info

        monkeypatch.setattr(train_module, "critic_loss", poisoned)

    def test_one_nan_rolls_back_reseeds_and_recovers(self, tmp_path, monkeypatch):
        out = tmp_path / "run"
        t = trainer(out)
        self._inject_nan(monkeypatch, t, lambda it, n: it == 3 and n == 1)    # first critic call of update 4
        t.run()
        seed = tiny().seed + ROLLBACK_SEED_STRIDE
        assert t.rollbacks == [{"failed_at_iteration": 4, "to_iteration": 3, "new_seed": seed}]
        assert load_checkpoint(out / LATEST).extra["rollbacks"] == t.rollbacks
        assert [r["iteration"] for r in jsonl(out / "metrics.jsonl")] == [1, 2, 3, 4, 5, 6]
        assert [e["event"] for e in jsonl(out / "events.jsonl")].count("rollback") == 1
        assert all(torch.isfinite(v).all() for v in weights(out / LATEST).values())

    def test_a_rolled_back_run_is_reproducible(self, tmp_path, monkeypatch):
        finals = []
        for name in ("a", "b"):
            t = trainer(tmp_path / name)
            self._inject_nan(monkeypatch, t, lambda it, n: it == 3 and n == 1)
            t.run()
            monkeypatch.undo()                                           # fresh patch for the second run
            finals.append(weights(tmp_path / name / LATEST))
        assert_same_weights(*finals)
        # The retry is re-seeded, so it must NOT replay the undisturbed run's draws.
        trainer(tmp_path / "clean").run()
        clean = weights(tmp_path / "clean" / LATEST)
        assert any(not torch.equal(clean[k], finals[0][k]) for k in clean)

    def test_persistent_nan_stops_after_three_consecutive_rollbacks(self, tmp_path, monkeypatch):
        out = tmp_path / "run"
        t = trainer(out)
        self._inject_nan(monkeypatch, t, lambda it, n: it >= 3)
        with pytest.raises(DivergenceError):
            t.run()
        assert MAX_CONSECUTIVE_ROLLBACKS == 3                            # the decided value (decisions.md)
        assert [e["event"] for e in jsonl(out / "events.jsonl")].count("rollback") == 3

    def test_rollbacks_to_different_checkpoints_do_not_accumulate(self, tmp_path, monkeypatch):
        """Two rollbacks to iteration 0 and two to iteration 3: four in total, never more than
        three in a row to the same checkpoint, so the run completes."""
        out = tmp_path / "run"
        t = trainer(out)
        fired = {0: 0, 3: 0}
        n_critic = tiny().n_critic

        def when(it, n):
            # n counts critic calls at this iteration across retries; n % n_critic == 1 is the
            # first critic call of each attempt. Poison the first two attempts at 0 and at 3.
            if it in fired and n % n_critic == 1 and fired[it] < 2:
                fired[it] += 1
                return True
            return False

        self._inject_nan(monkeypatch, t, when)
        t.run()
        assert t.iteration == 6 and len(t.rollbacks) == 4
        assert [r["to_iteration"] for r in t.rollbacks] == [0, 0, 3, 3]
        assert [r["new_seed"] for r in t.rollbacks] == [tiny().seed + ROLLBACK_SEED_STRIDE * k for k in (1, 2, 3, 4)]


# =============================================================================
class TestAblation:
    def test_unscaled_run_records_sigma_one(self, tmp_path):
        out = tmp_path / "run"
        c = tiny(scale_inputs=False, n_iterations=3)
        trainer(out, c).run()
        assert load_checkpoint(out / LATEST).sigma_nm == 1.0
        assert load_checkpoint(out / snapshot_name(3)).sigma_nm == 1.0

    def test_scaling_flag_must_agree(self, tmp_path):
        c = tiny(scale_inputs=False)
        with pytest.raises(ValueError, match="scale_inputs"):
            Trainer(c, WindowDataset(brownian(), 8, 1), tmp_path / "run")    # dataset scaled


# =============================================================================
class TestCLI:
    def test_build_config_layers_file_then_overrides(self, tmp_path):
        f = tmp_path / "c.json"
        f.write_text(json.dumps({"step_skip": 20, "n_critic": 4}))
        c = build_config(str(f), ["n_critic=2", "betas=[0.5, 0.9]", "phase_mode=all", "mean_operation=false"])
        assert (c.step_skip, c.n_critic, c.betas, c.phase_mode, c.mean_operation) == (20, 2, (0.5, 0.9), "all", False)
        with pytest.raises(ValueError, match="unknown"):
            build_config(None, ["learning_rate=0.1"])
        with pytest.raises(ValueError, match="key=value"):
            build_config(None, ["n_critic"])

    def test_new_run_then_resume(self, tmp_path, synthetic_system, capsys):
        sys_dir, _ = synthetic_system
        cfg = tmp_path / "tiny.json"
        cfg.write_text(json.dumps(tiny(n_iterations=3).to_dict()))
        out = tmp_path / "run"
        assert main([str(sys_dir), "--out", str(out), "--config", str(cfg), "--log-every", "1"]) == 0
        assert json.loads(capsys.readouterr().out.strip().splitlines()[-1])["iteration"] == 3
        assert (out / "evaluations.jsonl").exists()
        assert main([str(sys_dir), "--out", str(out), "--resume", "--set", "n_iterations=6", "--no-eval"]) == 0
        assert load_checkpoint(out / LATEST).iteration == 6
        with pytest.raises(SystemExit):
            main([str(sys_dir), "--out", str(out), "--resume", "--set", "lr=0.1"])


# =============================================================================
@pytest.mark.learning
def test_generator_learns_gaussian_step_statistics(tmp_path):
    """Data: i.i.d. Gaussian steps (unit std after scaling). After training, generated
    steps must have per-step std near 1 and no spurious lag-1 correlation. Bounds are
    heuristic (a first calibration of the training budget), not derived."""
    c = Config(latent_dim=8, noise_dim=8, block_len=8, n_channels=3, hidden_dim=64, batch_size=64,
               n_iterations=1500, checkpoint_every=1500, lr=5e-4, burn_in=2, pair_window=2,
               estimation_chains=64, estimation_time_ps=1000.0, err_window_ps=(20.0, 900.0),
               diffusion_fit_ps=(250.0, 500.0))
    t = Trainer(c, windows(brownian(n_traj=64, n_steps=400), c), tmp_path / "run", log_every=500)
    t.run()
    from mdgan.rollout import rollout
    steps = rollout(t.gz, t.gy, 512, 8, c.burn_in, generator=torch.Generator().manual_seed(0))
    std = steps.std(dim=0).mean().item()
    x = steps.flatten(1)
    lag1 = torch.corrcoef(torch.stack([steps[:, :-1].flatten(), steps[:, 1:].flatten()]))[0, 1].item()
    assert 0.8 < std < 1.2, std
    assert abs(lag1) < 0.1, lag1
    assert x.isfinite().all()


@pytest.mark.slow
@pytest.mark.learning
def test_brownian_diffusion_coefficient_is_recovered(tmp_path):
    """The first test of whether MD-GAN learns: train on a Brownian system with known D
    and evaluate the final checkpoint (blind: no checkpoint selection). Bounds are the
    review's 15 %; this run also calibrates the budget the persistent-walk test will need."""
    system = brownian(n_traj=64, n_steps=400)
    c = Config(latent_dim=8, noise_dim=8, block_len=8, n_channels=3, hidden_dim=128, batch_size=64,
               n_iterations=3000, checkpoint_every=1000, lr=5e-4, burn_in=4, pair_window=2,
               estimation_chains=2048, estimation_time_ps=4000.0, err_window_ps=(20.0, 3900.0),
               diffusion_fit_ps=(1000.0, 2000.0))
    out = tmp_path / "run"
    Trainer(c, windows(system, c), out, reference=reference_msd(system), log_every=500).run()
    final = jsonl(out / "evaluations.jsonl")[-1]
    print(json.dumps(final, indent=2))
    assert abs(final["D_rel_error"]) < 0.15, final
    assert final["err_percent"] < 15.0, final