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
    CHECKPOINT_EXIT_CODE, LATEST, MAX_CONSECUTIVE_ROLLBACKS, METRIC_NAMES, ROLLBACK_SEED_STRIDE, DivergenceError, Trainer,
    _frozen,
    build_config, critic_view, full_checkpoint_name, load_system, main, make_generator_optimizer, snapshot_name,
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
    return WindowDataset(system, c.block_len, c.step_skip, c.phase_mode, c.scale_inputs, c.window_offset_max_steps,
                         c.holdout_trajectories, c.critic_blocks)


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
        events = jsonl(out / "events.jsonl")
        assert [e["event"] for e in events].count("rollback") == 3
        assert events[-1]["event"] == "diverged"                         # the stop itself is recorded
        assert events[-1]["failed_at_iteration"] == 4 and events[-1]["rollbacks"] == 3

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
class TestBatch1Probes:
    PROBES = [{"gz_additive_noise_std": 0.5}, {"gz_weight_decay": 0.1}, {"gz_dropout": 0.2},
              {"block_len": 16}, {"gz_norm_position": "hidden"}, {"window_offset_max_steps": 5},
              {"critic_input": "positions"}, {"latent_noise_share": 0.25}, {"holdout_trajectories": (1, 4)},
              {"critic_blocks": 4, "window_offset_max_steps": 16}, {"noise_use_min": 0.9, "reward_chains": 16},
              {"corr_reward_blocks": 3, "window_offset_max_steps": 8, "reward_chains": 16}]

    def test_default_generator_optimizer_is_the_paper_adam(self):
        from mdgan.models import build_models
        gz, gy, _ = build_models(tiny())
        opt = make_generator_optimizer(gz, gy, tiny())
        assert type(opt) is torch.optim.Adam and len(opt.param_groups) == 1

    def test_weight_decay_is_decoupled_and_on_gz_only(self):
        from mdgan.models import build_models
        c = tiny(gz_weight_decay=0.1)
        gz, gy, _ = build_models(c)
        opt = make_generator_optimizer(gz, gy, c)
        assert type(opt) is torch.optim.AdamW
        assert [g["weight_decay"] for g in opt.param_groups] == [0.1, 0.0]
        assert [id(p) for p in opt.param_groups[0]["params"]] == [id(p) for p in gz.parameters()]
        assert [id(p) for p in opt.param_groups[1]["params"]] == [id(p) for p in gy.parameters()]
        before = {k: v.clone() for k, v in {**{f"gz.{n}": p for n, p in gz.named_parameters()},
                                            **{f"gy.{n}": p for n, p in gy.named_parameters()}}.items()}
        for p in [*gz.parameters(), *gy.parameters()]:
            p.grad = torch.zeros_like(p)                                 # zero gradient: only the decay acts
        opt.step()
        for n, p in gz.named_parameters():
            torch.testing.assert_close(p, before[f"gz.{n}"] * (1 - c.lr * 0.1), rtol=1e-6, atol=0)
        for n, p in gy.named_parameters():
            torch.testing.assert_close(p, before[f"gy.{n}"], rtol=0, atol=0)

    def test_holdout_must_agree_and_cannot_change_on_resume(self, tmp_path):
        with pytest.raises(ValueError, match="holdout_trajectories"):
            Trainer(tiny(holdout_trajectories=(2,)), WindowDataset(brownian(), 8, 1), tmp_path / "x")
        trainer(tmp_path / "ok", tiny(holdout_trajectories=(4, 2)), log_every=1)       # order does not matter
        out = tmp_path / "run"
        trainer(out, tiny(holdout_trajectories=(2,), n_iterations=3)).run()
        for other in ((), (3,)):
            with pytest.raises(ValueError, match="holdout_trajectories"):
                trainer(out, tiny(holdout_trajectories=other), resume=True)

    def test_resume_refuses_a_different_window_offset_maximum(self, tmp_path):
        """Caught by the first resume check (config equal to the checkpoint's except n_iterations); the
        dataset metadata records the maximum too, as a second line of defence (tests/test_dataset.py)."""
        out = tmp_path / "run"
        trainer(out, tiny(window_offset_max_steps=5, n_iterations=3)).run()
        for k in (0, 7):
            with pytest.raises(ValueError, match=r"config differs from the checkpoint.*'window_offset_max_steps': \(5, " + str(k)):
                trainer(out, tiny(window_offset_max_steps=k), resume=True)

    def test_dataset_and_config_must_agree_on_offsets(self, tmp_path):
        with pytest.raises(ValueError, match="window_offset_max_steps"):
            Trainer(tiny(window_offset_max_steps=5), WindowDataset(brownian(), 8, 1), tmp_path / "run")

    @pytest.mark.parametrize("position", ["output", "hidden"])
    def test_weight_decay_can_spare_batchnorm_gamma_beta(self, position):
        """gz_weight_decay_on_norm False: no BatchNorm parameter in the decayed group; they get zero decay; every
        other G_z parameter (biases included) is decayed; G_Y undecayed. A zero-gradient step shows it."""
        from mdgan.models import build_models
        c = tiny(gz_weight_decay=0.1, gz_weight_decay_on_norm=False, gz_norm_position=position)
        gz, gy, _ = build_models(c)
        opt = make_generator_optimizer(gz, gy, c)
        norm = {id(p) for m in gz.modules() if isinstance(m, torch.nn.BatchNorm1d) for p in m.parameters()}
        assert norm, "the model has BatchNorm parameters"
        decayed = [g for g in opt.param_groups if g["weight_decay"] > 0]
        assert len(decayed) == 1 and not {id(p) for p in decayed[0]["params"]} & norm
        assert sorted(id(p) for g in opt.param_groups for p in g["params"]) == \
            sorted(id(p) for p in [*gz.parameters(), *gy.parameters()])            # each parameter exactly once
        before = {n: p.detach().clone() for n, p in gz.named_parameters()}
        for p in [*gz.parameters(), *gy.parameters()]:
            p.grad = torch.zeros_like(p)
        opt.step()
        for n, p in gz.named_parameters():
            factor = 1.0 if id(p) in norm else 1 - c.lr * 0.1
            torch.testing.assert_close(p.detach(), before[n] * factor, rtol=1e-6, atol=0, msg=n)

    def test_weight_decay_on_norm_default_keeps_the_old_groups(self):
        from mdgan.models import build_models
        c = tiny(gz_weight_decay=0.1)
        assert c.gz_weight_decay_on_norm is True
        gz, gy, _ = build_models(c)
        opt = make_generator_optimizer(gz, gy, c)
        assert [[id(p) for p in g["params"]] for g in opt.param_groups] == \
            [[id(p) for p in gz.parameters()], [id(p) for p in gy.parameters()]]
        assert [g["weight_decay"] for g in opt.param_groups] == [0.1, 0.0]

    def test_resume_refuses_a_flipped_weight_decay_on_norm(self, tmp_path):
        out = tmp_path / "run"
        trainer(out, tiny(gz_weight_decay=0.1, n_iterations=3)).run()
        with pytest.raises(ValueError, match="gz_weight_decay_on_norm"):
            trainer(out, tiny(gz_weight_decay=0.1, gz_weight_decay_on_norm=False), resume=True)

    def test_weight_decay_leaves_the_critic_alone(self, tmp_path):
        t = trainer(tmp_path / "run", tiny(gz_weight_decay=0.1))
        assert type(t.opt_d) is torch.optim.Adam and t.opt_d.param_groups[0]["weight_decay"] == 0

    @pytest.mark.parametrize("probe", PROBES, ids=lambda p: next(iter(p)))
    def test_probe_trains_and_resumes_bit_exactly(self, tmp_path, probe):
        c = tiny(**probe)
        straight = tmp_path / "straight"
        trainer(straight, c, reference=reference_msd(brownian())).run()   # evaluates; training is unaffected
        short = tmp_path / "short"
        trainer(short, tiny(**{**probe, "n_iterations": 3})).run()
        trainer(short, c, resume=True).run()
        assert_same_weights(weights(straight / LATEST), weights(short / LATEST))
        ck = load_checkpoint(straight / LATEST)
        assert all(getattr(ck.config, k) == v for k, v in probe.items())
        evals = jsonl(straight / "evaluations.jsonl")
        assert [e["iteration"] for e in evals] == [3, 6] and all(np.isfinite(e["err_percent"]) for e in evals)


# =============================================================================
class TestRunningStatsTraining:
    def test_running_stats_do_not_change_training(self, tmp_path):
        """use_running_stats only adds buffers that training-mode BatchNorm updates but never reads, and no
        eval-mode call is reachable from a training step: every parameter and every logged W are bit-identical
        with and without it. (Evaluation, which reads the buffers, is where the arm differs.)"""
        trainer(tmp_path / "off", tiny(use_running_stats=False)).run()
        trainer(tmp_path / "on", tiny(use_running_stats=True)).run()
        off, on = load_checkpoint(tmp_path / "off" / LATEST), load_checkpoint(tmp_path / "on" / LATEST)
        buffers = ("running_mean", "running_var", "num_batches_tracked")
        for name in ("gz", "gy", "dg"):
            a, b = getattr(off, name).state_dict(), getattr(on, name).state_dict()
            params = [k for k in a if not k.endswith(buffers)]
            assert params == [k for k in b if not k.endswith(buffers)]
            for k in params:
                assert torch.equal(a[k], b[k]), f"{name}.{k}"
        w_off = [m["wasserstein"] for m in jsonl(tmp_path / "off" / "metrics.jsonl")]
        w_on = [m["wasserstein"] for m in jsonl(tmp_path / "on" / "metrics.jsonl")]
        assert w_off == w_on and len(w_off) > 0


# =============================================================================
class TestFullCheckpoints:
    """Config.full_checkpoint_iterations: the complete training state kept at chosen iterations."""

    def test_none_by_default(self, tmp_path):
        trainer(tmp_path / "run").run()
        assert not list((tmp_path / "run").glob("full_*.pt"))

    def test_kept_only_at_the_listed_iterations_and_complete(self, tmp_path):
        out = tmp_path / "run"
        trainer(out, tiny(full_checkpoint_iterations=(3,))).run()
        assert sorted(p.name for p in out.glob("full_*.pt")) == [full_checkpoint_name(3)]
        assert not list(out.glob("*.tmp"))
        ck = load_checkpoint(out / full_checkpoint_name(3))
        assert ck.kind == "full" and ck.iteration == 3 and ck.dg is not None
        assert set(ck.optimizer_states) == {"generator", "critic"} and ck.generator_state is not None

    def test_equals_the_state_a_run_stopped_there_would_hold(self, tmp_path):
        """full_000003.pt from a 6-iteration run == latest.pt of the same run stopped at 3: models, optimiser
        states and the training generator, so a resume from it continues exactly."""
        trainer(tmp_path / "long", tiny(full_checkpoint_iterations=(3,))).run()
        trainer(tmp_path / "short", tiny(full_checkpoint_iterations=(3,), n_iterations=3)).run()
        kept, stopped = load_checkpoint(tmp_path / "long" / full_checkpoint_name(3)), load_checkpoint(tmp_path / "short" / LATEST)
        assert_same_weights(weights(tmp_path / "long" / full_checkpoint_name(3)), weights(tmp_path / "short" / LATEST))
        assert torch.equal(kept.generator_state, stopped.generator_state)
        for name in ("generator", "critic"):
            a, b = kept.optimizer_states[name]["state"], stopped.optimizer_states[name]["state"]
            assert a.keys() == b.keys() and all(torch.equal(a[k][f], b[k][f]) for k in a for f in a[k])


# =============================================================================
class TestCriticInput:
    """Config.critic_input: what the critic sees of a window (train.critic_view)."""

    def test_increments_view_is_the_same_tensor(self):
        x = torch.randn(4, 16, 3)
        assert critic_view(x, "increments") is x                      # the default adds no operation at all

    def test_positions_of_constant_increments_grow_linearly(self):
        m = TINY["block_len"]
        x = torch.full((2, 2 * m, 3), 0.5)
        v = critic_view(x, "positions")
        expected = 0.5 * torch.arange(1, 2 * m + 1, dtype=torch.float32)[None, :, None].expand(2, 2 * m, 3)
        torch.testing.assert_close(v, expected, rtol=0, atol=1e-6)
        assert v[0, m, 0].item() == pytest.approx(0.5 * (m + 1))       # no reset at the block boundary

    def test_scaled_positions_divide_by_the_constant(self):
        m = TINY["block_len"]
        x = torch.full((2, 2 * m, 3), 0.5)
        torch.testing.assert_close(critic_view(x, "positions_scaled", 2.5), critic_view(x, "positions") / 2.5,
                                   rtol=0, atol=0)

    def _scaled(self, **kw):
        ratio = WindowDataset(brownian(), TINY["block_len"], 1).position_rms_ratio
        return tiny(critic_input="positions_scaled", critic_position_scale=ratio, **kw), ratio

    def test_scale_must_match_the_training_windows_within_1_percent(self, tmp_path):
        c, ratio = self._scaled()
        trainer(tmp_path / "ok", tiny(critic_input="positions_scaled", critic_position_scale=ratio * 1.009))
        with pytest.raises(ValueError, match="1 %"):
            trainer(tmp_path / "off", tiny(critic_input="positions_scaled", critic_position_scale=ratio * 1.02))

    def test_dataset_json_records_the_ratio_and_metadata_does_not(self, tmp_path):
        c, ratio = self._scaled(n_iterations=3)
        trainer(tmp_path / "run", c).run()
        written = json.loads((tmp_path / "run" / "dataset.json").read_text())
        assert written["position_rms_ratio"] == pytest.approx(ratio, rel=1e-12)
        assert "position_rms_ratio" not in WindowDataset(brownian(), TINY["block_len"], 1).metadata

    def test_dataset_json_records_the_analytic_D_of_a_synthetic_system(self, tmp_path):
        from mdgan.data.synthetic import CageWalk
        walk = CageWalk.miniature()
        trainer(tmp_path / "syn", tiny(n_iterations=3), system=walk.simulate(6, 200, seed=1)).run()
        written = json.loads((tmp_path / "syn" / "dataset.json").read_text())
        assert written["D_analytic_m2s"] == walk.diffusion_coefficient_m2s()
        trainer(tmp_path / "md", tiny(n_iterations=3)).run()                         # an MD-like system: none
        assert "D_analytic_m2s" not in json.loads((tmp_path / "md" / "dataset.json").read_text())

    def test_positions_scaled_trains_and_resumes_bit_exactly(self, tmp_path):
        c, _ = self._scaled()
        short, _ = self._scaled(n_iterations=3)
        trainer(tmp_path / "straight", c).run()
        trainer(tmp_path / "short", short).run()
        trainer(tmp_path / "short", c, resume=True).run()
        assert_same_weights(weights(tmp_path / "straight" / LATEST), weights(tmp_path / "short" / LATEST))

    def test_unknown_view_rejected(self):
        with pytest.raises(ValueError, match="critic_input"):
            critic_view(torch.zeros(1, 2, 3), "velocities")
        with pytest.raises(ValueError, match="critic_input"):
            tiny(critic_input="velocities")

    @pytest.mark.parametrize("view", ["positions", "positions_scaled"])
    def test_positions_reach_both_terms_of_training(self, tmp_path, monkeypatch, view):
        """Every critic call gets cumsum(real) and cumsum(fake) (divided by the scale for positions_scaled);
        the generator call gets the same of its fake."""
        import mdgan.train as train_module
        if view == "positions":
            c, div = tiny(critic_input="positions", n_iterations=1), 1.0
        else:
            c, div = self._scaled(n_iterations=1)
        t = trainer(tmp_path / "run", c)
        sampled, fakes, critic_in, gen_in = [], [], [], []
        sample, make, crit, gen = t.dataset.sample, train_module.make_fake_window, train_module.critic_loss, train_module.generator_loss

        def rec_sample(*a, **kw):
            out = sample(*a, **kw)
            sampled.append(out.detach().clone())
            return out

        def rec_make(*a, **kw):
            out = make(*a, **kw)
            fakes.append(out.detach().clone())
            return out

        def rec_crit(critic, real, fake, *a, **kw):
            critic_in.append((real.detach().clone(), fake.detach().clone()))
            return crit(critic, real, fake, *a, **kw)

        def rec_gen(critic, fake):
            gen_in.append(fake.detach().clone())
            return gen(critic, fake)

        monkeypatch.setattr(t.dataset, "sample", rec_sample)
        monkeypatch.setattr(train_module, "make_fake_window", rec_make)
        monkeypatch.setattr(train_module, "critic_loss", rec_crit)
        monkeypatch.setattr(train_module, "generator_loss", rec_gen)
        t.run()
        n = TINY["n_critic"]
        assert len(critic_in) == n and len(sampled) == n and len(gen_in) == 1 and len(fakes) == n + 1
        for i in range(n):
            torch.testing.assert_close(critic_in[i][0], torch.cumsum(sampled[i], dim=1) / div, rtol=0, atol=0)
            torch.testing.assert_close(critic_in[i][1], torch.cumsum(fakes[i], dim=1) / div, rtol=0, atol=0)
        torch.testing.assert_close(gen_in[0], torch.cumsum(fakes[n], dim=1) / div, rtol=0, atol=0)

    def test_gradient_penalty_is_measured_on_the_position_input(self):
        """With D(x) = sum(x), the input gradient is all ones: norm sqrt(2M C) on positions. Had the penalty
        been taken with respect to the increments, the norm would be sqrt(C * sum k^2)."""
        from mdgan.losses import critic_loss

        class SumCritic(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.w = torch.nn.Parameter(torch.ones(1))

            def forward(self, x):
                return (x * self.w).sum(dim=(1, 2))

        m, c = TINY["block_len"], 3
        real, fake = torch.randn(8, 2 * m, c), torch.randn(8, 2 * m, c)
        _, info = critic_loss(SumCritic(), critic_view(real, "positions"), critic_view(fake, "positions"), 10.0,
                              generator=torch.Generator().manual_seed(0))
        on_positions = (2 * m * c) ** 0.5
        on_increments = (c * sum(k * k for k in range(1, 2 * m + 1))) ** 0.5
        assert float(info["grad_norm"]) == pytest.approx(on_positions, rel=1e-5)
        assert abs(float(info["grad_norm"]) - on_increments) > 1.0


# =============================================================================
class TestLongCriticAndRewards:
    """critic_blocks, noise_use_min and corr_reward_blocks (not in the papers; off by default)."""

    ALL = {"critic_blocks": 4, "window_offset_max_steps": 16, "noise_use_min": 0.9, "corr_reward_blocks": 4,
           "reward_chains": 16}

    def test_default_run_never_touches_the_rewards(self, tmp_path, monkeypatch):
        import mdgan.train as train_module

        def forbidden(*a, **kw):
            raise AssertionError("generator_rewards called with both rewards off")

        monkeypatch.setattr(train_module, "generator_rewards", forbidden)
        t = trainer(tmp_path / "run")
        t.run()
        assert t.metric_names == METRIC_NAMES and t.corr_targets is None
        keys = {"iteration", *METRIC_NAMES, "iterations_per_second", "rollbacks"}
        assert all(set(r) == keys for r in jsonl(tmp_path / "run" / "metrics.jsonl"))
        saved = json.loads((tmp_path / "run" / "dataset.json").read_text())
        assert "corr_reward_targets" not in saved and "critic_blocks" not in saved

    def test_long_critic_on_positions_sees_the_cumsum_of_the_whole_window(self, tmp_path, monkeypatch):
        """critic_blocks 4 with critic_input positions: every critic call gets cumsum over all 4 M steps of the
        sampled real window and of the fake one; the generator call gets the same of its fake."""
        import mdgan.train as train_module
        t = trainer(tmp_path / "run", tiny(critic_blocks=4, window_offset_max_steps=16, critic_input="positions",
                                           n_iterations=1))
        sampled, fakes, critic_in, gen_in = [], [], [], []
        sample, make, crit, gen = (t.dataset.sample, train_module.make_fake_window, train_module.critic_loss,
                                   train_module.generator_loss)

        def rec_sample(*a, **kw):
            sampled.append(sample(*a, **kw))
            return sampled[-1]

        def rec_make(*a, **kw):
            out = make(*a, **kw)
            fakes.append(out.detach().clone())
            return out

        def rec_crit(critic, real, fake, *a, **kw):
            critic_in.append((real.detach().clone(), fake.detach().clone()))
            return crit(critic, real, fake, *a, **kw)

        def rec_gen(critic, fake):
            gen_in.append(fake.detach().clone())
            return gen(critic, fake)

        monkeypatch.setattr(t.dataset, "sample", rec_sample)
        monkeypatch.setattr(train_module, "make_fake_window", rec_make)
        monkeypatch.setattr(train_module, "critic_loss", rec_crit)
        monkeypatch.setattr(train_module, "generator_loss", rec_gen)
        t.run()
        n, steps = TINY["n_critic"], 4 * t.config.block_len
        assert len(critic_in) == n and len(sampled) == n and len(gen_in) == 1 and len(fakes) == n + 1
        for (real, fake), s, f in zip(critic_in, sampled, fakes, strict=False):
            assert real.shape[1] == steps and fake.shape[1] == steps
            torch.testing.assert_close(real, torch.cumsum(s, dim=1))
            torch.testing.assert_close(fake, torch.cumsum(f, dim=1))
        torch.testing.assert_close(gen_in[0], torch.cumsum(fakes[-1], dim=1))
        assert not torch.equal(critic_in[0][0][:, steps // 2], sampled[0][:, steps // 2])     # not the increments

    def test_reward_columns_targets_and_progress(self, tmp_path, capsys):
        c = tiny(**self.ALL)
        t = trainer(tmp_path / "run", c, progress=True)
        t.run()
        names = ("noise_use_share", "noise_use_penalty", "corr_error", "reward_loss")
        assert t.metric_names == METRIC_NAMES + names
        rows = jsonl(tmp_path / "run" / "metrics.jsonl")
        assert len(rows) == 6 and all(np.isfinite([r[k] for k in t.metric_names]).all() for r in rows)
        for r in rows:
            assert 0 <= r["noise_use_share"] <= 2 and r["noise_use_penalty"] >= 0 and r["corr_error"] >= 0
            assert r["reward_loss"] == pytest.approx(100 * r["noise_use_penalty"] + 10 * r["corr_error"], rel=1e-4)
        targets = t.dataset.block_sum_correlations(4)
        assert t.corr_targets.dtype == torch.float32
        np.testing.assert_allclose(t.corr_targets.numpy(), targets, rtol=1e-6)
        saved = json.loads((tmp_path / "run" / "dataset.json").read_text())
        np.testing.assert_allclose(saved["corr_reward_targets"], targets, rtol=1e-6)
        assert saved["critic_blocks"] == 4
        lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("it ")]
        assert len(lines) == 6 and all(f"| {k} " in ln for ln in lines for k in names)

    def test_a_constant_reward_changes_nothing_and_generator_loss_stays_the_critic_term(self, tmp_path, monkeypatch):
        """The reward acts through its gradient alone. A stand-in that returns a constant and draws nothing must
        leave every weight and every logged critic-side value as in the default run."""
        import mdgan.train as train_module
        trainer(tmp_path / "plain").run()
        const = {"noise_use_share": torch.tensor(0.5), "noise_use_penalty": torch.tensor(0.25), "reward_loss": torch.tensor(7.0)}
        monkeypatch.setattr(train_module, "generator_rewards", lambda *a, **kw: (torch.tensor(7.0), const))
        trainer(tmp_path / "const", tiny(noise_use_min=0.9)).run()
        assert_same_weights(weights(tmp_path / "plain" / LATEST), weights(tmp_path / "const" / LATEST))
        plain, const_rows = jsonl(tmp_path / "plain" / "metrics.jsonl"), jsonl(tmp_path / "const" / "metrics.jsonl")
        assert [[r[k] for k in METRIC_NAMES] for r in plain] == [[r[k] for k in METRIC_NAMES] for r in const_rows]
        assert [r["reward_loss"] for r in const_rows] == [7.0] * 6

    @pytest.mark.parametrize("on,off", [({"noise_use_min": 0.9}, {"noise_use_weight": 0.0}),
                                        ({"corr_reward_blocks": 3, "window_offset_max_steps": 8}, {"corr_reward_weight": 0.0})],
                             ids=["noise_use", "correlations"])
    def test_each_reward_moves_the_generators(self, tmp_path, on, off):
        """Weight 0 draws the same numbers and adds nothing: the difference from the weighted run is the
        reward's gradient. The critic sees the reward only through the generators it is trained against, so
        after one iteration its weights still agree."""
        base = {**on, "reward_chains": 16, "n_iterations": 1, "checkpoint_every": 1}
        trainer(tmp_path / "with", tiny(**base)).run()
        trainer(tmp_path / "zero", tiny(**base, **off)).run()
        a, b = weights(tmp_path / "with" / LATEST), weights(tmp_path / "zero" / LATEST)
        assert any(not torch.equal(a[k], b[k]) for k in a if k.startswith("gz."))
        if "corr_reward_blocks" in on:
            assert any(not torch.equal(a[k], b[k]) for k in a if k.startswith("gy."))
        else:
            assert all(torch.equal(a[k], b[k]) for k in a if k.startswith("gy."))       # the noise-use term has no G_Y
        assert all(torch.equal(a[k], b[k]) for k in a if k.startswith("dg."))

    def test_rewards_draw_from_the_trainer_generator_only(self, tmp_path, monkeypatch):
        """One generator drives data and model noise, the side batch included; training leaves the global
        torch RNG where the constructor put it."""
        import mdgan.train as train_module
        original, seen = train_module.generator_rewards, []

        def spy(*a, generator=None, **kw):
            seen.append(generator)
            return original(*a, generator=generator, **kw)

        monkeypatch.setattr(train_module, "generator_rewards", spy)
        t = trainer(tmp_path / "run", tiny(**self.ALL))
        before = torch.get_rng_state().clone()
        t.run()
        assert len(seen) == 6 and all(g is t.gen for g in seen)
        assert torch.equal(before, torch.get_rng_state())

    def test_non_finite_reward_rolls_back_at_that_iteration(self, tmp_path, monkeypatch):
        import mdgan.train as train_module
        original = train_module.generator_rewards
        out = tmp_path / "run"
        t = trainer(out, tiny(noise_use_min=0.9, reward_chains=16))
        seen = []

        def poisoned(*a, **kw):
            reward, stats = original(*a, **kw)
            seen.append(t.iteration)
            if t.iteration == 3 and seen.count(3) == 1:                   # update 4, first attempt
                return reward, {**stats, "reward_loss": stats["reward_loss"] * float("nan")}
            return reward, stats

        monkeypatch.setattr(train_module, "generator_rewards", poisoned)
        t.run()
        assert [(r["failed_at_iteration"], r["to_iteration"]) for r in t.rollbacks] == [(4, 3)]
        assert [r["iteration"] for r in jsonl(out / "metrics.jsonl")] == [1, 2, 3, 4, 5, 6]
        assert all(np.isfinite(r["reward_loss"]) for r in jsonl(out / "metrics.jsonl"))

    def test_critic_sees_L_blocks_real_and_fake(self, tmp_path, monkeypatch):
        import mdgan.train as train_module
        crit, gen, shapes = train_module.critic_loss, train_module.generator_loss, []

        def rec_crit(critic, real, fake, *a, **kw):
            shapes.append((tuple(real.shape), tuple(fake.shape)))
            return crit(critic, real, fake, *a, **kw)

        def rec_gen(critic, fake):
            shapes.append((tuple(fake.shape),))
            return gen(critic, fake)

        monkeypatch.setattr(train_module, "critic_loss", rec_crit)
        monkeypatch.setattr(train_module, "generator_loss", rec_gen)
        t = trainer(tmp_path / "run", tiny(critic_blocks=4, window_offset_max_steps=16, n_iterations=1))
        t.run()
        long = (TINY["batch_size"], 4 * TINY["block_len"], 3)
        assert shapes == [(long, long), (long, long), (long,)] and t.dg.window_len == 4 * TINY["block_len"]

    def test_dataset_and_config_must_agree_on_critic_blocks(self, tmp_path):
        two = WindowDataset(brownian(), 8, 1, window_offset_max_steps=16)
        four = WindowDataset(brownian(), 8, 1, window_offset_max_steps=16, critic_blocks=4)
        for c, ds in ((tiny(critic_blocks=4, window_offset_max_steps=16), two), (tiny(window_offset_max_steps=16), four)):
            with pytest.raises(ValueError, match="does not match.*critic_blocks"):
                Trainer(c, ds, tmp_path / "run")

    @pytest.mark.parametrize("change", [{"critic_blocks": 2}, {"noise_use_min": 0.5}, {"noise_use_weight": 1.0},
                                        {"corr_reward_blocks": 3}, {"corr_reward_weight": 1.0}, {"reward_chains": 8}],
                             ids=lambda p: next(iter(p)))
    def test_resume_refuses_a_change_of_any_new_option(self, tmp_path, change):
        out = tmp_path / "run"
        trainer(out, tiny(**self.ALL, n_iterations=3)).run()
        with pytest.raises(ValueError, match="config differs from the checkpoint.*" + next(iter(change))):
            trainer(out, tiny(**{**self.ALL, **change}), resume=True)

    def test_time_limit_restarts_with_everything_on_are_bit_exact(self, tmp_path):
        """The cluster cycle (one iteration per session) with the long critic and both rewards: the rewards
        draw from the checkpointed generator, so six sessions == one straight run, logs included."""
        c = tiny(**self.ALL)
        straight = tmp_path / "straight"
        trainer(straight, c).run()
        out = tmp_path / "chunked"
        status, sessions = trainer(out, c).run(max_seconds=0), 1
        while status == "time_limit" and sessions < 20:
            status, sessions = trainer(out, c, resume=True).run(max_seconds=0), sessions + 1
        assert status == "done" and sessions == 6
        assert_same_weights(weights(straight / LATEST), weights(out / LATEST))
        drop = lambda rows: [{k: v for k, v in r.items() if k != "iterations_per_second"} for r in rows]   # noqa: E731
        assert drop(jsonl(straight / "metrics.jsonl")) == drop(jsonl(out / "metrics.jsonl"))
        assert json.loads((straight / "dataset.json").read_text()) == json.loads((out / "dataset.json").read_text())

    def test_cluster_job_cycle_with_everything_on(self, tmp_path, capsys):
        """The command a CHTC job runs (cache file, --auto-resume, exit 85 until done) equals the straight run."""
        cache, system = TestCluster()._cache(tmp_path)
        c = tiny(**self.ALL)
        cfg = tmp_path / "all.json"
        cfg.write_text(json.dumps(c.to_dict()))
        out = tmp_path / "run"
        cmd = [str(cache), "--out", str(out), "--config", str(cfg), "--auto-resume", "--max-hours", "0",
               "--log-every", "1", "--no-eval"]
        codes = [main(cmd)]
        while codes[-1] == CHECKPOINT_EXIT_CODE and len(codes) < 20:
            codes.append(main(cmd))
        assert codes == [CHECKPOINT_EXIT_CODE] * 5 + [0]
        assert "| reward_loss " in capsys.readouterr().out
        straight = tmp_path / "straight"
        Trainer(c, windows(system, c), straight).run()
        assert_same_weights(weights(straight / LATEST), weights(out / LATEST))
        ck = load_checkpoint(out / LATEST)
        assert ck.dg.n_blocks == 4 and ck.config == c
        assert json.loads((out / "dataset.json").read_text())["critic_blocks"] == 4


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
        with pytest.raises(ValueError, match="only n_iterations"):
            main([str(sys_dir), "--out", str(out), "--resume", "--set", "lr=0.1"])


# =============================================================================
class TestCluster:
    """What an HTCondor job needs: a time limit that saves and signals, automatic resume,
    and training from a preprocessed cache .npz alone."""

    def _cache(self, tmp_path, system=None, version=None):
        from mdgan.data.preprocess import PREPROCESS_VERSION
        system = system or brownian()
        system.meta = {"preprocess_version": PREPROCESS_VERSION if version is None else version, "md5": ["0" * 32]}
        path = tmp_path / "system.npz"
        system.save(path)
        return path, system

    def test_time_limit_saves_latest_and_restarts_are_bit_exact(self, tmp_path):
        """Each session stops after one iteration (limit 0); six sessions == one straight run."""
        straight = tmp_path / "straight"
        trainer(straight).run()
        out = tmp_path / "chunked"
        status, sessions = trainer(out).run(max_seconds=0), 1
        while status == "time_limit" and sessions < 20:
            assert load_checkpoint(out / LATEST).iteration == sessions          # progress every session
            status, sessions = trainer(out, resume=True).run(max_seconds=0), sessions + 1
        assert status == "done" and sessions == 6
        assert_same_weights(weights(straight / LATEST), weights(out / LATEST))
        assert [r["iteration"] for r in jsonl(out / "metrics.jsonl")] == [1, 2, 3, 4, 5, 6]
        assert [e["event"] for e in jsonl(out / "events.jsonl")].count("time_limit") == 5

    def test_no_limit_means_done(self, tmp_path):
        assert trainer(tmp_path / "run", tiny(n_iterations=2)).run() == "done"

    def test_cli_job_cycle_from_a_cache_file(self, tmp_path):
        """What HTCondor does: run the same command until it exits 0; 85 means 'saved, restart me'."""
        cache, system = self._cache(tmp_path)
        cfg = tmp_path / "tiny.json"
        cfg.write_text(json.dumps(tiny().to_dict()))
        out = tmp_path / "run"
        cmd = [str(cache), "--out", str(out), "--config", str(cfg), "--auto-resume", "--max-hours", "0",
               "--log-every", "1"]
        codes = [main(cmd)]
        while codes[-1] == CHECKPOINT_EXIT_CODE and len(codes) < 20:
            codes.append(main(cmd))
        assert codes == [CHECKPOINT_EXIT_CODE] * 5 + [0]
        straight = tmp_path / "straight"
        Trainer(tiny(), windows(system, tiny()), straight).run()
        assert_same_weights(weights(straight / LATEST), weights(out / LATEST))
        assert [e["iteration"] for e in jsonl(out / "evaluations.jsonl")] == [3, 6]

    def test_provenance_falls_back_to_the_commit_stamp(self, tmp_path, monkeypatch):
        """Cluster copies have no .git; the stamp written before copying records the commit."""
        import subprocess

        import mdgan.train as train_module

        def no_git(*a, **kw):
            raise FileNotFoundError("git")

        monkeypatch.setattr(subprocess, "run", no_git)
        monkeypatch.setattr(train_module, "COMMIT_STAMP", tmp_path / "_commit.json")
        assert train_module._git_state()["git_commit"] is None
        (tmp_path / "_commit.json").write_text(json.dumps({"git_commit": "abc123", "git_dirty": False}))
        assert train_module._git_state() == {"git_commit": "abc123", "git_dirty": False, "git_source": "stamp"}

    def test_progress_lines_for_condor_tail(self, tmp_path, capsys):
        """A running cluster job is only visible through stdout; one line per log and evaluation."""
        system = brownian()
        trainer(tmp_path / "run", system=system, reference=reference_msd(system), progress=True).run()
        lines = capsys.readouterr().out.splitlines()
        assert [ln.split()[1] for ln in lines if ln.startswith("it ")] == ["1", "2", "3", "4", "5", "6"]
        assert [ln.split()[1] for ln in lines if ln.startswith("eval ")] == ["3", "6"]
        trainer(tmp_path / "quiet").run()
        assert capsys.readouterr().out == ""                         # library default stays silent

    def test_synthetic_npz_loads_and_trains_from_the_command_line(self, tmp_path):
        """The cage walk saved to a file passes load_system (generated data has no preprocessing version) and trains
        through main(), as python -m mdgan.train runs it; an unknown process without a version is still refused."""
        from mdgan.data.preprocess import SystemData
        from mdgan.data.synthetic import CageWalk
        path = tmp_path / "cage.npz"
        CageWalk.miniature().simulate(6, 200, seed=1).save(path)
        loaded = load_system(str(path))
        assert loaded.meta["process"] == "cage_walk" and "D_exact_m2s" in loaded.meta
        cfg = tmp_path / "c.json"
        cfg.write_text(json.dumps(tiny(n_iterations=3).to_dict()))
        out = tmp_path / "run"
        assert main([str(path), "--out", str(out), "--config", str(cfg), "--log-every", "1", "--no-eval"]) == 0
        assert json.loads((out / "dataset.json").read_text())["D_analytic_m2s"] == loaded.meta["D_exact_m2s"]
        cfg.write_text(json.dumps(tiny(n_iterations=3, holdout_trajectories=(0, 5)).to_dict()))
        held = tmp_path / "held"
        assert main([str(path), "--out", str(held), "--config", str(cfg), "--log-every", "1", "--no-eval"]) == 0
        written = json.loads((held / "dataset.json").read_text())
        assert written["holdout_trajectories"] == [0, 5] and written["n_trajectories"] == 4      # split in the folder
        other = SystemData(name="x", displacements=loaded.displacements, dt_ps=10.0, box_nm=loaded.box_nm,
                           meta={"process": "something_else"})
        other.save(tmp_path / "other.npz")
        with pytest.raises(ValueError, match="preprocess version"):
            load_system(str(tmp_path / "other.npz"))

    def test_load_system_from_cache_and_refuse_stale_versions(self, tmp_path):
        cache, system = self._cache(tmp_path)
        loaded = load_system(str(cache))
        np.testing.assert_array_equal(loaded.displacements, system.displacements)
        assert loaded.dt_ps == system.dt_ps
        stale, _ = self._cache(tmp_path / "old", version=0)
        with pytest.raises(ValueError, match="preprocess version"):
            load_system(str(stale))


# =============================================================================
@pytest.mark.learning
def test_generator_learns_gaussian_step_statistics(tmp_path):
    """Data: i.i.d. Gaussian steps (unit std after scaling). After training, generated
    steps must have per-step std near 1 and no spurious lag-1 correlation.

    2048 trajectories, i.e. 2048 training windows. With 64 windows in this 48-dim window
    space the Wasserstein objective itself favours a shrunken variance (exact W1 to 64
    points is smaller at std 0.5 than at 1.0; observed: std 0.70, W stuck at 6.9), so a
    small dataset tests data starvation, not learning. Bounds are a first calibration."""
    c = Config(latent_dim=8, noise_dim=8, block_len=8, n_channels=3, hidden_dim=64, batch_size=64,
               n_iterations=1500, checkpoint_every=1500, lr=5e-4, burn_in=2, pair_window=2,
               estimation_chains=64, estimation_time_ps=1000.0, err_window_ps=(20.0, 900.0),
               diffusion_fit_ps=(250.0, 500.0))
    t = Trainer(c, windows(brownian(n_traj=2048, n_steps=400), c), tmp_path / "run", log_every=500)
    t.run()
    from mdgan.rollout import rollout
    steps = rollout(t.gz, t.gy, 512, 8, c.burn_in, generator=torch.Generator().manual_seed(0))
    std = steps.std(dim=0).mean().item()
    lag1 = torch.corrcoef(torch.stack([steps[:, :-1].flatten(), steps[:, 1:].flatten()]))[0, 1].item()
    assert 0.9 < std < 1.1, std
    assert abs(lag1) < 0.1, lag1
    assert steps.isfinite().all()


@pytest.mark.slow
@pytest.mark.learning
def test_brownian_diffusion_coefficient_is_recovered(tmp_path):
    """Can MD-GAN learn a process with a known answer? Brownian steps, MSD = 6 D t.

    2048 trajectories (see the Gaussian test for why 64 is data-starved). Asserts on the
    BEST checkpoint by err, which is the paper's own protocol (best of its checkpoints,
    chosen against long MD). This test therefore claims "the model can learn the process",
    not "training is stable": the final checkpoint drifts (lr 5e-4: err 6.4 -> 13.5 -> 19.8 %,
    D_rel -0.04 -> -0.18 -> +0.81) because long-range block correlations are not
    constrained by a two-block critic. That is tracked separately (tools/diag_brownian.py,
    docs/decisions.md) until there is a fix with its own test."""
    system = brownian(n_traj=2048, n_steps=400)
    c = Config(latent_dim=8, noise_dim=8, block_len=8, n_channels=3, hidden_dim=128, batch_size=64,
               n_iterations=3000, checkpoint_every=1000, lr=5e-4, burn_in=4, pair_window=2,
               estimation_chains=2048, estimation_time_ps=4000.0, err_window_ps=(20.0, 3900.0),
               diffusion_fit_ps=(1000.0, 2000.0))
    out = tmp_path / "run"
    Trainer(c, windows(system, c), out, reference=reference_msd(system), log_every=500).run()
    evals = jsonl(out / "evaluations.jsonl")
    for e in evals:
        print(f"it {e['iteration']:5d}  err {e['err_percent']:5.1f}%  D_rel {e['D_rel_error']:+.3f}")
    best = min(evals, key=lambda e: e["err_percent"])
    assert best["err_percent"] < 15.0, best
    assert abs(best["D_rel_error"]) < 0.15, best