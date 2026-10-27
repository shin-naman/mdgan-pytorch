"""Tests for mdgan.config.Config and mdgan.models.build_models.

Config is the single home of every default. The drift guard at the bottom checks
that the constructor and function defaults elsewhere still agree with it.

Run: python -m pytest tests/test_config.py -v
"""

import inspect
import json

import pytest

from mdgan.config import Config


# =============================================================================
class TestDefaults:
    def test_defaults_are_the_documented_baseline(self):
        """docs/decisions.md. If one of these changes, decisions.md changes with it."""
        c = Config()
        assert (c.latent_dim, c.noise_dim, c.block_len, c.n_channels, c.hidden_dim) == (32, 32, 64, 3, 2048)
        assert c.negative_slope == 0.2 and c.use_running_stats is False and c.init == "pytorch"
        assert (c.step_skip, c.phase_mode, c.scale_inputs) == (1, "single", True)
        assert (c.batch_size, c.n_iterations, c.n_critic) == (64, 300_000, 5)
        assert (c.lr, c.betas, c.gp_lambda) == (1e-4, (0.0, 0.9), 10.0)
        assert (c.burn_in, c.pair_window, c.mean_operation) == (7, 4, True)
        assert (c.checkpoint_every, c.estimation_chains, c.estimation_time_ps) == (10_000, 8192, 80_000.0)
        assert c.err_window_ps == (200.0, 78_000.0) and c.diffusion_fit_ps == (10_000.0, 20_000.0)

    def test_frozen(self):
        with pytest.raises(Exception):
            Config().lr = 1.0

    def test_window_len(self):
        assert Config().window_len == 128 and Config(block_len=8).window_len == 16

    @pytest.mark.parametrize("step_skip,blocks", [(1, 125), (5, 25), (7, 18), (20, 7)])
    def test_estimation_blocks_cover_80_ns(self, step_skip, blocks):
        """80 ns at 10 ps fine steps: PE n_s=1 -> 8000 coarse steps -> exactly 125 blocks;
        other n_s round UP so the rollout always reaches 80 ns."""
        c = Config(step_skip=step_skip)
        n = c.estimation_blocks(10.0)
        assert n == blocks
        assert n * c.block_len * 10.0 * step_skip >= c.estimation_time_ps
        assert (n - 1) * c.block_len * 10.0 * step_skip < c.estimation_time_ps


# =============================================================================
class TestValidation:
    @pytest.mark.parametrize("bad", [
        dict(latent_dim=0), dict(block_len=-1), dict(n_critic=0), dict(batch_size=True),
        dict(step_skip=2.0), dict(pair_window=0), dict(estimation_chains=0),
        dict(burn_in=-1), dict(seed=-1), dict(phase_mode="some"),
        dict(lr=0.0), dict(betas=(0.0, 1.0)), dict(betas=(0.5,)), dict(gp_lambda=-1.0),
        dict(negative_slope=-0.1), dict(init="xavier"), dict(err_window_ps=(78_000.0, 200.0)),
        dict(diffusion_fit_ps=(0.0, 10.0)), dict(estimation_time_ps=0.0),
        dict(estimation_time_ps=10_000.0), dict(err_window_ps=(200.0, 90_000.0)),
        dict(diffusion_fit_ps=(10_000.0, 85_000.0)),
    ])
    def test_invalid_values_rejected(self, bad):
        with pytest.raises(ValueError):
            Config(**bad)

    def test_edge_values_accepted(self):
        Config(burn_in=0, gp_lambda=0.0, negative_slope=0.0, seed=0, phase_mode="all")


# =============================================================================
class TestSerialization:
    def test_round_trip_through_json(self):
        c = Config(latent_dim=7, step_skip=20, betas=(0.5, 0.99), mean_operation=False)
        d = json.loads(json.dumps(c.to_dict()))
        assert Config.from_dict(d) == c

    def test_to_dict_is_json_native(self):
        d = Config().to_dict()
        assert isinstance(d["betas"], list) and isinstance(d["err_window_ps"], list)
        json.dumps(d)

    def test_unknown_keys_rejected(self):
        d = Config().to_dict(); d["learning_rate"] = 1e-3
        with pytest.raises(ValueError, match="learning_rate"):
            Config.from_dict(d)

    def test_missing_keys_take_defaults(self):
        assert Config.from_dict({"latent_dim": 4}) == Config(latent_dim=4)

    def test_validation_applies_on_load(self):
        with pytest.raises(ValueError):
            Config.from_dict({"n_critic": 0})


# =============================================================================
class TestBuildModels:
    def test_shared_dimensions_agree(self):
        from mdgan.models import Critic, LatentTransitionGenerator, TrajectoryGenerator, build_models
        c = Config(latent_dim=5, noise_dim=4, block_len=8, n_channels=3, hidden_dim=16, negative_slope=0.1)
        gz, gy, dg = build_models(c)
        assert isinstance(gz, LatentTransitionGenerator) and isinstance(gy, TrajectoryGenerator)
        assert isinstance(dg, Critic)
        assert (gz.latent_dim, gz.noise_dim) == (gy.latent_dim, gy.noise_dim) == (5, 4)
        assert (gy.block_len, gy.n_channels) == (dg.block_len, dg.n_channels) == (8, 3)
        assert gz.hidden_dim == gy.hidden_dim == dg.hidden_dim == 16
        assert gz.act.negative_slope == gy.input_act.negative_slope == dg.act.negative_slope == 0.1

    def test_running_stats_option_is_passed_through(self):
        from mdgan.models import build_models
        gz, _, _ = build_models(Config(hidden_dim=8, use_running_stats=True))
        assert gz.output_norm.track_running_stats is True

    def test_built_models_compose(self):
        """G_z output feeds G_Y; two G_Y blocks make a critic window."""
        import torch

        from mdgan.models import build_models
        c = Config(latent_dim=5, noise_dim=4, block_len=8, hidden_dim=16)
        gz, gy, dg = build_models(c)
        z = gz.unroll(torch.rand(4, 5), steps=2)[-1]
        window = torch.cat([gy.generate(z), gy.generate(z)], dim=1)
        assert window.shape == (4, c.window_len, c.n_channels) and dg(window).shape == (4,)


# =============================================================================
class TestNoDrift:
    """Defaults also appear as keyword defaults in constructors and functions (so
    each module is usable on its own). They must agree with Config."""

    @pytest.mark.parametrize("target", ["LatentTransitionGenerator", "TrajectoryGenerator", "Critic", "WindowDataset"])
    def test_constructor_defaults_match_config(self, target):
        if target == "WindowDataset":
            from mdgan.data.dataset import WindowDataset as cls
        else:
            import mdgan.models as models
            cls = getattr(models, target)
        c = Config()
        checked = 0
        for name, param in inspect.signature(cls.__init__).parameters.items():
            if param.default is not inspect.Parameter.empty and hasattr(c, name):
                assert param.default == getattr(c, name), f"{target}.{name}"
                checked += 1
        assert checked >= 2

    def test_msd_window_defaults_match_config(self):
        from mdgan.data.msd import diffusion_coefficient, err
        c = Config()
        d = inspect.signature(diffusion_coefficient).parameters
        assert (d["t_min_ps"].default, d["t_max_ps"].default) == c.diffusion_fit_ps
        e = inspect.signature(err).parameters
        assert (e["t1_ps"].default, e["t2_ps"].default) == c.err_window_ps

    def test_loss_defaults_match_config(self):
        from mdgan.losses import critic_loss, gradient_penalty
        for fn in (gradient_penalty, critic_loss):
            assert inspect.signature(fn).parameters["gp_lambda"].default == Config().gp_lambda