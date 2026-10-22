"""Tests for mdgan.evaluate.

The oracle test replaces G_z / G_Y with stubs that emit Brownian steps with a KNOWN
D (in sigma units). The full evaluation pipeline must recover D and match the MD
reference MSD: this checks the unit and grid bookkeeping (sigma, n_s, dt, trimming,
MSD, D, err) end to end, independent of any learning.

Run: python -m pytest tests/test_evaluate.py -v
"""

import json

import numpy as np
import pytest
import torch
from torch import nn

from conftest import SMALL
from mdgan.config import Config
from mdgan.data.msd import diffusion_coefficient, err, msd_ensemble, msd_time_averaged
from mdgan.data.preprocess import SystemData
from mdgan.evaluate import EvaluationResult, evaluate, evaluate_checkpoint, generated_msd, main, reference_msd
from mdgan.rollout import rollout

D_TRUE = 1.3e-10                       # m^2/s
D_NM2_PS = D_TRUE / 1e-6               # nm^2/ps
DT = 10.0                              # ps, fine MD grid
DIMS = dict(latent_dim=SMALL["latent_dim"], noise_dim=SMALL["noise_dim"], block_len=SMALL["block_len"],
            n_channels=SMALL["n_channels"], hidden_dim=SMALL["hidden_dim"])
SHORT = dict(estimation_time_ps=4000.0, err_window_ps=(20.0, 3900.0), diffusion_fit_ps=(1000.0, 2000.0))


def cfg(**kw):
    return Config(**{**DIMS, **SHORT, "estimation_chains": 64, **kw})


def cfg_t(T, **kw):
    """Config whose evaluation windows fit inside a shorter estimation time T."""
    return cfg(estimation_time_ps=T, err_window_ps=(20.0, 0.9 * T), diffusion_fit_ps=(0.25 * T, 0.5 * T), **kw)


def brownian_system(n_traj, n_steps, seed=0):
    rng = np.random.default_rng(seed)
    d = rng.normal(scale=np.sqrt(2 * D_NM2_PS * DT), size=(n_traj, n_steps, 3)).astype(np.float32)
    return SystemData(name="brownian", displacements=d, dt_ps=DT, box_nm=np.full((n_traj, 3), 5.0), meta={})


class OracleGz(nn.Module):
    """Latent transition that does nothing; only the decoder matters for the oracle."""
    def __init__(self):
        super().__init__()
        self.latent_dim, self.noise_dim = 2, 1
        self._anchor = nn.Parameter(torch.zeros(1))

    def forward(self, z, r):
        return z

    def sample_noise(self, batch, *, device=None, dtype=None, generator=None):
        return torch.rand(batch, 1, device=device, dtype=dtype, generator=generator)


class OracleGy(nn.Module):
    """Emits i.i.d. Gaussian coarse steps with a given per-axis std (sigma units)."""
    def __init__(self, block_len, step_std):
        super().__init__()
        self.block_len, self.n_channels, self.step_std = block_len, 3, step_std

    def generate(self, z, generator=None):
        return torch.randn(z.shape[0], self.block_len, 3, generator=generator, dtype=z.dtype) * self.step_std


@pytest.fixture(scope="module")
def reference():
    """1024 Brownian trajectories x 400 fine steps (4 ns): msd_ref on the 10 ps grid."""
    return reference_msd(brownian_system(1024, 400, seed=1))


# =============================================================================
@pytest.mark.statistical
@pytest.mark.parametrize("step_skip", [1, 2])
def test_oracle_generator_recovers_known_D(reference, step_skip):
    """Tolerance. D from an ensemble of N chains has relative standard error ~sqrt(2/N)
    (test_msd.py derivation); the time-averaged reference over N_ref trajectories is
    bounded by the same expression with N_ref. err is a log-time average of relative
    MSD deviations whose per-lag standard error is ~sqrt(2/(3N)) for each curve.
    All bounds are 5 standard errors. A sigma-vs-sigma^2 or an n_s/dt bug is off by
    a factor of 2 or more."""
    msd_ref, dt_ref = reference
    N, N_ref, sigma = 4096, 1024, 0.05
    c = cfg(step_skip=step_skip, estimation_chains=N)
    std_sigma_units = np.sqrt(2 * D_NM2_PS * DT * step_skip) / sigma
    res = evaluate(OracleGz(), OracleGy(c.block_len, std_sigma_units), c, sigma, msd_ref, dt_ref,
                   generator=torch.Generator().manual_seed(0))
    assert res.dt_gen_ps == DT * step_skip and res.n_steps == int(4000 / (DT * step_skip))
    assert res.D_gen_m2s == pytest.approx(D_TRUE, rel=5 * np.sqrt(2 / N))
    assert abs(res.D_rel_error) < 5 * np.sqrt(2 / N + 2 / N_ref)
    assert res.err_percent < 100 * 5 * np.sqrt(2 / (3 * N) + 2 / (3 * N_ref))


# =============================================================================
class TestPipeline:
    def test_generated_msd_matches_the_manual_pipeline(self, gz, gy):
        c = cfg_t(1000.0, burn_in=3)
        msd, dt_gen, n = generated_msd(gz, gy, c, 0.17, DT, generator=torch.Generator().manual_seed(4))
        d = rollout(gz, gy, 64, c.estimation_blocks(DT), 3, mean_operation=True,
                    generator=torch.Generator().manual_seed(4))
        np.testing.assert_array_equal(msd, msd_ensemble(d[:, :100]) * 0.17 ** 2)
        assert (dt_gen, n) == (DT, 64)

    def test_mean_flag_and_burn_in_come_from_the_config(self, gz, gy):
        c = cfg_t(640.0, burn_in=0, mean_operation=False)
        msd, _, _ = generated_msd(gz, gy, c, 1.0, DT, generator=torch.Generator().manual_seed(1))
        d = rollout(gz, gy, 64, c.estimation_blocks(DT), 0, mean_operation=False,
                    generator=torch.Generator().manual_seed(1))
        np.testing.assert_array_equal(msd, msd_ensemble(d))

    def test_sigma_enters_squared(self, gz, gy):
        c = cfg_t(640.0)
        a, _, _ = generated_msd(gz, gy, c, 1.0, DT, generator=torch.Generator().manual_seed(2))
        b, _, _ = generated_msd(gz, gy, c, 3.0, DT, generator=torch.Generator().manual_seed(2))
        np.testing.assert_allclose(b, 9.0 * a, rtol=1e-12)

    def test_trims_to_the_estimation_time_when_blocks_round_up(self, gz, gy):
        """1000 ps at 10 ps is 100 steps; 13 blocks of 8 would give 104."""
        c = cfg_t(1000.0)
        assert c.estimation_blocks(DT) * c.block_len == 104
        msd, _, _ = generated_msd(gz, gy, c, 1.0, DT)
        assert msd.shape == (101,) and msd[0] == 0.0

    def test_chains_default_to_the_config(self, gz, gy):
        _, _, n = generated_msd(gz, gy, cfg_t(160.0, estimation_chains=16), 1.0, DT)
        assert n == 16

    def test_reference_msd(self):
        system = brownian_system(8, 50)
        msd, dt = reference_msd(system)
        np.testing.assert_array_equal(msd, msd_time_averaged(system.displacements)[0])
        assert dt == DT


# =============================================================================
class TestEvaluate:
    def test_result_fields_are_consistent(self, gz, gy, reference):
        msd_ref, dt_ref = reference
        c = cfg()
        res = evaluate(gz, gy, c, 0.17, msd_ref, dt_ref, iteration=70_000, generator=torch.Generator().manual_seed(0))
        assert isinstance(res, EvaluationResult) and res.iteration == 70_000 and res.sigma_nm == 0.17
        D_ref, s_ref = diffusion_coefficient(msd_ref, dt_ref, *c.diffusion_fit_ps)
        D_gen, _ = diffusion_coefficient(res.msd_gen_nm2, res.dt_gen_ps, *c.diffusion_fit_ps)
        assert res.D_ref_m2s == D_ref and res.slope_ref_nm2_per_ns == s_ref and res.D_gen_m2s == D_gen
        assert res.D_rel_error == pytest.approx(D_gen / D_ref - 1)
        assert res.err_percent == err(res.msd_gen_nm2, res.dt_gen_ps, msd_ref, dt_ref, *c.err_window_ps)
        assert np.isfinite(res.err_percent) and res.err_percent >= 0

    def test_to_dict_is_json_and_omits_the_curve(self, gz, gy, reference):
        res = evaluate(gz, gy, cfg(), 1.0, *reference)
        d = res.to_dict()
        json.dumps(d)
        assert "msd_gen_nm2" not in d and d["n_chains"] == 64

    def test_reference_too_short_is_rejected(self, gz, gy):
        msd_ref, dt_ref = reference_msd(brownian_system(4, 100))         # 1 ns < 3.9 ns window
        with pytest.raises(ValueError, match="reference MSD reaches"):
            evaluate(gz, gy, cfg(), 1.0, msd_ref, dt_ref)


# =============================================================================
class TestCheckpoint:
    @pytest.fixture
    def saved(self, tmp_path):
        from mdgan.checkpoint import save_checkpoint
        from mdgan.models import build_models
        c = cfg(seed=5)
        torch.manual_seed(0)
        models = build_models(c)
        p = tmp_path / "ck.pt"
        save_checkpoint(p, config=c, models=models, iteration=30_000, sigma_nm=0.17)
        return p, c, models

    def test_uses_the_checkpoint_config_sigma_and_iteration(self, saved):
        p, c, (gz, gy, _) = saved
        system = brownian_system(16, 400)
        res = evaluate_checkpoint(p, system, seed=3)
        direct = evaluate(gz, gy, c, 0.17, *reference_msd(system), iteration=30_000,
                          generator=torch.Generator().manual_seed(3))
        assert res.to_dict() == direct.to_dict()
        np.testing.assert_array_equal(res.msd_gen_nm2, direct.msd_gen_nm2)

    def test_seed_defaults_to_config_and_changes_the_ensemble(self, saved):
        p, _, _ = saved
        system = brownian_system(16, 400)
        msd_ref = reference_msd(system)[0]
        a = evaluate_checkpoint(p, system, msd_ref=msd_ref)
        b = evaluate_checkpoint(p, system, msd_ref=msd_ref, seed=5)       # config seed is 5
        c = evaluate_checkpoint(p, system, msd_ref=msd_ref, seed=6)
        np.testing.assert_array_equal(a.msd_gen_nm2, b.msd_gen_nm2)
        assert not np.array_equal(a.msd_gen_nm2, c.msd_gen_nm2)

    def test_cli(self, tmp_path, synthetic_system, capsys):
        """End to end from .xtc files: synthetic system is 3 trajectories x 199 steps (1.99 ns)."""
        from mdgan.checkpoint import save_checkpoint
        from mdgan.models import build_models
        c = cfg(estimation_chains=16, estimation_time_ps=1500.0, err_window_ps=(20.0, 1400.0),
                diffusion_fit_ps=(500.0, 1000.0))
        torch.manual_seed(0)
        save_checkpoint(tmp_path / "ck.pt", config=c, models=build_models(c), iteration=10, sigma_nm=0.2)
        sys_dir, _ = synthetic_system
        assert main([str(tmp_path / "ck.pt"), str(sys_dir), "--seed", "1"]) == 0
        out = json.loads(capsys.readouterr().out)
        assert out["iteration"] == 10 and out["n_chains"] == 16 and out["n_steps"] == 150