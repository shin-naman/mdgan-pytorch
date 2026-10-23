"""Tests for mdgan.checkpoint.

Run: python -m pytest tests/test_checkpoint.py -v
"""

import pytest
import torch

from conftest import SMALL
from mdgan.checkpoint import FORMAT_VERSION, MODULE_KEYS, Checkpoint, load_checkpoint, save_checkpoint, save_snapshot
from mdgan.config import Config
from mdgan.models import build_models

CFG = Config(latent_dim=SMALL["latent_dim"], noise_dim=SMALL["noise_dim"], block_len=SMALL["block_len"],
             n_channels=SMALL["n_channels"], hidden_dim=SMALL["hidden_dim"])


@pytest.fixture
def models():
    torch.manual_seed(11)
    return build_models(CFG)


def _outputs(gz, gy, dg):
    """Deterministic forward outputs of all three modules on fixed inputs."""
    g = torch.Generator().manual_seed(0)
    z, r = torch.rand(8, CFG.latent_dim, generator=g), torch.rand(8, CFG.noise_dim, generator=g)
    u, w = torch.randn(8, CFG.block_len, CFG.n_channels, generator=g), torch.randn(8, CFG.window_len, CFG.n_channels, generator=g)
    with torch.no_grad():
        return gz(z, r), gy(z, r, u), dg(w)


def _save(path, models, **kw):
    args = dict(config=CFG, models=models, iteration=1234, sigma_nm=0.1734)
    args.update(kw)
    save_checkpoint(path, **args)


def _payload(path):
    return torch.load(path, weights_only=True)


# =============================================================================
class TestRoundTrip:
    def test_models_and_metadata_survive(self, models, tmp_path):
        p = tmp_path / "ck.pt"
        _save(p, models, dataset={"system": "PE", "sigma_nm": 0.1734}, extra={"wasserstein": 0.5})
        ck = load_checkpoint(p)
        assert isinstance(ck, Checkpoint)
        assert ck.config == CFG and ck.iteration == 1234 and ck.sigma_nm == pytest.approx(0.1734)
        assert ck.dataset == {"system": "PE", "sigma_nm": 0.1734} and ck.extra == {"wasserstein": 0.5}
        for before, after in zip(_outputs(*models), _outputs(*ck.models), strict=True):
            torch.testing.assert_close(after, before, rtol=0, atol=0)

    def test_models_are_rebuilt_from_the_stored_config(self, tmp_path):
        other = Config(latent_dim=5, noise_dim=2, block_len=4, n_channels=3, hidden_dim=12)
        torch.manual_seed(0)
        p = tmp_path / "ck.pt"
        save_checkpoint(p, config=other, models=build_models(other), iteration=0, sigma_nm=1.0)
        ck = load_checkpoint(p)                                   # no config passed in
        assert ck.config == other and ck.gz.latent_dim == 5 and ck.dg.block_len == 4

    def test_optimizer_resume_is_bit_exact(self, models, tmp_path):
        """Two Adam steps straight == one step, save, load, one step."""
        def step(opt, net):
            loss = sum((p ** 2).sum() for p in net.parameters())
            opt.zero_grad(); loss.backward(); opt.step()

        gz, gy, dg = models
        opt = torch.optim.Adam(dg.parameters(), lr=1e-3, betas=(0.0, 0.9))
        step(opt, dg)
        p = tmp_path / "ck.pt"
        _save(p, models, optimizers={"critic": opt})
        step(opt, dg)                                            # reference: continue in place

        ck = load_checkpoint(p)
        opt2 = torch.optim.Adam(ck.dg.parameters(), lr=1e-3, betas=(0.0, 0.9))
        opt2.load_state_dict(ck.optimizer_states["critic"])
        step(opt2, ck.dg)
        for (n, a), (_, b) in zip(dg.state_dict().items(), ck.dg.state_dict().items(), strict=True):
            torch.testing.assert_close(b, a, rtol=0, atol=0, msg=n)

    def test_rng_states_survive(self, models, tmp_path):
        g = torch.Generator().manual_seed(42)
        torch.rand(5, generator=g)
        p = tmp_path / "ck.pt"
        _save(p, models, generator=g)
        expected_g, expected_global = torch.rand(5, generator=g), torch.rand(5)
        ck = load_checkpoint(p)
        g2 = torch.Generator(); g2.set_state(ck.generator_state)
        torch.set_rng_state(ck.torch_rng_state)
        torch.testing.assert_close(torch.rand(5, generator=g2), expected_g, rtol=0, atol=0)
        torch.testing.assert_close(torch.rand(5), expected_global, rtol=0, atol=0)

    def test_no_generator_means_no_generator_state(self, models, tmp_path):
        p = tmp_path / "ck.pt"
        _save(p, models)
        assert load_checkpoint(p).generator_state is None


# =============================================================================
class TestStrictLoading:
    """The three modules share state-dict key names, so the loader must refuse any
    cross-load or partial load rather than copy what happens to fit."""

    def test_the_premise_modules_share_key_names(self, models):
        gz, gy, dg = models
        assert set(gz.state_dict()) & set(gy.state_dict())
        assert set(gz.state_dict()) & set(dg.state_dict())

    def test_swapped_entries_refused_by_class_check(self, models, tmp_path):
        p = tmp_path / "ck.pt"
        _save(p, models)
        data = _payload(p)
        data["models"]["G_z"], data["models"]["G_Y"] = data["models"]["G_Y"], data["models"]["G_z"]
        torch.save(data, p)
        with pytest.raises(ValueError, match="holds a"):
            load_checkpoint(p)

    def test_forged_class_names_still_refused_by_strict_load(self, models, tmp_path):
        p = tmp_path / "ck.pt"
        _save(p, models)
        data = _payload(p)
        data["models"]["G_z"]["state_dict"] = data["models"]["D_G"]["state_dict"]   # class name still says G_z's
        torch.save(data, p)
        with pytest.raises(RuntimeError):
            load_checkpoint(p)

    @pytest.mark.parametrize("edit", ["missing_key", "extra_key"])
    def test_partial_or_padded_state_dict_refused(self, models, tmp_path, edit):
        """A non-strict loader would accept both silently."""
        p = tmp_path / "ck.pt"
        _save(p, models)
        data = _payload(p)
        sd = data["models"]["G_Y"]["state_dict"]
        if edit == "missing_key":
            del sd["output_layer.bias"]
        else:
            sd["stray.weight"] = torch.zeros(1)
        torch.save(data, p)
        with pytest.raises(RuntimeError):
            load_checkpoint(p)

    def test_missing_module_entry_refused(self, models, tmp_path):
        p = tmp_path / "ck.pt"
        _save(p, models)
        data = _payload(p); del data["models"]["D_G"]; torch.save(data, p)
        with pytest.raises(ValueError, match="modules"):
            load_checkpoint(p)

    def test_wrong_format_version_refused(self, models, tmp_path):
        p = tmp_path / "ck.pt"
        _save(p, models)
        data = _payload(p); data["format_version"] = FORMAT_VERSION + 1; torch.save(data, p)
        with pytest.raises(ValueError, match="format_version"):
            load_checkpoint(p)

    def test_unknown_config_key_refused(self, models, tmp_path):
        p = tmp_path / "ck.pt"
        _save(p, models)
        data = _payload(p); data["config"]["learning_rate"] = 1.0; torch.save(data, p)
        with pytest.raises(ValueError, match="learning_rate"):
            load_checkpoint(p)

    def test_file_is_readable_with_weights_only(self, models, tmp_path):
        """No arbitrary objects in the payload: safe to load with weights_only=True."""
        p = tmp_path / "ck.pt"
        _save(p, models, optimizers={"g": torch.optim.Adam(models[0].parameters())}, generator=torch.Generator())
        assert set(_payload(p)["models"]) == set(MODULE_KEYS)


# =============================================================================
class TestSave:
    def test_models_must_match_the_config(self, tmp_path):
        torch.manual_seed(0)
        other = build_models(Config(latent_dim=5, noise_dim=2, block_len=4, n_channels=3, hidden_dim=12))
        with pytest.raises(ValueError, match="do not match"):
            _save(tmp_path / "ck.pt", other)

    def test_models_must_be_in_order(self, models, tmp_path):
        gz, gy, dg = models
        with pytest.raises(ValueError, match="G_z must be"):
            _save(tmp_path / "ck.pt", (gy, gz, dg))

    @pytest.mark.parametrize("bad", [dict(iteration=-1), dict(iteration=1.5), dict(sigma_nm=0.0)])
    def test_invalid_scalars(self, models, tmp_path, bad):
        with pytest.raises(ValueError):
            _save(tmp_path / "ck.pt", models, **bad)

    def test_failed_save_leaves_the_previous_checkpoint_intact(self, models, tmp_path):
        p = tmp_path / "ck.pt"
        _save(p, models, iteration=10)
        with pytest.raises(Exception):
            _save(p, models, iteration=20, extra={"bad": lambda: None})    # not serializable
        assert load_checkpoint(p).iteration == 10
        assert sorted(x.name for x in tmp_path.iterdir()) == ["ck.pt"]       # no stray temporary file

    def test_overwrite_and_parent_directories(self, models, tmp_path):
        p = tmp_path / "runs" / "pe" / "ck.pt"
        _save(p, models, iteration=1)
        _save(p, models, iteration=2)
        assert load_checkpoint(p).iteration == 2
        assert sorted(x.name for x in p.parent.iterdir()) == ["ck.pt"]


# =============================================================================
class TestSnapshot:
    """Generators-only files: what evaluation needs, at a fraction of the size."""

    def _snap(self, path, models, **kw):
        gz, gy, _ = models
        save_snapshot(path, config=CFG, gz=gz, gy=gy, iteration=kw.pop("iteration", 50), sigma_nm=0.2, **kw)

    def test_round_trip(self, models, tmp_path):
        p = tmp_path / "snap.pt"
        self._snap(p, models, dataset={"system": "PE"})
        ck = load_checkpoint(p)
        assert ck.kind == "generators" and ck.dg is None and len(ck.models) == 2
        assert ck.iteration == 50 and ck.sigma_nm == 0.2 and ck.dataset == {"system": "PE"}
        assert ck.optimizer_states == {} and ck.generator_state is None
        before, after = _outputs(*models), _outputs(ck.gz, ck.gy, models[2])
        for a, b in zip(before[:2], after[:2], strict=True):
            torch.testing.assert_close(b, a, rtol=0, atol=0)

    def test_contains_no_critic_and_is_much_smaller_than_a_full_checkpoint(self, models, tmp_path):
        full, snap = tmp_path / "full.pt", tmp_path / "snap.pt"
        opts = {"g": torch.optim.Adam([*models[0].parameters(), *models[1].parameters()]),
                "d": torch.optim.Adam(models[2].parameters())}
        for o in opts.values():                                  # populate Adam's moment buffers
            o.zero_grad()
            sum((p ** 2).sum() for g in o.param_groups for p in g["params"]).backward()
            o.step()
        _save(full, models, optimizers=opts)
        self._snap(snap, models)
        assert set(_payload(snap)["models"]) == {"G_z", "G_Y"}
        assert snap.stat().st_size < 0.5 * full.stat().st_size

    def test_full_kind_reports_three_models(self, models, tmp_path):
        p = tmp_path / "ck.pt"
        _save(p, models)
        ck = load_checkpoint(p)
        assert ck.kind == "full" and len(ck.models) == 3 and ck.dg is not None

    def test_snapshot_with_a_critic_entry_refused(self, models, tmp_path):
        full, snap = tmp_path / "full.pt", tmp_path / "snap.pt"
        _save(full, models)
        self._snap(snap, models)
        data = _payload(snap); data["models"]["D_G"] = _payload(full)["models"]["D_G"]; torch.save(data, snap)
        with pytest.raises(ValueError, match="modules"):
            load_checkpoint(snap)

    def test_unknown_kind_refused(self, models, tmp_path):
        p = tmp_path / "snap.pt"
        self._snap(p, models)
        data = _payload(p); data["kind"] = "partial"; torch.save(data, p)
        with pytest.raises(ValueError, match="kind"):
            load_checkpoint(p)

    def test_strict_rules_apply_to_snapshots(self, models, tmp_path):
        p = tmp_path / "snap.pt"
        self._snap(p, models)
        data = _payload(p); del data["models"]["G_Y"]["state_dict"]["output_layer.bias"]; torch.save(data, p)
        with pytest.raises(RuntimeError):
            load_checkpoint(p)

    def test_save_snapshot_checks_the_models(self, tmp_path):
        torch.manual_seed(0)
        other = build_models(Config(latent_dim=5, noise_dim=2, block_len=4, n_channels=3, hidden_dim=12))
        with pytest.raises(ValueError, match="do not match"):
            save_snapshot(tmp_path / "s.pt", config=CFG, gz=other[0], gy=other[1], iteration=0, sigma_nm=1.0)

    def test_full_checkpoint_needs_three_models(self, models, tmp_path):
        with pytest.raises(ValueError, match="needs"):
            _save(tmp_path / "ck.pt", models[:2])