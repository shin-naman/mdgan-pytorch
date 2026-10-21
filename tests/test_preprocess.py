"""Unit tests for mdgan.data.preprocess.

A synthetic system directory (MD_data/<name>/trajectory/traj*.xtc) is built in
tmp_path from rigid-water random walks with known unwrapped displacements, so
every check runs without the dataset (fixture `synthetic_system` in conftest).
The real-data test uses the session `pe` fixture (real_data marker).

Run: python -m pytest tests/data/test_preprocess.py -v
"""

import numpy as np
import pytest

pytest.importorskip("mdtraj")

from mdgan.data.preprocess import (  # noqa: E402
    PREPROCESS_VERSION, SystemData, discover_trajectories, main, process_system, process_trajectory,
)
from synthetic_water import unwrapped_trajectory, wrap, write_xtc  # noqa: E402

XTC_PRECISION_NM = 1e-3
DISP_TOL_NM = 3e-3          # a displacement is a difference of two 1e-3-rounded coordinates
BOX = np.array([4.75, 4.75, 4.75])


# --- discovery ------------------------------------------------------------------------

def test_discover_natural_order_and_ignores_junk(synthetic_system):
    sys_dir, _ = synthetic_system
    (sys_dir / "trajectory" / "traj010.xtc").write_bytes(b"")
    (sys_dir / "trajectory" / "traj002b.xtc").write_bytes(b"")   # matches glob; sorted after traj002
    files = [p.name for p in discover_trajectories(sys_dir)]
    assert files == ["traj001.xtc", "traj002.xtc", "traj002b.xtc", "traj003.xtc", "traj010.xtc"]


def test_discover_requires_trajectory_subdir(tmp_path):
    with pytest.raises(FileNotFoundError, match="trajectory"):
        discover_trajectories(tmp_path)


def test_discover_requires_at_least_one_file(tmp_path):
    (tmp_path / "trajectory").mkdir()
    with pytest.raises(FileNotFoundError, match="traj"):
        discover_trajectories(tmp_path)


# --- process_trajectory ------------------------------------------------------------------

def test_process_trajectory_recovers_known_displacements(synthetic_system):
    sys_dir, truth = synthetic_system
    d, dt, box, n_frames = process_trajectory(sys_dir / "trajectory" / "traj001.xtc")
    assert d.shape == (199, 3) and d.dtype == np.float64
    assert dt == pytest.approx(10.0) and n_frames == 200
    np.testing.assert_allclose(box, BOX, atol=1e-5)
    np.testing.assert_allclose(d, truth[0], atol=DISP_TOL_NM)


def test_process_trajectory_rejects_wrong_atom_count(tmp_path):
    rng = np.random.default_rng(0)
    xyz = rng.uniform(0, 4, size=(20, 5, 3))
    write_xtc(tmp_path / "t.xtc", xyz, box=BOX)
    with pytest.raises(ValueError, match="3 atoms"):
        process_trajectory(tmp_path / "t.xtc")


def test_process_trajectory_rejects_varying_box(tmp_path):
    rng = np.random.default_rng(0)
    xyz, _, _ = unwrapped_trajectory(rng, 40, BOX)
    T = 40
    boxes = np.tile(np.diag(BOX), (T, 1, 1)).astype(np.float32)
    boxes[20:] *= 1.02
    write_xtc(tmp_path / "t.xtc", wrap(xyz, BOX), box=boxes)
    with pytest.raises(ValueError, match="box changes"):
        process_trajectory(tmp_path / "t.xtc")


# --- process_system ----------------------------------------------------------------------

def test_process_system_stacks_and_matches_truth(synthetic_system):
    sys_dir, truth = synthetic_system
    data = process_system(sys_dir)
    assert isinstance(data, SystemData)
    assert data.name == "synth"
    assert data.displacements.shape == (3, 199, 3) and data.displacements.dtype == np.float32
    assert data.n_trajectories == 3 and data.n_steps == 199
    assert data.dt_ps == pytest.approx(10.0)
    assert data.box_nm.shape == (3, 3)
    np.testing.assert_allclose(data.displacements, truth, atol=DISP_TOL_NM)


def test_process_system_metadata_is_complete(synthetic_system):
    sys_dir, _ = synthetic_system
    m = process_system(sys_dir, name="custom").meta
    assert m["preprocess_version"] == PREPROCESS_VERSION
    assert m["files"] == ["traj001.xtc", "traj002.xtc", "traj003.xtc"]
    assert len(m["md5"]) == 3 and all(len(h) == 32 for h in m["md5"])
    assert m["n_trajectories"] == 3 and m["n_frames"] == 200 and m["n_steps"] == 199
    assert m["dt_ps"] == pytest.approx(10.0)
    assert m["water_masses"] == [16.0, 1.008, 1.008]
    assert m["max_step_fraction"] == 0.45
    assert m["box_constant_across_trajectories"] is True
    assert 0 < m["max_step_nm"] < 0.45 * BOX[0]
    assert m["units"] == {"length": "nm", "time": "ps"}


def test_process_system_rejects_inconsistent_frame_counts(synthetic_system):
    sys_dir, _ = synthetic_system
    rng = np.random.default_rng(9)
    xyz, _, _ = unwrapped_trajectory(rng, 150, BOX)
    write_xtc(sys_dir / "trajectory" / "traj004.xtc", wrap(xyz, BOX), box=BOX)
    with pytest.raises(ValueError, match="frame counts"):
        process_system(sys_dir)


def test_process_system_rejects_inconsistent_dt(synthetic_system):
    sys_dir, _ = synthetic_system
    rng = np.random.default_rng(9)
    xyz, _, _ = unwrapped_trajectory(rng, 200, BOX)
    write_xtc(sys_dir / "trajectory" / "traj004.xtc", wrap(xyz, BOX), box=BOX, dt=20.0)
    with pytest.raises(ValueError, match="sampling intervals"):
        process_system(sys_dir)


def test_process_system_records_differing_boxes(synthetic_system):
    sys_dir, _ = synthetic_system
    rng = np.random.default_rng(9)
    big = BOX * 1.05
    xyz, _, _ = unwrapped_trajectory(rng, 200, big)
    write_xtc(sys_dir / "trajectory" / "traj004.xtc", wrap(xyz, big), box=big)
    data = process_system(sys_dir)
    assert data.meta["box_constant_across_trajectories"] is False
    np.testing.assert_allclose(data.box_nm[3], big, atol=1e-5)


# --- cache --------------------------------------------------------------------------------

def test_cache_round_trip_is_exact(synthetic_system, tmp_path):
    sys_dir, _ = synthetic_system
    cache = tmp_path / "cache" / "synth.npz"
    data = process_system(sys_dir, cache=cache)
    assert cache.is_file()
    loaded = SystemData.load(cache)
    assert loaded.name == data.name and loaded.dt_ps == data.dt_ps
    np.testing.assert_array_equal(loaded.displacements, data.displacements)
    np.testing.assert_array_equal(loaded.box_nm, data.box_nm)
    assert loaded.meta == data.meta


def test_cache_is_reused_when_sources_unchanged(synthetic_system, tmp_path, monkeypatch):
    sys_dir, _ = synthetic_system
    cache = tmp_path / "synth.npz"
    process_system(sys_dir, cache=cache)
    calls = []
    import mdgan.data.preprocess as mod
    monkeypatch.setattr(mod, "process_trajectory", lambda p: calls.append(p) or (_ for _ in ()).throw(AssertionError))
    process_system(sys_dir, cache=cache)
    assert calls == []


def test_cache_is_rebuilt_when_a_source_changes(synthetic_system, tmp_path):
    sys_dir, _ = synthetic_system
    cache = tmp_path / "synth.npz"
    first = process_system(sys_dir, cache=cache)
    rng = np.random.default_rng(77)
    xyz, _, _ = unwrapped_trajectory(rng, 200, BOX)
    write_xtc(sys_dir / "trajectory" / "traj002.xtc", wrap(xyz, BOX), box=BOX)     # overwrite one source
    second = process_system(sys_dir, cache=cache)
    assert second.meta["md5"][1] != first.meta["md5"][1]
    assert not np.array_equal(second.displacements[1], first.displacements[1])


def test_cache_is_rebuilt_when_version_changes(synthetic_system, tmp_path, monkeypatch):
    sys_dir, _ = synthetic_system
    cache = tmp_path / "synth.npz"
    process_system(sys_dir, cache=cache)
    import mdgan.data.preprocess as mod
    monkeypatch.setattr(mod, "PREPROCESS_VERSION", PREPROCESS_VERSION + 1)
    assert process_system(sys_dir, cache=cache).meta["preprocess_version"] == PREPROCESS_VERSION + 1


def test_force_reprocesses(synthetic_system, tmp_path, monkeypatch):
    sys_dir, _ = synthetic_system
    cache = tmp_path / "synth.npz"
    process_system(sys_dir, cache=cache)
    calls = []
    import mdgan.data.preprocess as mod
    real = mod.process_trajectory
    monkeypatch.setattr(mod, "process_trajectory", lambda p: calls.append(p) or real(p))
    process_system(sys_dir, cache=cache, force=True)
    assert len(calls) == 3


def test_cli(synthetic_system, tmp_path, capsys):
    sys_dir, _ = synthetic_system
    out = tmp_path / "cli.npz"
    assert main([str(sys_dir), "--out", str(out)]) == 0
    assert out.is_file()
    assert "synth: 3 trajectories x 199 steps" in capsys.readouterr().out


# --- real data (opt-in) -------------------------------------------------------------------

@pytest.mark.real_data
def test_real_pe_system(pe):
    data, _ = pe
    assert data.displacements.shape == (60, 8000, 3)
    assert data.dt_ps == pytest.approx(10.0)
    assert data.meta["box_constant_across_trajectories"] is False     # each run NPT-equilibrated separately
    assert (data.box_nm > 4.5).all() and (data.box_nm < 5.0).all()
    assert 1.8 < data.meta["max_step_nm"] < 1.9                        # the known largest hop
    assert np.isfinite(data.displacements).all()