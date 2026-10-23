"""Unit tests for mdgan.data.dataset.

A synthetic SystemData is built in memory. Its displacements carry a HUGE
value beyond the training region so that any leak of the reference region
into windows or sigma is unmistakable.

Run: python -m pytest tests/data/test_dataset.py -v
"""

import numpy as np
import pytest
import torch

from mdgan.data.dataset import PHASE_MODES, WindowDataset, coarsen
from mdgan.data.preprocess import SystemData

M, NS = 8, 3
LEAK = 1e6


def make_system(n_traj=5, n_steps=200, seed=0, dt=10.0):
    rng = np.random.default_rng(seed)
    d = rng.normal(scale=0.17, size=(n_traj, n_steps, 3)).astype(np.float32)
    region = 2 * M * NS + (NS - 1)                 # largest region any test uses
    d[:, region:] = LEAK                           # the reference region: must never be seen
    return SystemData(name="synth", displacements=d, dt_ps=dt,
                      box_nm=np.full((n_traj, 3), 4.75), meta={"md5": ["a" * 32] * n_traj})


# --- coarsen ---------------------------------------------------------------------------------

def test_coarsen_identity_for_step_skip_one():
    d = np.random.default_rng(1).normal(size=(2, 10, 3))
    np.testing.assert_array_equal(coarsen(d, 1), d)


def test_coarsen_equals_diff_of_subsampled_positions():
    d = np.random.default_rng(2).normal(size=(3, 50, 3))
    pos = np.concatenate([np.zeros((3, 1, 3)), np.cumsum(d, axis=1)], axis=1)   # (3, 51, 3)
    for phase in range(NS):
        sub = pos[:, phase::NS]                                                   # frames phase, phase+NS, ...
        np.testing.assert_allclose(coarsen(d, NS, phase), np.diff(sub, axis=1), atol=1e-12)


def test_coarsen_shape_and_hand_values():
    d = np.arange(2 * 7 * 3, dtype=float).reshape(2, 7, 3)
    c = coarsen(d, 3, phase=1)                      # steps 1..3 and 4..6 -> 2 coarse steps
    assert c.shape == (2, 2, 3)
    np.testing.assert_array_equal(c[0, 0], d[0, 1:4].sum(0))
    np.testing.assert_array_equal(c[1, 1], d[1, 4:7].sum(0))


@pytest.mark.parametrize("kw", [dict(step_skip=0), dict(step_skip=2.0), dict(step_skip=True),
                                dict(step_skip=3, phase=3), dict(step_skip=3, phase=-1)])
def test_coarsen_rejects_bad_arguments(kw):
    with pytest.raises(ValueError):
        coarsen(np.zeros((1, 10, 3)), **kw)


def test_coarsen_rejects_too_short():
    with pytest.raises(ValueError, match="too few"):
        coarsen(np.zeros((1, 2, 3)), 3)


# --- construction ----------------------------------------------------------------------------

def test_single_phase_counts_and_region():
    ds = WindowDataset(make_system(), block_len=M, step_skip=NS, phase_mode="single")
    assert ds.n_windows == 5 and ds.window_len == 2 * M
    assert ds.region_steps == 2 * M * NS
    assert ds.dt_ps == pytest.approx(10.0 * NS)
    assert ds.windows.shape == (5, 2 * M, 3) and ds.windows.dtype == torch.float32
    assert len(ds) == 5


def test_all_phases_counts_and_region():
    ds = WindowDataset(make_system(), block_len=M, step_skip=NS, phase_mode="all")
    assert ds.n_windows == 5 * NS
    assert ds.region_steps == 2 * M * NS + (NS - 1)


def test_phase_windows_are_shifted_by_exactly_one_fine_step():
    """With fine displacements d[t] = t (a ramp), the phase-p coarse step j is
    sum_{k<n_s} (p + j*n_s + k), so consecutive phases differ by exactly n_s per
    coarse step. Checks that phase p really starts p fine steps later, not just
    that the counts are right."""
    ns, n_traj = 2, 3
    ramp = np.arange(300, dtype=np.float32)[None, :, None].repeat(n_traj, 0).repeat(3, 2)
    sys_ = SystemData(name="ramp", displacements=ramp, dt_ps=10.0,
                      box_nm=np.full((n_traj, 3), 4.75), meta={})
    ds = WindowDataset(sys_, block_len=M, step_skip=ns, phase_mode="all")
    w = ds.windows.numpy().astype(np.float64) * ds.sigma_nm          # back to nm
    phase0, phase1 = w[:n_traj], w[n_traj:]
    j = np.arange(2 * M)[None, :, None]
    np.testing.assert_allclose(phase0, (2 * j * ns + 1) * np.ones_like(phase0), rtol=1e-5)
    np.testing.assert_allclose(phase1 - phase0, ns, rtol=1e-5)


def test_modes_identical_for_step_skip_one():
    a = WindowDataset(make_system(), block_len=M, step_skip=1, phase_mode="single")
    b = WindowDataset(make_system(), block_len=M, step_skip=1, phase_mode="all")
    assert a.n_windows == b.n_windows == 5
    torch.testing.assert_close(a.windows, b.windows, rtol=0, atol=0)
    assert a.sigma_nm == b.sigma_nm


def test_reference_region_never_leaks_into_windows_or_sigma():
    """The LEAK value (1e6) sits right after the region. Nothing may see it."""
    for mode in PHASE_MODES:
        ds = WindowDataset(make_system(), block_len=M, step_skip=NS, phase_mode=mode)
        assert ds.windows.abs().max().item() < 100
        assert ds.sigma_nm < 10


def test_windows_match_hand_built_coarsening_and_scaling():
    sys_ = make_system()
    ds = WindowDataset(sys_, block_len=M, step_skip=NS, phase_mode="all")
    region = sys_.displacements[:, :ds.region_steps].astype(np.float64)
    expected = np.concatenate([coarsen(region, NS, p)[:, :2 * M] for p in range(NS)], axis=0)
    assert ds.sigma_nm == pytest.approx(expected.std())
    torch.testing.assert_close(ds.windows, torch.from_numpy((expected / expected.std()).astype(np.float32)),
                               rtol=1e-6, atol=1e-6)


def test_sigma_is_one_scalar_and_windows_have_unit_spread():
    ds = WindowDataset(make_system(n_steps=400), block_len=M, step_skip=NS)
    assert isinstance(ds.sigma_nm, float)
    assert ds.windows.std(unbiased=False).item() == pytest.approx(1.0, abs=1e-5)


def test_unscaled_ablation_keeps_nm_and_records_the_raw_spread():
    """scale_inputs=False: windows stay in nm, sigma_nm = 1, window_std_nm still recorded."""
    sys_ = make_system()
    scaled = WindowDataset(sys_, block_len=M, step_skip=NS)
    raw = WindowDataset(sys_, block_len=M, step_skip=NS, scale_inputs=False)
    assert raw.sigma_nm == 1.0 and raw.scale_inputs is False
    assert raw.window_std_nm == scaled.window_std_nm == scaled.sigma_nm
    torch.testing.assert_close(raw.windows, scaled.windows * scaled.sigma_nm, rtol=1e-6, atol=1e-7)
    assert raw.metadata["scale_inputs"] is False and raw.metadata["sigma_nm"] == 1.0


def test_sigma_depends_on_step_skip():
    """Coarse steps are sums of n_s fine steps, so their spread grows ~sqrt(n_s)."""
    s1 = WindowDataset(make_system(n_steps=400), block_len=M, step_skip=1).sigma_nm
    s3 = WindowDataset(make_system(n_steps=400), block_len=M, step_skip=3).sigma_nm
    assert s3 / s1 == pytest.approx(np.sqrt(3), rel=0.2)


def test_windows_property_returns_a_copy():
    ds = WindowDataset(make_system(), block_len=M, step_skip=NS)
    w = ds.windows
    w.zero_()
    assert ds.windows.abs().sum().item() > 0


def test_region_too_long_rejected():
    with pytest.raises(ValueError, match="training region"):
        WindowDataset(make_system(n_steps=40), block_len=M, step_skip=NS)


@pytest.mark.parametrize("kw", [dict(block_len=0), dict(step_skip=0), dict(block_len=2.0),
                                dict(phase_mode="some"), dict(step_skip=True)])
def test_bad_arguments_rejected(kw):
    with pytest.raises(ValueError):
        WindowDataset(make_system(), **{"block_len": M, "step_skip": NS, **kw})


def test_zero_spread_rejected():
    sys_ = make_system()
    sys_.displacements[:, :2 * M * NS + NS] = 0.0
    with pytest.raises(ValueError, match="spread"):
        WindowDataset(sys_, block_len=M, step_skip=NS)


# --- sampling ---------------------------------------------------------------------------------

def test_sample_shape_dtype_and_with_replacement():
    ds = WindowDataset(make_system(), block_len=M, step_skip=NS)          # 5 windows
    b = ds.sample(64, generator=torch.Generator().manual_seed(0))
    assert b.shape == (64, 2 * M, 3) and b.dtype == torch.float32
    # every drawn window is one of the five, and with 64 draws some repeat
    flat = ds.windows.flatten(1)
    for row in b.flatten(1):
        assert any(torch.equal(row, w) for w in flat)


def test_sample_reproducible_and_leaves_global_rng():
    ds = WindowDataset(make_system(), block_len=M, step_skip=NS)
    before = torch.get_rng_state().clone()
    a = ds.sample(16, generator=torch.Generator().manual_seed(3))
    b = ds.sample(16, generator=torch.Generator().manual_seed(3))
    c = ds.sample(16, generator=torch.Generator().manual_seed(4))
    assert torch.equal(before, torch.get_rng_state())
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert not torch.equal(a, c)


def test_sample_without_generator_uses_global_rng():
    ds = WindowDataset(make_system(), block_len=M, step_skip=NS)
    before = torch.get_rng_state().clone()
    ds.sample(4)
    assert not torch.equal(before, torch.get_rng_state())


def test_sample_covers_all_windows_eventually():
    ds = WindowDataset(make_system(), block_len=M, step_skip=NS, phase_mode="all")   # 15 windows
    b = ds.sample(2000, generator=torch.Generator().manual_seed(0))
    seen = {tuple(row.tolist()) for row in b.flatten(1)}
    assert len(seen) == ds.n_windows


def test_sample_rejects_bad_batch():
    ds = WindowDataset(make_system(), block_len=M, step_skip=NS)
    for bad in (0, -1, 2.0, True):
        with pytest.raises(ValueError):
            ds.sample(bad)


def test_sample_does_not_alias_internal_storage():
    ds = WindowDataset(make_system(), block_len=M, step_skip=NS)
    b = ds.sample(4, generator=torch.Generator().manual_seed(0))
    b.zero_()
    assert ds.windows.abs().sum().item() > 0


# --- metadata --------------------------------------------------------------------------------------

def test_metadata_is_complete_and_json_serializable():
    import json
    ds = WindowDataset(make_system(), block_len=M, step_skip=NS, phase_mode="all")
    m = ds.metadata
    assert m == {
        "system": "synth", "block_len": M, "step_skip": NS, "phase_mode": "all",
        "region_steps": 2 * M * NS + NS - 1, "window_len": 2 * M, "n_windows": 15,
        "n_trajectories": 5, "dt_ps": 30.0, "scale_inputs": True, "window_std_nm": ds.window_std_nm,
        "sigma_nm": ds.sigma_nm, "source_md5": ["a" * 32] * 5,
    }
    json.dumps(m)
    assert "sigma" in repr(ds)


# --- real data (opt-in) --------------------------------------------------------------------------------

@pytest.mark.real_data
def test_real_pe_training_set(pe):
    """2023 paper, PE: M = 64, n_s = 1 -> 1.28 ns input = one window per trajectory."""
    ds = WindowDataset(pe[0], block_len=64, step_skip=1)
    assert ds.n_windows == 60 and ds.window_len == 128 and ds.region_steps == 128
    assert ds.dt_ps == 10.0
    assert 0.15 < ds.sigma_nm < 0.20            # measured ~0.17 nm per-axis step std
    b = ds.sample(64, generator=torch.Generator().manual_seed(0))
    assert b.shape == (64, 128, 3)