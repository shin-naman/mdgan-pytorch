"""Unit tests for mdgan.data.dataset.

A synthetic SystemData is built in memory. Its displacements carry a HUGE
value beyond the training region so that any leak of the reference region
into windows or sigma is unmistakable.

Run: python -m pytest tests/test_dataset.py -v
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


# =============================================================================
# Window offsets (Config.window_offset_max_steps)

K = 7


def test_position_rms_ratio_is_the_pooled_ratio_of_the_offset_zero_windows():
    sys_ = make_system()
    w = sys_.displacements[:, :2 * M].astype(np.float64)
    expected = np.sqrt((np.cumsum(w, axis=1) ** 2).mean()) / np.sqrt((w ** 2).mean())
    for kw in ({}, {"scale_inputs": False}, {"window_offset_max_steps": K}):
        ds = WindowDataset(sys_, block_len=M, step_skip=1, **kw)
        assert ds.position_rms_ratio == pytest.approx(expected, rel=1e-12), kw     # scale-free; offset-0 windows
        assert "position_rms_ratio" not in ds.metadata


def test_offset_zero_is_the_paper_dataset_bit_for_bit():
    a = WindowDataset(make_system(), block_len=M, step_skip=1)
    b = WindowDataset(make_system(), block_len=M, step_skip=1, window_offset_max_steps=0)
    assert torch.equal(a.windows, b.windows) and a.metadata == b.metadata
    assert "window_offset_max_steps" not in a.metadata                    # old runs' metadata unchanged
    ga, gb = torch.Generator().manual_seed(3), torch.Generator().manual_seed(3)
    assert torch.equal(a.sample(32, ga), b.sample(32, gb))
    assert torch.equal(ga.get_state(), gb.get_state())                   # nothing extra drawn


def test_sigma_and_offset_zero_windows_unchanged_by_offsets():
    a = WindowDataset(make_system(), block_len=M, step_skip=1)
    b = WindowDataset(make_system(), block_len=M, step_skip=1, window_offset_max_steps=K)
    assert b.sigma_nm == a.sigma_nm and b.window_std_nm == a.window_std_nm
    assert torch.equal(a.windows, b.windows)


def test_offset_region_and_metadata():
    ds = WindowDataset(make_system(), block_len=M, step_skip=1, window_offset_max_steps=K)
    assert ds.region_steps == 2 * M + K and ds.metadata["window_offset_max_steps"] == K
    ds = WindowDataset(make_system(), block_len=M, step_skip=NS, phase_mode="all", window_offset_max_steps=K)
    assert ds.region_steps == (2 * M + K) * NS + NS - 1


def test_sampled_windows_are_trajectory_slices_at_the_drawn_offsets():
    """Re-derive every sampled window from the indices and offsets the same generator draws."""
    sys_ = make_system()
    ds = WindowDataset(sys_, block_len=M, step_skip=1, window_offset_max_steps=K)
    out = ds.sample(64, torch.Generator().manual_seed(9))
    g = torch.Generator().manual_seed(9)
    idx = torch.randint(0, ds.n_windows, (64,), generator=g)
    off = torch.randint(0, K + 1, (64,), generator=g)
    d = sys_.displacements.astype(np.float64)
    for b in range(64):
        i, o = int(idx[b]), int(off[b])
        expected = (d[i, o:o + 2 * M] / ds.sigma_nm).astype(np.float32)
        np.testing.assert_array_equal(out[b].numpy(), expected)


def test_offsets_cover_zero_to_max_uniformly():
    """Recover each sampled window's offset from the window itself (match against the trajectory)."""
    sys_ = make_system()
    ds = WindowDataset(sys_, block_len=M, step_skip=1, window_offset_max_steps=K)
    out = ds.sample(1600, torch.Generator().manual_seed(1)).numpy()
    d = (sys_.displacements.astype(np.float64) / ds.sigma_nm).astype(np.float32)
    cand = np.stack([d[:, o:o + 2 * M] for o in range(K + 1)], axis=1)          # (W, K+1, 2M, C)
    found = []
    for w in out:
        hits = np.argwhere((cand == w).all(axis=(2, 3)))
        assert len(hits) >= 1
        found.append(int(hits[0, 1]))
    counts = np.bincount(found, minlength=K + 1)
    assert min(found) == 0 and max(found) == K and counts.min() > 0.6 * 1600 / (K + 1)


def test_offset_sampling_reproducible_and_leaves_global_rng():
    ds = WindowDataset(make_system(), block_len=M, step_skip=1, window_offset_max_steps=K)
    before = torch.get_rng_state().clone()
    a, b = ds.sample(16, torch.Generator().manual_seed(2)), ds.sample(16, torch.Generator().manual_seed(2))
    assert torch.equal(a, b) and torch.equal(before, torch.get_rng_state())


@pytest.mark.parametrize("bad", [-1, 2.5])
def test_offset_maximum_must_be_a_non_negative_integer(bad):
    with pytest.raises(ValueError, match="window_offset_max_steps"):
        WindowDataset(make_system(), block_len=M, step_skip=1, window_offset_max_steps=bad)


def test_offset_region_too_long_rejected():
    with pytest.raises(ValueError, match="training region"):
        WindowDataset(make_system(n_steps=2 * M + K - 1), block_len=M, step_skip=1, window_offset_max_steps=K)


# =============================================================================
# Held-out trajectories (Config.holdout_trajectories)

HOLD = (1, 3)


def _kept_system(sys_, hold):
    keep = [i for i in range(sys_.n_trajectories) if i not in hold]
    return SystemData(name=sys_.name, displacements=sys_.displacements[keep], dt_ps=sys_.dt_ps,
                      box_nm=sys_.box_nm[keep], meta=sys_.meta)


def test_empty_holdout_is_the_current_dataset_bit_for_bit():
    a = WindowDataset(make_system(), block_len=M, step_skip=1)
    b = WindowDataset(make_system(), block_len=M, step_skip=1, holdout_trajectories=())
    assert torch.equal(a.windows, b.windows) and a.metadata == b.metadata and a.sigma_nm == b.sigma_nm
    assert "holdout_trajectories" not in a.metadata and a.n_trajectories == 5
    ga, gb = torch.Generator().manual_seed(4), torch.Generator().manual_seed(4)
    assert torch.equal(a.sample(32, ga), b.sample(32, gb)) and torch.equal(ga.get_state(), gb.get_state())


def test_holdout_is_exactly_the_dataset_of_the_kept_trajectories():
    sys_ = make_system()
    held = WindowDataset(sys_, block_len=M, step_skip=1, holdout_trajectories=HOLD)
    kept = WindowDataset(_kept_system(sys_, HOLD), block_len=M, step_skip=1)
    assert torch.equal(held.windows, kept.windows) and held.sigma_nm == kept.sigma_nm   # sigma from the kept only
    assert held.position_rms_ratio == kept.position_rms_ratio
    assert held.n_windows == 3 and held.n_trajectories == 3
    assert held.metadata["holdout_trajectories"] == [1, 3]


def test_holdout_with_offsets_samples_only_kept_trajectories():
    sys_ = make_system()
    ds = WindowDataset(sys_, block_len=M, step_skip=1, window_offset_max_steps=K, holdout_trajectories=HOLD)
    out = ds.sample(400, torch.Generator().manual_seed(2)).numpy()
    d = (sys_.displacements.astype(np.float64) / ds.sigma_nm).astype(np.float32)
    owner = []
    for w in out:
        hits = [i for i in range(sys_.n_trajectories) for o in range(K + 1) if np.array_equal(d[i, o:o + 2 * M], w)]
        assert hits
        owner.append(hits[0])
    assert set(owner) == {0, 2, 4}                                        # never 1 or 3


@pytest.mark.parametrize("bad", [(1, 1), (-1,), (5,), (0, 1, 2, 3, 4)])
def test_invalid_holdout_rejected(bad):
    with pytest.raises(ValueError, match="holdout_trajectories"):
        WindowDataset(make_system(), block_len=M, step_skip=1, holdout_trajectories=bad)



# =============================================================================
# Critic windows of L blocks (Config.critic_blocks) and the correlation targets

L, KL = 4, 23                  # 4 blocks of M = 8 need K >= 16; region 2M + K = 39 steps, offsets 0..7


def test_two_blocks_is_the_current_dataset_bit_for_bit():
    for kw in ({}, {"window_offset_max_steps": K}):
        a = WindowDataset(make_system(), block_len=M, step_skip=1, **kw)
        b = WindowDataset(make_system(), block_len=M, step_skip=1, critic_blocks=2, **kw)
        assert torch.equal(a.windows, b.windows) and a.metadata == b.metadata and "critic_blocks" not in a.metadata
        ga, gb = torch.Generator().manual_seed(5), torch.Generator().manual_seed(5)
        assert torch.equal(a.sample(32, ga), b.sample(32, gb)) and torch.equal(ga.get_state(), gb.get_state())


def test_long_windows_leave_sigma_region_and_offset_zero_windows_unchanged():
    a = WindowDataset(make_system(), block_len=M, step_skip=1, window_offset_max_steps=KL)
    b = WindowDataset(make_system(), block_len=M, step_skip=1, window_offset_max_steps=KL, critic_blocks=L)
    assert b.sigma_nm == a.sigma_nm and b.region_steps == a.region_steps == 2 * M + KL
    assert torch.equal(a.windows, b.windows) and b.position_rms_ratio == a.position_rms_ratio
    assert b.metadata == {**a.metadata, "critic_blocks": L} and b.sample_len == L * M and a.sample_len == 2 * M


def test_long_windows_are_trajectory_slices_inside_the_region():
    """Re-derive every sampled window from the indices and offsets the same generator draws; the offsets
    stop at 2M + K - L*M, so no window leaves the training region (the system holds LEAK beyond it)."""
    sys_ = make_system()
    sys_.displacements[:, 2 * M + KL:] = LEAK
    sys_.displacements[:, :2 * M + KL] = np.random.default_rng(3).normal(scale=0.17, size=(5, 2 * M + KL, 3))
    ds = WindowDataset(sys_, block_len=M, step_skip=1, window_offset_max_steps=KL, critic_blocks=L)
    out = ds.sample(400, torch.Generator().manual_seed(9))
    assert out.shape == (400, L * M, 3) and out.abs().max() < 100
    g = torch.Generator().manual_seed(9)
    idx = torch.randint(0, ds.n_windows, (400,), generator=g)
    off = torch.randint(0, 2 * M + KL - L * M + 1, (400,), generator=g)
    assert int(off.min()) == 0 and int(off.max()) == 2 * M + KL - L * M == 7
    d = sys_.displacements.astype(np.float64)
    for b in range(400):
        i, o = int(idx[b]), int(off[b])
        np.testing.assert_array_equal(out[b].numpy(), (d[i, o:o + L * M] / ds.sigma_nm).astype(np.float32))


def test_long_windows_respect_the_holdout():
    sys_ = make_system()
    sys_.displacements[:, :2 * M + KL] = np.random.default_rng(3).normal(scale=0.17, size=(5, 2 * M + KL, 3))
    ds = WindowDataset(sys_, block_len=M, step_skip=1, window_offset_max_steps=KL, holdout_trajectories=HOLD,
                       critic_blocks=L)
    out = ds.sample(200, torch.Generator().manual_seed(2)).numpy()
    d = (sys_.displacements.astype(np.float64) / ds.sigma_nm).astype(np.float32)
    owner = {i for w in out for i in range(5) for o in range(8) if np.array_equal(d[i, o:o + L * M], w)}
    assert owner == {0, 2, 4}


@pytest.mark.parametrize("kw", [dict(critic_blocks=1), dict(critic_blocks=3), dict(critic_blocks=2.0),
                                dict(critic_blocks=L, window_offset_max_steps=(L - 2) * M - 1)])
def test_long_windows_need_enough_offsets(kw):
    with pytest.raises(ValueError, match="critic_blocks"):
        WindowDataset(make_system(), block_len=M, step_skip=1, **kw)
    WindowDataset(make_system(), block_len=M, step_skip=1, critic_blocks=L, window_offset_max_steps=(L - 2) * M)


def _ar_system(phi, n_traj=40, n_blocks=60, seed=0):
    """Displacements whose M-step block sums are an AR(1) chain with coefficient phi: rho_k = phi^k.
    Each block's steps are that block's sum spread evenly plus noise that sums to zero within the block."""
    rng = np.random.default_rng(seed)
    s = np.zeros((n_traj, n_blocks, 3))
    s[:, 0] = rng.normal(size=(n_traj, 3))
    for t in range(1, n_blocks):
        s[:, t] = phi * s[:, t - 1] + np.sqrt(1 - phi ** 2) * rng.normal(size=(n_traj, 3))
    e = rng.normal(size=(n_traj, n_blocks, M, 3))
    d = (s[:, :, None, :] / M + (e - e.mean(axis=2, keepdims=True))).reshape(n_traj, n_blocks * M, 3)
    return SystemData(name="ar", displacements=d.astype(np.float32), dt_ps=10.0,
                      box_nm=np.full((n_traj, 3), 4.75), meta={})


def test_block_sum_correlations_equal_an_independent_computation():
    """np.corrcoef on explicitly summed blocks, per grid position, averaged over the M positions."""
    sys_ = _ar_system(-0.4)
    ds = WindowDataset(sys_, block_len=M, step_skip=1, window_offset_max_steps=58 * M)       # the whole system
    d = sys_.displacements.astype(np.float64)
    expected = np.zeros(3)
    for o in range(M):
        nb = (d.shape[1] - o) // M
        sums = np.stack([d[:, o + b * M:o + (b + 1) * M].sum(axis=1) for b in range(nb)], axis=1)   # (N, nb, C)
        expected += [np.corrcoef(sums[:, :-k].ravel(), sums[:, k:].ravel())[0, 1] for k in (1, 2, 3)]
    got = ds.block_sum_correlations(4)
    assert got.shape == (3,)
    np.testing.assert_allclose(got, expected / M, rtol=0, atol=1e-6)        # float32 windows against float64


@pytest.mark.statistical
def test_block_sum_correlations_recover_planted_values():
    """Independent steps: rho_k = 0 (standard error near 1 / sqrt(200 * 3 * 28) = 0.008 per grid position).
    Steps mu_i + eps with one mu ~ N(0, a^2) per trajectory and axis: a block sum is M mu + sum(eps), so
    rho_k = M a^2 / (M a^2 + 1) at every lag and grid position; 0.5 for a^2 = 1 / M. 600 values of mu put
    the standard error near 0.03. Tolerances are 4 standard errors."""
    rng = np.random.default_rng(0)
    eps = rng.normal(size=(200, 30 * M, 3))
    mu = rng.normal(scale=np.sqrt(1 / M), size=(200, 1, 3))
    for d, expected, tol in ((eps, 0.0, 0.03), (eps + mu, 0.5, 0.12)):
        sys_ = SystemData(name="planted", displacements=d.astype(np.float32), dt_ps=10.0,
                          box_nm=np.full((200, 3), 4.75), meta={})
        ds = WindowDataset(sys_, block_len=M, step_skip=1, window_offset_max_steps=28 * M)
        np.testing.assert_allclose(ds.block_sum_correlations(6), np.full(5, expected), rtol=0, atol=tol)


def test_block_sum_correlations_use_training_trajectories_and_the_region_only():
    sys_ = _ar_system(-0.4)
    full = WindowDataset(sys_, block_len=M, step_skip=1, window_offset_max_steps=58 * M)
    held = WindowDataset(sys_, block_len=M, step_skip=1, window_offset_max_steps=58 * M, holdout_trajectories=(0, 1, 2))
    kept = WindowDataset(_kept_system(sys_, (0, 1, 2)), block_len=M, step_skip=1, window_offset_max_steps=58 * M)
    np.testing.assert_array_equal(held.block_sum_correlations(4), kept.block_sum_correlations(4))
    assert not np.array_equal(held.block_sum_correlations(4), full.block_sum_correlations(4))
    short = WindowDataset(sys_, block_len=M, step_skip=1, window_offset_max_steps=2 * M)       # 4 blocks of region
    assert not np.allclose(short.block_sum_correlations(4), full.block_sum_correlations(4))
    assert np.isfinite(short.block_sum_correlations(4)).all()
    assert np.isfinite(WindowDataset(sys_, block_len=M, step_skip=1).block_sum_correlations(2)).all()   # lag 1, K = 0


@pytest.mark.parametrize("n", [1, 5, 2.0])
def test_block_sum_correlations_need_the_blocks_in_the_region(n):
    ds = WindowDataset(make_system(), block_len=M, step_skip=1, window_offset_max_steps=2 * M)    # 4 blocks
    with pytest.raises(ValueError, match="n_blocks"):
        ds.block_sum_correlations(n)


# --- block-sum variance (the variance reward's target) ------------------------------------------------------

def test_block_sum_variance_equals_an_independent_computation():
    """np.var of explicitly summed blocks pooled over trajectories, blocks and axes, per grid position that holds
    n_blocks blocks, averaged over those positions; in the model's units (here nm, scale_inputs False)."""
    sys_ = _ar_system(-0.4)
    d = sys_.displacements.astype(np.float64)
    for k_max, n_blocks in ((58 * M, 4), (2 * M, 4), (0, 2)):           # whole system; exactly 4 blocks; the papers' 2
        ds = WindowDataset(sys_, block_len=M, step_skip=1, scale_inputs=False, window_offset_max_steps=k_max)
        span = 2 * M + k_max
        vals = []
        for o in range(M):
            nb = (span - o) // M
            if nb < n_blocks:
                break
            sums = np.stack([d[:, o + b * M:o + (b + 1) * M].sum(axis=1) for b in range(nb)], axis=1)
            vals.append(np.var(sums))
        assert len(vals) == (M if k_max == 58 * M else 1)
        np.testing.assert_allclose(ds.block_sum_variance(n_blocks), np.mean(vals), rtol=1e-5)    # float32 windows


@pytest.mark.statistical
def test_block_sum_variance_recovers_a_known_variance():
    """Independent normal steps of standard deviation s: a block sum of M steps has variance M s^2. 200 x 3
    series of 30 blocks give 18,000 sums per grid position, a relative standard error of sqrt(2 / 18,000) =
    1.1 %; the tolerance is 5 % (above 4 standard errors). With scaling, the same number divided by sigma^2."""
    s = 0.17
    d = np.random.default_rng(1).normal(scale=s, size=(200, 30 * M, 3))
    sys_ = SystemData(name="iid", displacements=d.astype(np.float32), dt_ps=10.0, box_nm=np.full((200, 3), 4.75), meta={})
    raw = WindowDataset(sys_, block_len=M, step_skip=1, scale_inputs=False, window_offset_max_steps=28 * M)
    assert raw.block_sum_variance(4) == pytest.approx(M * s ** 2, rel=0.05)
    scaled = WindowDataset(sys_, block_len=M, step_skip=1, window_offset_max_steps=28 * M)
    assert scaled.block_sum_variance(4) == pytest.approx(raw.block_sum_variance(4) / scaled.sigma_nm ** 2, rel=1e-5)


def test_block_sum_variance_uses_training_trajectories_and_the_region_only():
    sys_ = _ar_system(-0.4)
    full = WindowDataset(sys_, block_len=M, step_skip=1, window_offset_max_steps=58 * M)
    held = WindowDataset(sys_, block_len=M, step_skip=1, window_offset_max_steps=58 * M, holdout_trajectories=(0, 1, 2))
    kept = WindowDataset(_kept_system(sys_, (0, 1, 2)), block_len=M, step_skip=1, window_offset_max_steps=58 * M)
    assert held.block_sum_variance(4) == kept.block_sum_variance(4) != full.block_sum_variance(4)
    short = WindowDataset(sys_, block_len=M, step_skip=1, window_offset_max_steps=2 * M)
    assert short.block_sum_variance(4) != full.block_sum_variance(4) and short.block_sum_variance(4) > 0


@pytest.mark.parametrize("n", [1, 5, 2.0])
def test_block_sum_variance_needs_the_blocks_in_the_region(n):
    ds = WindowDataset(make_system(), block_len=M, step_skip=1, window_offset_max_steps=2 * M)    # 4 blocks
    with pytest.raises(ValueError, match="n_blocks"):
        ds.block_sum_variance(n)


# --- region start (not in the papers) ----------------------------------------------------------------------

def _plain_system(n_steps=200):
    """Normal steps everywhere (make_system fills its reference region with a constant)."""
    d = np.random.default_rng(3).normal(scale=0.17, size=(5, n_steps, 3)).astype(np.float32)
    return SystemData(name="plain", displacements=d, dt_ps=10.0, box_nm=np.full((5, 3), 4.75), meta={})


def test_region_start_zero_is_the_current_dataset_bit_for_bit():
    a = WindowDataset(make_system(), block_len=M, step_skip=NS, window_offset_max_steps=2 * M)
    b = WindowDataset(make_system(), block_len=M, step_skip=NS, window_offset_max_steps=2 * M, region_start_steps=0)
    assert a.metadata == b.metadata and "region_start_steps" not in b.metadata
    assert torch.equal(a.windows, b.windows) and torch.equal(a._series, b._series) and a.sigma_nm == b.sigma_nm


@pytest.mark.parametrize("ns", [1, NS])
def test_region_start_moves_the_whole_region(ns):
    """Starting s coarse steps in equals dropping the first s * n_s fine steps of every trajectory: same windows,
    series, sigma, position ratio and reward targets; only the metadata records the start."""
    sys_ = _plain_system()
    s, k = 5, 2 * M
    moved = WindowDataset(sys_, block_len=M, step_skip=ns, window_offset_max_steps=k, region_start_steps=s)
    cut = SystemData(name=sys_.name, displacements=sys_.displacements[:, s * ns:], dt_ps=sys_.dt_ps,
                     box_nm=sys_.box_nm, meta=sys_.meta)
    ref = WindowDataset(cut, block_len=M, step_skip=ns, window_offset_max_steps=k)
    assert torch.equal(moved.windows, ref.windows) and torch.equal(moved._series, ref._series)
    assert moved.sigma_nm == ref.sigma_nm and moved.position_rms_ratio == ref.position_rms_ratio
    np.testing.assert_array_equal(moved.block_sum_correlations(4), ref.block_sum_correlations(4))
    assert moved.block_sum_variance(4) == ref.block_sum_variance(4)
    assert moved.metadata == {**ref.metadata, "region_start_steps": s}
    origin = WindowDataset(sys_, block_len=M, step_skip=ns, window_offset_max_steps=k)
    assert not torch.equal(moved.windows, origin.windows)


def test_region_start_must_leave_room_for_the_region():
    sys_ = _plain_system()
    WindowDataset(sys_, block_len=M, step_skip=1, window_offset_max_steps=2 * M, region_start_steps=200 - 4 * M)
    with pytest.raises(ValueError, match="training region needs"):
        WindowDataset(sys_, block_len=M, step_skip=1, window_offset_max_steps=2 * M, region_start_steps=200 - 4 * M + 1)
    for bad in (-1, 2.5, True):
        with pytest.raises(ValueError, match="region_start_steps"):
            WindowDataset(sys_, block_len=M, step_skip=1, region_start_steps=bad)


# --- sub-block sums (the sub-block correlation reward) ---------------------------------------------------------

def _ar_sub_system(phi, steps, n_traj=40, n_sub=64, seed=0):
    """Displacements whose consecutive `steps`-step sub-block sums are an AR(1) chain with coefficient phi (rho_k =
    phi^k at the sub-block level): each sub-block's sum spread evenly plus noise that sums to zero inside it."""
    rng = np.random.default_rng(seed)
    s = np.zeros((n_traj, n_sub, 3))
    s[:, 0] = rng.normal(size=(n_traj, 3))
    for t in range(1, n_sub):
        s[:, t] = phi * s[:, t - 1] + np.sqrt(1 - phi ** 2) * rng.normal(size=(n_traj, 3))
    e = rng.normal(size=(n_traj, n_sub, steps, 3))
    d = (s[:, :, None, :] / steps + (e - e.mean(axis=2, keepdims=True))).reshape(n_traj, n_sub * steps, 3)
    return SystemData(name="ar_sub", displacements=d.astype(np.float32), dt_ps=10.0,
                      box_nm=np.full((n_traj, 3), 4.75), meta={})


def test_subblock_targets_equal_an_independent_computation():
    """np.corrcoef on explicitly summed sub-blocks, pairs within each trajectory, pooled; np.var for the variance;
    pair and sample counts by hand. Both the papers' region (K = 0) and a longer one."""
    s = 2
    sys_ = _ar_sub_system(-0.4, s)
    d = sys_.displacements.astype(np.float64)
    for k_max in (0, 2 * M):
        ds = WindowDataset(sys_, block_len=M, step_skip=1, scale_inputs=False, window_offset_max_steps=k_max)
        span = 2 * M + k_max
        sums = np.stack([d[:, i * s:(i + 1) * s].sum(axis=1) for i in range(span // s)], axis=1)     # (N, nb, C)
        nb = sums.shape[1]
        expected = [np.corrcoef(sums[:, :-k].ravel(), sums[:, k:].ravel())[0, 1] for k in range(1, 8)]
        np.testing.assert_allclose(ds.subblock_sum_correlations(s, 7), expected, rtol=0, atol=1e-6)
        assert ds.subblock_pair_counts(s, 7) == [40 * (nb - k) * 3 for k in range(1, 8)]
        v, n = ds.subblock_sum_variance(s)
        np.testing.assert_allclose(v, np.var(sums), rtol=1e-5)
        assert n == 40 * nb * 3


@pytest.mark.statistical
def test_subblock_targets_recover_a_planted_ar1():
    """Sub-block sums AR(1) with phi = -0.4: rho_k = (-0.4)^k. 200 x 3 series of 120 sub-blocks give at least
    200 * 3 * 113 = 67,800 pairs per lag, a standard error near 1 / sqrt(67,800) = 0.004; tolerance 0.02."""
    sys_ = _ar_sub_system(-0.4, 2, n_traj=200, n_sub=120)
    ds = WindowDataset(sys_, block_len=M, step_skip=1, window_offset_max_steps=120 * 2 - 2 * M)
    np.testing.assert_allclose(ds.subblock_sum_correlations(2, 7), (-0.4) ** np.arange(1, 8), rtol=0, atol=0.02)


def _block_lag1_from_subblocks(rho, q):
    """Block lag-1 correlation of sums of q consecutive sub-blocks from the sub-block correlations rho_1..rho_{2q-1}
    of a stationary series: Cov(B_j, B_j+1) = sum_d (q - |d - q|) gamma_d, Var(B) = q gamma_0 + 2 sum_{d<q} (q - d) gamma_d."""
    num = sum((q - abs(d - q)) * rho[d - 1] for d in range(1, 2 * q))
    den = q + 2 * sum((q - d) * rho[d - 1] for d in range(1, q))
    return num / den


@pytest.mark.statistical
@pytest.mark.parametrize("phi", [-0.4, 0.3])
def test_subblock_lags_1_to_7_fix_the_block_lag_1(phi):
    """With M = 4 s, the block lag-1 correlation of a stationary series is a fixed combination of its sub-block
    correlations at lags 1-7, so the sub-block reward constrains what the block reward does. Steps AR(1) with
    coefficient phi (stationary step by step, so every block grid sees the same statistics). 300 x 3 series of 40
    blocks give about 35,000 block pairs per grid (standard error near 0.005); both sides are estimates, and the
    tolerance 0.02 is four of those standard errors."""
    rng = np.random.default_rng(5)
    n_traj, n = 300, 40 * M
    d = np.zeros((n_traj, n, 3))
    d[:, 0] = rng.normal(size=(n_traj, 3))
    for t in range(1, n):
        d[:, t] = phi * d[:, t - 1] + np.sqrt(1 - phi ** 2) * rng.normal(size=(n_traj, 3))
    sys_ = SystemData(name="ar_step", displacements=d.astype(np.float32), dt_ps=10.0,
                      box_nm=np.full((n_traj, 3), 4.75), meta={})
    ds = WindowDataset(sys_, block_len=M, step_skip=1, window_offset_max_steps=n - 2 * M)
    implied = _block_lag1_from_subblocks(ds.subblock_sum_correlations(M // 4, 7), 4)
    assert implied == pytest.approx(ds.block_sum_correlations(2)[0], abs=0.02)
    assert abs(implied) > 0.03                                # a real correlation, not two zeros agreeing


@pytest.mark.parametrize("phi", [-0.6, -0.4, 0.3, 0.8])
def test_subblock_identity_is_exact_for_population_values(phi):
    """The same identity with no sampling: AR(1) steps have gamma_k = phi^|k|, so every sub-block and block
    covariance is a finite sum; the combination of the exact sub-block correlations equals the exact block lag 1."""
    s = M // 4
    cov = lambda a, b, n: sum(phi ** abs(i - j) for i in range(a, a + n) for j in range(b, b + n))   # noqa: E731
    rho_sub = [cov(0, d * s, s) / cov(0, 0, s) for d in range(1, 8)]
    assert _block_lag1_from_subblocks(rho_sub, 4) == pytest.approx(cov(0, M, M) / cov(0, 0, M), rel=1e-12)


def test_subblock_targets_use_training_trajectories_and_the_region_only():
    sys_ = _ar_sub_system(-0.4, 2)
    full = WindowDataset(sys_, block_len=M, step_skip=1, window_offset_max_steps=6 * M)
    held = WindowDataset(sys_, block_len=M, step_skip=1, window_offset_max_steps=6 * M, holdout_trajectories=(0, 1, 2))
    kept = WindowDataset(_kept_system(sys_, (0, 1, 2)), block_len=M, step_skip=1, window_offset_max_steps=6 * M)
    np.testing.assert_array_equal(held.subblock_sum_correlations(2, 7), kept.subblock_sum_correlations(2, 7))
    assert held.subblock_sum_variance(2) == kept.subblock_sum_variance(2) != full.subblock_sum_variance(2)
    short = WindowDataset(sys_, block_len=M, step_skip=1)
    assert not np.allclose(short.subblock_sum_correlations(2, 7), full.subblock_sum_correlations(2, 7))


@pytest.mark.parametrize("steps,lags", [(2, 8), (2, 0), (16, 1), (0, 1)])
def test_subblock_targets_need_the_sub_blocks(steps, lags):
    ds = WindowDataset(_ar_sub_system(-0.4, 2), block_len=M, step_skip=1)            # 16 steps: 8 sub-blocks of 2
    with pytest.raises(ValueError):
        ds.subblock_sum_correlations(steps, lags)
