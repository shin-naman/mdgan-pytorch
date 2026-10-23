"""Unit tests for mdgan.data.xtc.

Part 1 tests the validation and the RawTrajectory container directly on
in-memory arrays: no files, no mdtraj. Part 2 writes small synthetic .xtc
files with mdtraj into tmp_path and reads them back (skipped if mdtraj is
absent). Part 3 uses the `pe_dir` fixture (real_data; skips unless MDGAN_PE_DIR is set).

Run: python -m pytest tests/test_xtc.py -v
"""

import numpy as np
import pytest

from mdgan.data.xtc import (
    ORTHORHOMBIC_TOL_NM,
    TIME_TOL_PS,
    RawTrajectory,
    _validate,
    read_xtc,
)


def arrays(n_frames=50, n_atoms=3, dt=10.0, L=(4.7, 4.7, 4.7), seed=0):
    """In-memory stand-in for what XTCTrajectoryFile.read() returns."""
    rng = np.random.default_rng(seed)
    xyz = rng.uniform(0, 4.0, size=(n_frames, n_atoms, 3)).astype(np.float32)
    time = (np.arange(n_frames) * dt).astype(np.float32)
    box = np.tile(np.diag(L), (n_frames, 1, 1)).astype(np.float32)
    return xyz, time, box


# =============================================================================
# Part 1: _validate and RawTrajectory, no files
# =============================================================================

def test_validate_builds_record_with_float64_and_diagonal_box():
    xyz, time, box = arrays(L=(3.0, 5.0, 7.0))
    raw = _validate(xyz, time, box, "mem")
    assert isinstance(raw, RawTrajectory)
    assert raw.xyz.dtype == raw.time_ps.dtype == raw.box_nm.dtype == np.float64
    assert raw.xyz.shape == (50, 3, 3) and raw.box_nm.shape == (50, 3)
    np.testing.assert_allclose(raw.box_nm[0], [3.0, 5.0, 7.0], atol=1e-6)
    np.testing.assert_allclose(raw.xyz, xyz, atol=1e-6)
    assert raw.source == "mem"


def test_properties():
    xyz, time, box = arrays(n_frames=12, n_atoms=7, dt=2.5)
    raw = _validate(xyz, time, box, "mem")
    assert raw.n_frames == 12 and raw.n_atoms == 7
    assert raw.dt_ps == pytest.approx(2.5)
    assert raw.box_is_constant


def test_box_is_constant_false_when_box_changes():
    xyz, time, box = arrays()
    box[25:] *= 1.01
    raw = _validate(xyz, time, box, "mem")
    assert not raw.box_is_constant
    assert raw.box_nm[30, 0] > raw.box_nm[0, 0]


def test_record_is_frozen():
    raw = _validate(*arrays(), "mem")
    with pytest.raises(Exception):
        raw.xyz = None


def test_many_atoms_accepted():
    """The reader is generic; the atom-count contract belongs to water.py."""
    assert _validate(*arrays(n_atoms=500), "mem").n_atoms == 500


# --- check 3: sane data ---------------------------------------------------------------

@pytest.mark.parametrize("shape", [(50, 3), (50, 3, 2), (50, 3, 3, 1)])
def test_bad_coordinate_shape_rejected(shape):
    xyz, time, box = arrays()
    with pytest.raises(ValueError, match=r"\(T, A, 3\)"):
        _validate(np.zeros(shape, np.float32), time, box, "mem")


def test_single_frame_rejected():
    xyz, time, box = arrays(n_frames=1)
    with pytest.raises(ValueError, match="two frames"):
        _validate(xyz, time, box, "mem")


def test_non_finite_coordinates_rejected():
    xyz, time, box = arrays()
    xyz[3, 1, 2] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        _validate(xyz, time, box, "mem")


def test_non_positive_box_length_rejected():
    xyz, time, box = arrays()
    box[:, 1, 1] = 0.0
    with pytest.raises(ValueError, match="non-positive"):
        _validate(xyz, time, box, "mem")


# --- check 2: uniform increasing time -------------------------------------------------

def test_time_length_mismatch_rejected():
    xyz, time, box = arrays()
    with pytest.raises(ValueError, match="time array"):
        _validate(xyz, time[:-1], box, "mem")


def test_non_uniform_time_rejected():
    xyz, time, box = arrays()
    time[10:] += 5.0
    with pytest.raises(ValueError, match="non-uniform"):
        _validate(xyz, time, box, "mem")


def test_repeated_time_stamp_rejected():
    xyz, time, box = arrays()
    time[10] = time[9]
    with pytest.raises(ValueError, match="increasing"):
        _validate(xyz, time, box, "mem")


def test_decreasing_time_rejected():
    xyz, time, box = arrays()
    with pytest.raises(ValueError, match="increasing"):
        _validate(xyz, time[::-1].copy(), box, "mem")


def test_time_jitter_within_tolerance_accepted():
    xyz, time, box = arrays()
    time = time.astype(np.float64) + np.random.default_rng(1).uniform(-TIME_TOL_PS / 3, TIME_TOL_PS / 3, 50)
    time.sort()
    _validate(xyz, time, box, "mem")


# --- check 1: orthorhombic box ---------------------------------------------------------

def test_triclinic_box_rejected():
    xyz, time, box = arrays()
    box[:, 1, 0] = 0.5                # tilt: box vector b has an x component
    with pytest.raises(ValueError, match="orthorhombic"):
        _validate(xyz, time, box, "mem")


def test_single_tilted_frame_is_enough_to_reject():
    xyz, time, box = arrays()
    box[17, 2, 0] = 0.01
    with pytest.raises(ValueError, match="orthorhombic"):
        _validate(xyz, time, box, "mem")


def test_float32_ulp_jitter_still_counts_as_constant_box():
    """XTC stores boxes as float32. One ulp at 4.75 nm is ~4.8e-7 nm, just below
    ORTHORHOMBIC_TOL_NM = 1e-6. A real file whose box differs only by float32
    rounding must not be reported as a varying box (which preprocess rejects)."""
    xyz, time, box = arrays()
    L = np.float32(4.7)
    jitter = np.where(np.arange(50) % 2 == 0, L, np.nextafter(L, np.float32(10)))   # alternate by one ulp
    box[:, 0, 0] = jitter
    raw = _validate(xyz, time, box, "mem")
    assert np.ptp(raw.box_nm[:, 0]) > 0                   # the jitter is really there
    assert raw.box_is_constant
    box[25:, 0, 0] = L * np.float32(1.0001)               # a genuine 0.5 pm-per-nm change
    assert not _validate(xyz, time, box, "mem").box_is_constant


def test_off_diagonal_within_tolerance_accepted():
    xyz, time, box = arrays()
    box[:, 0, 1] = ORTHORHOMBIC_TOL_NM / 10
    _validate(xyz, time, box, "mem")


def test_bad_box_shape_rejected():
    xyz, time, box = arrays()
    with pytest.raises(ValueError, match=r"\(T, 3, 3\)"):
        _validate(xyz, time, box[:, :, 0], "mem")


# =============================================================================
# Part 2: real .xtc files written with mdtraj
# =============================================================================

from synthetic_water import write_xtc  # noqa: E402

XTC_PRECISION_NM = 1e-3   # the format stores coordinates with 3 decimals


def test_file_round_trip(tmp_path):
    pytest.importorskip("mdtraj")
    xyz, time, box = arrays()
    p = tmp_path / "t.xtc"
    write_xtc(p, xyz, time=time, box=box)
    raw = read_xtc(p)
    assert raw.n_frames == 50 and raw.n_atoms == 3 and raw.dt_ps == pytest.approx(10.0)
    np.testing.assert_allclose(raw.xyz, xyz, atol=XTC_PRECISION_NM)
    np.testing.assert_allclose(raw.time_ps, time, atol=1e-4)
    np.testing.assert_allclose(raw.box_nm, 4.7, atol=1e-5)
    assert raw.box_is_constant and raw.source == str(p)


def test_file_validation_errors_carry_the_path(tmp_path):
    pytest.importorskip("mdtraj")
    xyz, time, box = arrays()
    box[:, 1, 0] = 0.5
    p = tmp_path / "tilted.xtc"
    write_xtc(p, xyz, time=time, box=box)
    with pytest.raises(ValueError, match="tilted.xtc"):
        read_xtc(p)


def test_missing_file():
    pytest.importorskip("mdtraj")
    with pytest.raises(FileNotFoundError):
        read_xtc("/nonexistent/traj.xtc")


# =============================================================================
# Part 3: real data (opt-in)
# =============================================================================

@pytest.mark.real_data
def test_real_pe_trajectory_facts(pe_dir):
    """Facts established by inspection and relied on downstream."""
    raw = read_xtc(pe_dir / "trajectory" / "traj001.xtc")
    assert raw.n_frames == 8001 and raw.n_atoms == 3
    assert raw.dt_ps == pytest.approx(10.0)
    assert raw.time_ps[0] == 0.0 and raw.time_ps[-1] == pytest.approx(80000.0)
    assert raw.box_is_constant
    np.testing.assert_allclose(raw.box_nm[0], raw.box_nm[0, 0], rtol=1e-6)   # cubic
    assert 4.5 < raw.box_nm[0, 0] < 5.0