"""Shared test configuration.

SMALL: the one set of tiny model dimensions used by every model test. M = 8 and
C = 3 are deliberately different so that a (B, T, C) <-> (B, C, T) transpose
bug cannot pass a shape check by accident; hidden_dim = 24 keeps tests fast.

Determinism policy (docs/testing.md): every test runs under a seed derived from
its own node id, so no assertion depends on an unseeded RNG. Tests that need a
specific seed call torch.manual_seed / np.random.default_rng themselves.

Data fixtures: `synthetic_system` builds a Zenodo-layout system directory of
wrapped rigid-water trajectories with known displacements. `pe_dir` / `pe` give
the real PE system when MDGAN_PE_DIR points at MD_data/PE and skip otherwise;
`pe` is processed once per session.
"""

import os
import zlib
from pathlib import Path

import numpy as np
import pytest
import torch

SMALL = dict(latent_dim=7, noise_dim=3, block_len=8, n_channels=3, hidden_dim=24)


@pytest.fixture(autouse=True)
def _seed_everything(request):
    seed = zlib.crc32(request.node.nodeid.encode()) & 0xFFFF_FFFF
    torch.manual_seed(seed)
    np.random.seed(seed % (2**32 - 1))


@pytest.fixture
def gz():
    from mdgan.models.latent_transition import LatentTransitionGenerator
    torch.manual_seed(717)
    return LatentTransitionGenerator(latent_dim=SMALL["latent_dim"], noise_dim=SMALL["noise_dim"],
                                     hidden_dim=SMALL["hidden_dim"])


@pytest.fixture
def gy():
    from mdgan.models.trajectory_generator import TrajectoryGenerator
    torch.manual_seed(717)
    return TrajectoryGenerator(**SMALL)


@pytest.fixture
def dg():
    from mdgan.models.critic import Critic
    torch.manual_seed(717)
    return Critic(block_len=SMALL["block_len"], n_channels=SMALL["n_channels"], hidden_dim=SMALL["hidden_dim"])


@pytest.fixture
def three_nets(gz, gy, dg):
    return gz, gy, dg


# --- data fixtures -------------------------------------------------------------------------

SYNTH_BOX = np.array([4.75, 4.75, 4.75])


@pytest.fixture
def synthetic_system(tmp_path):
    """MD_data/synth/{msd.out, trajectory/traj001..003.xtc} with wrapped coordinates.

    Returns (system_dir, true_displacements (3, 199, 3)). Junk files that discovery
    must ignore are included.
    """
    pytest.importorskip("mdtraj")
    from synthetic_water import unwrapped_trajectory, wrap, write_xtc
    sys_dir = tmp_path / "MD_data" / "synth"
    (sys_dir / "trajectory").mkdir(parents=True)
    (sys_dir / "msd.out").write_text("# placeholder\n")
    truth = []
    for i in range(1, 4):
        xyz, _, steps = unwrapped_trajectory(np.random.default_rng(i), 200, SYNTH_BOX, step_std=0.15)
        write_xtc(sys_dir / "trajectory" / f"traj{i:03d}.xtc", wrap(xyz, SYNTH_BOX), box=SYNTH_BOX)
        truth.append(steps)
    (sys_dir / "trajectory" / "._traj001.xtc").write_bytes(b"\x00\x05\x16\x07")
    (sys_dir / "trajectory" / "notes.txt").write_text("ignore me")
    return sys_dir, np.stack(truth)


@pytest.fixture(scope="session")
def pe_dir():
    """MD_data/PE from MDGAN_PE_DIR; skips when unset, fails loudly when wrong."""
    if "MDGAN_PE_DIR" not in os.environ:
        pytest.skip("set MDGAN_PE_DIR to MD_data/PE")
    d = Path(os.environ["MDGAN_PE_DIR"])
    if not (d / "trajectory").is_dir() or not (d / "msd.out").is_file():
        raise FileNotFoundError(f"MDGAN_PE_DIR={d} must contain trajectory/ and msd.out")
    return d


@pytest.fixture(scope="session")
def pe(pe_dir, tmp_path_factory):
    """(SystemData, msd.out array) for the real PE system, processed once per session."""
    pytest.importorskip("mdtraj")
    from mdgan.data.preprocess import process_system
    system = process_system(pe_dir, cache=tmp_path_factory.mktemp("pe") / "PE.npz")
    return system, np.loadtxt(pe_dir / "msd.out")