"""Shared test configuration.

SMALL: the one set of tiny model dimensions used by every model test. M = 8 and
C = 3 are deliberately different so that a (B, T, C) <-> (B, C, T) transpose
bug cannot pass a shape check by accident; hidden_dim = 24 keeps tests fast.

Determinism policy (docs/testing.md): every test runs under a seed derived from
its own node id, so no assertion depends on an unseeded RNG. Tests that need a
specific seed call torch.manual_seed / np.random.default_rng themselves.
"""

import zlib

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