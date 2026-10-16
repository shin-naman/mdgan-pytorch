"""Intentionally strict RED tests: proposed fail-fast API requirements.

Unlike test_gz_contract.py, these tests MAY FAIL against a minimal working
G_z. That is their point. Decide whether these safeguards belong in G_z or
at the dataset/training boundary before modifying production code.

Run: python -m pytest tests/G_z/test_gz_hardening.py -v
"""

import pytest
import torch

from mdgan.models.latent_transition import LatentTransitionGenerator


@pytest.mark.parametrize("name,kwargs", [
    ("zero latent dimension", dict(latent_dim=0, noise_dim=3, hidden_dim=16)),
    ("negative latent dimension", dict(latent_dim=-1, noise_dim=3, hidden_dim=16)),
    ("zero noise dimension", dict(latent_dim=7, noise_dim=0, hidden_dim=16)),
    ("negative noise dimension", dict(latent_dim=7, noise_dim=-1, hidden_dim=16)),
    ("zero hidden width", dict(latent_dim=7, noise_dim=3, hidden_dim=0)),
    ("negative hidden width", dict(latent_dim=7, noise_dim=3, hidden_dim=-1)),
])
def test_invalid_dimensions_rejected_at_construction(name, kwargs):
    """A stochastic latent generator should reject nonpositive dimensions early."""
    with pytest.raises(ValueError, match="dimension|dim|positive|noise|latent|hidden"):
        LatentTransitionGenerator(**kwargs)


@pytest.mark.parametrize("which,bad_value", [
    ("z", float("nan")),
    ("z", float("inf")),
    ("r", float("nan")),
    ("r", -float("inf")),
])
def test_nonfinite_inputs_rejected_before_poisoning_batchnorm(which, bad_value):
    """Opt-in debug check (validate_finite=True). It costs a device sync per
    call, so it is OFF by default; the trainer owns divergence detection.
    """
    gz = LatentTransitionGenerator(7, 3, hidden_dim=16, validate_finite=True)
    gz.train()
    z = torch.rand(8, 7)
    r = torch.rand(8, 3)
    if which == "z":
        z[0, 0] = bad_value
    else:
        r[0, 0] = bad_value
    with pytest.raises(ValueError, match="finite|NaN|Inf|invalid"):
        gz(z, r)


def test_training_batch_of_one_has_an_explicit_error_policy():
    """Require our own actionable error, not the low-level BatchNorm exception."""
    gz = LatentTransitionGenerator(7, 3, hidden_dim=16)
    gz.train()
    with pytest.raises(ValueError, match="batch size|batch_size|at least 2"):
        gz(torch.rand(1, 7), torch.rand(1, 3))