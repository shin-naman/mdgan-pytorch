"""Tests for mdgan.ops.mean_operation (Kawada et al. 2023, eq. 9).

Run: python -m pytest tests/test_ops.py -v
"""

import torch

from mdgan.ops import mean_operation


def test_zero_mean_over_batch_at_every_step_and_axis():
    x = torch.randn(64, 128, 3) + 5.0
    y = mean_operation(x)
    torch.testing.assert_close(y.mean(0), torch.zeros(128, 3), atol=1e-5, rtol=0)
    assert y.shape == x.shape


def test_only_the_batch_mean_is_removed():
    x = torch.randn(64, 128, 3)
    torch.testing.assert_close(mean_operation(x), x - x.mean(0, keepdim=True), rtol=0, atol=0)


def test_idempotent():
    x = torch.randn(32, 16, 3)
    torch.testing.assert_close(mean_operation(mean_operation(x)), mean_operation(x), atol=1e-6, rtol=0)


def test_does_not_modify_input():
    x = torch.randn(8, 4, 3)
    orig = x.clone()
    mean_operation(x)
    torch.testing.assert_close(x, orig, rtol=0, atol=0)


def test_commutes_with_cumsum_over_time():
    """Centering displacements per step == centering positions per time."""
    d = torch.randn(32, 128, 3)
    lhs = torch.cumsum(mean_operation(d), dim=1)
    rhs = mean_operation(torch.cumsum(d, dim=1))
    torch.testing.assert_close(lhs, rhs, atol=1e-5, rtol=1e-5)


def test_exact_jacobian():
    """grad of sum(y * w) w.r.t. x is w - mean(w): each sample's gradient is
    reduced by the batch mean, i.e. the op is a projection, not a detach."""
    x = torch.randn(8, 4, 3, requires_grad=True)
    w = torch.randn(8, 4, 3)
    (mean_operation(x) * w).sum().backward()
    torch.testing.assert_close(x.grad, w - w.mean(0, keepdim=True), atol=1e-6, rtol=0)


def test_batch_of_one_becomes_zero():
    x = torch.randn(1, 10, 3)
    torch.testing.assert_close(mean_operation(x), torch.zeros_like(x), rtol=0, atol=0)


def test_removed_mean_scales_as_one_over_sqrt_n():
    """Documents why the gain is a training-time regularizer: ~sigma/sqrt(N)."""
    torch.manual_seed(0)
    x = torch.randn(8192, 16, 3)
    removed_est = (x - mean_operation(x)).abs().mean().item()
    removed_train = (x[:64] - mean_operation(x[:64])).abs().mean().item()
    assert removed_est < 0.02 and removed_train > 0.05


def test_custom_batch_dim():
    x = torch.randn(5, 7, 3)
    torch.testing.assert_close(mean_operation(x, batch_dim=1).mean(1), torch.zeros(5, 3), atol=1e-6, rtol=0)


def test_works_under_no_grad_and_with_requires_grad():
    x = torch.randn(4, 3, 3, requires_grad=True)
    assert mean_operation(x).requires_grad
    with torch.no_grad():
        assert not mean_operation(x).requires_grad