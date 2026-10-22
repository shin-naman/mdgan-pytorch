"""Tests for mdgan._util."""

import pytest

from mdgan._util import positive_int


@pytest.mark.parametrize("value", [1, 2, 64, 10**9])
def test_positive_int_returns_valid_values(value):
    assert positive_int("x", value) == value


@pytest.mark.parametrize("value", [0, -1, True, False, 2.0, "3", None, 1.5])
def test_positive_int_rejects_everything_else_and_names_the_argument(value):
    with pytest.raises(ValueError, match="block_len must be a positive integer"):
        positive_int("block_len", value)