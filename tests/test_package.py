"""Basic environment smoke test; G_z tests will be added next."""

import mdgan


def test_package_import():
    assert mdgan.__doc__
