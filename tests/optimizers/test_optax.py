"""Tests for Optax that are not covered by the generic tests."""
import pytest

from robotics_optimization_benchmarks.optimizers import make


def test_optax_invalid_algorithm():
    """Test that Optax will not initialize with an invalid algorithm name."""
    with pytest.raises(ValueError):
        make("Optax")(optimizer_name="doesn't exist", params={})
