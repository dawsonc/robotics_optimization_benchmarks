"""Tests for MCMC that are not covered by the generic tests."""
import pytest

from robotics_optimization_benchmarks.optimizers import make


def test_mcmc_invalid_algorithm():
    """Test that MCMC will not initialize with an invalid set of hyperparameters."""
    with pytest.raises(ValueError):
        make("MCMC")(use_gradients=False, use_metropolis=False)
