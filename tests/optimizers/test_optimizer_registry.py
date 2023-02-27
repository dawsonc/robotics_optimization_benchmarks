"""Test the optimizer registry."""
import pytest

import robotics_optimization_benchmarks.optimizers.registry as br
from robotics_optimization_benchmarks.optimizers.optimizer import Optimizer


class SuperAwesomeOptimizer(Optimizer):
    """A dummy optimizer for testing."""

    _name: str = "super_awesome_optimizer"


# Disable abstract methods for these tests
SuperAwesomeOptimizer.__abstractmethods__ = set()  # type: ignore


@pytest.fixture(autouse=True)
def mock_registry(monkeypatch) -> None:
    """Monkeypatch the registry to start fresh for each test."""
    monkeypatch.setattr(br, "_optimizer_registry", {})


def test_optimizer_registry() -> None:
    """Test adding new optimizers to the optimizer registry and accessing them."""
    # Register a new optimizer
    br.register(SuperAwesomeOptimizer.name, SuperAwesomeOptimizer)

    # We should be able to reference it later
    optimizer = br.make(SuperAwesomeOptimizer.name)
    assert optimizer is SuperAwesomeOptimizer


def test_optimizer_registry_register_invalid_name() -> None:
    """Test that the optimizer registry gives an error when adding a duplicate."""
    # Add the first version
    br.register(SuperAwesomeOptimizer.name, SuperAwesomeOptimizer)

    # Try to add a duplicate; should give an error
    with pytest.raises(ValueError):
        br.register(SuperAwesomeOptimizer.name, SuperAwesomeOptimizer)


def test_optimizer_registry_make_invalid_name() -> None:
    """Test that the optimizer registry errors when accessing an invalid name."""
    # Try to access an entry that's not there; should give an error
    with pytest.raises(KeyError):
        br.make(SuperAwesomeOptimizer.name)
