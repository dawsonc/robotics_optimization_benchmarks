"""Test the benchmark registry."""
import pytest
from beartype.typing import Type

import robotics_optimization_benchmarks.benchmarks.registry as br
from robotics_optimization_benchmarks.benchmarks.benchmark import Benchmark
from robotics_optimization_benchmarks.registry import Registry


class SuperAwesomeBenchmark(Benchmark):
    """A dummy benchmark for testing."""

    _name: str = "super_awesome_benchmark"


# Disable abstract methods for these tests
SuperAwesomeBenchmark.__abstractmethods__ = set()  # type: ignore


@pytest.fixture(autouse=True)
def mock_registry(monkeypatch) -> None:
    """Monkeypatch the registry to start fresh for each test."""
    monkeypatch.setattr(br, "_benchmark_registry", Registry[Type[Benchmark]]())


def test_benchmark_registry() -> None:
    """Test adding new benchmarks to the benchmark registry and accessing them."""
    # Register a new benchmark
    br.register(SuperAwesomeBenchmark.name, SuperAwesomeBenchmark)

    # We should be able to reference it later
    benchmark = br.make(SuperAwesomeBenchmark.name)
    assert benchmark is SuperAwesomeBenchmark


def test_benchmark_registry_register_invalid_name() -> None:
    """Test that the benchmark registry gives an error when adding a duplicate."""
    # Add the first version
    br.register(SuperAwesomeBenchmark.name, SuperAwesomeBenchmark)

    # Try to add a duplicate; should give an error
    with pytest.raises(ValueError):
        br.register(SuperAwesomeBenchmark.name, SuperAwesomeBenchmark)


def test_benchmark_registry_make_invalid_name() -> None:
    """Test that the benchmark registry errors when accessing an invalid name."""
    # Try to access an entry that's not there; should give an error
    with pytest.raises(KeyError):
        br.make(SuperAwesomeBenchmark.name)
