"""Test the benchmark registry."""
from copy import deepcopy

import pytest
from beartype.typing import Generator

import robotics_optimization_benchmarks.benchmarks.registry as br
from robotics_optimization_benchmarks.benchmarks.benchmark import Benchmark


class SuperAwesomeBenchmark(Benchmark):
    """A dummy benchmark for testing."""

    _name: str = "super_awesome_benchmark"


# Disable abstract methods for these tests
SuperAwesomeBenchmark.__abstractmethods__ = set()  # type: ignore


@pytest.fixture(autouse=True)
def setup_test_registry() -> Generator[None, None, None]:
    """Clear the benchmark registry before each test and restore it after."""
    # Stop Pylance from screaming at us while we do bad things
    old_entries = deepcopy(br._benchmark_registry._entries)  # type: ignore
    br._benchmark_registry._entries.clear()  # type: ignore

    yield  # tests run here

    br._benchmark_registry._entries = old_entries  # type: ignore


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
