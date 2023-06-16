"""Test the double well benchmark."""
import io

import jax.random as jrandom
import pytest

from robotics_optimization_benchmarks.benchmarks import make
from robotics_optimization_benchmarks.benchmarks.nl_opt import DoubleWell


dimensions_to_test = [1, 10, 100]


def test_make_double_well():
    """Test making a double well benchmark from the registry."""
    benchmark = make("double_well")
    assert benchmark == DoubleWell


@pytest.mark.parametrize("dimension", dimensions_to_test)
def test_double_well_init(dimension: int):
    """Test double well benchmark initialization."""
    benchmark = DoubleWell(dimension=dimension)
    assert benchmark.dimension == dimension
    assert benchmark.name == "double_well"


@pytest.mark.parametrize("dimension", dimensions_to_test)
def test_double_well_to_dict(dimension: int):
    """Test double well benchmark to dict."""
    benchmark = DoubleWell(dimension=dimension)
    assert benchmark.to_dict() == {"dimension": dimension}


@pytest.mark.parametrize("dimension", dimensions_to_test)
def test_double_well_from_dict(dimension: int):
    """Test creating a double_well benchmark from a dictionary."""
    benchmark = DoubleWell.from_dict({"dimension": dimension})
    assert benchmark.dimension == dimension
    assert benchmark.name == "double_well"


@pytest.mark.parametrize("dimension", dimensions_to_test)
def test_double_well_sample_initial_guess(dimension):
    """Test sampling an initial guess."""
    benchmark = DoubleWell(dimension=dimension)
    key = jrandom.PRNGKey(0)
    initial_guess = benchmark.sample_initial_guess(key)
    assert initial_guess.shape == (dimension,)


@pytest.mark.parametrize("dimension", dimensions_to_test)
def test_double_well_evaluate_solution(dimension):
    """Test evaluating the benchmark."""
    benchmark = DoubleWell(dimension=dimension)
    solution = benchmark.sample_initial_guess(jrandom.PRNGKey(0))
    value = benchmark.evaluate_solution(solution)

    # Solution should be scalar
    assert value.shape == ()


@pytest.mark.parametrize("dimension", dimensions_to_test)
def test_double_well_render_solution(dimension):
    """Test rendering the benchmark."""
    benchmark = DoubleWell(dimension=dimension)
    solution = benchmark.sample_initial_guess(jrandom.PRNGKey(0))
    benchmark.render_solution(solution, io.BytesIO())
