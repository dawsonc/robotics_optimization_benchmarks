"""Test the Ackley benchmark."""
import io

import jax.random as jrandom
import pytest

from robotics_optimization_benchmarks.benchmarks import make
from robotics_optimization_benchmarks.benchmarks.nl_opt import Ackley


dimensions_to_test = [1, 10, 100]


def test_make_ackley():
    """Test making a ackley benchmark from the registry."""
    benchmark = make("ackley")
    assert benchmark == Ackley


@pytest.mark.parametrize("dimension", dimensions_to_test)
def test_ackley_init(dimension: int):
    """Test ackley benchmark initialization."""
    benchmark = Ackley(dimension=dimension)
    assert benchmark.dimension == dimension
    assert benchmark.name == "ackley"


@pytest.mark.parametrize("dimension", dimensions_to_test)
def test_ackley_to_dict(dimension: int):
    """Test ackley benchmark to dict."""
    benchmark = Ackley(dimension=dimension)
    assert benchmark.to_dict() == {"dimension": dimension}


@pytest.mark.parametrize("dimension", dimensions_to_test)
def test_ackley_from_dict(dimension: int):
    """Test creating a ackley benchmark from a dictionary."""
    benchmark = Ackley.from_dict({"dimension": dimension})
    assert benchmark.dimension == dimension
    assert benchmark.name == "ackley"


@pytest.mark.parametrize("dimension", dimensions_to_test)
def test_ackley_sample_initial_guess(dimension):
    """Test sampling an initial guess."""
    benchmark = Ackley(dimension=dimension)
    key = jrandom.PRNGKey(0)
    initial_guess = benchmark.sample_initial_guess(key)
    assert initial_guess.shape == (dimension,)


@pytest.mark.parametrize("dimension", dimensions_to_test)
def test_ackley_evaluate_solution(dimension):
    """Test evaluating the benchmark."""
    benchmark = Ackley(dimension=dimension)
    solution = benchmark.sample_initial_guess(jrandom.PRNGKey(0))
    value = benchmark.evaluate_solution(solution)

    # Solution should be scalar
    assert value.shape == ()


@pytest.mark.parametrize("dimension", dimensions_to_test)
def test_ackley_render_solution(dimension):
    """Test rendering the benchmark."""
    benchmark = Ackley(dimension=dimension)
    solution = benchmark.sample_initial_guess(jrandom.PRNGKey(0))
    benchmark.render_solution(solution, io.BytesIO())
