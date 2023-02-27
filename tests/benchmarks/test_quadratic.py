"""Test the quadratic benchmark."""
import io

import jax.random as jrandom
import pytest

from robotics_optimization_benchmarks.benchmarks import make
from robotics_optimization_benchmarks.benchmarks.quadratic import Quadratic


dimensions_to_test = [1, 10, 100]


def test_make_quadratic():
    """Test making a quadratic benchmark from the registry."""
    benchmark = make("quadratic")
    assert benchmark == Quadratic


@pytest.mark.parametrize("dimension", dimensions_to_test)
def test_quadratic_init(dimension: int):
    """Test quadratic benchmark initialization."""
    benchmark = Quadratic(dimension=dimension)
    assert benchmark.dimension == dimension
    assert benchmark.name == "quadratic"


@pytest.mark.parametrize("dimension", dimensions_to_test)
def test_quadratic_to_dict(dimension: int):
    """Test quadratic benchmark to dict."""
    benchmark = Quadratic(dimension=dimension)
    assert benchmark.to_dict() == {"dimension": dimension}


@pytest.mark.parametrize("dimension", dimensions_to_test)
def test_quadratic_from_dict(dimension: int):
    """Test creating a quadratic benchmark from a dictionary."""
    benchmark = Quadratic.from_dict({"dimension": dimension})
    assert benchmark.dimension == dimension
    assert benchmark.name == "quadratic"


@pytest.mark.parametrize("dimension", dimensions_to_test)
def test_quadratic_sample_initial_guess(dimension):
    """Test sampling an initial guess."""
    benchmark = Quadratic(dimension=dimension)
    key = jrandom.PRNGKey(0)
    initial_guess = benchmark.sample_initial_guess(key)
    assert initial_guess.shape == (dimension,)


@pytest.mark.parametrize("dimension", dimensions_to_test)
def test_quadratic_evaluate_solution(dimension):
    """Test evaluating the benchmark."""
    benchmark = Quadratic(dimension=dimension)
    solution = benchmark.sample_initial_guess(jrandom.PRNGKey(0))
    value = benchmark.evaluate_solution(solution)

    # Solution should be scalar
    assert value.shape == ()


@pytest.mark.parametrize("dimension", dimensions_to_test)
def test_quadratic_render_solution(dimension):
    """Test rendering the benchmark."""
    benchmark = Quadratic(dimension=dimension)
    solution = benchmark.sample_initial_guess(jrandom.PRNGKey(0))
    benchmark.render_solution(solution, io.BytesIO())
