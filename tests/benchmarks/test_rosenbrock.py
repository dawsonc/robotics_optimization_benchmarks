"""Test the Rosenbrock benchmark."""
import io

import jax.random as jrandom
import pytest

from robotics_optimization_benchmarks.benchmarks import make
from robotics_optimization_benchmarks.benchmarks.nl_opt import Rosenbrock


# This benchmark is only defined for even dimensions.
even_dimensions_to_test = [2, 10, 100]
odd_dimensions_to_test = [1, 3, 11]


def test_make_rosenbrock():
    """Test making a rosenbrock benchmark from the registry."""
    benchmark = make("rosenbrock")
    assert benchmark == Rosenbrock


@pytest.mark.parametrize("dimension", even_dimensions_to_test)
def test_rosenbrock_init(dimension: int):
    """Test rosenbrock benchmark initialization."""
    benchmark = Rosenbrock(dimension=dimension)
    assert benchmark.dimension == dimension
    assert benchmark.name == "rosenbrock"


@pytest.mark.parametrize("dimension", odd_dimensions_to_test)
def test_rosenbrock_init_fails_with_odd_dimension(dimension: int):
    """Test rosenbrock benchmark initialization."""
    with pytest.raises(ValueError):
        Rosenbrock(dimension=dimension)


@pytest.mark.parametrize("dimension", even_dimensions_to_test)
def test_rosenbrock_to_dict(dimension: int):
    """Test rosenbrock benchmark to dict."""
    benchmark = Rosenbrock(dimension=dimension)
    assert benchmark.to_dict() == {"dimension": dimension}


@pytest.mark.parametrize("dimension", even_dimensions_to_test)
def test_rosenbrock_from_dict(dimension: int):
    """Test creating a rosenbrock benchmark from a dictionary."""
    benchmark = Rosenbrock.from_dict({"dimension": dimension})
    assert benchmark.dimension == dimension
    assert benchmark.name == "rosenbrock"


@pytest.mark.parametrize("dimension", even_dimensions_to_test)
def test_rosenbrock_sample_initial_guess(dimension):
    """Test sampling an initial guess."""
    benchmark = Rosenbrock(dimension=dimension)
    key = jrandom.PRNGKey(0)
    initial_guess = benchmark.sample_initial_guess(key)
    assert initial_guess.shape == (dimension,)


@pytest.mark.parametrize("dimension", even_dimensions_to_test)
def test_rosenbrock_evaluate_solution(dimension):
    """Test evaluating the benchmark."""
    benchmark = Rosenbrock(dimension=dimension)
    solution = benchmark.sample_initial_guess(jrandom.PRNGKey(0))
    value = benchmark.evaluate_solution(solution)

    # Solution should be scalar
    assert value.shape == ()


@pytest.mark.parametrize("dimension", even_dimensions_to_test)
def test_rosenbrock_render_solution(dimension):
    """Test rendering the benchmark."""
    benchmark = Rosenbrock(dimension=dimension)
    solution = benchmark.sample_initial_guess(jrandom.PRNGKey(0))
    benchmark.render_solution(solution, io.BytesIO())
