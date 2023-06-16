"""Test the Himmelblau benchmark."""
import io

import jax.random as jrandom
import pytest

from robotics_optimization_benchmarks.benchmarks import make
from robotics_optimization_benchmarks.benchmarks.nl_opt import Himmelblau


# This benchmark is only defined for even dimensions.
even_dimensions_to_test = [2, 10, 100]
odd_dimensions_to_test = [1, 3, 11]


def test_make_himmelblau():
    """Test making a himmelblau benchmark from the registry."""
    benchmark = make("himmelblau")
    assert benchmark == Himmelblau


@pytest.mark.parametrize("dimension", even_dimensions_to_test)
def test_himmelblau_init(dimension: int):
    """Test himmelblau benchmark initialization."""
    benchmark = Himmelblau(dimension=dimension)
    assert benchmark.dimension == dimension
    assert benchmark.name == "himmelblau"


@pytest.mark.parametrize("dimension", odd_dimensions_to_test)
def test_himmelblau_init_fails_with_odd_dimension(dimension: int):
    """Test himmelblau benchmark initialization."""
    with pytest.raises(ValueError):
        Himmelblau(dimension=dimension)


@pytest.mark.parametrize("dimension", even_dimensions_to_test)
def test_himmelblau_to_dict(dimension: int):
    """Test himmelblau benchmark to dict."""
    benchmark = Himmelblau(dimension=dimension)
    assert benchmark.to_dict() == {"dimension": dimension}


@pytest.mark.parametrize("dimension", even_dimensions_to_test)
def test_himmelblau_from_dict(dimension: int):
    """Test creating a himmelblau benchmark from a dictionary."""
    benchmark = Himmelblau.from_dict({"dimension": dimension})
    assert benchmark.dimension == dimension
    assert benchmark.name == "himmelblau"


@pytest.mark.parametrize("dimension", even_dimensions_to_test)
def test_himmelblau_sample_initial_guess(dimension):
    """Test sampling an initial guess."""
    benchmark = Himmelblau(dimension=dimension)
    key = jrandom.PRNGKey(0)
    initial_guess = benchmark.sample_initial_guess(key)
    assert initial_guess.shape == (dimension,)


@pytest.mark.parametrize("dimension", even_dimensions_to_test)
def test_himmelblau_evaluate_solution(dimension):
    """Test evaluating the benchmark."""
    benchmark = Himmelblau(dimension=dimension)
    solution = benchmark.sample_initial_guess(jrandom.PRNGKey(0))
    value = benchmark.evaluate_solution(solution)

    # Solution should be scalar
    assert value.shape == ()


@pytest.mark.parametrize("dimension", even_dimensions_to_test)
def test_himmelblau_render_solution(dimension):
    """Test rendering the benchmark."""
    benchmark = Himmelblau(dimension=dimension)
    solution = benchmark.sample_initial_guess(jrandom.PRNGKey(0))
    benchmark.render_solution(solution, io.BytesIO())
