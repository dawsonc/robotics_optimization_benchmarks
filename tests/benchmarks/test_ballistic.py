"""Test the ballistic benchmark."""
import io

import jax.random as jrandom
import pytest

from robotics_optimization_benchmarks.benchmarks import make
from robotics_optimization_benchmarks.benchmarks.ballistic import Ballistic


dimensions_to_test = [1, 10, 100]


def test_make_ballistic():
    """Test making a ballistic benchmark from the registry."""
    benchmark = make("ballistic")
    assert benchmark == Ballistic


@pytest.mark.parametrize("dimension", dimensions_to_test)
def test_ballistic_init(dimension: int):
    """Test ballistic benchmark initialization."""
    benchmark = Ballistic(dimension=dimension)
    assert benchmark.dimension == dimension
    assert benchmark.name == "ballistic"


@pytest.mark.parametrize("dimension", dimensions_to_test)
def test_ballistic_from_dict(dimension: int):
    """Test creating a ballistic benchmark from a dictionary."""
    benchmark = Ballistic.from_dict({"dimension": dimension})
    assert benchmark.dimension == dimension
    assert benchmark.name == "ballistic"


@pytest.mark.parametrize("dimension", dimensions_to_test)
def test_ballistic_sample_initial_guess(dimension):
    """Test sampling an initial guess."""
    benchmark = Ballistic(dimension=dimension)
    key = jrandom.PRNGKey(0)
    initial_guess = benchmark.sample_initial_guess(key)
    assert initial_guess.shape == (dimension,)


@pytest.mark.parametrize("dimension", dimensions_to_test)
def test_ballistic_evaluate_solution(dimension):
    """Test evaluating the benchmark."""
    benchmark = Ballistic(dimension=dimension)
    solution = benchmark.sample_initial_guess(jrandom.PRNGKey(0))
    value = benchmark.evaluate_solution(solution)

    # Solution should be scalar
    assert value.shape == ()


@pytest.mark.parametrize("dimension", dimensions_to_test)
def test_ballistic_render_solution(dimension):
    """Test rendering the benchmark."""
    benchmark = Ballistic(dimension=dimension)
    solution = benchmark.sample_initial_guess(jrandom.PRNGKey(0))
    benchmark.render_solution(solution, io.BytesIO())
