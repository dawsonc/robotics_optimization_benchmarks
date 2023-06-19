"""Test the Heaviside benchmark."""
import io

import jax.random as jrandom
import pytest

from robotics_optimization_benchmarks.benchmarks import make
from robotics_optimization_benchmarks.benchmarks.nl_opt import Heaviside


dimensions_to_test = [1, 10, 100]


def test_make_heaviside():
    """Test making a heaviside benchmark from the registry."""
    benchmark = make("heaviside")
    assert benchmark == Heaviside


@pytest.mark.parametrize("dimension", dimensions_to_test)
def test_heaviside_init(dimension: int):
    """Test heaviside benchmark initialization."""
    benchmark = Heaviside(dimension=dimension)
    assert benchmark.dimension == dimension
    assert benchmark.name == "heaviside"


@pytest.mark.parametrize("dimension", dimensions_to_test)
def test_heaviside_to_dict(dimension: int):
    """Test heaviside benchmark to dict."""
    benchmark = Heaviside(dimension=dimension)
    assert benchmark.to_dict() == {"dimension": dimension}


@pytest.mark.parametrize("dimension", dimensions_to_test)
def test_heaviside_from_dict(dimension: int):
    """Test creating a heaviside benchmark from a dictionary."""
    benchmark = Heaviside.from_dict({"dimension": dimension})
    assert benchmark.dimension == dimension
    assert benchmark.name == "heaviside"


@pytest.mark.parametrize("dimension", dimensions_to_test)
def test_heaviside_sample_initial_guess(dimension):
    """Test sampling an initial guess."""
    benchmark = Heaviside(dimension=dimension)
    key = jrandom.PRNGKey(0)
    initial_guess = benchmark.sample_initial_guess(key)
    assert initial_guess.shape == (dimension,)


@pytest.mark.parametrize("dimension", dimensions_to_test)
def test_heaviside_evaluate_solution(dimension):
    """Test evaluating the benchmark."""
    benchmark = Heaviside(dimension=dimension)
    solution = benchmark.sample_initial_guess(jrandom.PRNGKey(0))
    value = benchmark.evaluate_solution(solution)

    # Solution should be scalar
    assert value.shape == ()


@pytest.mark.parametrize("dimension", dimensions_to_test)
def test_heaviside_render_solution(dimension):
    """Test rendering the benchmark."""
    benchmark = Heaviside(dimension=dimension)
    solution = benchmark.sample_initial_guess(jrandom.PRNGKey(0))
    benchmark.render_solution(solution, io.BytesIO())
