"""Test the pong benchmark."""
import io

import jax.random as jrandom
import pytest

from robotics_optimization_benchmarks.benchmarks import make
from robotics_optimization_benchmarks.benchmarks.pong import Pong


dimensions_to_test = [1, 10, 100]


def test_make_pong():
    """Test making a pong benchmark from the registry."""
    benchmark = make("pong")
    assert benchmark == Pong


@pytest.mark.parametrize("dimension", dimensions_to_test)
def test_pong_init(dimension: int):
    """Test pong benchmark initialization."""
    benchmark = Pong(dimension=dimension)
    assert benchmark.dimension == dimension
    assert benchmark.name == "pong"


@pytest.mark.parametrize("dimension", dimensions_to_test)
def test_pong_from_dict(dimension: int):
    """Test creating a pong benchmark from a dictionary."""
    benchmark = Pong.from_dict({"dimension": dimension})
    assert benchmark.dimension == dimension
    assert benchmark.name == "pong"


@pytest.mark.parametrize("dimension", dimensions_to_test)
def test_pong_to_dict(dimension: int):
    """Test pong benchmark to dict."""
    benchmark = Pong(dimension=dimension)
    assert benchmark.to_dict() == {"dimension": dimension}


@pytest.mark.parametrize("dimension", dimensions_to_test)
def test_pong_sample_initial_guess(dimension):
    """Test sampling an initial guess."""
    benchmark = Pong(dimension=dimension)
    key = jrandom.PRNGKey(0)
    initial_guess = benchmark.sample_initial_guess(key)
    assert initial_guess.shape == (dimension, 3)


@pytest.mark.parametrize("dimension", dimensions_to_test)
def test_pong_evaluate_solution(dimension):
    """Test evaluating the benchmark."""
    benchmark = Pong(dimension=dimension)
    solution = benchmark.sample_initial_guess(jrandom.PRNGKey(0))
    value = benchmark.evaluate_solution(solution)

    # Solution should be scalar
    assert value.shape == ()


@pytest.mark.parametrize("dimension", dimensions_to_test)
def test_pong_render_solution(dimension):
    """Test rendering the benchmark."""
    benchmark = Pong(dimension=dimension)
    solution = benchmark.sample_initial_guess(jrandom.PRNGKey(0))
    benchmark.render_solution(solution, io.BytesIO())
