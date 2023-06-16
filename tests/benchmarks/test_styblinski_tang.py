"""Test the StyblinskiTang benchmark."""
import io

import jax.random as jrandom
import pytest

from robotics_optimization_benchmarks.benchmarks import make
from robotics_optimization_benchmarks.benchmarks.nl_opt import StyblinskiTang


dimensions_to_test = [1, 10, 100]


def test_make_styblinski_tang():
    """Test making a styblinski_tang benchmark from the registry."""
    benchmark = make("styblinski_tang")
    assert benchmark == StyblinskiTang


@pytest.mark.parametrize("dimension", dimensions_to_test)
def test_styblinski_tang_init(dimension: int):
    """Test styblinski_tang benchmark initialization."""
    benchmark = StyblinskiTang(dimension=dimension)
    assert benchmark.dimension == dimension
    assert benchmark.name == "styblinski_tang"


@pytest.mark.parametrize("dimension", dimensions_to_test)
def test_styblinski_tang_to_dict(dimension: int):
    """Test styblinski_tang benchmark to dict."""
    benchmark = StyblinskiTang(dimension=dimension)
    assert benchmark.to_dict() == {"dimension": dimension}


@pytest.mark.parametrize("dimension", dimensions_to_test)
def test_styblinski_tang_from_dict(dimension: int):
    """Test creating a styblinski_tang benchmark from a dictionary."""
    benchmark = StyblinskiTang.from_dict({"dimension": dimension})
    assert benchmark.dimension == dimension
    assert benchmark.name == "styblinski_tang"


@pytest.mark.parametrize("dimension", dimensions_to_test)
def test_styblinski_tang_sample_initial_guess(dimension):
    """Test sampling an initial guess."""
    benchmark = StyblinskiTang(dimension=dimension)
    key = jrandom.PRNGKey(0)
    initial_guess = benchmark.sample_initial_guess(key)
    assert initial_guess.shape == (dimension,)


@pytest.mark.parametrize("dimension", dimensions_to_test)
def test_styblinski_tang_evaluate_solution(dimension):
    """Test evaluating the benchmark."""
    benchmark = StyblinskiTang(dimension=dimension)
    solution = benchmark.sample_initial_guess(jrandom.PRNGKey(0))
    value = benchmark.evaluate_solution(solution)

    # Solution should be scalar
    assert value.shape == ()


@pytest.mark.parametrize("dimension", dimensions_to_test)
def test_styblinski_tang_render_solution(dimension):
    """Test rendering the benchmark."""
    benchmark = StyblinskiTang(dimension=dimension)
    solution = benchmark.sample_initial_guess(jrandom.PRNGKey(0))
    benchmark.render_solution(solution, io.BytesIO())
