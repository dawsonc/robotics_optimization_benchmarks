"""Test the three hump camel benchmark."""
import io

import jax.random as jrandom
import pytest

from robotics_optimization_benchmarks.benchmarks import make
from robotics_optimization_benchmarks.benchmarks.nl_opt import ThreeHumpCamel


# This benchmark is only defined for even dimensions.
even_dimensions_to_test = [2, 10, 100]
odd_dimensions_to_test = [1, 3, 11]


def test_make_three_hump_camel():
    """Test making a three_hump_camel benchmark from the registry."""
    benchmark = make("three_hump_camel")
    assert benchmark == ThreeHumpCamel


@pytest.mark.parametrize("dimension", even_dimensions_to_test)
def test_three_hump_camel_init(dimension: int):
    """Test three_hump_camel benchmark initialization."""
    benchmark = ThreeHumpCamel(dimension=dimension)
    assert benchmark.dimension == dimension
    assert benchmark.name == "three_hump_camel"


@pytest.mark.parametrize("dimension", odd_dimensions_to_test)
def test_three_hump_camel_init_fails_with_odd_dimension(dimension: int):
    """Test three_hump_camel benchmark initialization."""
    with pytest.raises(ValueError):
        ThreeHumpCamel(dimension=dimension)


@pytest.mark.parametrize("dimension", even_dimensions_to_test)
def test_three_hump_camel_to_dict(dimension: int):
    """Test three_hump_camel benchmark to dict."""
    benchmark = ThreeHumpCamel(dimension=dimension)
    assert benchmark.to_dict() == {"dimension": dimension}


@pytest.mark.parametrize("dimension", even_dimensions_to_test)
def test_three_hump_camel_from_dict(dimension: int):
    """Test creating a three_hump_camel benchmark from a dictionary."""
    benchmark = ThreeHumpCamel.from_dict({"dimension": dimension})
    assert benchmark.dimension == dimension
    assert benchmark.name == "three_hump_camel"


@pytest.mark.parametrize("dimension", even_dimensions_to_test)
def test_three_hump_camel_sample_initial_guess(dimension):
    """Test sampling an initial guess."""
    benchmark = ThreeHumpCamel(dimension=dimension)
    key = jrandom.PRNGKey(0)
    initial_guess = benchmark.sample_initial_guess(key)
    assert initial_guess.shape == (dimension,)


@pytest.mark.parametrize("dimension", even_dimensions_to_test)
def test_three_hump_camel_evaluate_solution(dimension):
    """Test evaluating the benchmark."""
    benchmark = ThreeHumpCamel(dimension=dimension)
    solution = benchmark.sample_initial_guess(jrandom.PRNGKey(0))
    value = benchmark.evaluate_solution(solution)

    # Solution should be scalar
    assert value.shape == ()


@pytest.mark.parametrize("dimension", even_dimensions_to_test)
def test_three_hump_camel_render_solution(dimension):
    """Test rendering the benchmark."""
    benchmark = ThreeHumpCamel(dimension=dimension)
    solution = benchmark.sample_initial_guess(jrandom.PRNGKey(0))
    benchmark.render_solution(solution, io.BytesIO())
