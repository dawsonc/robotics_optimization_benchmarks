"""Test the quadratic benchmark contaminated with high frequency noise."""
import io

import jax.random as jrandom
import pytest

from robotics_optimization_benchmarks.benchmarks import make
from robotics_optimization_benchmarks.benchmarks.nl_opt import HFQuadratic


dimensions_to_test = [1, 10, 100]


def test_make_quadratic():
    """Test making a quadratic benchmark from the registry."""
    benchmark = make("hf_quadratic")
    assert benchmark == HFQuadratic


@pytest.mark.parametrize("dimension", dimensions_to_test)
def test_hf_quadratic_init(dimension: int):
    """Test quadratic benchmark initialization."""
    benchmark = HFQuadratic(dimension=dimension)
    assert benchmark.dimension == dimension
    assert benchmark.name == "hf_quadratic"


@pytest.mark.parametrize("dimension", dimensions_to_test)
def test_hf_quadratic_to_dict(dimension: int):
    """Test quadratic benchmark to dict."""
    benchmark = HFQuadratic(dimension=dimension)
    assert "dimension" in benchmark.to_dict()
    assert "period" in benchmark.to_dict()
    assert "n_components" in benchmark.to_dict()
    assert "noise_scale" in benchmark.to_dict()


@pytest.mark.parametrize("dimension", dimensions_to_test)
def test_hf_quadratic_from_dict(dimension: int):
    """Test creating a quadratic benchmark from a dictionary."""
    benchmark = HFQuadratic.from_dict({"dimension": dimension})
    assert benchmark.dimension == dimension
    assert benchmark.name == "hf_quadratic"


@pytest.mark.parametrize("dimension", dimensions_to_test)
def test_hf_quadratic_sample_initial_guess(dimension):
    """Test sampling an initial guess."""
    benchmark = HFQuadratic(dimension=dimension)
    key = jrandom.PRNGKey(0)
    initial_guess = benchmark.sample_initial_guess(key)
    assert initial_guess.shape == (dimension,)


@pytest.mark.parametrize("dimension", dimensions_to_test)
def test_hf_quadratic_evaluate_solution(dimension):
    """Test evaluating the benchmark."""
    benchmark = HFQuadratic(dimension=dimension)
    solution = benchmark.sample_initial_guess(jrandom.PRNGKey(0))
    value = benchmark.evaluate_solution(solution)

    # Solution should be scalar
    assert value.shape == ()


@pytest.mark.parametrize("dimension", dimensions_to_test)
def test_hf_quadratic_render_solution(dimension):
    """Test rendering the benchmark."""
    benchmark = HFQuadratic(dimension=dimension)
    solution = benchmark.sample_initial_guess(jrandom.PRNGKey(0))
    benchmark.render_solution(solution, io.BytesIO())


@pytest.mark.parametrize("dimension", dimensions_to_test)
def test_hf_quadratic_noise_lipschitz_constant(dimension):
    """Test getting the Lipschitz constant of the benchmark noise."""
    benchmark = HFQuadratic(dimension=dimension)
    noise_lipschitz = benchmark.noise_lipschitz_constant
    assert noise_lipschitz.shape == ()
