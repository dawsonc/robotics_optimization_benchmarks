"""Test the quadratic benchmark."""
import io

import jax.numpy as jnp
import jax.random as jrandom
import pytest

from robotics_optimization_benchmarks.benchmarks.reacher.reacher import MLP
from robotics_optimization_benchmarks.benchmarks.reacher.reacher import Reacher


mlp_depths_to_test = [2, 4]


@pytest.mark.parametrize("policy_network_depth", mlp_depths_to_test)
@pytest.mark.parametrize("policy_network_width", [32, 64])
def test_mlp(policy_network_depth, policy_network_width):
    """Test the MLP."""
    in_size, out_size = 11, 2
    key = jrandom.PRNGKey(0)
    mlp = MLP(in_size, out_size, [policy_network_width] * policy_network_depth, key)
    assert mlp is not None

    # Make sure we can call the neural network
    action = mlp(jnp.zeros((in_size,)))
    assert action.shape == (out_size,)


@pytest.mark.parametrize("policy_network_depth", mlp_depths_to_test)
def test_reacher_init(policy_network_depth):
    """Test initializing the benchmark."""
    benchmark = Reacher(policy_network_depth=policy_network_depth)
    assert benchmark.name == "reacher"


@pytest.mark.parametrize("policy_network_depth", mlp_depths_to_test)
def test_reacher_from_dict(policy_network_depth):
    """Test creating a quadratic benchmark from a dictionary."""
    benchmark = Reacher.from_dict(
        {
            "policy_network_depth": policy_network_depth,
            "policy_network_width": 32,
            "horizon": 100,
        }
    )
    assert benchmark is not None


@pytest.mark.parametrize("policy_network_depth", mlp_depths_to_test)
def test_reacher_sample_initial_guess(policy_network_depth):
    """Test sampling an initial guess."""
    benchmark = Reacher(policy_network_depth=policy_network_depth)
    initial_guess = benchmark.sample_initial_guess(jrandom.PRNGKey(0))

    # Make sure we can call the neural network
    n_obs = benchmark._n_observations  # pylint: disable=protected-access
    n_actions = benchmark._n_actions  # pylint: disable=protected-access
    action = initial_guess(jnp.zeros((n_obs,)))
    assert action.shape == (n_actions,)


@pytest.mark.parametrize("policy_network_depth", mlp_depths_to_test)
def test_reacher_evaluate_solution(policy_network_depth):
    """Test evaluating the benchmark."""
    benchmark = Reacher(horizon=10, policy_network_depth=policy_network_depth)
    initial_guess = benchmark.sample_initial_guess(jrandom.PRNGKey(0))
    value = benchmark.evaluate_solution(initial_guess)

    # Solution should be scalar
    assert value.shape == ()


@pytest.mark.slow
def test_reacher_render_solution_to_binary_io():
    """Test rendering the benchmark, saving to a binary IO."""
    benchmark = Reacher(
        horizon=10,
    )
    initial_guess = benchmark.sample_initial_guess(jrandom.PRNGKey(0))
    benchmark.render_solution(initial_guess, io.BytesIO())


@pytest.mark.slow
def test_reacher_render_solution_to_file(monkeypatch):
    """Test rendering the benchmark, saving to a file."""
    # We need to monkeypatch the file open function to provide a binary IO object
    monkeypatch.setattr(__builtins__, "open", lambda *args, **kwargs: io.BytesIO())

    benchmark = Reacher(
        horizon=10,
    )
    initial_guess = benchmark.sample_initial_guess(jrandom.PRNGKey(0))
    benchmark.render_solution(initial_guess, "test_file.gif")
