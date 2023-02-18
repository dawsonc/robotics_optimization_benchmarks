"""Test the brax benchmarks."""
import io

import jax.numpy as jnp
import jax.random as jrandom
import pytest

from robotics_optimization_benchmarks.benchmarks import make
from robotics_optimization_benchmarks.benchmarks.brax.brax import MLP
from robotics_optimization_benchmarks.benchmarks.brax.brax import Brax


envs_to_test = [
    "ant",
    "halfcheetah",
    "hopper",
    "humanoid",
    "reacher",
    "walker2d",
    "fetch",
    "grasp",
    "ur5e",
]


def test_make_brax():
    """Test making a brax benchmark."""
    benchmark = make("brax")
    assert benchmark == Brax


@pytest.mark.parametrize("in_size", [5, 10])
@pytest.mark.parametrize("out_size", [2, 4])
@pytest.mark.parametrize("policy_network_depth", [2, 4])
@pytest.mark.parametrize("policy_network_width", [32, 64])
def test_mlp(in_size, out_size, policy_network_depth, policy_network_width):
    """Test the MLP."""
    key = jrandom.PRNGKey(0)
    mlp = MLP(in_size, out_size, [policy_network_width] * policy_network_depth, key)
    assert mlp is not None

    # Make sure we can call the neural network
    action = mlp(jnp.zeros((in_size,)))
    assert action.shape == (out_size,)


@pytest.mark.parametrize("env", envs_to_test)
def test_brax_init(env):
    """Test initializing the benchmark."""
    benchmark = Brax(env)
    assert benchmark.task == env


@pytest.mark.parametrize("env", envs_to_test)
def test_brax_from_dict(env):
    """Test creating a brax benchmark from a dictionary."""
    benchmark = Brax.from_dict(
        {
            "task": env,
            "policy_network_depth": 2,
            "policy_network_width": 32,
            "horizon": 100,
        }
    )
    assert benchmark is not None


@pytest.mark.parametrize("env", envs_to_test)
def test_brax_sample_initial_guess(env):
    """Test sampling an initial guess."""
    benchmark = Brax(env)
    initial_guess = benchmark.sample_initial_guess(jrandom.PRNGKey(0))

    # Make sure we can call the neural network
    n_obs = benchmark._n_observations  # pylint: disable=protected-access
    n_actions = benchmark._n_actions  # pylint: disable=protected-access
    action = initial_guess(jnp.zeros((n_obs,)))
    assert action.shape == (n_actions,)


@pytest.mark.parametrize("env", envs_to_test)
def test_brax_evaluate_solution(env):
    """Test evaluating the benchmark."""
    benchmark = Brax(env, horizon=10)
    initial_guess = benchmark.sample_initial_guess(jrandom.PRNGKey(0))
    value = benchmark.evaluate_solution(initial_guess)

    # Solution should be scalar
    assert value.shape == ()


@pytest.mark.slow
@pytest.mark.parametrize("env", envs_to_test)
def test_brax_render_solution_to_binary_io(env):
    """Test rendering the benchmark, saving to a binary IO."""
    benchmark = Brax(task=env, horizon=10)
    initial_guess = benchmark.sample_initial_guess(jrandom.PRNGKey(0))
    benchmark.render_solution(initial_guess, io.BytesIO())


@pytest.mark.slow
@pytest.mark.parametrize("env", envs_to_test)
def test_brax_render_solution_to_file(env, tmpdir):
    """Test rendering the benchmark, saving to a file."""
    # Save to a temporary directory for testing
    save_path = tmpdir.join("test_render.gif").strpath

    benchmark = Brax(task=env, horizon=10)
    initial_guess = benchmark.sample_initial_guess(jrandom.PRNGKey(0))
    benchmark.render_solution(initial_guess, save_path)
