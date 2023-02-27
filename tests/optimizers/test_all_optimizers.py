"""Define common tests that should be run on each optimizer."""
import jax
import jax.random as jrandom
import pytest
from chex import assert_trees_all_close
from jax.config import config as jconfig

from robotics_optimization_benchmarks.benchmarks.quadratic import Quadratic
from robotics_optimization_benchmarks.optimizers import make


# Turn on NaN checking
jconfig.update("jax_debug_nans", True)

# Define a list of optimizers that we should test, in the format (name, param_dict)
optimizers_to_test = [
    ("BGD", {}),
    ("GD", {}),
    ("MCMC", {"use_gradients": True, "use_metropolis": True}),  # MALA
    ("MCMC", {"use_gradients": True, "use_metropolis": False}),  # ULA
    ("MCMC", {"use_gradients": False, "use_metropolis": True}),  # RMH
    ("VPG", {}),
]


# Define a pytest fixture to construct a quadratic benchmark instance
@pytest.fixture(name="quadratic_benchmark")
def fixture_quadratic_benchmark():
    """Create a quadratic benchmark instance."""
    return Quadratic(dimension=10)


@pytest.mark.parametrize("optimizer_name,_", optimizers_to_test)
def test_make_optimizer(optimizer_name, _):
    """Test making an optimizer from the registry."""
    optimizer = make(optimizer_name)
    assert optimizer.name == optimizer_name


@pytest.mark.parametrize("optimizer_name,params", optimizers_to_test)
def test_optimizer_from_dict(optimizer_name, params):
    """Test optimizer initialization from dictionary."""
    optimizer = make(optimizer_name).from_dict(params)
    assert optimizer.name == optimizer_name


@pytest.mark.parametrize("optimizer_name,params", optimizers_to_test)
def test_optimizer_description(optimizer_name, params):
    """Test that the optimizer description exists."""
    optimizer = make(optimizer_name).from_dict(params)
    assert optimizer.description is not None


@pytest.mark.parametrize("optimizer_name,params", optimizers_to_test)
def test_optimizer_make_step(optimizer_name, params, quadratic_benchmark):
    """Test initialization of optimizer state."""
    optimizer = make(optimizer_name).from_dict(params)

    key = jrandom.PRNGKey(0)
    initial_guess = quadratic_benchmark.sample_initial_guess(key)
    state, step = optimizer.make_step(
        quadratic_benchmark.evaluate_solution, initial_guess
    )

    assert state is not None
    assert_trees_all_close(state.solution, initial_guess)
    assert step is not None


@pytest.mark.parametrize("optimizer_name,params", optimizers_to_test)
@pytest.mark.parametrize("variant", [jax.jit, lambda f: f])  # test with and without jit
def test_optimizer_step(optimizer_name, params, variant, quadratic_benchmark):
    """Test stepping the optimizer."""
    optimizer = make(optimizer_name).from_dict(params)

    key = jrandom.PRNGKey(0)
    initial_guess = quadratic_benchmark.sample_initial_guess(key)
    state, step = optimizer.make_step(
        quadratic_benchmark.evaluate_solution, initial_guess
    )

    # Run the step function a few times to make sure it's self-compatible
    for _ in range(10):
        state = variant(step)(state, key)

    assert state is not None


@pytest.mark.parametrize("optimizer_name,params", optimizers_to_test)
def test_optimizer_optimization(optimizer_name, params, quadratic_benchmark):
    """Test that the optimizer can minimize a simple convex function."""
    # Initialize the optimizer and JIT the step function
    optimizer = make(optimizer_name).from_dict(params)
    key = jrandom.PRNGKey(0)
    initial_guess = quadratic_benchmark.sample_initial_guess(key)

    state, step = optimizer.make_step(
        quadratic_benchmark.evaluate_solution, initial_guess
    )
    step = jax.jit(step)

    # Run the optimization loop for a fixed number of steps
    n_steps = 1000
    for _ in range(n_steps):
        subkey, key = jrandom.split(key)
        state = step(state, subkey)

    # We should find a solution with a lower objective value than the initial guess
    minimum_acceptable_decrease = 0.9  # we expect at least a 90% decrease
    final_objective = quadratic_benchmark.evaluate_solution(state.solution)
    initial_objective = quadratic_benchmark.evaluate_solution(initial_guess)

    assert final_objective < minimum_acceptable_decrease * initial_objective
