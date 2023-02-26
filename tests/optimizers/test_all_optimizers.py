"""Define common tests that should be run on each optimizer."""
import jax
import jax.random as jrandom
import pytest
from chex import assert_trees_all_close

from robotics_optimization_benchmarks.benchmarks.quadratic import Quadratic
from robotics_optimization_benchmarks.optimizers import make


# Define a list of optimizers that we should test, in the format (name, param_dict)
optimizers_to_test = [
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
def test_optimizer_init(optimizer_name, params, quadratic_benchmark):
    """Test initialization of optimizer state."""
    optimizer = make(optimizer_name).from_dict(params)

    key = jrandom.PRNGKey(0)
    initial_guess = quadratic_benchmark.sample_initial_guess(key)
    state, step = optimizer.init(quadratic_benchmark.evaluate_solution, initial_guess)

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
    state, step = optimizer.init(quadratic_benchmark.evaluate_solution, initial_guess)

    # Run the step function a few times to make sure it's self-compatible
    for _ in range(10):
        state = variant(step)(state, key)

    assert state is not None
