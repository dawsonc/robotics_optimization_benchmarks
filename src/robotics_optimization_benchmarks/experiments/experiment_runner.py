"""Define functions to run a single experiment."""
import jax
import jax.random as jrandom
from beartype import beartype

from robotics_optimization_benchmarks.benchmarks import Benchmark
from robotics_optimization_benchmarks.experiments.loggers import Logger
from robotics_optimization_benchmarks.optimizers import Optimizer
from robotics_optimization_benchmarks.types import DecisionVariable


@beartype
def run_experiment(
    benchmark: Benchmark,
    optimizer: Optimizer,
    seed: int,
    max_steps: int,
    logger: Logger,
) -> DecisionVariable:
    """Run the given optimizer on the given benchmark, starting with the given seed.

    Logs results to WandB

    Args:
        benchmark: the benchmark to run the optimizer on.
        optimizer: the optimizer to run on the benchmark.
        seed: the random seed to use for the experiment.
        max_steps: the maximum number of steps to run the optimizer for.
        logger: the object to use to log the results

    Returns:
        - The solution found by the optimizer.
    """
    # Get a JAX random key from the given seed and split it for use in initialization
    # and running the optimizer
    init_key, opt_key = jrandom.split(jrandom.PRNGKey(seed))

    # Sample an initial state and initialize the optimizer
    opt_state, step_fn = optimizer.make_step(
        benchmark.evaluate_solution, benchmark.sample_initial_guess(init_key)
    )
    best_objective = opt_state.objective_value

    # Log initial data
    log_packet = {
        "Cumulative objective calls": opt_state.cumulative_function_calls,
        "Objective": opt_state.objective_value,
        "Best objective": best_objective,
    }
    logger.log(log_packet)

    # Run the optimizer starting from this seed, which gives us a trace of solutions and
    # objectives for each step.
    for key in jrandom.split(opt_key, max_steps):
        opt_state = jax.jit(step_fn)(opt_state, key)

        # Update the best solution found
        best_objective = min(best_objective, opt_state.objective_value)

        # Log the data from this step
        log_packet = {
            "Cumulative objective calls": opt_state.cumulative_function_calls,
            "Objective": opt_state.objective_value,
            "Best objective": best_objective,
        }
        logger.log(log_packet)

    # Return the logs and solution
    return opt_state.solution
