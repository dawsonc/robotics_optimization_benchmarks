"""Define functions to run a single experiment."""
import time

import jax
import jax.random as jrandom
import jax.tree_util as jtu
import pandas as pd
from beartype import beartype
from beartype.typing import Tuple

from robotics_optimization_benchmarks.benchmarks import Benchmark
from robotics_optimization_benchmarks.optimizers import Optimizer
from robotics_optimization_benchmarks.types import DecisionVariable


# Start by defining some useful functions for running experiments.


def _wrap_for_scan(step_function):
    """Make an optimizer step function compatible with `jax.lax.scan`."""  # noqa: D202

    # scan expects the function to return a tuple of (output, carry), both of which
    # are the same in this case.
    def wrapped(state, step_key):
        next_state = step_function(state, step_key)
        return next_state, next_state

    return wrapped


@beartype
def run_experiment(
    benchmark: Benchmark,
    optimizer: Optimizer,
    optimizer_name: str,
    seed: int,
    max_steps: int,
) -> Tuple[pd.DataFrame, DecisionVariable]:
    """Run the given optimizer on the given benchmark, starting with the given seed.

    Args:
        benchmark: the benchmark to run the optimizer on.
        optimizer: the optimizer to run on the benchmark.
        optimizer_name: the name of the optimizer to use in the output dataframe.
        seed: the random seed to use for the experiment.
        max_steps: the maximum number of steps to run the optimizer for.

    Returns:
        - A dataframe of the optimization history
        - The solution found by the optimizer.
    """
    # Get a JAX random key from the given seed and split it for use in initialization
    # and running the optimizer
    init_key, opt_key = jrandom.split(jrandom.PRNGKey(seed))

    # Sample an initial state and initialize the optimizer
    initial_opt_state, step_fn = optimizer.make_step(
        benchmark.evaluate_solution, benchmark.sample_initial_guess(init_key)
    )

    # Run the optimizer starting from this seed, which gives us a trace of solutions and
    # objectives for each step. Make sure to record the time this takes!
    start = time.perf_counter()
    _, states = jax.lax.scan(
        jax.jit(_wrap_for_scan(step_fn)),
        initial_opt_state,
        jrandom.split(opt_key, max_steps),
    )
    end = time.perf_counter()
    total_time = end - start

    # Format the optimizer progress into a dataframe
    optimizer_trace_df = pd.DataFrame(
        {
            "Optimizer name": optimizer_name,
            "Optimizer type": optimizer.name,
            "Seed": seed,
            "Steps": range(max_steps),
            "Objective evaluations": states.cumulative_function_calls,
            "Objective": states.objective_value,
            "Time per step": total_time / max_steps,
        }
    )
    # Extract the solution at the last step
    solution = jtu.tree_map(lambda x: x[-1], states.solution)

    return optimizer_trace_df, solution
