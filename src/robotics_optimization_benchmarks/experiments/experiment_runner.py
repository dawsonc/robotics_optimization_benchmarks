"""Define functions to run a single experiment."""
import time

import jax
import jax.random as jrandom
import pandas as pd
from beartype import beartype
from beartype.typing import Tuple

from robotics_optimization_benchmarks.benchmarks import Benchmark
from robotics_optimization_benchmarks.optimizers import Optimizer
from robotics_optimization_benchmarks.types import DecisionVariable


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

    # Pre-run the jitted step function to compile it (key doesn't matter since we
    # don't use the results of this step)
    step_fn = jax.jit(step_fn)
    step_fn(initial_opt_state, init_key)

    # Run the optimizer starting from this seed, which gives us a trace of solutions and
    # objectives for each step. Make sure to record the time this takes!
    start = time.perf_counter()
    keys = jrandom.split(opt_key, max_steps)
    opt_state = initial_opt_state
    values = []
    cumulative_function_calls = []
    for key in keys:
        opt_state = step_fn(opt_state, key)
        values.append(opt_state.objective_value)
        cumulative_function_calls.append(opt_state.cumulative_function_calls)
    end = time.perf_counter()
    total_time = end - start

    # Clear the compilation cache to avoid OOM errors
    step_fn._clear_cache()  # pylint: disable=protected-access

    # Format the optimizer progress into a dataframe
    optimizer_trace_df = pd.DataFrame(
        {
            "Optimizer name": optimizer_name,
            "Optimizer type": optimizer.name,
            "Seed": seed,
            "Steps": range(max_steps),
            "Cumulative objective calls": cumulative_function_calls,
            "Objective": values,
            "Avg. time per step (s)": total_time / max_steps,
        }
    )

    return optimizer_trace_df, opt_state.solution
