"""Define a class to setup and run suites of experiments."""
import json
import os
import time

import equinox as eqx
import jax
import jax.random as jrandom
import pandas as pd
from beartype import beartype
from beartype.typing import Any
from beartype.typing import Dict
from beartype.typing import List
from beartype.typing import Tuple

from robotics_optimization_benchmarks.benchmarks import Benchmark
from robotics_optimization_benchmarks.optimizers import Optimizer


@beartype
class ExperimentSuite:
    """A suite of experiments that are run on a single benchmark.

    An `ExperimentSuite` allows the user to compare the performance of a set of
    optimizers (including multiple instances of the same optimizer with different
    hyperparameters) on the same benchmark problem, with a focus on reproducibility and
    ease of use.

    Example:
        >>> from robotics_optimization_benchmarks.optimizers import make as make_bench
        >>> from robotics_optimization_benchmarks.optimizers import make as make_opt
        >>> experiment_suite = ExperimentSuite(
        ...     name="AwesomeExperiments",
        ...     description="Provide a brief description of the experiment suite.",
        ...     seeds=[0, 1],  # run each optimizer with each seed
        ...     max_steps=10,
        ...     benchmark=make_bench("Quadratic").from_dict({"dimension": 10}),
        ...     optimizers={
        ...         "Opt1": make_opt("GD").from_dict({"step_size": 0.01})},
        ...         "Opt2": make_opt("GD").from_dict({"step_size": 0.1})},
        ...     }
        ... )
    """

    def __init__(
        self,
        name: str,
        description: str,
        seeds: List[int],
        max_steps: int,
        benchmark: Benchmark,
        optimizers: Dict[str, Optimizer],
    ) -> None:
        """Initialize an experiment suite.

        Args:
            name: the name of the experiment suite.
            description: a brief description of the experiment suite.
            seeds: the random seeds to use for the experiment suite. Each optimizer will
                be run once with each seed.
            max_steps: the maximum number of steps to run for each optimizer.
            benchmark: the benchmark to use for the experiment suite. Each optimizer
                will be run on this benchmark.
            optimizers: a dictionary of optimizers, where the keys are the names of the
                optimizers (as you would like to see them e.g. in a plot legend) and the
                values are the optimizers themselves.
        """
        self._name = name
        self._description = description
        self._seeds = seeds
        self._max_steps = max_steps
        self._benchmark = benchmark
        self._optimizers = optimizers

    def to_dict(self) -> Dict[str, Any]:
        """Return a dict of all parameters required to recreate this ExperimentSuite."""
        return {
            "name": self._name,
            "description": self._description,
            "seeds": self._seeds,
            "max_steps": self._max_steps,
            "benchmark_name": self._benchmark.name,
            "benchmark_hparams": self._benchmark.to_dict(),
            "optimizer_specs": [
                {
                    "name": name,
                    "type": optimizer.name,
                    "hparams": optimizer.to_dict(),
                }
                for name, optimizer in self._optimizers.items()
            ],
        }

    def run(self, results_dir: str) -> Tuple[str, List[str], List[str]]:
        """Run the experiment suite.

        Args:
            results_dir: the directory to save the results to.

        Returns:
            The filename of the JSON file that contains the parameters of this suite.
            A list of filenames that were created to store the optimizer traces.
            A list of filenames that were created to store the solutions.
        """
        # Running the experiment suite has the following steps:
        # 1. Make sure that a directory exists at the given path (create it if not).
        # 2. Save the parameters of this experiment suite to a JSON file in the results
        #    directory.
        # 3. Run each optimizer for `max_steps` steps on the benchmark, restarting for
        #    each seed.
        # 4. Save the results of each optimizer to a CSV file in the results directory.

        # Start by making sure the results directory exists.
        os.makedirs(results_dir, exist_ok=True)

        # Next, save any parameters we need for reproducibility (i.e. all kwargs for
        # the experiment suite factory) to a JSON file in the results directory.
        params_file_name = os.path.join(results_dir, "experiment_suite_params.json")
        with open(params_file_name, "w", encoding="utf-8") as params_file:
            json.dump(self.to_dict(), params_file, indent=4)

        # For each optimizer, run it on the benchmark for `max_steps` steps, repeating
        # for each seed.
        trace_file_names = []
        solution_file_names = []
        for optimizer_name, optimizer in self._optimizers.items():
            # Make the step function for this optimizer
            step_fn = optimizer.make_step(self._benchmark.evaluate_solution)

            # Run on each seed and accumulate the results
            optimizer_traces = []
            for seed in self._seeds:
                print(f"Running {optimizer_name} with seed {seed}.")

                # Get a JAX random key from the given seed and split it for use in
                # initialization and running the optimizer
                init_key, opt_key = jrandom.split(jrandom.PRNGKey(seed))

                # Sample an initial state and initialize the optimizer
                start = time.perf_counter()
                initial_opt_state = jax.jit(optimizer.init_state, static_argnums=0)(
                    self._benchmark.evaluate_solution,
                    self._benchmark.sample_initial_guess(init_key),
                )

                # Pre-run the jitted step function to compile it (key doesn't matter
                # since we don't use the results of this step)
                step_fn = jax.jit(step_fn)
                step_fn(initial_opt_state, init_key)
                end = time.perf_counter()
                print(f"Init state and compile in {end - start:.2f} seconds.")

                # Run the optimization
                start = time.perf_counter()
                keys = jrandom.split(opt_key, self._max_steps)
                opt_state = initial_opt_state
                values = []
                cumulative_function_calls = []
                for key in keys:
                    opt_state = step_fn(opt_state, key)
                    values.append(opt_state.objective_value)
                    cumulative_function_calls.append(
                        opt_state.cumulative_function_calls
                    )
                end = time.perf_counter()
                total_time = end - start
                print(f"Ran in {total_time:.2f} seconds")

                # Format the optimizer progress into a dataframe
                optimizer_trace_df = pd.DataFrame(
                    {
                        "Optimizer name": optimizer_name,
                        "Optimizer type": optimizer.name,
                        "Seed": seed,
                        "Steps": range(self._max_steps),
                        "Cumulative objective calls": cumulative_function_calls,
                        "Objective": values,
                        "Avg. time per step (s)": total_time / self._max_steps,
                    }
                )
                optimizer_traces.append(optimizer_trace_df)

                # Save the solutions from each seed to a Equinox serialized file
                solution_file_name = os.path.join(
                    results_dir, f"{optimizer_name}_solution_{seed}.eqx"
                )
                eqx.tree_serialise_leaves(solution_file_name, opt_state.solution)
                solution_file_names.append(solution_file_name)

            # Concatenate the optimizer_traces from each seed into one DataFrame and
            # save it to a CSV file in the optimizer_traces directory.
            optimizer_traces = pd.concat(optimizer_traces, ignore_index=True)
            trace_file_name = os.path.join(results_dir, f"{optimizer_name}_trace.csv")
            optimizer_traces.to_csv(trace_file_name, index=False)
            trace_file_names.append(trace_file_name)

        return params_file_name, trace_file_names, solution_file_names
