"""Define a class to setup and run suites of experiments."""
import json
import os

import pandas as pd
from beartype import beartype
from beartype.typing import Any
from beartype.typing import Dict
from beartype.typing import List
from beartype.typing import Tuple
from jaxtyping import Array
from jaxtyping import Shaped

from robotics_optimization_benchmarks.benchmarks import Benchmark
from robotics_optimization_benchmarks.experiments.experiment_runner import (
    run_experiment,
)
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
            json.dump(self.to_dict(), params_file)

        # For each optimizer, run it on the benchmark for `max_steps` steps, repeating
        # for each seed.
        trace_file_names = []
        solution_file_names = []
        for optimizer_name, optimizer in self._optimizers.items():
            # Run on each seed and accumulate the results
            optimizer_traces = []
            solutions = []
            for seed in self._seeds:
                optimizer_trace_df, solution = run_experiment(
                    self._benchmark, optimizer, optimizer_name, seed, self._max_steps
                )
                optimizer_traces.append(optimizer_trace_df)
                solutions.append(solution)

            # Concatenate the optimizer_traces from each seed into one DataFrame and
            # save it to a CSV file in the optimizer_traces directory.
            optimizer_traces = pd.concat(optimizer_traces, ignore_index=True)
            trace_file_name = os.path.join(results_dir, f"{optimizer_name}_trace.csv")
            optimizer_traces.to_csv(trace_file_name, index=False)
            trace_file_names.append(trace_file_name)

            # Save the solutions from each seed to a JSON file in the solutions
            # directory.
            solution_file_name = os.path.join(
                results_dir, f"{optimizer_name}_solution.json"
            )
            with open(
                solution_file_name,
                "w",
                encoding="utf-8",
            ) as solutions_file:
                json.dump(
                    solutions,
                    solutions_file,
                    default=lambda obj: obj.tolist()
                    if isinstance(obj, Shaped[Array, "..."])
                    else obj,
                )
            solution_file_names.append(solution_file_name)

        return params_file_name, trace_file_names, solution_file_names
