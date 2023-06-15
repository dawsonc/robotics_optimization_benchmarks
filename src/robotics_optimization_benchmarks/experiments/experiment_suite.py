"""Define a class to setup and run suites of experiments."""
from beartype import beartype
from beartype.typing import Any
from beartype.typing import Dict
from beartype.typing import List

from robotics_optimization_benchmarks.benchmarks import Benchmark
from robotics_optimization_benchmarks.experiments.experiment_runner import (
    run_experiment,
)
from robotics_optimization_benchmarks.experiments.loggers import Logger
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

    def run(self, logger: Logger) -> None:
        """Run the experiment suite.

        Args:
            logger: the object to use to log the results
        """
        # Run each optimizer, logging the results as we go.
        for optimizer_name, optimizer in self._optimizers.items():
            # Run on each seed
            for seed in self._seeds:
                # Assemble a dictionary of hyperparams
                config = (
                    self.to_dict()
                    | {
                        "benchmark_name": self._benchmark.name,
                        "optimizer_name": optimizer_name,
                        "seed": seed,
                        "max_steps": self._max_steps,
                    }
                    | optimizer.to_dict()
                    | self._benchmark.to_dict()
                )

                # Start the logger
                logger.start(self._name, config)

                # Run the experiment
                solution = run_experiment(
                    self._benchmark,
                    optimizer,
                    seed,
                    self._max_steps,
                    logger,
                )

                logger.save_artifact("solution", solution)

                # Finish logging
                logger.finish()
