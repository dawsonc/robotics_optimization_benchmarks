"""A set of factory methods for creating suites of experiments."""
import json

from beartype import beartype
from beartype.typing import Any
from beartype.typing import Dict
from beartype.typing import List
from beartype.typing import Union

from robotics_optimization_benchmarks.benchmarks.registry import make as make_benchmark
from robotics_optimization_benchmarks.experiment_suite import ExperimentSuite
from robotics_optimization_benchmarks.optimizers.registry import make as make_optimizer


@beartype
def create_experiment_suite(
    name: str,
    description: str,
    seeds: List[int],
    benchmark_name: str,
    benchmark_hparams: Dict[str, Any],
    max_steps: int,
    optimizer_specs: List[Dict[str, Union[str, Dict[str, Any]]]],
) -> ExperimentSuite:
    """Create an experiment suite.

    Example:
        >>> experiment_suite = create_experiment_suite(
        ...     name="test_suite",
        ...     description="Test suite for integration tests.",
        ...     seeds=[0, 1, 2],
        ...     benchmark_name="quadratic",
        ...     benchmark_hparams={"dimension": 10},
        ...     max_steps=100,
        ...     optimizer_specs=[
        ...         {
        ...             "name": "GradientDescent_1",
        ...             "type": "GD",
        ...             "hparams": {"step_size": 0.01},
        ...         },
        ...      ],
        ... )

    Args:
        name: the name of the experiment suite.
        description: a brief description of the experiment suite.
        seeds: the random seeds to use for the experiment suite. Each optimizer will
            be run once with each seed.
        benchmark_name: the name of the benchmark to use for the experiment suite. Each
            optimizer will be run on this benchmark.
        benchmark_hparams: the hyperparameters for the benchmark.
        max_steps: the maximum number of optimization steps to run for each optimizer.
        optimizer_specs: a list of dictionaries, each containing the name, type, and
            hyperparameters dictionary for each optimizer.

    Returns:
        An experiment suite.

    Raises:
        ValueError: if the optimizers have duplicate names.
    """
    # Input validation: optimizers must have unique names.
    optimizer_names = [optimizer_spec["name"] for optimizer_spec in optimizer_specs]
    if len(optimizer_names) != len(set(optimizer_names)):
        raise ValueError(
            "The optimizers must have unique names, but the following names were "
            f"provided: {optimizer_names}."
        )

    # Create the benchmark and all the optimizers, which we'll then inject into
    # the experiment suite.
    benchmark = make_benchmark(benchmark_name).from_dict(benchmark_hparams)
    optimizers = {
        optimizer_spec["name"]: make_optimizer(optimizer_spec["type"]).from_dict(
            optimizer_spec["hparams"]
        )
        for optimizer_spec in optimizer_specs
    }

    experiment_suite = ExperimentSuite(
        name=name,
        description=description,
        seeds=seeds,
        max_steps=max_steps,
        benchmark=benchmark,
        optimizers=optimizers,
    )

    return experiment_suite


@beartype
def create_experiment_suite_from_dict(
    experiment_suite_spec: Dict[str, Any]
) -> ExperimentSuite:
    """Create an experiment suite from a dictionary.

    Args:
        experiment_suite_spec: a dictionary containing the following keys:
            - name: the name of the experiment suite.
            - description: a brief description of the experiment suite.
            - seeds: a list of random seeds to use for the experiment suite.
            - benchmark_name: the name of the benchmark to use for the experiment suite.
            - benchmark_hparams: a dict of hyperparameters for the benchmark.
            - max_steps: the maximum number of optimization steps to run for each
                optimizer.
            - optimizer_specs: a list of dictionaries, each containing the name, type,
                and hyperparameters dictionary for each optimizer.

    Returns:
        An experiment suite.
    """
    return create_experiment_suite(
        name=experiment_suite_spec["name"],
        description=experiment_suite_spec["description"],
        seeds=experiment_suite_spec["seeds"],
        benchmark_name=experiment_suite_spec["benchmark_name"],
        benchmark_hparams=experiment_suite_spec["benchmark_hparams"],
        max_steps=experiment_suite_spec["max_steps"],
        optimizer_specs=experiment_suite_spec["optimizer_specs"],
    )


def create_experiment_suite_from_file(json_file_path: str) -> ExperimentSuite:
    """Create an experiment suite from a JSON file.

    Args:
        json_file_path: the path to the JSON file containing the parameters of the
            experiment suite.

    Returns:
        An experiment suite.
    """
    with open(json_file_path, encoding="utf-8") as params_f:
        experiment_suite_spec = json.load(params_f)

    return create_experiment_suite_from_dict(experiment_suite_spec)
