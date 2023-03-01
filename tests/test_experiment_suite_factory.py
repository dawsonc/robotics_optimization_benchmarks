"""Define tests for the experiment suite factory."""
import json
import pathlib

import pandas as pd
import pytest

from robotics_optimization_benchmarks import experiment_suite_factory


def test_experiment_suite_factory_user_story(tmpdir) -> None:
    """Integration test: test creating and running an experiment suite.

    As a user, I want to run suites of multiple experiments to compare the performance
    of different optimizers on the same benchmark problem, so that I can easily generate
    the results for my paper.
    """
    experiment_suite = experiment_suite_factory.create_experiment_suite(
        # As a user, I can name this test suite and provide a brief description to
        # help me remember what I was testing with this experiment.
        name="test_suite",
        description="Test suite for integration tests.",
        #
        # As a user, I want to specify the random seeds I use for this experiment suite,
        # so I can test across different seeds and reproduce my results.
        seeds=[0, 1, 2],
        #
        # As a user, I want to specify which benchmark problem I want to use for this
        # experiment suite, so that I can compare optimizers on the same problem.
        benchmark_name="quadratic",
        benchmark_hparams={"dimension": 10},
        #
        # As a user, I want to specify how many optimization steps each optimizer should
        # run for, to enable an apples-to-apples comparison
        max_steps=100,
        #
        # As a user, I want to specify which optimizers I want to use for this
        # experiment suite and specify the hyperparameters for each optimizer, so that
        # I can compare the performance of different optimizers.
        optimizer_specs=[
            {
                "name": "GradientDescent_1",  # a name to label this optimizer
                "type": "GD",  # what type of optimizer is this? should match registry
                "hparams": {"step_size": 0.01},
            },
            {
                "name": "MALA_1",
                "type": "MCMC",
                "hparams": {
                    "step_size": 0.01,
                    "use_gradients": True,
                    "use_metropolis": True,
                },
            },
        ],
    )

    # Make sure that initialization took place
    assert experiment_suite is not None

    # As a user, I want to run the experiment suite and save the results to a file, so
    # that I can analyze the results later and reproduce my results.
    params_path, trace_paths, solution_paths = experiment_suite.run(
        results_dir=tmpdir.strpath
    )
    # There should be files for:
    # - a JSON that allows us to re-create the experiment suite
    # - a CSV file for each optimizer that contains the progress of each optimizer
    # - a JSON file for each optimizer that contains the final solution of each optimizer
    assert pathlib.Path(params_path).exists()
    for trace_path, solution_path in zip(trace_paths, solution_paths, strict=True):
        assert pathlib.Path(trace_path).exists()
        assert pathlib.Path(solution_path).exists()

    # Check the contents of the parameter file
    with open(params_path, encoding="utf-8") as params_f:
        params = json.load(params_f)
        assert params == experiment_suite.to_dict()

    # Can we create a new experiment suite from the params?
    experiment_suite_2 = experiment_suite_factory.create_experiment_suite_from_file(
        params_path
    )
    assert experiment_suite_2.to_dict() == experiment_suite.to_dict()

    # Check the contents of the CSV files
    gd_df = pd.read_csv(trace_paths[0])
    mala_df = pd.read_csv(trace_paths[1])
    assert gd_df.columns.tolist() == mala_df.columns.tolist()


# Test that the experiment suite factory raises an error if the optimizers have
# duplicate names.
def test_experiment_suite_factory_duplicate_optimizer_names() -> None:
    """Test that we cannot create an experiment suite with duplicate optimizer names."""
    with pytest.raises(ValueError):
        experiment_suite_factory.create_experiment_suite(
            name="test_suite",
            description="Test suite for integration tests.",
            seeds=[0, 1, 2],
            benchmark_name="quadratic",
            benchmark_hparams={"dimension": 10},
            max_steps=100,
            optimizer_specs=[
                {
                    "name": "GradientDescent_1",
                    "type": "GD",
                    "hparams": {"step_size": 0.01},
                },
                {
                    "name": "GradientDescent_1",
                    "type": "GD",
                    "hparams": {"step_size": 0.1},
                },
            ],
        )
