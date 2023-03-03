"""Define tests for the visualization of experiment suites."""
import pathlib

import matplotlib.pyplot as plt
import pytest

from robotics_optimization_benchmarks import experiment_suite_factory
from robotics_optimization_benchmarks import visualization


@pytest.fixture(name="experiment_suite")
def fixture_experiment_suite():
    """Create a simple experiment suite."""
    return experiment_suite_factory.create_experiment_suite(
        name="test_suite",
        description="Test suite for integration tests.",
        seeds=list(range(5)),
        benchmark_name="quadratic",
        benchmark_hparams={"dimension": 10},
        max_steps=100,
        optimizer_specs=[
            {
                "name": "GradientDescent_1",  # a name to label this optimizer
                "type": "GD",  # what type of optimizer is this? should match registry
                "hparams": {"step_size": 5e-2},
            },
            {
                "name": "MALA_1",
                "type": "MCMC",
                "hparams": {
                    "step_size": 5e-2,
                    "use_gradients": True,
                    "use_metropolis": True,
                },
            },
        ],
    )


def test_plot_optimizer_progress(tmpdir, experiment_suite) -> None:
    """Test that we can plot the results of an experiment suite."""
    # Run the experiments
    params_path, trace_paths, _ = experiment_suite.run(tmpdir.strpath)

    # Plot the results
    fig = visualization.plot_optimizer_progress(params_path, trace_paths)
    assert fig is not None
    plt.close(fig)


def test_render_results(tmpdir, experiment_suite) -> None:
    """Test that we can render the results of an experiment suite."""
    # Run the experiments
    params_path, _, solution_paths = experiment_suite.run(tmpdir.strpath)

    # Render the results to files in the results directory
    render_paths = visualization.render_results_to_file(params_path, solution_paths)

    # Make sure the files got saved
    assert len(render_paths) == len(solution_paths)
    for render_path in render_paths:
        assert pathlib.Path(render_path).exists()
