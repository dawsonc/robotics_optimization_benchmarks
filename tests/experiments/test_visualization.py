"""Define tests for the visualization of experiment suites."""
import matplotlib.pyplot as plt

from robotics_optimization_benchmarks import experiment_suite_factory
from robotics_optimization_benchmarks import visualization


def test_experiment_suite_factory_plot(tmpdir) -> None:
    """Test that we can plot the results of an experiment suite."""
    # Create a simple experiment suite and run it
    experiment_suite = experiment_suite_factory.create_experiment_suite(
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
    params_path, trace_paths, _ = experiment_suite.run(tmpdir.strpath)

    # Plot the results
    fig = visualization.plot_optimizer_progress(params_path, trace_paths)
    assert fig is not None
    plt.close(fig)
