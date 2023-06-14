"""Run a suite of experiments on the double well example."""
import matplotlib.pyplot as plt

from robotics_optimization_benchmarks.experiments import experiment_suite_factory
from robotics_optimization_benchmarks.experiments import visualization as viz


if __name__ == "__main__":
    # Create an experiment suite for the double well
    experiment_suite = experiment_suite_factory.create_experiment_suite(
        name="double_well",
        description="Compare optimization and inference methods on the double-well.",
        seeds=[0, 1, 2, 3],
        benchmark_name="double_well",
        benchmark_hparams={"dimension": 1},
        max_steps=50,
        optimizer_specs=[
            {
                "name": "Gradient-based optimization",  # a name to label this optimizer
                "type": "GD",  # what type of optimizer is this? should match registry
                "hparams": {"step_size": 0.02},
            },
            {
                "name": "Gradient-based inference",
                "type": "MCMC",
                "hparams": {
                    "step_size": 0.02,
                    "use_gradients": True,
                    "use_metropolis": True,
                },
            },
        ],
    )

    # Run the experiments
    (params_filename, trace_filenames, solution_filenames) = experiment_suite.run(
        "./results/double_well"
    )

    # Plot the results
    fig = viz.plot_optimizer_progress(
        params_filename, trace_filenames, y_field="Best objective"
    )
    plt.show()
