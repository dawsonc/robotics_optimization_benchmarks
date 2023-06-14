"""Run a suite of experiments on the ballistic example."""
import matplotlib.pyplot as plt

from robotics_optimization_benchmarks.experiments import experiment_suite_factory
from robotics_optimization_benchmarks.experiments import visualization as viz


if __name__ == "__main__":
    # Create an experiment suite for the ballistic
    experiment_suite = experiment_suite_factory.create_experiment_suite(
        name="ballistic",
        description="Compare optimization and inference methods on the ballistic env.",
        seeds=[0, 1, 2, 3],
        benchmark_name="ballistic",
        benchmark_hparams={"dimension": 1000},
        max_steps=50,
        optimizer_specs=[
            {
                "name": "Gradient-based optimization",  # a name to label this optimizer
                "type": "GD",  # what type of optimizer is this? should match registry
                "hparams": {"step_size": 1e-2},
            },
            {
                "name": "Gradient-free optimization",  # a name to label this optimizer
                "type": "BGD",  # what type of optimizer is this? should match registry
                "hparams": {"step_size": 1e-2, "n_samples": 1000},
            },
            {
                "name": "Gradient-based inference",
                "type": "MCMC",
                "hparams": {
                    "step_size": 1e-2,
                    "use_gradients": True,
                    "use_metropolis": True,
                },
            },
            {
                "name": "Gradient-free inference",
                "type": "MCMC",
                "hparams": {
                    "step_size": 1e-2,
                    "use_gradients": False,
                    "use_metropolis": True,
                },
            },
        ],
    )

    # Run the experiments
    (params_filename, trace_filenames, solution_filenames) = experiment_suite.run(
        "./results/ballistic"
    )

    # Plot the results
    fig = viz.plot_optimizer_progress(
        params_filename, trace_filenames, y_field="Best objective", size=(12.0, 12.0)
    )
    plt.show()
