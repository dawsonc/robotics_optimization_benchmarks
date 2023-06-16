"""Run a suite of experiments on the ballistic example."""
from robotics_optimization_benchmarks.experiments import experiment_suite_factory
from robotics_optimization_benchmarks.experiments.loggers import WandbLogger


if __name__ == "__main__":
    # Create an experiment suite for the ballistic
    experiment_suite = experiment_suite_factory.create_experiment_suite(
        name="ballistic",
        description="Compare optimization and inference methods on the ballistic env.",
        seeds=[0, 1, 2, 3],
        benchmark_name="ballistic",
        benchmark_hparams={"dimension": 100},
        max_steps=50,
        optimizer_specs=[
            {
                "name": "Gradient-based optimization",  # a name to label this optimizer
                "type": "GD",  # what type of optimizer is this? should match registry
                "hparams": {"step_size": 1e-2},
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

    # Create a logger to save the results to Weights & Biases
    logger = WandbLogger(results_dir="./results/double_well")

    # Run the experiments
    experiment_suite.run(logger)
