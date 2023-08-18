"""Run a suite of experiments on the pong example."""
import argparse
import os

import matplotlib.pyplot as plt
import seaborn as sns

from robotics_optimization_benchmarks.experiments import experiment_suite_factory
from robotics_optimization_benchmarks.experiments.loggers import FileLogger


if __name__ == "__main__":
    # Define the command line arguments
    parser = argparse.ArgumentParser(
        description="Run a suite of experiments on the pong example."
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="./results/pong",
        help="The directory to save the results to.",
    )
    parser.add_argument(
        "--load_data",
        action="store_true",
        help="Whether to load the data from disk or re-run the experiments.",
    )
    args = parser.parse_args()

    # We want to run this expeirment with a range of different dimensions, but using
    # the same logger each time so that we can aggregate the results.
    logger = FileLogger(results_dir=args.results_dir)

    dimensions = [1, 10, 100, 1000]

    if not args.load_data:
        for dimension in dimensions:
            # Create an experiment suite for the pong
            experiment_suite = experiment_suite_factory.create_experiment_suite(
                name=f"pong_{dimension}",
                description="Compare optimization and inference methods on the pong env.",
                seeds=list(range(10)),
                benchmark_name="pong",
                benchmark_hparams={"dimension": dimension},
                max_steps=500,
                optimizer_specs=[
                    {
                        "name": "Gradient-based optimization",  # a name to label this optimizer
                        "type": "GD",  # what type of optimizer is this? should match registry
                        "hparams": {"step_size": 1e-2},
                    },
                    {
                        "name": "Gradient-based inference (L 1)",
                        "type": "MCMC",
                        "hparams": {
                            "step_size": 1e-2,
                            "objective_scale": 1e0,
                            "use_gradients": True,
                            "use_metropolis": True,
                        },
                    },
                    {
                        "name": "Gradient-based inference (L 10)",
                        "type": "MCMC",
                        "hparams": {
                            "step_size": 1e-2,
                            "objective_scale": 1e1,
                            "use_gradients": True,
                            "use_metropolis": True,
                        },
                    },
                    {
                        "name": "Gradient-free inference",
                        "type": "MCMC",
                        "hparams": {
                            "step_size": 1e-2,
                            "objective_scale": 1e1,
                            "use_gradients": False,
                            "use_metropolis": True,
                        },
                    },
                ],
            )

            # Run the experiments
            experiment_suite.run(logger)

    # Get the logged data as a pandas dataframe
    log_df = logger.get_logs()

    # Get the best objective seen for each seed
    min_df = log_df.groupby(
        ["seed", "optimizer_name", "dimension"], as_index=False
    ).min()

    # Normalize cost by dimension
    min_df["Best objective"] /= min_df["dimension"]

    # Plot the results
    sns.lineplot(
        data=min_df,
        x="dimension",
        y="Best objective",
        hue="optimizer_name",
        err_style="bars",
    )
    plt.xscale("log")
    plt.title("pong")

    # Save the plot
    plt.savefig(os.path.join(args.results_dir, "0_results.png"))
