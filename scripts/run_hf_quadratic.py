"""Run a suite of experiments on the quadratic example."""
import argparse
import os

import jax.numpy as jnp
import matplotlib.pyplot as plt
import seaborn as sns

from robotics_optimization_benchmarks.experiments import experiment_suite_factory
from robotics_optimization_benchmarks.experiments.loggers import FileLogger


if __name__ == "__main__":
    # Define the command line arguments
    parser = argparse.ArgumentParser(
        description="Run a suite of experiments on the quadratic example."
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="./results/hf_quadratic",
        help="The directory to save the results to.",
    )
    parser.add_argument(
        "--load_data",
        action="store_true",
        help="Whether to load the data from disk or re-run the experiments.",
    )
    parser.add_argument(
        "--dimension",
        type=int,
        default=100,
    )
    args = parser.parse_args()

    # We want to run this expeirment with a range of different dimensions, but using
    # the same logger each time so that we can aggregate the results.
    logger = FileLogger(results_dir=args.results_dir)

    n_noise_components = 2
    noise_scale = 0.1
    period = 0.1

    if not args.load_data:
        for noise_scale in jnp.logspace(-4, 0, 100).tolist():
            # Create an experiment suite for the quadratic
            experiment_suite = experiment_suite_factory.create_experiment_suite(
                name="hf_quadratic",
                description="Compare optimization and inference methods on the hf_quadratic env.",
                seeds=list(range(10)),
                benchmark_name="hf_quadratic",
                benchmark_hparams={
                    "dimension": args.dimension,
                    "n_components": n_noise_components,
                    "period": period,
                    "noise_scale": noise_scale,
                },
                max_steps=200,
                optimizer_specs=[
                    {
                        "name": "GD (1e-1)",  # a name to label this optimizer
                        "type": "GD",  # what type of optimizer is this? should match registry
                        "hparams": {"step_size": 1e-1},
                    },
                    {
                        "name": "MALA (1e-1)",
                        "type": "MCMC",
                        "hparams": {
                            "step_size": 1e-1,
                            "use_gradients": True,
                            "use_metropolis": True,
                        },
                    },
                    {
                        "name": "RMH (1e-1)",
                        "type": "MCMC",
                        "hparams": {
                            "step_size": 1e-1,
                            "use_gradients": False,
                            "use_metropolis": True,
                        },
                    },
                    # 1e-2 steps
                    {
                        "name": "GD (1e-2)",
                        "type": "GD",
                        "hparams": {"step_size": 1e-2},
                    },
                    {
                        "name": "MALA (1e-2)",
                        "type": "MCMC",
                        "hparams": {
                            "step_size": 1e-2,
                            "use_gradients": True,
                            "use_metropolis": True,
                        },
                    },
                    {
                        "name": "RMH (1e-2)",
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
            experiment_suite.run(logger)

    # Get the logged data as a pandas dataframe
    log_df = logger.get_logs()
    print("Loaded data from disk.")

    # Overwrite optimizer type to reflect MALA vs RMH
    log_df["optimizer_type"] = log_df["optimizer_name"].apply(
        lambda x: "GD"
        if not ("MALA" in x or "RMH" in x)
        else ("MALA" if "MALA" in x else "RMH")
    )

    # # Plot convergence
    # sns.lineplot(
    #     data=log_df,
    #     x="Cumulative objective calls",
    #     y="Objective",
    #     hue="optimizer_name",
    #     style="lipschitz_constant",
    # )

    # # Save the plot
    # plt.savefig(os.path.join(args.results_dir, "0_learning_curves.png"))
    # print("done plotting learning curves.")

    # # Clear the plot
    # plt.clf()

    def last_10_avg(group):
        """Compute the average objective seen in the last 10 iterations."""
        return group.nlargest(10, "Cumulative objective calls")["Objective"].mean()

    results_df = (
        log_df.groupby(["lipschitz_constant", "optimizer_type", "step_size", "seed"])
        .apply(last_10_avg)
        .reset_index(name="Last 10 Avg Objective")
    )
    sns.lineplot(
        data=results_df,
        x="lipschitz_constant",
        y="Last 10 Avg Objective",
        hue="optimizer_type",
        style="step_size",
        err_style="bars",
    )
    plt.xscale("log")
    # plt.yscale("log")

    # Save the plot
    plt.savefig(os.path.join(args.results_dir, "0_best_objectives.png"))
