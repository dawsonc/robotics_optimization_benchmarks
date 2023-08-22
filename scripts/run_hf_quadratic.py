"""Run a suite of experiments on the quadratic example."""
import argparse
import os
from functools import partial

import jax.numpy as jnp
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LogNorm

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
        for noise_scale in jnp.logspace(0, 1, 25).tolist():
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
                max_steps=400,
                optimizer_specs=[
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
                        "name": "MALA (1e-2)",
                        "type": "MCMC",
                        "hparams": {
                            "step_size": 1e-2,
                            "use_gradients": True,
                            "use_metropolis": True,
                        },
                    },
                    {
                        "name": "MALA (1e-3)",
                        "type": "MCMC",
                        "hparams": {
                            "step_size": 1e-3,
                            "use_gradients": True,
                            "use_metropolis": True,
                        },
                    },
                    {
                        "name": "MALA (1e-4)",
                        "type": "MCMC",
                        "hparams": {
                            "step_size": 1e-4,
                            "use_gradients": True,
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

    # # Overwrite optimizer type to reflect MALA vs RMH
    # log_df["optimizer_type"] = log_df["optimizer_name"].apply(
    #     lambda x: "GD"
    #     if not ("MALA" in x or "RMH" in x)
    #     else ("MALA" if "MALA" in x else "RMH")
    # )

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

    # # Plot convergence
    # sns.lineplot(
    #     data=log_df,
    #     x="Cumulative objective calls",
    #     y="acceptance_rate",
    #     hue="optimizer_name",
    #     style="lipschitz_constant",
    # )

    # # Save the plot
    # plt.savefig(os.path.join(args.results_dir, "0_accept_rate.png"))

    # # Clear the plot
    # plt.clf()

    def last_50_avg(group, metric):
        """Compute the average of a metric over the last 50 iterations."""
        return group.nlargest(50, "Cumulative objective calls")[metric].mean()

    results_df = (
        log_df.groupby(["lipschitz_constant", "optimizer_type", "step_size", "seed"])
        .apply(partial(last_50_avg, metric="acceptance_rate"))
        .reset_index(name="Last 50 Avg Acceptance Rate")
    )
    sns.lineplot(
        data=results_df,
        x="lipschitz_constant",
        y="Last 50 Avg Acceptance Rate",
        hue="step_size",
        err_style="bars",
        errorbar=("ci", 95),
        hue_norm=LogNorm(),
    )
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Lipschitz Constant")
    plt.ylabel("MCMC Acceptance Rate")
    plt.gca().get_legend().set_title(r"MALA step size $\tau$")

    # Save the plot
    plt.savefig(os.path.join(args.results_dir, "0_mean_accept_rate.png"))

    plt.clf()

    results_df = (
        log_df.groupby(["lipschitz_constant", "optimizer_type", "step_size", "seed"])
        .apply(partial(last_50_avg, metric="Objective"))
        .reset_index(name="Last 50 Avg Objective")
    )
    sns.lineplot(
        data=results_df,
        x="lipschitz_constant",
        y="Last 50 Avg Objective",
        hue="step_size",
        err_style="bars",
        errorbar=("ci", 95),
        hue_norm=LogNorm(),
    )
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Lipschitz Constant")
    plt.ylabel("Final objective value")
    plt.gca().get_legend().set_title(r"MALA step size $\tau$")

    # Save the plot
    plt.savefig(os.path.join(args.results_dir, "0_mean_objective.png"))
