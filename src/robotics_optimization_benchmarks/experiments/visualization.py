"""Provide functionality for visualizing the results of experiments."""
import json
import os

import equinox as eqx
import jax.random as jrandom
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from beartype import beartype
from beartype.typing import List
from beartype.typing import Tuple

from robotics_optimization_benchmarks.benchmarks.registry import make as make_benchmark


def set_plot_style() -> None:
    """Initialize the plotting style for pretty, production-ready plots."""
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.5)


@beartype
def plot_optimizer_progress(
    params_file: str,
    optimizer_trace_files: List[str],
    size: Tuple[float, float] = (24, 12),
    x_time: bool = False,
    y_field: str = "Objective",
) -> plt.Figure:
    """Plot the progress of the different optimizers run on the same benchmark.

    Args:
        params_file: the path to the JSON file containing the parameters of the
            experiment suite.
        optimizer_trace_files: a list of paths to the CSV files containing the traces
            of the different optimizers.
        size: the size of the figure to plot the results on (width, height in inches).
        x_time: whether to plot the x axis as elapsed time rather than cumulative
            objective calls.
        y_field: the field to plot on the y axis. Should be one of "Objective",
            "Best objective"

    Returns:
        A figure containing the plots.
    """
    # Create a figure to plot the results on
    set_plot_style()
    fig, axes = plt.subplots(1, 1, figsize=size)

    # Load the params of the experiment suite (mainly so we can get the name of the
    # benchmark to use as a title for the plot)
    with open(params_file, encoding="utf=8") as params_file:
        params = json.load(params_file)
    fig.suptitle(params["benchmark_name"])

    # Load the results from each optimizer and concatenate into a large dataframe
    trace_df = pd.concat(
        [pd.read_csv(f) for f in optimizer_trace_files], ignore_index=True
    )

    if not x_time:
        # Plot the performance of each optimizer as a function of the total number of
        # objective evaluations and as a function of time
        sns.lineplot(
            data=trace_df,
            x="Cumulative objective calls",
            y=y_field,
            hue="Algorithm",
            ax=axes,
        )
    else:
        # To ensure consistent values for the x axis (allowing us to compute confidence
        # intervals), we don't show exact elapsed time but rather scale by the average
        # time per step across all seeds
        trace_df["Elapsed time (s)"] = 0.0  # we'll fill this in as we go
        names = trace_df["Algorithm"].unique()
        for name in names:
            filtered_df = trace_df.loc[trace_df["Algorithm"] == name]
            avg_time_per_step = filtered_df["Avg. time per step (s)"].mean()
            elapsed_time = filtered_df["Steps"] * avg_time_per_step
            trace_df.loc[
                trace_df["Algorithm"] == name, "Elapsed time (s)"
            ] = elapsed_time

        sns.lineplot(
            data=trace_df,
            x="Elapsed time (s)",
            y=y_field,
            hue="Algorithm",
            ax=axes,
        )

    return fig


@beartype
def render_results_to_file(
    params_file: str,
    solution_files: List[str],
) -> List[str]:
    """Render the solutions of different optimizers, saving the results to files.

    Args:
        params_file: the path to the JSON file containing the parameters of the
            experiment suite.
        solution_files: a list of paths to the JSON files containing the solutions found
            by different optimizers.

    Returns:
        A list of paths where the results have been saved.
    """
    # Load the params of the experiment suite and create a matching benchmark
    with open(params_file, encoding="utf=8") as params_file:
        benchmark_params = json.load(params_file)
    benchmark = make_benchmark(benchmark_params["benchmark_name"]).from_dict(
        benchmark_params["benchmark_hparams"]
    )

    # Load the results from each optimizer, render it, and save it. To deserialize the
    # solution, we need to get an example PyTree solution from the benchmark
    example_solution = benchmark.sample_initial_guess(jrandom.PRNGKey(0))
    render_files = []
    for solution_file in solution_files:
        # Use equinox to deserialize the pytree solution (assumed to have the same
        # structure as the example solution)
        solution = eqx.tree_deserialise_leaves(solution_file, example_solution)

        # Render the solution and save it to a file with the same name as the solution
        # file but with a .png extension
        render_file = (
            os.path.splitext(solution_file)[0] + "." + benchmark.render_extension
        )
        benchmark.render_solution(solution, save_to=render_file)
        render_files.append(render_file)

    return render_files
