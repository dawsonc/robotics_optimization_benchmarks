"""Provide functionality for visualizing the results of experiments."""
import json

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from beartype import beartype
from beartype.typing import List
from beartype.typing import Tuple


def set_plot_style() -> None:
    """Initialize the plotting style for pretty, production-ready plots."""
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.5)


@beartype
def plot_optimizer_progress(
    params_file: str,
    optimizer_trace_files: List[str],
    size: Tuple[float, float] = (24, 12),
) -> plt.Figure:
    """Plot the progress of the different optimizers run on the same benchmark.

    Args:
        params_file: the path to the JSON file containing the parameters of the
            experiment suite.
        optimizer_trace_files: a list of paths to the JSON files containing the traces
            of the different optimizers.
        size: the size of the figure to plot the results on (width, height in inches).

    Returns:
        A figure containing the plots.
    """
    # Create a figure to plot the results on
    fig, axes = plt.subplots(1, 2, figsize=size)

    # Load the params of the experiment suite (mainly so we can get the name of the
    # benchmark to use as a title for the plot)
    with open(params_file, encoding="utf=8") as params_file:
        params = json.load(params_file)
    fig.suptitle(params["benchmark_name"])

    # Load the results from each optimizer and concatenate into a large dataframe
    trace_df = pd.concat(
        [pd.read_csv(f) for f in optimizer_trace_files], ignore_index=True
    )

    # Plot the performance of each optimizer as a function of the total number of
    # objective evaluations and as a function of time
    sns.lineplot(
        data=trace_df,
        x="Cumulative objective calls",
        y="Objective",
        hue="Optimizer name",
        ax=axes[0],
    )

    # To ensure consistent values for the x axis (allowing us to compute confidence
    # intervals), we don't show exact elapsed time but rather scale by the average
    # time per step across all seeds
    trace_df["Elapsed time (s)"] = 0.0  # we'll fill this in as we go
    names = trace_df["Optimizer name"].unique()
    for name in names:
        filtered_df = trace_df.loc[trace_df["Optimizer name"] == name]
        avg_time_per_step = filtered_df["Avg. time per step (s)"].mean()
        elapsed_time = filtered_df["Steps"] * avg_time_per_step
        trace_df.loc[
            trace_df["Optimizer name"] == name, "Elapsed time (s)"
        ] = elapsed_time

    sns.lineplot(
        data=trace_df,
        x="Elapsed time (s)",
        y="Objective",
        hue="Optimizer name",
        ax=axes[1],
    )

    return fig
