"""Hyperparameter tuning for n=10 ballistic benchmark."""
import os

from beartype.typing import Any
from beartype.typing import Dict
from beartype.typing import List

from robotics_optimization_benchmarks import experiment_suite_factory
from robotics_optimization_benchmarks import visualization


BASE_RESULTS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "results", "ballistic_10_sweep")
)


def sweep_ballistic_10(
    optimizer_type: str, optimizer_name: str, hparams_list: List[Dict[str, Any]]
) -> None:
    """Run a parameter sweep experiment for the ballistic benchmark with 10 dimensions.

    Args:
        optimizer_type: The type of the optimizer to use.
        optimizer_name: The name to display for this optimizer.
        hparams_list: A list of hyperparameter dictionaries to sweep over.
    """
    experiment_suite = experiment_suite_factory.create_experiment_suite(
        name="ballistic_10",
        description="10-dimensional ballistic benchmark.",
        seeds=list(range(5)),
        benchmark_name="ballistic",
        benchmark_hparams={"dimension": 10},
        max_steps=1000,
        optimizer_specs=[
            {
                "name": f"{optimizer_name} {hparams}",
                "type": optimizer_type,
                "hparams": hparams,
            }
            for hparams in hparams_list
        ],
    )
    # Run the experiment suite, saving to a sub-directory specifically for the optimizer
    # being tested.
    results_dir = os.path.join(BASE_RESULTS_DIR, optimizer_name)
    params_path, trace_paths, _ = experiment_suite.run(results_dir)
    fig = visualization.plot_optimizer_progress(params_path, trace_paths)
    fig.savefig(os.path.join(results_dir, "optimizer_progress.png"))


if __name__ == "__main__":
    # Define optimizers and the range of parameters to sweep over
    optimizers_to_test = [
        # (Type, Name, Hyperparameters)
        # # Gradient descent
        # ("GD", "GD", [{"step_size": step_size} for step_size in [1e-1, 1e-2, 1e-3]]),
        # # MALA
        # (
        #     "MCMC",
        #     "MALA",
        #     [
        #         {
        #             "use_gradients": True,
        #             "use_metropolis": True,
        #             "step_size": step_size,
        #         }
        #         for step_size in [1e-1, 1e-2, 1e-3]
        #     ],
        # ),
        # # RMH
        # (
        #     "MCMC",
        #     "RMH",
        #     [
        #         {
        #             "use_gradients": False,
        #             "use_metropolis": True,
        #             "step_size": step_size,
        #         }
        #         for step_size in [1e-1, 1e-2, 1e-3]
        #     ],
        # ),
        # # REINFORCE/vanilla policy gradient
        # (
        #     "VPG",
        #     "REINFORCE",
        #     [
        #         {"step_size": step_size, "perturbation_stddev": stddev}
        #         for step_size in [1e-1, 1e-2, 1e-3]
        #         for stddev in [0.5, 0.1, 0.05]
        #     ],
        # ),
        # batched gradient descent
        (
            "BGD",
            "BGD",
            [
                {
                    "step_size": step_size,
                    "smoothing_std": stddev,
                    "n_samples": n_samples,
                }
                for step_size in [1e-3, 1e-4]
                for stddev in [0.1, 0.05]
                for n_samples in [5, 10]
            ],
        ),
    ]

    for optimizer_type, optimizer_name, hparams in optimizers_to_test:
        print(f"Running parameter sweep for {optimizer_name}...")
        sweep_ballistic_10(optimizer_type, optimizer_name, hparams)
