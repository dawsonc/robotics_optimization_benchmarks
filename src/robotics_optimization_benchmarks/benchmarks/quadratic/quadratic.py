"""Define a simple quadratic optimization benchmark."""
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from beartype import beartype
from jaxtyping import Array
from jaxtyping import Float
from jaxtyping import jaxtyped

from robotics_optimization_benchmarks.benchmarks.benchmark import Benchmark
from robotics_optimization_benchmarks.benchmarks.benchmark import DecisionVariable
from robotics_optimization_benchmarks.types import PRNGKeyArray


class Quadratic(Benchmark):
    """Define a simple quadratic optimization benchmark.

    Mimizes the problem

    U(x) = x^T x / n

    with x in R^n. Not intended to be a challenging problem, but rather a simple test
    case for optimizers.

    Attributes:
        dimension: The dimension of the problem.
    """

    _name = "quadratic"

    def __init__(self, dimension: int = 10):
        """Initialize the benchmark.

        Args:
            dimension: The dimension of the problem.
        """
        self.dimension = dimension

    @jaxtyped
    @beartype
    def sample_initial_guess(self, key: PRNGKeyArray) -> DecisionVariable:
        """Sample a random initial solution to the problem.

        x_0 ~ N(1, I_{n x n})

        Args:
            key: a JAX PRNG key used to sample the solution.

        Returns:
            A random initial solution to the problem in R^{self.dimension}.
        """
        return jax.random.normal(key, shape=(self.dimension,)) + 1.0

    @jaxtyped
    @beartype
    def evaluate_solution(self, solution: DecisionVariable) -> Float[Array, ""]:
        """Evaluate the objective function at a given solution.

        Args:
            solution: the solution at which to evaluate the objective function.

        Returns:
            The objective function evaluated at the given solution.
        """
        return jnp.dot(solution, solution) / self.dimension

    def render_solution(self, solution: DecisionVariable) -> plt.figure:
        """Visualize a solution to the problem.

        Args:
            solution: the solution to visualize.

        Returns:
            A matplotlib figure containing the visualization.
        """
        # Create the figure to plot (we'll return this to the caller)
        fig, axis = plt.subplots(1, 1, figsize=(8, 8))

        # Plot the solution relative to the optimal cost
        axis.scatter(0, 0, color="k", marker="_", s=200, label="Optimal")
        axis.scatter(
            0,
            self.evaluate_solution(solution),
            color="r",
            marker="x",
            s=200,
            label="Solution",
        )

        return fig
