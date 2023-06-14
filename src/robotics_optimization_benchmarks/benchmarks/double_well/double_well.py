"""Define a double-well optimization benchmark."""
import jax
import matplotlib.pyplot as plt
from beartype import beartype
from beartype.typing import Any
from beartype.typing import BinaryIO
from beartype.typing import Dict
from jaxtyping import Array
from jaxtyping import Float
from jaxtyping import jaxtyped

from robotics_optimization_benchmarks.benchmarks import Benchmark
from robotics_optimization_benchmarks.types import DecisionVariable
from robotics_optimization_benchmarks.types import PRNGKeyArray


class DoubleWell(Benchmark):
    """Define a simple double well optimization benchmark.

    Mimizes the problem

    U(x) = x^4 - x^2 - 0.5 * x^3

    with x in R^n. Not intended to be a challenging problem, but rather a simple test
    case for optimizers (but it does have a local minimum around x=-0.5, the global
    optimum is at x=1).

    Attributes:
        dimension: The dimension of the problem.
    """

    _name: str = "double_well"

    dimension: int

    def __init__(self, dimension: int = 10):
        """Initialize the benchmark.

        Args:
            dimension: The dimension of the problem.
        """
        self.dimension = dimension

    @beartype
    def to_dict(self) -> Dict[str, Any]:
        """Get a dictionary containing the parameters to initialize the benchmark."""
        return {"dimension": self.dimension}

    @jaxtyped
    @beartype
    def sample_initial_guess(self, key: PRNGKeyArray) -> DecisionVariable:
        """Sample a random initial solution to the problem.

        x_0 ~ N(-1.5, 0.1 * I_{n x n})

        Args:
            key: a JAX PRNG key used to sample the solution.

        Returns:
            A random initial solution to the problem in R^{self.dimension}.
        """
        return 0.1 * jax.random.normal(key, shape=(self.dimension,)) - 1.5

    @jaxtyped
    @beartype
    def evaluate_solution(self, solution: DecisionVariable) -> Float[Array, ""]:
        """Evaluate the objective function at a given solution.

        Args:
            solution: the solution at which to evaluate the objective function.

        Returns:
            The objective function evaluated at the given solution.
        """
        return (solution**4 - solution**2 - 0.5 * solution**3).sum()

    def render_solution(
        self, solution: DecisionVariable, save_to: str | BinaryIO
    ) -> None:
        """Visualize a solution to the problem, saving the visualization.

        Args:
            solution: the solution to visualize.
            save_to: the path or file-like object to save the visualization to.
        """
        # Create the figure to plot (we'll return this to the caller)
        fig, axis = plt.subplots(1, 1, figsize=(8, 8))

        # Plot the solution relative to the optimal cost
        axis.scatter(
            0,
            -0.5194 * self.dimension,
            color="k",
            marker="_",
            s=200,
            label="Global optimum",
        )
        axis.scatter(
            0,
            -0.1279 * self.dimension,
            color="b",
            marker="_",
            s=200,
            label="Local optimum",
        )
        axis.scatter(
            0,
            self.evaluate_solution(solution),
            color="r",
            marker="x",
            s=200,
            label="Solution",
        )

        # Save the figure and clean up
        fig.savefig(save_to, format="png")
        plt.close(fig)
