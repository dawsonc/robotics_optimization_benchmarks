"""Define the Himmelblau optimization benchmark."""
import jax
import jax.numpy as jnp
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


class Himmelblau(Benchmark):
    """Define a benchmark for the Himmelblau function.

    Mimizes the function defined here:
    https://en.wikipedia.org/wiki/Himmelblau%27s_function

    This function has four identical global minima at f = 0 in 2D.

    Only permits even dimensions.

    Attributes:
        dimension: The dimension of the problem.
    """

    _name: str = "himmelblau"

    dimension: int

    def __init__(self, dimension: int = 10):
        """Initialize the benchmark.

        Args:
            dimension: The dimension of the problem.

        Raises:
            ValueError: if the dimension is not even.
        """
        if dimension % 2 != 0:
            raise ValueError("Only even dimensions are permitted.")

        self.dimension = dimension

    @beartype
    def to_dict(self) -> Dict[str, Any]:
        """Get a dictionary containing the parameters to initialize the benchmark."""
        return {"dimension": self.dimension}

    @jaxtyped
    @beartype
    def sample_initial_guess(self, key: PRNGKeyArray) -> DecisionVariable:
        """Sample a random initial solution to the problem.

        x_0 ~ Uniform[-5, 5]

        Args:
            key: a JAX PRNG key used to sample the solution.

        Returns:
            A random initial solution to the problem in R^{self.dimension}.
        """
        return jax.random.uniform(key, shape=(self.dimension,), minval=-5, maxval=5)

    @jaxtyped
    @beartype
    def evaluate_solution(self, solution: DecisionVariable) -> Float[Array, ""]:
        """Evaluate the objective function at a given solution.

        Args:
            solution: the solution at which to evaluate the objective function.

        Returns:
            The objective function evaluated at the given solution.
        """
        x, y = solution[::2], solution[1::2]
        return jnp.sum((x**2 + y - 11) ** 2 + (x + y**2 - 7) ** 2)

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
        axis.scatter(0, 0, color="k", marker="_", s=200, label="Optimal")
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
