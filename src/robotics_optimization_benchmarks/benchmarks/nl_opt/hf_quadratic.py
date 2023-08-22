"""Define a quadratic objective contaminated with high frequency noise."""
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


class HFQuadratic(Benchmark):
    r"""Define a quadratic objective contaminated with high frequency noise.

    Given noise frequency limit N and period L, mimize the problem

    U(x) = \\sum_i [ x_i ^ 2 + \\frac{4}{\\pi} \\sum_{n=1,3,...,N} sin(n \\pi x_i / L) / n

    with x in R^n. This is intended to induce changes in behavior as noise strength is
    varied.

    Attributes:
        dimension: The dimension of the problem.
        n_components: The frequency limit of the noise.
        period: The period of the noise.
        noise_scale: The scale of the noise.
    """

    _name: str = "hf_quadratic"

    dimension: int
    n_components: int
    period: float
    noise_scale: float

    def __init__(
        self,
        dimension: int = 10,
        n_components: int = 10,
        period: float = 0.1,
        noise_scale: float = 0.1,
        **_,
    ):
        """Initialize the benchmark.

        Args:
            dimension: The dimension of the problem.
            n_components: The frequency limit of the noise.
            period: The period of the noise.
            noise_scale: The scale of the noise.
            Additional keyword arguments are unused
        """
        self.dimension = dimension
        self.n_components = n_components
        self.period = period
        self.noise_scale = noise_scale

    @beartype
    def to_dict(self) -> Dict[str, Any]:
        """Get a dictionary containing the parameters to initialize the benchmark."""
        return {
            "dimension": self.dimension,
            "n_components": self.n_components,
            "period": self.period,
            "noise_scale": self.noise_scale,
            "lipschitz_constant": self.noise_lipschitz_constant.item(),
        }

    @jaxtyped
    @beartype
    def sample_initial_guess(self, key: PRNGKeyArray) -> DecisionVariable:
        """Sample a random initial solution to the problem.

        x_0 ~ Uniform[-3, 3]

        Args:
            key: a JAX PRNG key used to sample the solution.

        Returns:
            A random initial solution to the problem in R^{self.dimension}.
        """
        return jax.random.uniform(key, shape=(self.dimension,), minval=-3, maxval=3)

    @jaxtyped
    @beartype
    def noise(self, solution: DecisionVariable) -> Float[Array, ""]:
        """Compute the high-frequency noise term.

        Args:
            solution: the solution at which to evaluate the noise.

        Returns:
            The high-frequency noise term.
        """
        # The noise is \\frac{4}{\\pi} \\sum_{n=1,3,...,N} sin(n \\pi x_i / L) / n

        # Get an array of all frequencies to include in the noise
        frequencies = jnp.arange(1, self.n_components * 2, 2)

        # Construct the fourier series (n_components terms for each dimension)
        # Individual series term: Z x R -> R
        fourier_term = (
            lambda n, x_i: jnp.sin(  # pylint: disable=unnecessary-lambda-assignment
                n * jnp.pi * x_i / self.period
            )
            / n
        )
        # Full series for individual dimension: R -> R
        noise_terms = (
            lambda x_i: jax.vmap(  # pylint: disable=unnecessary-lambda-assignment
                fourier_term, in_axes=(0, None)
            )(frequencies, x_i).sum()
        )
        # Full series for all dimensions: R^n -> R
        noise = jax.vmap(noise_terms)(solution).sum()

        return noise * 4 / jnp.pi * self.noise_scale

    @property
    def noise_lipschitz_constant(self) -> Float[Array, ""]:
        """Get the Lipschitz constant of the noise term."""
        # Maximum slope occurs at x = 0, so we can just evaluate the derivative there
        max_grad = jax.grad(self.noise)(jnp.zeros(self.dimension))
        return jnp.linalg.norm(max_grad)

    @jaxtyped
    @beartype
    def evaluate_solution(self, solution: DecisionVariable) -> Float[Array, ""]:
        """Evaluate the objective function at a given solution.

        Args:
            solution: the solution at which to evaluate the objective function.

        Returns:
            The objective function evaluated at the given solution.
        """
        return jnp.dot(solution, solution) + self.noise(solution)

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


if __name__ == "__main__":
    # Plot the cost for a 1D problem for a range of solutions
    problem = HFQuadratic(dimension=1, n_components=2, period=0.1, noise_scale=0.1)

    # Scan over a bunch of x values
    N = 1000
    x = jnp.linspace(-3.0, 3.0, N).reshape(-1, 1)
    y = jax.vmap(problem.evaluate_solution)(x)

    # Plot the cost surface
    fig, ax = plt.subplots()
    ax.plot(x, y)

    # Pretty up the plot
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"Cost")

    plt.show()
