"""Define a base class for implementing benchmark optimization problems."""
from abc import ABC
from abc import abstractmethod

from beartype import beartype
from beartype.typing import Any
from beartype.typing import Dict
from jax.random import PRNGKeyArray
from jaxtyping import Array
from jaxtyping import Float
from jaxtyping import PyTree
from jaxtyping import jaxtyped


# Define a convenience type for decision variables
DecisionVariable = PyTree[Float[Array, "..."]]


class Benchmark(ABC):
    """An abstract base class for implementing benchmark optimization problems.

    Subclass `Benchmark` to implement new optimization problems. Each subclass must
    define the following methods:
        * `sample_initial_guess`: sample a random initial solution to the problem.
        * `evaluate_solution`: evaluate the objective function at a given solution,
            returning a scalar that should be minimized.
        * `render_solution`: visualize a solution to the problem.

    Benchmarks are assumed to define minimization problems.

    Benchmarks must accept PyTrees (i.e. arrays or nested lists/dicts of arrays) as
    solutions/decision variables to ensure compatibility with JAX.
    """

    _name: str = "AbstractBenchmark"

    @classmethod
    @property
    @beartype
    def name(cls) -> str:
        """Get the name of the benchmark."""
        return cls._name

    @classmethod
    @beartype
    def from_dict(cls, dict: Dict[str, Any]) -> "Benchmark":
        """Create a new benchmark instance from a dictionary.

        Args:
            dict: a dictionary containing the parameters to initialize the benchmark.

        Raises:
            TypeError: if the dictionary contains invalid parameters.  # noqa: DAR402

        Returns:
            A new benchmark instance.
        """
        return cls(**dict)

    @abstractmethod
    @jaxtyped
    @beartype
    def sample_initial_guess(self, key: PRNGKeyArray) -> DecisionVariable:
        """Sample a random initial solution to the problem.

        Args:
            key: a JAX PRNG key used to sample the solution.

        Returns:
            A random initial solution to the problem.
        """

    @abstractmethod
    @jaxtyped
    @beartype
    def evaluate_solution(self, solution: DecisionVariable) -> Float[Array, ""]:
        """Evaluate the objective function at a given solution.

        Args:
            solution: the solution at which to evaluate the objective function.

        Returns:
            The objective function evaluated at the given solution.
        """

    @abstractmethod
    @beartype
    def render_solution(self, solution: DecisionVariable) -> None:
        """Visualize a solution to the problem.

        Args:
            solution: the solution to visualize.
        """
