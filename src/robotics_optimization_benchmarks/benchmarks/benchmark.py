"""Define a base class for implementing benchmark optimization problems."""
from abc import ABC
from abc import abstractmethod

from beartype import beartype
from beartype.typing import Any
from beartype.typing import BinaryIO
from beartype.typing import Dict
from jaxtyping import Array
from jaxtyping import Float
from jaxtyping import jaxtyped

from robotics_optimization_benchmarks.types import DecisionVariable
from robotics_optimization_benchmarks.types import PRNGKeyArray


class Benchmark(ABC):
    """An abstract base class for implementing benchmark optimization problems.

    Subclass `Benchmark` to implement new optimization problems. Each subclass must
    define the abstract methods defined in this class for sampling random solutions,
    evaluating potential solutions, and rendering solutions.

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
    def from_dict(cls, params: Dict[str, Any]) -> "Benchmark":
        """Create a new benchmark instance from a dictionary.

        Args:
            params: a dictionary containing the parameters to initialize the benchmark.

        Raises:
            TypeError: if the dictionary contains invalid parameters.  # noqa: DAR402

        Returns:
            A new benchmark instance.
        """
        return cls(**params)

    @abstractmethod
    @beartype
    def to_dict(self) -> Dict[str, Any]:
        """Get a dictionary containing the parameters to initialize the benchmark."""

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
    def render_solution(
        self, solution: DecisionVariable, save_to: str | BinaryIO
    ) -> None:
        """Visualize a solution to the problem, saving the visualization.

        Args:
            solution: the solution to visualize.
            save_to: the path or file-like object to save the visualization to.
        """
