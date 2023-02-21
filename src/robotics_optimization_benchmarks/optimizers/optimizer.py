"""Define a base class for implementing optimization algorithms."""
from abc import ABC
from abc import abstractmethod

from beartype import beartype
from beartype.typing import Any
from beartype.typing import Callable
from beartype.typing import Dict
from jaxtyping import Array
from jaxtyping import Float
from jaxtyping import jaxtyped

from robotics_optimization_benchmarks.benchmarks.benchmark import DecisionVariable
from robotics_optimization_benchmarks.types import PRNGKeyArray


class Optimizer(ABC):
    """An abstract base class for implementing optimization algorithms.

    Subclass `Optimizer` to implement new optimization algorithms. Each subclass must
    implement the abstract methods defined in this class for solving minimization
    problems.
    """

    _name: str = "AbstractOptimizer"

    @classmethod
    @property
    @beartype
    def name(cls) -> str:
        """Get the name of the optimizer."""
        return cls._name

    @classmethod
    @beartype
    def from_dict(cls, params: Dict[str, Any]) -> "Optimizer":
        """Create a new optimizer instance from a dictionary.

        Args:
            params: a dictionary containing the parameters to initialize the optimizer.

        Raises:
            TypeError: if the dictionary contains invalid parameters.  # noqa: DAR402

        Returns:
            A new optimizer instance.
        """
        return cls(**params)

    @abstractmethod
    @jaxtyped
    @beartype
    def solve(
        self,
        objective: Callable[[DecisionVariable], Float[Array, ""]],
        initial_solution: DecisionVariable,
        rng_key: PRNGKeyArray,
    ) -> DecisionVariable:
        """Solve the optimization problem by minimizing the objective.

        Args:
            objective: a function mapping decision variables to scalar objective values.
            initial_solution: the initial solution from which to start the optimization.
            rng_key: a random number generator key.

        Returns:
            The solution to the optimization problem.
        """
        raise NotImplementedError
