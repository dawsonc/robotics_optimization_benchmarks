"""Define a base class for implementing optimization algorithms."""
from abc import ABC
from abc import abstractmethod

import chex
from beartype import beartype
from beartype.typing import Any
from beartype.typing import Callable
from beartype.typing import Dict
from beartype.typing import Tuple
from jaxtyping import Array
from jaxtyping import Float
from jaxtyping import jaxtyped

from robotics_optimization_benchmarks.types import DecisionVariable
from robotics_optimization_benchmarks.types import PRNGKeyArray


@chex.dataclass
class OptimizerState:
    """A struct for storing the state of an optimizer.

    Subclasses of `Optimizer` may define specialized subclasses of `OptimizerState`

    Attributes:
        solution: the current solution.
        objective_value: the value of the objective function at the current solution.
        cumulative_function_calls: the cumulative number of objective function or
            gradient calls.
    """

    solution: DecisionVariable
    objective_value: Float[Array, ""]
    cumulative_function_calls: int


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

    @property
    @beartype
    def description(self) -> str:
        """Get a string description of this optimizer."""
        return self.name

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
    @beartype
    def to_dict(self) -> Dict[str, Any]:
        """Get a dictionary containing the parameters to initialize this optimizer."""

    @abstractmethod
    @jaxtyped
    @beartype
    def make_step(
        self,
        objective_fn: Callable[[DecisionVariable], Float[Array, ""]],
        initial_solution: DecisionVariable,
    ) -> Tuple[
        OptimizerState, Callable[[OptimizerState, PRNGKeyArray], OptimizerState]
    ]:
        """Initialize the state of the optimizer and return the step function.

        Args:
            objective_fn: the objective function to minimize.
            initial_solution: the initial solution.

        Returns:
            initial_state: The initial state of the optimizer.
            step_fn: A function that takes the current state of the optimizer and a PRNG
                key and returns the next state of the optimizer, executing one step of
                the optimization algorithm.
        """
