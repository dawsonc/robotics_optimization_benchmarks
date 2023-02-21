"""Define a base class for implementing optimization algorithms."""
from abc import ABC
from abc import abstractmethod

from beartype import beartype
from beartype.typing import Any
from beartype.typing import Callable
from beartype.typing import Dict
from beartype.typing import NamedTuple
from jaxtyping import Array
from jaxtyping import Float
from jaxtyping import jaxtyped

from robotics_optimization_benchmarks.benchmarks.benchmark import DecisionVariable
from robotics_optimization_benchmarks.types import PRNGKeyArray


class OptimizerState(NamedTuple):
    """A struct for storing the state of an optimizer.

    Subclasses of `Optimizer` may define specialized subclasses of `OptimizerState`

    Attributes:
        objective_fn: the objective function to minimize.
        solution: the current solution.
        objective: the current objective value.
        cumulative_objective_calls: the cumulative number of objective function calls.
        cumulative_objective_gradient_calls: the cumulative number of evaluations of
            the gradient of the objective function.
    """

    objective_fn: Callable[[DecisionVariable], Float[Array, ""]]
    solution: DecisionVariable
    objective: Float[Array, ""]

    cumulative_objective_calls: int = 0
    cumulative_objective_gradient_calls: int = 0


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
    def init(
        self,
        objective_fn: Callable[[DecisionVariable], Float[Array, ""]],
        initial_solution: DecisionVariable,
    ) -> OptimizerState:
        """Initialize the state of the optimizer.

        Args:
            objective_fn: the objective function to minimize.
            initial_solution: the initial solution.

        Returns:
            The initial state of the optimizer.
        """
        raise NotImplementedError

    @abstractmethod
    @jaxtyped
    @beartype
    def step(
        self,
        state: OptimizerState,
        rng_key: PRNGKeyArray,
    ) -> OptimizerState:
        """Take one step towards minimizing the objective.

        Args:
            state: the current state of the optimizer.
            rng_key: a random number generator key.

        Returns:
            The solution to the optimization problem.
            The trace of solutions at each step of the optimization process
        """
        raise NotImplementedError
