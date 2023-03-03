"""Implement an interface to `Optax <https://optax.readthedocs.io/en/latest/api.html>`_."""
import chex
import jax
import jax.tree_util as jtu
import optax
from beartype import beartype
from beartype.typing import Any
from beartype.typing import Callable
from beartype.typing import Dict
from beartype.typing import NamedTuple
from beartype.typing import Tuple
from jaxtyping import Array
from jaxtyping import Float
from jaxtyping import jaxtyped

from robotics_optimization_benchmarks.optimizers.optimizer import Optimizer
from robotics_optimization_benchmarks.optimizers.optimizer import OptimizerState
from robotics_optimization_benchmarks.types import DecisionVariable
from robotics_optimization_benchmarks.types import PRNGKeyArray


@chex.dataclass
class OptaxOptimizerState(OptimizerState):
    """A struct for storing the state of an Optax-derived optimizer.

    Attributes:
        solution: the current solution.
        cumulative_objective_calls: the cumulative number of objective function calls.
        cumulative_gradient_calls: the cumulative number of evaluations of the gradient
        optax_state: the state of the wrapped Optax optimizer.
    """

    optax_state: NamedTuple


class Optax(Optimizer):
    """Wrapper around arbitrary Optax optimizers."""

    _name: str = "Optax"
    _algorithm_name: str
    _algorithm: optax.GradientTransformation
    _algorithm_params: Dict[str, Any]

    @beartype
    def __init__(self, optimizer_name: str, params: dict[str, Any]):
        """Initialize the optimizer.

        Args:
            optimizer_name: the name of the Optax optimizer to use.
            params: a dictionary of keyword arguments to pass to the initializer of the
                Optax optimizer.
        """
        super().__init__()
        if not hasattr(optax, optimizer_name):
            raise ValueError(f"Optax optimizer {optimizer_name} not found.")
        self._algorithm_name = optimizer_name
        self._algorithm = getattr(optax, optimizer_name)(**params)
        self._algorithm_params = params

    @property
    @beartype
    def description(self) -> str:
        """Get a string description of this optimizer."""
        return f"{self.name} ({self._algorithm_name})"

    @jaxtyped
    @beartype
    def make_step(
        self,
        objective_fn: Callable[[DecisionVariable], Float[Array, ""]],
        initial_solution: DecisionVariable,
    ) -> Tuple[
        OptaxOptimizerState,
        Callable[[OptaxOptimizerState, PRNGKeyArray], OptaxOptimizerState],
    ]:
        """Initialize the state of the optimizer.

        Args:
            objective_fn: the objective function to minimize.
            initial_solution: the initial solution.

        Returns:
            initial_state: The initial state of the optimizer.
            step_fn: A function that takes the current state of the optimizer and a PRNG
                key and returns the next state of the optimizer, executing one step of
                the optimization algorithm.
        """
        # Create the initial state of the optimizer.
        initial_state = OptaxOptimizerState(
            solution=initial_solution,
            cumulative_objective_calls=0,
            cumulative_gradient_calls=0,
            optax_state=self._algorithm.init(initial_solution),
        )

        # Auto-diff the objective to pass into our step function
        grad_fn = jax.grad(objective_fn)

        # Define the step function (baking in the objective and gradient functions).
        @jaxtyped
        @beartype
        def step(state: OptimizerState, _: PRNGKeyArray) -> OptimizerState:
            """Take one step towards minimizing the objective.

            Args:
                state: the current state of the optimizer.
                _: a random number generator key (unused).

            Returns:
                The solution to the optimization problem.
            """
            gradient = grad_fn(state.solution)
            next_solution = jtu.tree_map(
                lambda x, grad: x - self._step_size * grad, state.solution, gradient
            )

            return OptimizerState(
                solution=next_solution,
                # We evaluated the gradient once to step to the next solution.
                cumulative_gradient_calls=state.cumulative_gradient_calls + 1,
                # We didn't need to call the objective function itself.
                cumulative_objective_calls=state.cumulative_objective_calls,
            )

        return initial_state, step
