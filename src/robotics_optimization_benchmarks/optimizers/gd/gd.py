"""Implement the gradient descent optimizer."""
import chex
import jax
import jax.tree_util as jtu
from beartype import beartype
from beartype.typing import Any
from beartype.typing import Callable
from beartype.typing import Dict
from beartype.typing import Tuple
from jaxtyping import Array
from jaxtyping import Float
from jaxtyping import jaxtyped

from robotics_optimization_benchmarks.optimizers.optimizer import Optimizer
from robotics_optimization_benchmarks.optimizers.optimizer import OptimizerState
from robotics_optimization_benchmarks.types import DecisionVariable
from robotics_optimization_benchmarks.types import PRNGKeyArray


@chex.dataclass
class GDOptimizerState(OptimizerState):
    """A struct for storing the state of a GD optimizer.

    Attributes:
        solution: the current solution.
        cumulative_objective_calls: the cumulative number of objective function calls.
        cumulative_gradient_calls: the cumulative number of evaluations of the gradient
        grad: the gradient of the objective function at the current
            solution.
    """

    grad: DecisionVariable


class GD(Optimizer):
    """Minimize an objective function using gradient descent."""

    _name: str = "GD"

    @beartype
    def __init__(self, step_size: float = 0.01):
        """Initialize the optimizer.

        Args:
            step_size: the learning rate for gradient descent.
        """
        super().__init__()
        self._step_size = step_size

    @beartype
    def to_dict(self) -> Dict[str, Any]:
        """Get a dictionary containing the parameters to initialize this optimizer."""
        return {"step_size": self._step_size}

    @jaxtyped
    @beartype
    def make_step(
        self,
        objective_fn: Callable[[DecisionVariable], Float[Array, ""]],
        initial_solution: DecisionVariable,
    ) -> Tuple[
        OptimizerState, Callable[[OptimizerState, PRNGKeyArray], OptimizerState]
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
        # Auto-diff the objective to pass into our step function
        value_and_grad_fn = jax.value_and_grad(objective_fn)

        # Create the initial state of the optimizer.
        value, grad = value_and_grad_fn(initial_solution)
        initial_state = GDOptimizerState(
            solution=initial_solution,
            objective_value=value,
            cumulative_function_calls=0,
            grad=grad,
        )

        # Define the step function (baking in the objective and gradient functions).
        @jaxtyped
        @beartype
        def step(state: GDOptimizerState, _: PRNGKeyArray) -> GDOptimizerState:
            """Take one step towards minimizing the objective.

            Args:
                state: the current state of the optimizer.
                _: a random number generator key (unused).

            Returns:
                The solution to the optimization problem.
            """
            next_solution = jtu.tree_map(
                lambda x, grad: x - self._step_size * grad, state.solution, state.grad
            )

            # Update the value and gradient
            value, gradient = value_and_grad_fn(next_solution)

            return GDOptimizerState(
                solution=next_solution,
                objective_value=value,
                # We evaluated the gradient once to step to the next solution.
                cumulative_function_calls=state.cumulative_function_calls + 1,
                grad=gradient,
            )

        return initial_state, step
