"""Implement the gradient descent optimizer."""
import jax
import jax.tree_util as jtu
from beartype import beartype
from beartype.typing import Callable
from beartype.typing import Tuple, Dict, Any
from jaxtyping import Array
from jaxtyping import Float
from jaxtyping import jaxtyped

from robotics_optimization_benchmarks.optimizers.optimizer import Optimizer
from robotics_optimization_benchmarks.optimizers.optimizer import OptimizerState
from robotics_optimization_benchmarks.types import DecisionVariable
from robotics_optimization_benchmarks.types import PRNGKeyArray


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
        # Create the initial state of the optimizer.
        initial_state = OptimizerState(
            solution=initial_solution,
            cumulative_objective_calls=0,
            cumulative_gradient_calls=0,
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
