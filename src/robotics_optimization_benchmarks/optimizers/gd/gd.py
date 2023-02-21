"""Implement the gradient descent optimizer."""
import jax
import jax.tree_util as jtu
from beartype import beartype
from beartype.typing import Callable
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
        return OptimizerState(
            objective_fn=objective_fn,
            solution=initial_solution,
        )

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
        """
        gradient = jax.grad(state.objective_fn)(state.solution)
        next_solution = jtu.tree_map(
            lambda x, grad: x - self._step_size * grad, state.solution, gradient
        )

        return OptimizerState(
            objective_fn=state.objective_fn,
            solution=next_solution,
            # We evaluated the gradient once to step to the next solution.
            cumulative_gradient_calls=state.cumulative_gradient_calls + 1,
            # We didn't need to call the objective function itself.
            cumulative_objective_calls=state.cumulative_objective_calls,
        )
