"""Implement an interface to `Optax <https://optax.readthedocs.io/en/latest/api.html>`_."""
import chex
import jax
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
        objective_value: the value of the objective function at the current solution.
        cumulative_function_calls: the cumulative number of objective function calls.
        debug: any debug information to log
        optax_state: the state of the wrapped Optax optimizer.
    """

    optax_state: NamedTuple


class Optax(Optimizer):
    """Wrapper around arbitrary Optax optimizers.

    Args:
        optimizer_name: the name of the Optax optimizer to use.
        params: a dictionary of keyword arguments to pass to the initializer of the
            Optax optimizer.

    Raises:
        ValueError: if the specified Optax optimizer is not found.
    """

    _name: str = "Optax"
    _optimizer_name: str
    _optimizer: optax.GradientTransformation
    _optimizer_params: Dict[str, Any]

    @beartype
    def __init__(self, optimizer_name: str, params: dict[str, Any]):
        """Initialize the optimizer."""
        super().__init__()
        if not hasattr(optax, optimizer_name):
            raise ValueError(f"Optax optimizer {optimizer_name} not found.")
        self._optimizer_name = optimizer_name
        self._optimizer = getattr(optax, optimizer_name)(**params)
        self._optimizer_params = params

    @property
    @beartype
    def description(self) -> str:
        """Get a string description of this optimizer."""
        return f"{self.name} ({self._optimizer_name})"

    @beartype
    def to_dict(self) -> Dict[str, Any]:
        """Get a dictionary containing the parameters to initialize this optimizer."""
        return {
            "optimizer_name": self._optimizer_name,
            "params": self._optimizer_params,
        }

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
            objective_value=objective_fn(initial_solution),
            cumulative_function_calls=0,
            optax_state=self._optimizer.init(initial_solution),
            debug={},  # nothing special to log
        )

        # Auto-diff the objective to pass into our step function
        value_and_grad_fn = jax.value_and_grad(objective_fn)

        # Define the step function (baking in the objective and gradient functions).
        @jaxtyped
        @beartype
        def step(state: OptaxOptimizerState, _: PRNGKeyArray) -> OptaxOptimizerState:
            """Take one step towards minimizing the objective.

            Args:
                state: the current state of the optimizer.
                _: a random number generator key (unused).

            Returns:
                The solution to the optimization problem.
            """
            value, gradient = value_and_grad_fn(state.solution)
            updates, next_optax_state = self._optimizer.update(
                gradient, state.optax_state, state.solution
            )
            next_solution = optax.apply_updates(state.solution, updates)

            return OptaxOptimizerState(
                solution=next_solution,
                objective_value=value,
                # We evaluated the gradient once to step to the next solution.
                cumulative_function_calls=state.cumulative_function_calls + 1,
                optax_state=next_optax_state,
                debug={},  # nothing special to log
            )

        return initial_state, step
