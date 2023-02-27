"""Implement the vanilla policy gradient optimizer with Gaussian additive noise."""
import chex
import jax
import jax.flatten_util
import jax.numpy as jnp
import jax.random as jrandom
import jax.tree_util as jtu
from beartype import beartype
from beartype.typing import Callable
from beartype.typing import Tuple
from jaxtyping import Array
from jaxtyping import Float
from jaxtyping import jaxtyped

from robotics_optimization_benchmarks.optimizers.optimizer import Optimizer
from robotics_optimization_benchmarks.optimizers.optimizer import OptimizerState
from robotics_optimization_benchmarks.types import DecisionVariable
from robotics_optimization_benchmarks.types import PRNGKeyArray


@chex.dataclass
class VPGOptimizerState(OptimizerState):
    """A struct for storing the state of a VPG optimizer.

    Attributes:
        solution: the current solution.
        cumulative_objective_calls: the cumulative number of objective function calls.
        cumulative_gradient_calls: the cumulative number of evaluations of the gradient.
        baseline: the current baseline.
    """

    baseline: Float[Array, ""]


class VPG(Optimizer):
    """Minimize an objective function using the vanilla policy gradient algorithm.

    Uses a moving average of all previously-observed costs as the baseline.

    This is equivalent to the "weight perturbation with an estimated baseline" algorithm
    referenced `here <https://underactuated.mit.edu/rl_policy_search.html>`_, or it
    could be seen as a variant of the vanilla policy gradient algorithm.
    """

    _name: str = "VPG"

    _step_size: float
    _perturbation_stddev: float
    _baseline_update_rate: float

    @beartype
    def __init__(
        self,
        step_size: float = 1e-3,
        perturbation_stddev: float = 0.1,
        baseline_update_rate: float = 0.5,
    ):
        """Initialize the optimizer.

        Args:
            step_size: the learning rate for gradient descent.
            perturbation_stddev: the strength of the perturbation.
            baseline_update_rate: the rate at which the baseline moving average is
                updated. 0 means that the baseline is constant at 0 and 1 means that the
                baseline is equal to the most-recently-observed cost.
        """
        super().__init__()
        self._step_size = step_size
        self._perturbation_stddev = perturbation_stddev
        self._baseline_update_rate = baseline_update_rate

    @beartype
    def to_dict(self) -> dict:
        """Get a dictionary containing the parameters to initialize this optimizer."""
        return {
            "step_size": self._step_size,
            "perturbation_stddev": self._perturbation_stddev,
            "baseline_update_rate": self._baseline_update_rate,
        }

    @jaxtyped
    @beartype
    def make_step(
        self,
        objective_fn: Callable[[DecisionVariable], Float[Array, ""]],
        initial_solution: DecisionVariable,
    ) -> Tuple[
        VPGOptimizerState,
        Callable[[VPGOptimizerState, PRNGKeyArray], VPGOptimizerState],
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
        initial_state = VPGOptimizerState(
            solution=initial_solution,
            baseline=jnp.array(0.0),  # Initialize baseline moving average to zero
            cumulative_objective_calls=0,
            cumulative_gradient_calls=0,
        )

        # Define the step function (baking in the objective and gradient functions).
        @jaxtyped
        @beartype
        def step(state: VPGOptimizerState, key: PRNGKeyArray) -> VPGOptimizerState:
            """Take one step towards minimizing the objective.

            Args:
                state: the current state of the optimizer.
                key: a random number generator key (unused).

            Returns:
                The solution to the optimization problem.
            """
            # Perturb the solution with noise of the same shape
            flat_solution, unravel_fn = jax.flatten_util.ravel_pytree(state.solution)
            noise = self._perturbation_stddev * jrandom.normal(
                key, shape=flat_solution.shape, dtype=flat_solution.dtype
            )
            # Re-shape to match solution and perturb
            beta = unravel_fn(noise)
            perturbed_solution = jtu.tree_map(
                lambda x, beta: x + beta,
                state.solution,
                beta,
            )

            # Get the cost at the perturbed solution
            perturbed_objective_value = objective_fn(perturbed_solution)

            # Step in the direction of the approximate gradient
            advantage = perturbed_objective_value - state.baseline
            next_solution = jtu.tree_map(
                lambda x, b: x
                - self._step_size / self._perturbation_stddev**2 * advantage * b,
                state.solution,
                beta,
            )

            return VPGOptimizerState(
                solution=next_solution,
                # We did not evaluate the gradient.
                cumulative_gradient_calls=state.cumulative_gradient_calls,
                # We called the objective function once.
                cumulative_objective_calls=state.cumulative_objective_calls + 1,
                # Update the moving average baseline
                baseline=(
                    (1 - self._baseline_update_rate) * state.baseline
                    + self._baseline_update_rate * perturbed_objective_value
                ),
            )

        return initial_state, step
