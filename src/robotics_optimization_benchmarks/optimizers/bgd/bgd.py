"""Implement the batched-gradient descent optimizer."""
import jax
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


class BGD(Optimizer):
    """Minimize an objective function using batched-gradient descent.

    Applies the batched-gradient descent algorithm, which is equivalent to the
    "weight-perturbation" algorithm referenced
    `here <underactuated.mit.edu/rl_policy_search.html>`_ but averaging over
    :code:`n_samples` perturbed samples to reduce the variance of the estimate.

    This optimizer does not use gradients, but uses `n_samples + 1` objective
    evaluations per step (because we use the exact objective value at the current
    solution as our baseline).
    """

    _name: str = "BGD"

    _step_size: float
    _smoothing_std: float
    _n_samples: int

    @beartype
    def __init__(
        self, step_size: float = 0.01, smoothing_std: float = 0.1, n_samples: int = 10
    ):
        """Initialize the optimizer.

        Args:
            step_size: the learning rate for gradient descent.
            smoothing_std: the standard deviation of the smoothing Gaussian.
            n_samples: the number of samples to use for estimating the batched gradient.
        """
        super().__init__()
        self._step_size = step_size
        self._smoothing_std = smoothing_std
        self._n_samples = n_samples

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

        # The zero-order batched gradient estimator is the average of a bunch of
        # individual zero-order gradient estimates, so we need a function to compute
        # those individual estimates.
        def zero_order_grad_estimate(
            solution: DecisionVariable, objective: Float[Array, ""], key: PRNGKeyArray
        ) -> DecisionVariable:
            """Estimate gradient using a single perturbation around the given solution.

            Args:
                solution: the current solution.
                objective: the objective value at the current solution.
                key: a PRNG key.

            Returns:
                The zero-order gradient estimate based on a single perturbation.
            """
            # Perturb the solution using noise with the same shape.
            flat_solution, unravel_fn = jax.flatten_util.ravel_pytree(solution)
            sample = self._smoothing_std * jrandom.normal(
                key, shape=flat_solution.shape, dtype=flat_solution.dtype
            )
            noise = unravel_fn(sample)
            perturbed_solution = jtu.tree_map(lambda x, y: x + y, solution, noise)

            # Evaluate the objective function at the perturbed solution.
            perturbed_objective = objective_fn(perturbed_solution)

            # Compute the zero-order gradient estimate.
            grad_estimate = jtu.tree_map(
                lambda b: (perturbed_objective - objective)
                / self._smoothing_std**2
                * b,
                noise,
            )

            return grad_estimate

        def zero_order_batched_grad_estimate(
            solution: DecisionVariable, objective: Float[Array, ""], key: PRNGKeyArray
        ) -> DecisionVariable:
            """Estimate the batched zero-order gradient of the objective function.

            Do this by perturbing the solution a number of times and estimating the
            gradient from the objective values of the perturbed solutions.

            Args:
                solution: the current solution.
                objective: the objective value at the current solution.
                key: a PRNG key.

            Returns:
                The gradient estimate of the objective function at the given solution.
            """
            # Evaluate the gradient at a bunch of randomly perturbed samples
            keys = jrandom.split(key, self._n_samples)
            grad_estimates = jax.vmap(
                zero_order_grad_estimate, in_axes=(None, None, 0)
            )(solution, objective, keys)

            # Average these gradients to get the batched estimate.
            batched_grad_estimate = jtu.tree_map(
                lambda g: jnp.mean(g, axis=0), grad_estimates
            )

            return batched_grad_estimate

        # Define the step function (baking in the objective and gradient functions).
        @jaxtyped
        @beartype
        def step(state: OptimizerState, key: PRNGKeyArray) -> OptimizerState:
            """Take one step towards minimizing the objective.

            Args:
                state: the current state of the optimizer.
                key: a random number generator key.

            Returns:
                The solution to the optimization problem.
            """
            # Implement gradient descent with the batched zero-order estimate
            current_objective = objective_fn(state.solution)
            gradient = zero_order_batched_grad_estimate(
                state.solution, current_objective, key
            )
            next_solution = jtu.tree_map(
                lambda x, grad: x - self._step_size * grad, state.solution, gradient
            )

            return OptimizerState(
                solution=next_solution,
                # We didn't evaluate the gradient.
                cumulative_gradient_calls=state.cumulative_gradient_calls,
                # We called the objective once to get a baseline and then
                # self._n_samples additional times
                cumulative_objective_calls=state.cumulative_objective_calls
                + 1
                + self._n_samples,
            )

        return initial_state, step
