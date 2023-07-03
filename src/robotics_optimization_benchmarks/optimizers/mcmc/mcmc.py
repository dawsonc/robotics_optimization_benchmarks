"""Define a sampling based MCMC optimizer."""
import operator

import chex
import jax
import jax.flatten_util
import jax.numpy as jnp
import jax.random as jrandom
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
class MCMCOptimizerState(OptimizerState):
    """A struct for storing the state of an MCMC optimizer.

    Attributes:
        solution: the current solution.
        cumulative_objective_calls: the cumulative number of objective function calls.
        cumulative_gradient_calls: the cumulative number of evaluations of the gradient
        logdensity: the negative of the objective value at the current solution.
        logdensity_grad: the negative gradient of the objective function at the current
            solution.
    """

    logdensity: Float[Array, ""]
    logdensity_grad: DecisionVariable


class MCMC(Optimizer):
    """Define a base class for various MCMC samplers."""

    _name: str = "MCMC"

    _step_size: float
    # Determine whether this is MALA (Metropolis-Adjusted Langevin Algorithm; uses
    # gradients and Metropolis accept/reject), ULA (Unadjusted Langevin Algorithm; uses
    # gradients but no accept/reject), or RMH (random-walk Metropolis-Hastings; no
    # gradients)
    _uses_gradients: bool
    _metropolis_adjusted: bool
    # Add scaling to the objective (sometimes helps MCMC methods find optima)
    _objective_scale: float

    @property
    def description(self) -> str:
        """Get a string description of this optimizer."""
        if self._uses_gradients and self._metropolis_adjusted:
            return "MALA"
        if self._uses_gradients and not self._metropolis_adjusted:
            return "ULA"
        else:
            return "RMH"

    @beartype
    def __init__(
        self,
        use_gradients: bool,
        use_metropolis: bool,
        step_size: float = 1e-3,
        objective_scale: float = 1.0,
    ):
        """Initialize the MCMC optimizer.

        To get a MALA sampler, set `use_gradients=True` and `use_metropolis=True`.
        To get a ULA sampler, set `use_gradients=True` and `use_metropolis=False`.
        To get an RMH sampler, set `use_gradients=False` and `use_metropolis=True`.

        Args:
            use_gradients: whether to use gradients.
            use_metropolis: whether to use a Metropolis accept/reject step.
            step_size: the step size of the MCMC sampler.
            objective_scale: a scaling factor to apply to the objective function.

        Raises:
            ValueError: if both `use_gradients` and `use_metropolis` are False.
        """
        # Sanity check: if both use_gradients and use_metropolis are False, then
        # this just executes a random walk in state space and that's no good to anybody
        if not use_gradients and not use_metropolis:
            raise ValueError(
                "Cannot create an MCMC optimizer with use_gradients and use_metroplis "
                "both False."
            )

        super().__init__()
        self._step_size = step_size
        self._uses_gradients = use_gradients
        self._metropolis_adjusted = use_metropolis
        self._objective_scale = objective_scale

    @beartype
    def to_dict(self) -> Dict[str, Any]:
        """Get a dictionary containing the parameters to initialize this optimizer."""
        return {
            "use_gradients": self._uses_gradients,
            "use_metropolis": self._metropolis_adjusted,
            "step_size": self._step_size,
            "objective_scale": self._objective_scale,
        }

    @jaxtyped
    @beartype
    def transition_log_probability(
        self,
        state: MCMCOptimizerState,
        new_state: MCMCOptimizerState,
    ) -> Float[Array, ""]:
        """Compute the log probability of transitioning from one state to another.

        Args:
            state: the current state.
            new_state: the proposed state.

        Returns:
            The log probability of transitioning from the current state to the proposed
            state.
        """
        # Based on the MALA implementation in Blackjax, but extended to allow
        # turning it into RMH/gradient descent

        # If we're not using gradients, zero out the gradient in a JIT compatible way
        grad = jax.lax.cond(
            self._uses_gradients,
            lambda x: x,  # if using gradients, pass them through
            lambda x: jtu.tree_map(jnp.zeros_like, x),  # otherwise, zero them out
            state.logdensity_grad,
        )

        # Theta is the difference between the new and old states, adjusted by the
        # gradient, which gives just the random diffusion component of the proposed move
        theta = jtu.tree_map(
            lambda new_x, x, g: new_x - x - self._step_size * g,
            new_state.solution,
            state.solution,
            grad,
        )
        # theta_dot is the squared magnitude of the diffusion component
        theta_dot = jtu.tree_reduce(
            operator.add, jtu.tree_map(lambda x: jnp.sum(x * x), theta)
        )

        transition_log_probability = -0.25 * (1.0 / self._step_size) * theta_dot
        return transition_log_probability

    @jaxtyped
    @beartype
    def make_step(
        self,
        objective_fn: Callable[[DecisionVariable], Float[Array, ""]],
        initial_solution: DecisionVariable,
    ) -> Tuple[
        MCMCOptimizerState,
        Callable[[MCMCOptimizerState, PRNGKeyArray], MCMCOptimizerState],
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
        # Convert the objective function to a log density function by negating it
        # This means that low costs -> high densitites -> more likely samples
        logdensity_fn = lambda x: -self._objective_scale * objective_fn(x)

        # Wrap the objective function to return its value and gradient, then get
        # the initial objective value and gradient (caching these in the state
        # halves the number of objective function and gradient calls we need).
        logdensity_and_grad_fn = jax.value_and_grad(logdensity_fn)
        logdensity, logdensity_grad = logdensity_and_grad_fn(initial_solution)

        # Initialize the state.
        initial_state = MCMCOptimizerState(
            solution=initial_solution,
            objective_value=-logdensity,
            logdensity=logdensity,
            logdensity_grad=logdensity_grad,
            cumulative_function_calls=1,
        )

        # Define the step function (baking in the objective and gradient functions).
        @jaxtyped
        @beartype
        def step(
            state: MCMCOptimizerState, prng_key: PRNGKeyArray
        ) -> MCMCOptimizerState:
            """Take one step of the MCMC sampler.

            Args:
                state: the current state of the optimizer.
                prng_key: a PRNG key.

            Returns:
                The next state of the optimizer.
            """
            # Split the key
            proposal_key, acceptance_key = jrandom.split(prng_key)

            # Propose a new state with some random perturbation + (optional) gradient-
            # based drift

            # Generate noise with the same shape as the solution pytree
            flat_solution, unravel_fn = jax.flatten_util.ravel_pytree(state.solution)
            sample = jrandom.normal(
                proposal_key, shape=flat_solution.shape, dtype=flat_solution.dtype
            )
            noise = unravel_fn(sample)
            proposed_delta_x = jnp.sqrt(2 * self._step_size) * noise

            # Add gradients to the proposed move if using
            proposed_delta_x = jax.lax.cond(
                self._uses_gradients,
                # Add gradients if using
                lambda dx: jtu.tree_map(
                    lambda leaf, grad: leaf + self._step_size * grad,
                    dx,
                    state.logdensity_grad,
                ),
                # Otherwise, don't add anything
                lambda dx: dx,
                proposed_delta_x,
            )

            # Use the proposed change in x to propose a new state
            proposed_solution = jtu.tree_map(
                lambda leaf, dx: leaf + dx, state.solution, proposed_delta_x
            )
            proposed_logdensity, proposed_logdensity_grad = logdensity_and_grad_fn(
                proposed_solution
            )
            proposed_state = MCMCOptimizerState(
                solution=proposed_solution,
                objective_value=-proposed_logdensity,
                logdensity=proposed_logdensity,
                logdensity_grad=proposed_logdensity_grad,
                cumulative_function_calls=state.cumulative_function_calls + 1,
            )

            # Update the old state to increase the number of objective and gradient
            # calls (since even if we reject the proposed state, we still needed
            # to evaluate the objective and gradient during this step).
            old_state = MCMCOptimizerState(
                solution=state.solution,
                objective_value=state.objective_value,
                logdensity=state.logdensity,
                logdensity_grad=state.logdensity_grad,
                cumulative_function_calls=state.cumulative_function_calls + 1,
            )

            # Compute the acceptance probability per the Metropolis-Hastings algorithm
            # (work with log probabilities to avoid underflow)
            log_p_accept = (
                proposed_state.logdensity
                - old_state.logdensity
                + self.transition_log_probability(proposed_state, old_state)
                - self.transition_log_probability(old_state, proposed_state)
            )
            log_p_accept = jnp.where(jnp.isnan(log_p_accept), -jnp.inf, log_p_accept)
            p_accept = jnp.clip(jnp.exp(log_p_accept), a_max=1)

            # Based on this probability, decide whether to accept the proposed state
            # If we're not using Metropolis adjustment, then always accept
            do_accept = jax.lax.cond(
                self._metropolis_adjusted,
                lambda: jrandom.bernoulli(acceptance_key, p_accept),
                lambda: jnp.array(True),
            )

            # Accept (or not) the new state
            accepted_state = jax.lax.cond(
                do_accept, lambda: proposed_state, lambda: old_state
            )

            return accepted_state

        return initial_state, step
