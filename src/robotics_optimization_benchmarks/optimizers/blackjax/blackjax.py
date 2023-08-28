"""Implement an interface to `BlackJax <https://blackjax-devs.github.io/blackjax/>`_."""
import blackjax
import chex
import jax
import jax.numpy as jnp
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
class BlackJaxState(OptimizerState):
    """A struct for storing the state of a BlackJax-derived optimizer/sampler.

    Attributes:
        solution: the current solution.
        objective_value: the value of the objective function at the current solution.
        cumulative_function_calls: the cumulative number of objective function calls.
        debug: any debug information to log
        blackjax_state: the state of the wrapped BlackJax sampler.
        num_steps: the number of steps taken by the optimizer.
    """

    blackjax_state: NamedTuple
    num_steps: int = 0


class NUTS(Optimizer):
    """Wrapper around BlackJax HMC with NUTS.

    The notation here is going to blur the lines between "samplers" and "optimizers".

    Args:
        step_size: the step size to use for the NUTS sampler.
    """

    _name: str = "NUTS"
    _sampler: blackjax.base.SamplingAlgorithm
    _step_size: float

    @beartype
    def __init__(
        self,
        step_size: float,
    ):
        """Initialize the sampler."""
        super().__init__()

        self._step_size = step_size

    @property
    @beartype
    def description(self) -> str:
        """Get a string description of this optimizer."""
        return "NUTS"

    @beartype
    def to_dict(self) -> Dict[str, Any]:
        """Get a dictionary containing the parameters to initialize this optimizer."""
        return {
            "step_size": self._step_size,
        }

    @jaxtyped
    @beartype
    def make_step(
        self,
        objective_fn: Callable[[DecisionVariable], Float[Array, ""]],
        initial_solution: DecisionVariable,
    ) -> Tuple[BlackJaxState, Callable[[BlackJaxState, PRNGKeyArray], BlackJaxState]]:
        """Initialize the state of the sampler.

        Args:
            objective_fn: the objective function to minimize.
            initial_solution: the initial solution.

        Returns:
            initial_state: The initial state of the optimizer.

            step_fn: A function that takes the current state of the optimizer and a PRNG
            key and returns the next state of the optimizer, executing one step of
            the optimization algorithm.
        """
        # Start by computing the inverse mass matrix for this problem
        n_vars = jax.flatten_util.ravel_pytree(initial_solution)[0].size
        inverse_mass_matrix = jnp.ones(n_vars)

        # Build the kernel
        def logdensity_fn(x):
            return -objective_fn(x)

        self._sampler = blackjax.nuts(
            logdensity_fn, self._step_size, inverse_mass_matrix
        )

        # Initialize the sampler
        initial_nuts_state = self._sampler.init(initial_solution)

        # Wrap the state in our own state struct
        initial_state = BlackJaxState(
            solution=initial_solution,
            objective_value=objective_fn(initial_solution),
            cumulative_function_calls=0,
            blackjax_state=initial_nuts_state,
            debug={"acceptance_rate": 1.0},
        )

        # Define the step function wrapping blackjax.
        @jaxtyped
        @beartype
        def step(state: BlackJaxState, key: PRNGKeyArray) -> BlackJaxState:
            """Take one step.

            Args:
                state: the current state of the optimizer.
                key: a random number generator key.

            Returns:
                A sample.
            """
            # Take a step
            (
                next_nuts_state,
                nuts_info,
            ) = self._sampler.step(  # pylint: disable=no-member
                key, state.blackjax_state
            )

            # Update the cumulative average acceptance rate
            old_accept_rate = state.debug["acceptance_rate"]
            new_accept_rate = nuts_info.acceptance_probability
            new_cumulative_accept_rate = (
                state.num_steps * old_accept_rate + new_accept_rate
            ) / (state.num_steps + 1)

            return BlackJaxState(
                solution=next_nuts_state.position,
                objective_value=next_nuts_state.potential_energy,
                # We evaluated the gradient once to step to the next solution.
                cumulative_function_calls=(
                    state.cumulative_function_calls + nuts_info.num_integration_steps
                ),
                blackjax_state=next_nuts_state,
                num_steps=state.num_steps + 1,
                debug={"acceptance_rate": new_cumulative_accept_rate},
            )

        return initial_state, step


class HMC(Optimizer):
    """Wrapper around BlackJax vanilla HMC.

    The notation here is going to blur the lines between "samplers" and "optimizers".

    Args:
        step_size: the step size to use for the HMC sampler.
        num_integration_steps: the number of integration steps to use for the HMC sampler.
    """

    _name: str = "HMC"
    _sampler: blackjax.base.SamplingAlgorithm
    _step_size: float
    _num_integration_steps: int

    @beartype
    def __init__(
        self,
        step_size: float,
        num_integration_steps: int = 20,
    ):
        """Initialize the sampler."""
        super().__init__()

        self._step_size = step_size
        self._num_integration_steps = num_integration_steps

    @property
    @beartype
    def description(self) -> str:
        """Get a string description of this optimizer."""
        return "HMC"

    @beartype
    def to_dict(self) -> Dict[str, Any]:
        """Get a dictionary containing the parameters to initialize this optimizer."""
        return {
            "step_size": self._step_size,
            "num_integration_steps": self._num_integration_steps,
        }

    @jaxtyped
    @beartype
    def make_step(
        self,
        objective_fn: Callable[[DecisionVariable], Float[Array, ""]],
        initial_solution: DecisionVariable,
    ) -> Tuple[BlackJaxState, Callable[[BlackJaxState, PRNGKeyArray], BlackJaxState]]:
        """Initialize the state of the sampler.

        Args:
            objective_fn: the objective function to minimize.
            initial_solution: the initial solution.

        Returns:
            initial_state: The initial state of the optimizer.

            step_fn: A function that takes the current state of the optimizer and a PRNG
            key and returns the next state of the optimizer, executing one step of
            the optimization algorithm.
        """
        # Start by computing the inverse mass matrix for this problem
        n_vars = jax.flatten_util.ravel_pytree(initial_solution)[0].size
        inverse_mass_matrix = jnp.ones(n_vars)

        # Build the kernel
        def logdensity_fn(x):
            return -objective_fn(x)

        self._sampler = blackjax.hmc(
            logdensity_fn,
            self._step_size,
            inverse_mass_matrix,
            self._num_integration_steps,
        )

        # Initialize the sampler
        initial_hmc_state = self._sampler.init(initial_solution)

        # Wrap the state in our own state struct
        initial_state = BlackJaxState(
            solution=initial_solution,
            objective_value=objective_fn(initial_solution),
            cumulative_function_calls=0,
            blackjax_state=initial_hmc_state,
            debug={"acceptance_rate": 1.0},
        )

        # Define the step function wrapping blackjax.
        @jaxtyped
        @beartype
        def step(state: BlackJaxState, key: PRNGKeyArray) -> BlackJaxState:
            """Take one step.

            Args:
                state: the current state of the optimizer.
                key: a random number generator key.

            Returns:
                A sample.
            """
            # Take a step
            (
                next_hmc_state,
                hmc_info,
            ) = self._sampler.step(  # pylint: disable=no-member
                key, state.blackjax_state
            )

            # Update the cumulative average acceptance rate
            old_accept_rate = state.debug["acceptance_rate"]
            new_accept_rate = hmc_info.acceptance_probability
            new_cumulative_accept_rate = (
                state.num_steps * old_accept_rate + new_accept_rate
            ) / (state.num_steps + 1)

            return BlackJaxState(
                solution=next_hmc_state.position,
                objective_value=next_hmc_state.potential_energy,
                # We evaluated the gradient once to step to the next solution.
                cumulative_function_calls=(
                    state.cumulative_function_calls + hmc_info.num_integration_steps
                ),
                blackjax_state=next_hmc_state,
                num_steps=state.num_steps + 1,
                debug={"acceptance_rate": new_cumulative_accept_rate},
            )

        return initial_state, step
