"""Define a simple quadratic optimization benchmark."""
import brax
import brax.envs
import brax.io.image
import equinox as eqx
import jax
import jax.random as jrandom
from beartype import beartype
from beartype.typing import BinaryIO
from beartype.typing import List
from jaxtyping import Array
from jaxtyping import Float
from jaxtyping import jaxtyped

from robotics_optimization_benchmarks.benchmarks import Benchmark
from robotics_optimization_benchmarks.types import PRNGKeyArray


class MLP(eqx.Module):
    """Define a simple multi-layer perceptron."""

    layers: List[eqx.Module]

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_sizes: List[int],
        key: PRNGKeyArray,
    ) -> None:
        """Initialize the MLP.

        Args:
            input_size: The size of the input.
            output_size: The size of the output.
            hidden_sizes: A list of hidden layer sizes.
            key: A JAX PRNG key used to initialize the network weights.
        """
        self.layers = []
        for hidden_size in hidden_sizes:
            self.layers.append(eqx.nn.Linear(input_size, hidden_size, key=key))
            input_size = hidden_size

        self.layers.append(eqx.nn.Linear(input_size, output_size, key=key))

    def __call__(self, obs: Float[Array, " n_in"]) -> Float[Array, " n_out"]:
        """Evaluate the MLP."""
        for layer in self.layers[:-1]:
            obs = jax.nn.relu(layer(obs))
        return self.layers[-1](obs)


class Reacher(Benchmark):
    """Define a benchmark based on the reacher task from the OpenAI Gym.

    The reacher task is a 2D task in which the goal is to move a 2D arm to a
    target location. The arm is controlled by a 2D action space, and the
    observation space is a 11D vector containing the position and velocity of the
    arm and the target.

    The goal is to minimize the distance between the end effector and the target
    location.

    The control action is the output of a neural network, the parameters of which
    form the decision variable.

    This benchmark is implemented using the Brax reacher environment.

    Attributes:
        policy_network_width: The number of neurons in each hidden layer of the
            policy network.
        policy_network_depth: The number of hidden layers in the policy network.
    """

    _name: str = "reacher"

    # The action space is 2D, and the observation space is 11D.
    _n_actions: int = 2
    _n_observations: int = 11

    policy_network_depth: int
    policy_network_width: int
    horizon: int

    def __init__(
        self,
        policy_network_width: int = 32,
        policy_network_depth: int = 2,
        horizon: int = 100,
    ):
        """Initialize the benchmark.

        Args:
            policy_network_width: The number of neurons in each hidden layer of the
                policy network.
            policy_network_depth: The number of hidden layers in the policy network.
            horizon: The number of time steps to run the simulation for.
        """
        self.policy_network_width = policy_network_width
        self.policy_network_depth = policy_network_depth
        self.horizon = horizon

    @jaxtyped
    @beartype
    def sample_initial_guess(self, key: PRNGKeyArray) -> MLP:
        """Sample a random initial solution to the problem.

        The solution is represented as an Equinox module (i.e. callable PyTree)
        that is compatible with jax.grad and related transformations.

        Args:
            key: a JAX PRNG key used to initialize a random network.

        Returns:
            A randomly initialized multi-layer perceptron (MLP)
        """
        return MLP(
            input_size=self._n_observations,
            output_size=self._n_actions,
            hidden_sizes=[self.policy_network_width] * self.policy_network_depth,
            key=key,
        )

    @jaxtyped
    @beartype
    def rollout(
        self, env: brax.envs.Env, state: brax.envs.State, solution: MLP
    ) -> brax.envs.State:
        """Rollout a solution to the problem.

        Args:
            env: the Brax environment to rollout in.
            state: the initial state of the environment.
            solution: the solution to rollout.

        Returns:
            A PyTree of states during the rollout.
        """
        # Scan over the states

        @jax.jit
        def step(state, _):
            next_state = jax.jit(env.step)(state, solution(state.obs))
            return next_state, next_state

        _, rollout = jax.lax.scan(step, state, None, length=self.horizon)

        return rollout

    @jaxtyped
    @beartype
    def evaluate_solution(self, solution: MLP) -> Float[Array, ""]:
        """Evaluate the objective function at a given solution.

        Args:
            solution: the solution at which to evaluate the objective function.

        Returns:
            The objective function evaluated at the given solution.
        """
        # Create the Brax environment for the reacher environment and reset it
        # using a fixed arbitrary key
        env = brax.envs.create(env_name="reacher")
        state = env.reset(rng=jrandom.PRNGKey(0))

        # Return the cumulative reward (negative so it gets treated like a cost)
        rollout = self.rollout(env, state, solution)
        return -rollout.reward.sum()

    def render_solution(self, solution: MLP, save_to: str | BinaryIO) -> None:
        """Visualize a solution to the problem, saving the visualization.

        Args:
            solution: the solution to visualize.
            save_to: the path or file-like object to save the visualization to.
        """
        # Create the Brax environment for the reacher environment and reset it
        # using a fixed arbitrary key
        env = brax.envs.create(env_name="reacher")
        state = env.reset(rng=jrandom.PRNGKey(0))

        # Get the rollout and convert from a PyTree with a leading axis for the
        # time dimension to a list of PyTrees with one entry for each time step.
        rollout = self.rollout(env, state, solution)
        rollout = jax.tree_util.tree_transpose(
            outer_treedef=jax.tree_util.tree_structure(state),
            inner_treedef=jax.tree_util.tree_structure(list(range(self.horizon))),
            pytree_to_transpose=jax.tree_util.tree_map(list, rollout),
        )

        # Render, which yields the bytes of a GIF animation, which we have to handle
        # differently depending on whether we're saving to a file or a file-like
        render_bytes = brax.io.image.render(
            env.sys, [s.qp for s in rollout], width=1280, height=960
        )
        if isinstance(save_to, str):
            with open(save_to, "wb") as file:
                file.write(render_bytes)
        else:
            save_to.write(render_bytes)
