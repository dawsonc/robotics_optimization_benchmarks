"""Define a benchmark for a generic Brax environment."""
import brax
import brax.envs
import brax.io.image
import equinox as eqx
import jax
import jax.random as jrandom
from beartype import beartype
from beartype.typing import Any
from beartype.typing import BinaryIO
from beartype.typing import Dict
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


class Brax(Benchmark):
    """Define a benchmark based on the environments included with Brax.

    These environments are intended for RL, but we can repurpose them for optimization
    by rolling the out for multiple steps and taking the negative cumulative reward as
    our cost.

    We'll equip each environment with a multi-layer perceptron control policy; the
    parameters of this policy will form the decision variables.

    Available tasks include:
        - `ant <github.com/google/brax/blob/main/brax/envs/ant.py>`_ from
          `OpenAI Gym Ant-v2 <gym.openai.com/envs/Ant-v2/>`_: make a four-legged
          creature walk forward as fast as possible.
        - `halfcheetah <github.com/google/brax/blob/main/brax/envs/halfcheetah.py>`_
          from `OpenAI Gym HalfCheetah-v2 <gym.openai.com/envs/HalfCheetah-v2/>`_: make
          a two-dimensional two-legged creature walk forward as fast as possible.
        - `hopper <github.com/google/brax/blob/main/brax/envs/hopper.py>`_ from
          `OpenAI Gym Hopper-v2 <gym.openai.com/envs/Hopper-v2/>`_: make a
          two-dimensional one-legged robot hop forward as fast as possible.
        - `humanoid <github.com/google/brax/blob/main/brax/envs/humanoid.py>`_ from
          `OpenAI Gym Humanoid-v2 <gym.openai.com/envs/Humanoid-v2/>`_: make a
          three-dimensional bipedal robot walk forward as fast as possible, without
          falling over.
        - `reacher <github.com/google/brax/blob/main/brax/envs/reacher.py>`_: from
          `OpenAI Gym Reacher-v2 <gym.openai.com/envs/Reacher-v2/>`_: makes a two-joint
          reacher arm move its tip to a target.
        - `pusher <github.com/google/brax/blob/main/brax/envs/pusher.py>`_: a robot arm
            pushes an object to a target location.
        - `walker2d <github.com/google/brax/blob/main/brax/envs/walker2d.py>`_ from
          `OpenAI Gym Walker2d-v2 <gym.openai.com/envs/Walker2d-v2/>`_: make a
          two-dimensional bipedal robot walk forward as fast as possible
        - `inverted_pendulum <github.com/google/brax/blob/main/brax/envs/inverted_pendulum.py>`_:
            an inverted pendulum.
        - `inverted_double_pendulum <github.com/google/brax/blob/main/brax/envs/inverted_double_pendulum.py>`_:  # noqa
            an inverted double pendulum.


    Attributes:
        policy_network_width: The number of neurons in each hidden layer of the
            policy network.
        policy_network_depth: The number of hidden layers in the policy network.
        horizon: The number of time steps to run the simulation for.
    """

    _name: str = "brax"
    _task: str

    render_extension = "gif"

    # These are set based on the task
    _n_actions: int
    _n_observations: int

    policy_network_depth: int
    policy_network_width: int
    horizon: int

    def __init__(
        self,
        task: str,
        policy_network_width: int = 32,
        policy_network_depth: int = 2,
        horizon: int = 100,
    ):
        """Initialize the benchmark.

        Args:
            task: The name of the Brax task environment to use.
            policy_network_width: The number of neurons in each hidden layer of the
                policy network.
            policy_network_depth: The number of hidden layers in the policy network.
            horizon: The number of time steps to run the simulation for.
        """
        self._task = task
        self.policy_network_width = policy_network_width
        self.policy_network_depth = policy_network_depth
        self.horizon = horizon

        # Initialize the environment and use it to specify the input and output
        # dimensions of the policy network.
        self._env = brax.envs.get_environment(env_name=task, backend="positional")
        self._n_actions = self._env.action_size
        self._n_observations = self._env.observation_size

    @beartype
    def to_dict(self) -> Dict[str, Any]:
        """Get a dictionary containing the parameters to initialize the benchmark."""
        return {
            "task": self._task,
            "policy_network_width": self.policy_network_width,
            "policy_network_depth": self.policy_network_depth,
            "horizon": self.horizon,
        }

    @property
    def task(self) -> str:
        """Return the name of the task."""
        return self._task

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
        # Reset the brax environment using a fixed arbitrary key
        state = self._env.reset(rng=jrandom.PRNGKey(0))

        # Return the cumulative reward (negative so it gets treated like a cost)
        rollout = self.rollout(self._env, state, solution)
        return -rollout.reward.sum()

    def render_solution(self, solution: MLP, save_to: str | BinaryIO) -> None:
        """Visualize a solution to the problem, saving the visualization.

        Args:
            solution: the solution to visualize.
            save_to: the path or file-like object to save the visualization to.
        """
        # Reset the brax environment using a fixed arbitrary key
        state = self._env.reset(rng=jrandom.PRNGKey(0))

        # Get the rollout and convert from a PyTree with a leading axis for the
        # time dimension to a list of PyTrees with one entry for each time step.
        rollout = self.rollout(self._env, state, solution)
        rollout = jax.tree_util.tree_transpose(
            outer_treedef=jax.tree_util.tree_structure(state),
            inner_treedef=jax.tree_util.tree_structure(list(range(self.horizon))),
            pytree_to_transpose=jax.tree_util.tree_map(list, rollout),
        )

        # Render, which yields the bytes of a GIF animation, which we have to handle
        # differently depending on whether we're saving to a file or a file-like
        render_bytes = brax.io.image.render(
            self._env.sys, [s.qp for s in rollout], width=1280, height=960
        )
        if isinstance(save_to, str):
            with open(save_to, "wb") as file:
                file.write(render_bytes)
        else:
            save_to.write(render_bytes)
