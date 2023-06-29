"""Define a discontinuous optimization benchmark for bouncing a ball at a target."""
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from beartype import beartype
from beartype.typing import Any
from beartype.typing import BinaryIO
from beartype.typing import Dict
from jaxtyping import Array
from jaxtyping import Float
from jaxtyping import jaxtyped

from robotics_optimization_benchmarks.benchmarks import Benchmark
from robotics_optimization_benchmarks.types import PRNGKeyArray


class Pong(Benchmark):
    """Define a benchmark for the pong trajectory optimization from Suh 2022.

    In this benchmark, `dimension` balls are bounced off of `dimension` paddles towards
    a target. The balls do not interact; we run `dimension` simulations in parallel.

    The objective is to get the balls as close as possible to the target.

    The decision variables are the [x, y, theta] pose of the paddle for each ball.

    Attributes:
        dimension: The dimension of the problem.
    """

    _name: str = "pong"

    dimension: int

    _paddle_width = 0.2  # m, width of the paddle
    _paddle_thickness = 0.05  # m, thickness of the paddle

    goal_x = 1.0  # m, x-coordinate of the goal
    goal_y = 0.0  # m, y-coordinate of the goal

    def __init__(self, dimension: int = 10):
        """Initialize the benchmark.

        Args:
            dimension: The dimension of the problem.
        """
        self.dimension = dimension

    @beartype
    def to_dict(self) -> Dict[str, Any]:
        """Get a dictionary containing the parameters to initialize the benchmark."""
        return {"dimension": self.dimension}

    @jaxtyped
    @beartype
    def sample_initial_guess(self, key: PRNGKeyArray) -> Float[Array, " dimension"]:
        """Sample a random initial solution to the problem.

        x ~ Uniform(-1, 1)
        y ~ Uniform(-1, 1)
        theta ~ Uniform(-pi, pi)

        Args:
            key: a JAX PRNG key used to sample the solution.

        Returns:
            A random initial solution to the problem in R^{3 x self.dimension}.
        """
        initial_guess = jnp.zeros((self.dimension, 3))
        initial_guess = initial_guess.at[:, 0].set(
            jax.random.uniform(key, shape=(self.dimension,), minval=-1, maxval=1)
        )
        initial_guess = initial_guess.at[:, 1].set(
            jax.random.uniform(key, shape=(self.dimension,), minval=-1, maxval=1)
        )
        initial_guess = initial_guess.at[:, 2].set(
            jax.random.uniform(
                key, shape=(self.dimension,), minval=-jnp.pi, maxval=jnp.pi
            )
        )

        return initial_guess

    @jaxtyped
    @beartype
    def simulate(self, solution: Float[Array, "dimension 3"]):
        """Simulate the ball flying through the air and interacting with the paddle.

        Args:
            solution: The [x, y, theta] pose of the paddle for each ball.

        Returns:
            - The state (p_x, p_y, v_x, v_y) of each ball at the final time step.
            - The trace of states throughout the simulation
        """
        # These parameters define the physics of the problem
        v_0 = 1.0  # m/s, initial ball velocity
        timestep = 0.02  # s, time step
        horizon = 150  # number of steps

        # Start all balls from (1, 1) and travelling in the [-1, -1] direction
        p_x0 = jnp.ones(self.dimension)
        p_y0 = jnp.ones(self.dimension)
        v_x0 = -v_0 * jnp.cos(jnp.pi / 4.0) * jnp.ones(self.dimension)
        v_y0 = -v_0 * jnp.sin(jnp.pi / 4.0) * jnp.ones(self.dimension)

        # Extract solution
        paddle_x, paddle_y, paddle_theta = solution.T

        # Simulate the trajectory of all of the balls in free flight,
        # but if they hit the paddle, bounce them back

        # Define a step function for simulation using jax.lax.scan
        def step_fn(carry, _):
            # Extract states
            p_x, p_y, v_x, v_y = carry

            # Update states (assuming no collision)
            p_x_no_collision = p_x + timestep * v_x
            p_y_no_collision = p_y + timestep * v_y

            # Check for collision
            # p2b = paddle_to_ball
            p2b_x = p_x_no_collision - paddle_x
            p2b_y = p_y_no_collision - paddle_y
            p2b_normal = p2b_x * jnp.cos(paddle_theta) + p2b_y * jnp.sin(paddle_theta)
            p2b_tangent = p2b_x * jnp.sin(paddle_theta) - p2b_y * jnp.cos(paddle_theta)

            collision = jnp.logical_and(
                jnp.logical_and(
                    p2b_normal <= 0, jnp.abs(p2b_normal) <= self._paddle_thickness
                ),
                jnp.abs(p2b_tangent) <= self._paddle_width / 2,
            )

            # Get the time of contact
            v_normal = v_x * jnp.cos(paddle_theta) + v_y * jnp.sin(paddle_theta)
            v_normal = jnp.where(jnp.abs(v_normal) <= 1e-2, 1e-2, v_normal)  # avoid nan
            time_of_contact = jnp.where(
                collision,
                timestep - p2b_normal / v_normal,
                timestep,
            )

            # Update the position up to the time of contact
            p_x_next = p_x + time_of_contact * v_x
            p_y_next = p_y + time_of_contact * v_y

            # Flip velocities about the contact plane if there was a collision
            v_x_next = jnp.where(
                collision, v_x - 2 * v_normal * jnp.cos(paddle_theta), v_x
            )
            v_y_next = jnp.where(
                collision, v_y - 2 * v_normal * jnp.sin(paddle_theta), v_y
            )

            # Update the positions with the remaining time
            p_x_next = p_x_next + (timestep - time_of_contact) * v_x_next
            p_y_next = p_y_next + (timestep - time_of_contact) * v_y_next

            # Update the carry
            carry = (p_x_next, p_y_next, v_x_next, v_y_next)
            return carry, carry

        return jax.lax.scan(step_fn, (p_x0, p_y0, v_x0, v_y0), None, length=horizon)

    @jaxtyped
    @beartype
    def evaluate_solution(
        self, solution: Float[Array, " dimension"]
    ) -> Float[Array, ""]:
        """Evaluate the objective function at a given solution.

        Args:
            solution: the solution at which to evaluate the objective function.

        Returns:
            The objective function evaluated at the given solution.
        """
        (final_x, final_y, _, _), _ = self.simulate(solution)

        # The cost here is the squared distance to the goal
        return jnp.sum((final_x - self.goal_x) ** 2 + (final_y - self.goal_y) ** 2)

    def render_solution(
        self, solution: Float[Array, " dimension"], save_to: str | BinaryIO
    ) -> None:
        """Visualize a solution to the problem, saving the visualization.

        Args:
            solution: the solution to visualize.
            save_to: the path or file-like object to save the visualization to.
        """
        # Simulate to get the trace of states
        _, (p_x, p_y, _, _) = self.simulate(solution)

        # Create the figure to plot (we'll return this to the caller)
        fig, axis = plt.subplots(1, 1, figsize=(12, 12))

        # Plot the trajectory of the ball
        axis.plot(p_x, p_y, color="blue", linewidth=1.0, label="Trajectory")

        # Plot the paddle
        paddle_x, paddle_y, paddle_theta = solution.T
        paddle_center = jnp.vstack((paddle_x, paddle_y))
        paddle_tangent = (
            jnp.vstack((jnp.sin(paddle_theta), -jnp.cos(paddle_theta)))
            * self._paddle_width
            / 2
        )
        paddle_left = paddle_center - paddle_tangent
        paddle_right = paddle_center + paddle_tangent
        paddle_points = jnp.hstack((paddle_left, paddle_right))
        axis.plot(
            paddle_points[0],
            paddle_points[1],
            color="red",
            linewidth=2.0,
            label="Paddle",
        )

        # Plot the goal
        axis.plot(
            self.goal_x,
            self.goal_y,
            color="green",
            marker="o",
            markersize=10,
            label="Goal",
        )

        # Make axes equal for intuitive visualization
        axis.set_aspect("equal")

        # Save the figure and clean up
        fig.savefig(save_to, format="png")
        plt.close(fig)


if __name__ == "__main__":
    # Plot a 1D problem
    pong = Pong(1)
    pong.render_solution(jnp.array([[-0.25, -0.25, 0.5]]), "test_pong.png")
