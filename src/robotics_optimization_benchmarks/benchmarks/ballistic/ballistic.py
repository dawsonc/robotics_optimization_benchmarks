"""Define a discontinuous optimization benchmark for throwing a ball over a wall."""
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


class Ballistic(Benchmark):
    """Define a benchmark for the ballistic trajectory optimization from Suh 2022.

    In this benchmark, `dimension` balls are thrown through a gap in a wall.

    The objective is to get the balls to land as far away as possible. The decision
    variables are the launch angle for each ball.

    Attributes:
        dimension: The dimension of the problem.
    """

    _name: str = "ballistic"

    dimension: int

    _wall_x = 5.5  # m, x position of the wall
    _gap_height = 2.0  # m, height of the wall
    _gap_width = 1.0  # m, width of the gap
    _wall_width = 0.1  # m, thickness of the wall

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

        x_0 ~ Uniform(0, pi/2)

        Args:
            key: a JAX PRNG key used to sample the solution.

        Returns:
            A random initial solution to the problem in R^{self.dimension}.
        """
        return jax.random.uniform(
            key, shape=(self.dimension,), minval=0, maxval=jnp.pi / 2
        )

    @jaxtyped
    @beartype
    def simulate(self, solution: Float[Array, " dimension"]):
        """Simulate the ball flying through the air and interacting with the wall.

        Args:
            solution: The launch angle for each ball.

        Returns:
            - The state (p_x, p_y, v_x, v_y) of each ball at the final time step.
            - The trace of states throughout the simulation
        """
        # These parameters define the physics of the problem
        v_0 = 10.0  # m/s, initial ball velocity
        timestep = 0.01  # s, time step
        gravity = 9.81  # m/s^2, gravitational acceleration
        max_t_flight = v_0 * 2 / gravity  # s, maximum flight time
        n_steps = int(max_t_flight / timestep + 1)  # number of time steps

        # The decision variables are the launch angles of the balls.
        # Convert that into initial velocity
        p_x0 = jnp.zeros_like(solution)
        p_y0 = jnp.zeros_like(solution)
        v_x0 = v_0 * jnp.cos(solution)
        v_y0 = v_0 * jnp.sin(solution)

        # Simulate the trajectory of all of the balls in free flight,
        # but if any of them hit the wall, their x velocity is set to zero.
        # If any hit the ground, their x and y velocities are set to zero

        # Define a step function for simulation using jax.lax.scan
        def step_fn(carry, _):
            # Extract states
            p_x, p_y, v_x, v_y = carry

            # Update states
            p_x = p_x + timestep * v_x
            p_y = p_y + timestep * v_y
            v_y = v_y - timestep * gravity

            # Check for collisions
            wall_collision = jnp.logical_and(
                jnp.abs(p_x - self._wall_x) <= self._wall_width / 2.0,
                jnp.logical_or(
                    p_y <= self._gap_height, p_y >= self._gap_height + self._gap_width
                ),
            )
            ground_collision = p_y <= 0.0

            # Update state to reflect possible collision with the wall
            p_x = jnp.where(wall_collision, self._wall_x + jnp.zeros_like(p_x), p_x)
            v_x = jnp.where(wall_collision, jnp.zeros_like(v_x), v_x)

            # If there was a collision with the ground, we should find the moment of
            # collison and reverse back to it. If we don't do this, the gradients will
            # be OPPOSITE of what they should be (not exactly opposite, but generally
            # a negative dot product with the true gradient).
            ground_penetration = jnp.minimum(p_y, 0.0)
            t_ground_collision = ground_penetration / (1e-3 + jnp.abs(v_y))

            p_y = jnp.where(ground_collision, jnp.zeros_like(p_y), p_y)
            p_x = jnp.where(ground_collision, p_x + t_ground_collision * v_x, p_x)
            v_x = jnp.where(ground_collision, jnp.zeros_like(v_x), v_x)
            v_y = jnp.where(ground_collision, jnp.zeros_like(v_y), v_y)

            # Update the carry
            carry = (p_x, p_y, v_x, v_y)
            return carry, carry

        return jax.lax.scan(step_fn, (p_x0, p_y0, v_x0, v_y0), None, length=n_steps)

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
        (final_x, _, _, _), _ = self.simulate(solution)

        # The cost here is the negative total distance travelled by the balls
        return -final_x.sum()

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

        # Plot the ground and the wall
        axis.plot([-1.0, 11.0], [0.0, 0.0], "k-", linewidth=2.0)
        axis.fill_between(
            [self._wall_x - self._wall_width / 2, self._wall_x + self._wall_width / 2],
            0.0,
            self._gap_height,
            color="black",
            alpha=0.2,
        )
        axis.fill_between(
            [self._wall_x - self._wall_width / 2, self._wall_x + self._wall_width / 2],
            self._gap_height + self._gap_width,
            10.0,
            color="black",
            alpha=0.2,
        )

        # Save the figure and clean up
        fig.savefig(save_to, format="png")
        plt.close(fig)


if __name__ == "__main__":
    # Plot the cost function in 1D
    problem = Ballistic(dimension=1)
    x = jnp.linspace(0.0, jnp.pi / 2, 1000).reshape(-1, 1)
    y = jax.vmap(problem.evaluate_solution)(x)
    plt.plot(x, y)
    plt.show()
