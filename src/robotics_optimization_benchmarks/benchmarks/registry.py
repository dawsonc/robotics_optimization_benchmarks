"""Define a public API for managing the benchmark registry."""
from beartype import beartype
from beartype.typing import Dict
from beartype.typing import Type

from robotics_optimization_benchmarks.benchmarks.ballistic import Ballistic
from robotics_optimization_benchmarks.benchmarks.benchmark import Benchmark
from robotics_optimization_benchmarks.benchmarks.brax import Brax
from robotics_optimization_benchmarks.benchmarks.nl_opt import Ackley
from robotics_optimization_benchmarks.benchmarks.nl_opt import DoubleWell
from robotics_optimization_benchmarks.benchmarks.nl_opt import Heaviside
from robotics_optimization_benchmarks.benchmarks.nl_opt import HFQuadratic
from robotics_optimization_benchmarks.benchmarks.nl_opt import Himmelblau
from robotics_optimization_benchmarks.benchmarks.nl_opt import Quadratic
from robotics_optimization_benchmarks.benchmarks.nl_opt import Rosenbrock
from robotics_optimization_benchmarks.benchmarks.nl_opt import StyblinskiTang
from robotics_optimization_benchmarks.benchmarks.nl_opt import ThreeHumpCamel
from robotics_optimization_benchmarks.benchmarks.pong import Pong


# Make a registry to store the benchmarks
# WARNING: global mutable state is usually frowned upon, but here we'll manage
# access to it using public functions to mitigate some of the risk
_benchmark_registry: Dict[str, Type[Benchmark]] = {}


# Define public functions for accessing the benchmark registry
@beartype
def make(name: str) -> Type[Benchmark]:
    """Access a benchmark stored in the registry.

    Benchmarks can be constructed by chaining with the initializer or `from_dict`
    class methods, e.g.:

    .. code-block:: python

        my_benchmark = benchmarks.make("benchmark")(arg1, arg2)
        my_benchmark = benchmarks.make("benchmark").from_dict({"arg1": 1, "arg2": 2})

    Args:
        name: the name of the benchmark to access.

    Raises:
        KeyError: if the benchmark name is not registered.  # noqa: DAR402

    Returns:
        The `Benchmark` class stored in the registry under the given name.
    """
    return _benchmark_registry[name]


@beartype
def register(name: str, benchmark: Type[Benchmark]) -> None:
    """Register a benchmark with the given name.

    Args:
        name: the name to register the benchmark under.
        benchmark: the benchmark class to register.

    Raises:
        ValueError: if there is already a benchmark registered under this name.
    """
    if name in _benchmark_registry:
        raise ValueError(f"Benchmark {name} is already registered!")
    _benchmark_registry[name] = benchmark


###############################################################################
# Register built-in benchmarks
###############################################################################

# Contact problems
register(Ballistic.name, Ballistic)
register(Pong.name, Pong)
register(Brax.name, Brax)

# Nonlinear optimization test functions
register(Quadratic.name, Quadratic)
register(HFQuadratic.name, HFQuadratic)
register(DoubleWell.name, DoubleWell)
register(Ackley.name, Ackley)
register(Rosenbrock.name, Rosenbrock)
register(Heaviside.name, Heaviside)
register(Himmelblau.name, Himmelblau)
register(StyblinskiTang.name, StyblinskiTang)
register(ThreeHumpCamel.name, ThreeHumpCamel)
