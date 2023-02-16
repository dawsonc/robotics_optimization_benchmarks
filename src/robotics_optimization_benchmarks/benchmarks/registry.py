"""Define a public API for managing the benchmark registry."""
from beartype import beartype  # TODO how to get rid of this warning???
from beartype.typing import Type

from robotics_optimization_benchmarks.benchmarks.benchmark import Benchmark
from robotics_optimization_benchmarks.registry import Registry


# Make a registry to store the benchmarks
# WARNING: global mutable state is usually frowned upon, but here we'll manage
# access to it using public functions to mitigate some of the risk
_benchmark_registry: Registry[Type[Benchmark]] = Registry[Type[Benchmark]]()


# Define public functions for accessing the benchmark registry
@beartype
def make(name: str) -> Type[Benchmark]:
    """Access a benchmark stored in the registry.

    Benchmarks can be constructed by chaining with the initializer or `from_dict`
    class methods, e.g.:
    ```
    my_benchmark = benchmarks.make("benchmark")(arg1, arg2)
    my_benchmark = benchmarks.make("benchmark").from_dict({"arg1": 1, "arg2": 2})
    ```

    Args:
        name: the name of the benchmark to access.

    Raises:
        KeyError: if the benchmark name is not registered.  # noqa: DAR402

    Returns:
        The benchmark class stored in the registry.
    """
    return _benchmark_registry.get_by_name(name)


@beartype
def register(name: str, benchmark: Type[Benchmark]) -> None:
    """Register a benchmark with the given name.

    Args:
        name: the name to register the benchmark under.
        benchmark: the benchmark class to register.

    Raises:
        ValueError:  # noqa: DAR402
            if there is already a benchmark registered under this name.
    """
    _benchmark_registry.register(name, benchmark)
