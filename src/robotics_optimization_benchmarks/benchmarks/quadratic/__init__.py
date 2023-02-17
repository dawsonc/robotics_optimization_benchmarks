"""Define a simple benchmark for a convex (quadratic) optimization."""

from robotics_optimization_benchmarks.benchmarks.quadratic.quadratic import Quadratic
from robotics_optimization_benchmarks.benchmarks.registry import register


__all__ = [
    "Quadratic",
]

# Register the benchmark in the benchmark registry
register(Quadratic.name, Quadratic)
