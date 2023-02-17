"""Define a benchmark for the 2-link reacher environment."""

from robotics_optimization_benchmarks.benchmarks.reacher.reacher import Reacher
from robotics_optimization_benchmarks.benchmarks.registry import register


__all__ = [
    "Reacher",
]

# Register the benchmark in the benchmark registry
register(Reacher.name, Reacher)
