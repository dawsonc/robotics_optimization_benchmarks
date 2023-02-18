"""Testing the performance of sampling and optimization algorithms."""

from robotics_optimization_benchmarks.benchmarks import make as make_benchmark
from robotics_optimization_benchmarks.benchmarks import register as register_benchmark


__all__ = [
    "make_benchmark",
    "register_benchmark",
]
