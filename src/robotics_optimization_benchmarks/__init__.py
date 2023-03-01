"""Testing the performance of sampling and optimization algorithms."""

from robotics_optimization_benchmarks.benchmarks import make as make_benchmark
from robotics_optimization_benchmarks.benchmarks import register as register_benchmark
from robotics_optimization_benchmarks.optimizers import make as make_optimizer
from robotics_optimization_benchmarks.optimizers import register as register_optimizer


__all__ = [
    "make_benchmark",
    "register_benchmark",
    "make_optimizer",
    "register_optimizer",
]
