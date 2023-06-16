"""Define some benchmarks for standard nonconvex optimization test problems."""

from robotics_optimization_benchmarks.benchmarks.nl_opt.ackley import Ackley
from robotics_optimization_benchmarks.benchmarks.nl_opt.double_well import DoubleWell
from robotics_optimization_benchmarks.benchmarks.nl_opt.quadratic import Quadratic
from robotics_optimization_benchmarks.benchmarks.nl_opt.rosenbrock import Rosenbrock


__all__ = [
    "Ackley",
    "Quadratic",
    "DoubleWell",
    "Rosenbrock",
]
