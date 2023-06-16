"""Define some benchmarks for standard nonconvex optimization test problems."""

from robotics_optimization_benchmarks.benchmarks.nl_opt.ackley import Ackley
from robotics_optimization_benchmarks.benchmarks.nl_opt.double_well import DoubleWell
from robotics_optimization_benchmarks.benchmarks.nl_opt.himmelblau import Himmelblau
from robotics_optimization_benchmarks.benchmarks.nl_opt.quadratic import Quadratic
from robotics_optimization_benchmarks.benchmarks.nl_opt.rosenbrock import Rosenbrock
from robotics_optimization_benchmarks.benchmarks.nl_opt.three_hump_camel import (
    ThreeHumpCamel,
)


__all__ = [
    "Ackley",
    "DoubleWell",
    "Himmelblau",
    "Quadratic",
    "Rosenbrock",
    "ThreeHumpCamel",
]
