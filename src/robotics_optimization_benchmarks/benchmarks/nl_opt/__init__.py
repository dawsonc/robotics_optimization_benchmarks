"""Define some benchmarks for standard nonconvex optimization test problems."""

from robotics_optimization_benchmarks.benchmarks.nl_opt.ackley import Ackley
from robotics_optimization_benchmarks.benchmarks.nl_opt.double_well import DoubleWell
from robotics_optimization_benchmarks.benchmarks.nl_opt.heaviside import Heaviside
from robotics_optimization_benchmarks.benchmarks.nl_opt.hf_quadratic import HFQuadratic
from robotics_optimization_benchmarks.benchmarks.nl_opt.himmelblau import Himmelblau
from robotics_optimization_benchmarks.benchmarks.nl_opt.quadratic import Quadratic
from robotics_optimization_benchmarks.benchmarks.nl_opt.rosenbrock import Rosenbrock
from robotics_optimization_benchmarks.benchmarks.nl_opt.styblinski_tang import (
    StyblinskiTang,
)
from robotics_optimization_benchmarks.benchmarks.nl_opt.three_hump_camel import (
    ThreeHumpCamel,
)


__all__ = [
    "Ackley",
    "DoubleWell",
    "Heaviside",
    "HFQuadratic",
    "Himmelblau",
    "Quadratic",
    "Rosenbrock",
    "StyblinskiTang",
    "ThreeHumpCamel",
]
