"""A set of robotics- and control-relevant optimization benchmarks, implemented in JAX.

Benchmarks are accessed using the `make` function (inspired by the OpenAI Gym API).

.. code-block:: python

    my_benchmark = benchmarks.make("super_cool_benchmark")

When implementing new benchmarks, make sure to add them to the registry!
Benchmarks can be registered using the `register` function.

.. code-block:: python

    benchmarks.register("super_cool_benchmark", Benchmark)  # Register class not instance!

If you add benchmarks as sub-modules here, you can register them in this __init__ file.
If you add benchmarks in another package, you can register them there.
"""
from robotics_optimization_benchmarks.benchmarks.benchmark import Benchmark
from robotics_optimization_benchmarks.benchmarks.registry import make
from robotics_optimization_benchmarks.benchmarks.registry import register


__all__ = [
    "make",
    "register",
    "Benchmark",
]
