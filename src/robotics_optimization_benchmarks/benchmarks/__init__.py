"""A set of robotics- and control-relevant optimization benchmarks, implemented in JAX.

Benchmarks are accessed using the `make` function (inspired by the OpenAI Gym API).
```
my_benchmark = benchmarks.make("super_cool_benchmark")
```

When implementing new benchmarks, make sure to add them to the registry in this file!
Benchmarks can be registered using the `register` function.
```
benchmarks.register("super_cool_benchmark", Benchmark)  # Register class not instance!
```
"""
from robotics_optimization_benchmarks.benchmarks.benchmark import Benchmark
from robotics_optimization_benchmarks.benchmarks.registry import make
from robotics_optimization_benchmarks.benchmarks.registry import register


__all__ = [
    "Benchmark",
    "make",
    "register",
]

# -----------------------------------------------------
# Add benchmarks to registry below
# -----------------------------------------------------
