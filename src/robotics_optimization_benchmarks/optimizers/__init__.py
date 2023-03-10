"""A set of optimization algorithms implemented in JAX.

Optimizers are accessed using the :code:`make` function (inspired by the OpenAI Gym API).

.. code-block:: python

    my_optimizer = optimizers.make("super_cool_optimizer")

When implementing new optimizers, make sure to add them to the registry!
Benchmarks can be registered using the :code:`register` function.

.. code-block:: python

    optimizers.register("super_cool_optimizer", Optimizer)  # Register class not instance!

If you add optimizers as sub-modules here, you can register them in this
:code:`__init__` file. If you add optimizers in another package, you can register them
in their home package.
"""
from robotics_optimization_benchmarks.optimizers.optimizer import Optimizer
from robotics_optimization_benchmarks.optimizers.optimizer import OptimizerState
from robotics_optimization_benchmarks.optimizers.registry import make
from robotics_optimization_benchmarks.optimizers.registry import register


__all__ = [
    "make",
    "register",
    "Optimizer",
    "OptimizerState",
]
