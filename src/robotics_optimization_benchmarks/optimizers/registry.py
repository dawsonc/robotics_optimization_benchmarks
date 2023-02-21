"""Define a public API for managing the optimizer registry."""
from beartype import beartype
from beartype.typing import Dict
from beartype.typing import Type

from robotics_optimization_benchmarks.optimizers.gd import GD
from robotics_optimization_benchmarks.optimizers.optimizer import Optimizer


# Make a registry to store the optimizers
# WARNING: global mutable state is usually frowned upon, but here we'll manage
# access to it using public functions to mitigate some of the risk
_optimizer_registry: Dict[str, Type[Optimizer]] = {}


# Define public functions for accessing the optimizer registry
@beartype
def make(name: str) -> Type[Optimizer]:
    """Access an optimizer stored in the registry.

    Optimizers can be constructed by chaining with the initializer or `from_dict`
    class methods, e.g.:
    ```
    my_optimizer = optimizers.make("optimizer")(arg1, arg2)
    my_optimizer = optimizers.make("optimizer").from_dict({"arg1": 1, "arg2": 2})
    ```

    Args:
        name: the name of the optimizer to access.

    Raises:
        KeyError: if the optimizer name is not registered.  # noqa: DAR402

    Returns:
        The optimizer class stored in the registry.
    """
    return _optimizer_registry[name]


@beartype
def register(name: str, optimizer: Type[Optimizer]) -> None:
    """Register an optimizer with the given name.

    Args:
        name: the name to register the optimizer under.
        optimizer: the optimzer class to register.

    Raises:
        ValueError: if there is already a optimizer registered under this name.
    """
    if name in _optimizer_registry:
        raise ValueError(f"Optimizer {name} is already registered!")
    _optimizer_registry[name] = optimizer


###############################################################################
# Register built-in optimizers
###############################################################################

register(GD.name, GD)
