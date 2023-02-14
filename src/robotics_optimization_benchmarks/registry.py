"""Registry to track various benchmarks and optimizers."""
from copy import deepcopy

import beartype.typing as typing
from beartype import beartype


# Define a generic type (since this registry can be used to register different types
# of underlying object)
T = typing.TypeVar("T")


@beartype
class Registry(typing.Generic[T]):
    """Allows registering objects (e.g. benchmarks) and retrieving by name later.

    Registry entries are immutable; they can be overwritten but retrieving an entry will
    return a copy of the entry, rather than the entry itself.

    Example:
        >>> my_registry = Registry[int]()  # a simple registry that saves int
        >>> my_registry.register("my_special_int", 1)
        >>> my_registry.get_by_name("my_special_int")
        1
        >>> my_registry.names
        ['my_special_int']
    """

    # Private attribute for tracking the entries
    _entries: typing.Dict[str, T]

    def __init__(self) -> None:
        """Create an empty registry."""
        self._entries = {}

    @property
    def names(self) -> typing.List[str]:
        """Return a list of names of entries in the registry."""
        return list(self._entries.keys())

    def get_by_name(self, name: str) -> T:
        """Retrieve an object by name.

        Args:
            name: the name of the entry to retrieve.

        Returns:
            The object registered under the given name.

        Raises:
            KeyError: if there is no object registered with this name.  # noqa: DAR402
        """
        return deepcopy(self._entries[name])

    def register(self, name: str, entry: T) -> None:
        """Register an object with the given name.

        Args:
            name: the name that we should save the entry under.
            entry: the object to register

        Raises:
            ValueError: if there is already an object registered under this name.
        """
        if name in self._entries:
            raise ValueError(f"Name {name} has already been registered!")

        self._entries[name] = entry
