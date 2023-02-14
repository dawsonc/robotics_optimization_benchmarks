"""Test cases for the Registry class."""
import pytest
from beartype import beartype

from robotics_optimization_benchmarks.registry import Registry


class SuperCoolObject:
    """A super cool object that doesn't do anything (used for testing)."""

    def __init__(self) -> None:
        """Initialize the object with a list attribute."""
        self.attribute = [1, 2, 3]


@pytest.fixture
@beartype
def registry() -> Registry[SuperCoolObject]:
    """Fixture for creating a registry pre-populated with some objects."""
    my_registry = Registry[SuperCoolObject]()
    my_registry.register("first", SuperCoolObject())
    my_registry.register("second", SuperCoolObject())

    return my_registry


def test_create_registry() -> None:
    """Test creating a registry, adding some items to it, and getting a list of items."""
    # Create some objects to save in the registry
    first = SuperCoolObject()
    second = SuperCoolObject()

    my_registry = Registry[SuperCoolObject]()
    my_registry.register("first", first)
    my_registry.register("second", second)

    assert len(my_registry.names) == 2


def test_access_registry_valid_name(registry: Registry[SuperCoolObject]) -> None:
    """Test accessing an items in a registry."""
    name = registry.names[0]
    assert registry.get_by_name(name) is not None


def test_access_registry_invalid_name(registry: Registry[SuperCoolObject]) -> None:
    """Test accessing an items in a registry that isn't actually there."""
    name = "nonexistent"
    with pytest.raises(KeyError):
        registry.get_by_name(name)


def test_register_valid_name(registry: Registry[SuperCoolObject]) -> None:
    """Test adding to a registry."""
    initial_len = len(registry.names)
    registry.register("never before seen", SuperCoolObject())
    assert len(registry.names) == initial_len + 1


def test_register_invalid_name(registry: Registry[SuperCoolObject]) -> None:
    """Test adding to a registry using a name that's already in there."""
    name = registry.names[0]
    with pytest.raises(ValueError):
        registry.register(name, SuperCoolObject())


def test_registry_items_immutable(registry: Registry[SuperCoolObject]) -> None:
    """Test that registry entries are immutable.

    This means that modifying the object retrieved from the registry should not modify
    the entry saved in the registry.
    """
    # Get the first entry and modify it
    name = registry.names[0]
    first_entry = registry.get_by_name(name)
    original_attribute = list(
        first_entry.attribute
    )  # save a copy to compare with later
    first_entry.attribute.append(4)

    # If we get this entry again, it should not have been modified
    second_entry = registry.get_by_name(name)
    assert second_entry.attribute == original_attribute
    assert second_entry.attribute != first_entry.attribute
