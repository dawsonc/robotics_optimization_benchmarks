"""Define an interface for a logging class."""

from abc import ABC
from abc import abstractmethod

from beartype import beartype
from beartype.typing import Any
from beartype.typing import Dict
from beartype.typing import Optional
from jaxtyping import PyTree
from jaxtyping import jaxtyped


class Logger(ABC):
    """Define an abstract base class for a logger."""

    @abstractmethod
    @beartype
    def start(
        self, benchmark: str, config: Dict[str, Any], group: Optional[str] = None
    ) -> None:
        """Start logging.

        Args:
            benchmark: the name used to experiments on the same benchmark
            config: a dictionary of hyperparameters to save
            group: the name of this group of experiments
        """

    @abstractmethod
    def finish(self) -> None:
        """Finish logging and save anything that needs to be persisted."""

    @abstractmethod
    @beartype
    @jaxtyped
    def log(self, data: Dict[str, Any]) -> None:
        """Log the given data.

        Args:
            data: a dictionary of data to log, in the form of JAX scalars.
        """

    @abstractmethod
    @beartype
    @jaxtyped
    def save_artifact(self, name: str, data: PyTree, type: str = "generic") -> str:
        """Save an artifact to the logger.

        Args:
            name: the name of the artifact
            data: the data to save
            type: the type of the artifact

        Returns:
            the string identifier for the saved artifact

        Raises:
            RuntimeError: if the logger is not running
        """

    @abstractmethod
    @beartype
    @jaxtyped
    def load_artifact(self, artifact_path: str, example_pytree: PyTree) -> PyTree:
        """Load an artifact from the logger.

        Args:
            artifact_path: the path to the artifact
            example_pytree: an example of the PyTree structure of the artifact

        Returns:
            the loaded artifact
        """
