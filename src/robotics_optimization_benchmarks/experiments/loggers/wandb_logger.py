"""Define an interface for logging using WandB."""
from beartype import beartype
from beartype.typing import Any
from beartype.typing import Dict
from jaxtyping import PyTree
from jaxtyping import jaxtyped

import wandb
from robotics_optimization_benchmarks.experiments.loggers.file_logger import FileLogger


class WandbLogger(FileLogger):
    """Define a logger that saves to WandB.

    Subclasses FileLogger, so it also saves everything locally.
    """

    _run = None

    @beartype
    def __init__(self, results_dir: str) -> None:
        """Initialize the logger.

        Args:
            results_dir: the local directory to save results to
        """
        super().__init__(results_dir)

    @beartype
    def start(self, label: str, config: Dict[str, Any]) -> None:
        """Start logging.

        Args:
            label: the name used to group similar experiments
            config: a dictionary of hyperparameters to save
        """
        super().start(label, config)
        self._run = wandb.init(project=label, config=config)

    def finish(self) -> None:
        """Finish logging and save anything that needs to be persisted."""
        self._run.finish()
        self._run = None  # reset to prevent access to a finished run
        super().finish()

    @beartype
    @jaxtyped
    def log(self, data: Dict[str, Any]) -> None:
        """Log the given data.

        Args:
            data: a dictionary of data to log, in the form of JAX scalars.
        """
        super().log(data)
        self._run.log(data)

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
        """
        # Save the artifact to a local file
        save_path = super().save_artifact(name, data, type)

        # Upload that file to WandB
        artifact = wandb.Artifact(name=name, type=type)
        artifact.add_file(save_path)
        self._run.log_artifact(artifact)

        # Return the path to the local file
        return save_path
