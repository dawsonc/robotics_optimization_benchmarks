"""Define a logger that saves to local files."""
import json
import os
from datetime import datetime

import equinox as eqx
import pandas as pd
from beartype import beartype
from beartype.typing import Any
from beartype.typing import Dict
from beartype.typing import List
from beartype.typing import Optional
from jaxtyping import Array
from jaxtyping import Float
from jaxtyping import PyTree
from jaxtyping import jaxtyped

from robotics_optimization_benchmarks.experiments.loggers.abstract_logger import Logger


class FileLogger(Logger):
    """A logger that saves results to local files."""

    _results_dir: str
    _save_prefix: str
    _log_data: List[Dict[str, Float[Array, ""]]]

    @beartype
    def __init__(self, results_dir: str) -> None:
        """Initialize the logger.

        Args:
            results_dir: the directory to save results to
        """
        self._results_dir = results_dir
        # Make sure the save directory exists; create it if not
        os.makedirs(self._results_dir, exist_ok=True)

        # Make the save prefix empty to indicate that no logging is happening
        self._save_prefix = ""

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
        # Create a subdirectory for these logs with a unique name
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S.%f")
        if group is not None:
            timestamp += "_" + group

        self._save_prefix = os.path.join(self._results_dir, benchmark, timestamp)
        os.makedirs(self._save_prefix, exist_ok=True)

        # We're going to save things in memory in this list of log packets, then
        # save to a CSV file when we finish. Start with an empty list
        self._log_data = []

        # Save the parameters to a JSON
        params_path = os.path.join(self._save_prefix, "config.json")
        with open(params_path, "w", encoding="utf-8") as params_file:
            json.dump(config, params_file)

    def finish(self) -> None:
        """Finish logging and save anything that needs to be persisted."""
        # Assemble the logged data into a dataframe and save to csv
        log_df = pd.DataFrame(self._log_data)
        log_path = os.path.join(self._save_prefix, "log.csv")
        log_df.to_csv(log_path, index=False)

        # Set the save prefix to empty to indicate that no logging is happening
        self._save_prefix = ""

    @beartype
    @jaxtyped
    def log(self, data: Dict[str, Any]) -> None:
        """Log the given data.

        Args:
            data: a dictionary of data to log, in the form of JAX scalars.
        """
        self._log_data.append(data)

    @beartype
    @jaxtyped
    def get_logs(self) -> pd.DataFrame:
        """Get the logged data as a single pandas dataframe.

        Returns:
            All of the logged data consolidated into a single pandas dataframe,
            using the tidy data format to include all config information in the
            dataframe.
        """
        # All of the log and config files are saved as csv and json files in the sub-sub
        # directories of the results directory. We need to find all of these directories
        # and load the data from them.
        log_dirs = []
        for root, _, files in os.walk(self._results_dir):
            # Check if this directory contains a log.csv file
            if "log.csv" in files:
                log_dirs.append(root)

        # Load the data and config from each directory
        logs = []
        for log_dir in log_dirs:
            log_file = os.path.join(log_dir, "log.csv")
            config_file = os.path.join(log_dir, "config.json")

            # Load the config
            with open(config_file, encoding="utf-8") as config_file:
                config = json.load(config_file)

            # Load the log
            log = pd.read_csv(log_file)

            # Flatten the config in case it contains nested dicts, then convert to a
            # dataframe with the same shape as logs, so that we can add the config info
            # to each row of the logs
            config = pd.json_normalize(config)
            config = config.loc[config.index.repeat(log.shape[0])]
            config = config.reset_index(drop=True)

            # Add the config to the log
            log = pd.concat([log, config], axis=1)

            # Add the log to the list of logs
            logs.append(log)

        # Concatenate all of the logs into a single dataframe
        logs = pd.concat(logs, axis=0)

        return logs

    @beartype
    @jaxtyped
    def save_artifact(self, name: str, data: PyTree, log_type: str = "generic") -> str:
        """Save an artifact to the logger.

        Args:
            name: the name of the artifact
            data: the data to save
            log_type: the type of artifact to save

        Returns:
            the string identifier for the saved artifact

        Raises:
            RuntimeError: if the logger is not currently logging
        """
        # Only allow saving if logging is going on
        if self._save_prefix == "":
            raise RuntimeError("Cannot save artifact when not logging")

        # Save the data to a file
        artifact_path = os.path.join(self._save_prefix, name + ".eqx")
        eqx.tree_serialise_leaves(artifact_path, data)

        return artifact_path

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
        return eqx.tree_deserialise_leaves(artifact_path, example_pytree)
