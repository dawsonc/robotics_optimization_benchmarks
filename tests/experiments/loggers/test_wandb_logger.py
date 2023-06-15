"""Define tests for the WandB logger."""
import os
import pathlib

import jax.numpy as jnp
import pytest

from robotics_optimization_benchmarks.experiments.loggers.wandb_logger import (
    WandbLogger,
)


def test_wandb_logger(tmpdir) -> None:
    """Test that we can log data to a file."""
    # Set up a logger
    save_dir = os.path.join(tmpdir, "results")
    logger = WandbLogger(save_dir)

    # Start logging and log a few data packets
    logger.start("test", {"learning_rate": 1e-3})
    for i in range(10):
        logger.log({"a number": i})
    logger.finish()

    # Check that the hyperparameters and log files got saved
    assert len(list(pathlib.Path(save_dir).glob("**/*.json"))) == 1  # hyperparameters
    assert len(list(pathlib.Path(save_dir).glob("**/*.csv"))) == 1  # logs


def test_wandb_logger_multiple_runs(tmpdir) -> None:
    """Test that we can log data to a file."""
    # Set up a logger
    save_dir = os.path.join(tmpdir, "results")
    logger = WandbLogger(save_dir)

    # Start logging and log a few data packets
    logger.start("test1", {"learning_rate": 1e-3})
    for i in range(10):
        logger.log({"a number": i})
    logger.finish()

    # Start logging again and log a few data packets
    logger.start("test2", {"learning_rate": 1e-3})
    for i in range(10):
        logger.log({"a number": i})
    logger.finish()

    # Check that the hyperparameters and log files got saved
    assert len(list(pathlib.Path(save_dir).glob("**/*.json"))) == 2  # hyperparameters
    assert len(list(pathlib.Path(save_dir).glob("**/*.csv"))) == 2  # logs


def test_wandb_logger_save_and_load_artifact(tmpdir) -> None:
    """Test saving an artifact."""
    # Create a logger
    save_dir = os.path.join(tmpdir, "results")
    logger = WandbLogger(save_dir)

    # Make an artifact to try to save
    my_pytree = {"a": 1, "b": jnp.array([1, 2, 3])}

    # Saving before starting should raise an error
    with pytest.raises(RuntimeError):
        artifact_path = logger.save_artifact("my_pytree", my_pytree)

    # Start logging and save an artifact
    logger.start("test", {"learning_rate": 1e-3})
    artifact_path = logger.save_artifact("my_pytree", my_pytree)
    logger.finish()

    # Check that the artifact got saved
    assert pathlib.Path(artifact_path).exists()

    # Test that we can load the artifact
    loaded_pytree = logger.load_artifact(artifact_path, my_pytree)
    assert loaded_pytree is not None
