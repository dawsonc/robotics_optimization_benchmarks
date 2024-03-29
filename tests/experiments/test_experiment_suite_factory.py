"""Define tests for the experiment suite factory."""
import pytest
from jaxtyping import PyTree

from robotics_optimization_benchmarks import experiment_suite_factory
from robotics_optimization_benchmarks.experiments.loggers import Logger


# Make a mock logger
class MockLogger(Logger):
    """A mock logger."""

    started: bool = False
    finished: bool = False
    config: dict
    log_data: list
    saved_artifacts: dict

    def start(self, label: str, config: dict, group: str) -> None:
        """Start the logger."""
        self.started = True
        self.config = config
        self.log_data = []
        self.saved_artifacts = {}

    def finish(self) -> None:
        """Finish the logger."""
        self.finished = True

    def log(self, data) -> None:
        """Log data."""
        print(f"Logging {data}...")
        self.log_data.append(data)
        print(len(self.log_data))

    def save_artifact(self, name: str, data: PyTree, log_type: str = "generic") -> str:
        """Save an artifact."""
        self.saved_artifacts[name] = data
        return name

    def load_artifact(self, artifact_path: str, example_pytree: PyTree) -> PyTree:
        """Load an artifact (noop)."""

    def get_logs(self):
        """Get the logs."""


# Make a fixture for a logger
@pytest.fixture(name="logger")
def fixture_file_logger(tmpdir):
    """Create a file logger instance."""
    return MockLogger()


@pytest.mark.parametrize("save_artifacts", [True, False])
def test_experiment_suite_factory_user_story(save_artifacts, logger) -> None:
    """Integration test: test creating and running an experiment suite.

    As a user, I want to run suites of multiple experiments to compare the performance
    of different optimizers on the same benchmark problem, so that I can easily generate
    the results for my paper.
    """
    experiment_suite = experiment_suite_factory.create_experiment_suite(
        # As a user, I can name this test suite and provide a brief description to
        # help me remember what I was testing with this experiment.
        name="test_suite",
        description="Test suite for integration tests.",
        #
        # As a user, I want to specify the random seeds I use for this experiment suite,
        # so I can test across different seeds and reproduce my results.
        seeds=[0, 1, 2],
        #
        # As a user, I want to specify which benchmark problem I want to use for this
        # experiment suite, so that I can compare optimizers on the same problem.
        benchmark_name="quadratic",
        benchmark_hparams={"dimension": 10},
        #
        # As a user, I want to specify how many optimization steps each optimizer should
        # run for, to enable an apples-to-apples comparison
        max_steps=100,
        #
        # As a user, I want to specify which optimizers I want to use for this
        # experiment suite and specify the hyperparameters for each optimizer, so that
        # I can compare the performance of different optimizers.
        optimizer_specs=[
            {
                "name": "GradientDescent_1",  # a name to label this optimizer
                "type": "GD",  # what type of optimizer is this? should match registry
                "hparams": {"step_size": 0.01},
            },
            {
                "name": "MALA_1",
                "type": "MCMC",
                "hparams": {
                    "step_size": 0.01,
                    "use_gradients": True,
                    "use_metropolis": True,
                },
            },
        ],
    )

    import json

    with open(
        "tests/experiments/test_data/experiment_suite.json", "w", encoding="utf-8"
    ) as params_file:
        json.dump(experiment_suite.to_dict(), params_file)

    # Make sure that initialization took place
    assert experiment_suite is not None
    assert experiment_suite.to_dict()["name"] == "test_suite"

    # I want to be able to run these experiments without saving the solution
    experiment_suite.run(logger, save_solution=save_artifacts)

    # We should have logged some data packets
    assert len(logger.log_data) > 1

    # Whether or not we saved artifacts depends on the parameterized flag
    assert len(logger.saved_artifacts) == (1 if save_artifacts else 0)

    # Make sure that the logger was started and finished
    assert logger.started
    assert logger.finished


def test_experiment_suite_factory_duplicate_optimizer_names() -> None:
    """Test that we cannot create an experiment suite with duplicate optimizer names."""
    with pytest.raises(ValueError):
        experiment_suite_factory.create_experiment_suite(
            name="test_suite",
            description="Test suite for integration tests.",
            seeds=[0, 1, 2],
            benchmark_name="quadratic",
            benchmark_hparams={"dimension": 10},
            max_steps=100,
            optimizer_specs=[
                {
                    "name": "GradientDescent_1",
                    "type": "GD",
                    "hparams": {"step_size": 0.01},
                },
                {
                    "name": "GradientDescent_1",
                    "type": "GD",
                    "hparams": {"step_size": 0.1},
                },
            ],
        )


def test_experiment_suite_factory_from_file() -> None:
    """Test that we can create an experiment suite from a file."""
    experiment_suite = experiment_suite_factory.create_experiment_suite_from_file(
        "tests/experiments/test_data/experiment_suite.json"
    )
    assert experiment_suite is not None
    assert experiment_suite.to_dict()["name"] == "test_suite"
