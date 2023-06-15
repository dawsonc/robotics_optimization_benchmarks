"""This module contains code for running and plotting experiments.

The intended usage is to create an experiment suite using the factory methods,
run the suite using the methods provided by :code:`ExperimentSuite`.
"""
from robotics_optimization_benchmarks.experiments import experiment_suite_factory
from robotics_optimization_benchmarks.experiments import loggers


__all__ = [
    "experiment_suite_factory",
    "loggers",
]
