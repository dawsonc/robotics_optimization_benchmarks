"""This module contains code for running and plotting experiments.

The intended usage is to create an experiment suite using the factory methods,
run the suite using the methods provided by :code:`ExperimentSuite`, then visualize and
render the results of the experiments using the methods provided by the
:code:`visualization` sub-module.
"""
from robotics_optimization_benchmarks.experiments import experiment_suite_factory
from robotics_optimization_benchmarks.experiments import visualization


__all__ = [
    "experiment_suite_factory",
    "visualization",
]
