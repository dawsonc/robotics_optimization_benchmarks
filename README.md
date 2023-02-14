# Robotics Optimization Benchmarks

<!-- [![PyPI](https://img.shields.io/pypi/v/robotics_optimization_benchmarks.svg)][pypi_]
[![Status](https://img.shields.io/pypi/status/robotics_optimization_benchmarks.svg)][status]
[![Python Version](https://img.shields.io/pypi/pyversions/robotics_optimization_benchmarks)][python version]
[![License](https://img.shields.io/pypi/l/robotics_optimization_benchmarks)][license]

[![Read the documentation at https://robotics_optimization_benchmarks.readthedocs.io/](https://img.shields.io/readthedocs/robotics_optimization_benchmarks/latest.svg?label=Read%20the%20Docs)][read the docs] -->

[![Tests](https://github.com/dawsonc/robotics_optimization_benchmarks/workflows/Tests/badge.svg)][tests]
[![Codecov](https://codecov.io/gh/dawsonc/robotics_optimization_benchmarks/branch/main/graph/badge.svg)][codecov]
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)][pre-commit]
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)][black]

[pypi_]: https://pypi.org/project/robotics_optimization_benchmarks/
[status]: https://pypi.org/project/robotics_optimization_benchmarks/
[python version]: https://pypi.org/project/robotics_optimization_benchmarks
[read the docs]: https://robotics_optimization_benchmarks.readthedocs.io/
[tests]: https://github.com/dawsonc/robotics_optimization_benchmarks/actions?workflow=Tests
[codecov]: https://app.codecov.io/gh/dawsonc/robotics_optimization_benchmarks
[pre-commit]: https://github.com/pre-commit/pre-commit
[black]: https://github.com/psf/black

This project aims to do 3 things:

1. Maintain a suite of robotics- and control-relevant optimization benchmarks. This is similar to the aims of the [Gymnasium](https://gymnasium.farama.org) project, but is a) more specific to controls and robotics, b) intended for more general optimization rather than strictly reinforcement learning, and c) is implemented in [JAX](https://jax.readthedocs.io) to allow for easy acceleration, parallelization, and automatic-differentiation.
2. Maintain a set of baseline optimization algorithms (ranging from exact gradient-based to gradient free) against which researchers can compare their new algorithms.
3. Publish up-to-date comparisons of the performance of the baseline algorithms on the benchmark suite.

## Installation

For reproducibility, these benchmarks are intended to be run from within a development container. To install, clone the code

```bash
git clone git@github.com:dawsonc/robotics_optimization_benchmarks.git
```

Open directory in VSCode, then use the `Rebuild and Reopen in container` command. The package should be installed in the container. You can verify that installation was successful by running the tests with `poetry run nox`

## Usage

Please see the [Command-line Reference] for details.

## Citing

If you find this useful in your own research, please cite our publication on this topic

```bibtex
TODO
```

## Contributing

Contributions are very welcome. This includes both bug-fixes and new features for the main codebase as well as contributions of new baseline algorithms and benchmark problems (although these will require some discussion on what value-add we get from committing to maintain a new algorithm/benchmark).
To learn more, see the [Contributor Guide].

## License

Distributed under the terms of the [BSD 3-clause license][license],
_robotics_optimization_Benchmarks_ is free and open source software.

## Issues

If you encounter any problems,
please [file an issue] along with a detailed description.

## Credits

This project was generated from [@cjolowicz]'s [Hypermodern Python Cookiecutter] template.

[@cjolowicz]: https://github.com/cjolowicz
[pypi]: https://pypi.org/
[hypermodern python cookiecutter]: https://github.com/cjolowicz/cookiecutter-hypermodern-python
[file an issue]: https://github.com/dawsonc/robotics_optimization_benchmarks/issues
[pip]: https://pip.pypa.io/

<!-- github-only -->

[license]: https://github.com/dawsonc/robotics_optimization_benchmarks/blob/main/LICENSE
[contributor guide]: https://github.com/dawsonc/robotics_optimization_benchmarks/blob/main/CONTRIBUTING.md
[command-line reference]: https://robotics_optimization_benchmarks.readthedocs.io/en/latest/usage.html
