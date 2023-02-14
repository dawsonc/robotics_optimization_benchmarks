"""Command-line interface."""
import click


@click.command()
@click.version_option()
def main() -> None:
    """Robotics Optimization Benchmarks."""


if __name__ == "__main__":
    main(prog_name="robotics_optimization_benchmarks")  # pragma: no cover
