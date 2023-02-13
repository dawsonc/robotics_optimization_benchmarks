"""Command-line interface."""
import click


@click.command()
@click.version_option()
def main() -> None:
    """Sampling_Benchmarks."""


if __name__ == "__main__":
    main(prog_name="sampling_benchmarks")  # pragma: no cover
