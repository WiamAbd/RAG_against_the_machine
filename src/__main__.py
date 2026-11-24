"""Run the CLI and report errors without Python tracebacks."""

import sys

import fire

from .cli import RagCLI


def main() -> None:
    """Run a command and handle failures at the application boundary."""
    try:
        fire.Fire(RagCLI)
    except Exception as error:
        print(
            f"Error: {str(error) or type(error).__name__}",
            file=sys.stderr,
        )
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
