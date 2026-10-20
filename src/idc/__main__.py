"""Module entry point: `python -m idc`, which `make run ARGS="…"` forwards to.

The entry point exists so that the invocation documented in `README.md` and the
`run` target of the `Makefile` reach the package from the moment the repository
is scaffolded. The commands themselves are implemented in ticket T10.1; until
then every invocation exits with status 2 and says so, rather than failing with
an import error.
"""

import sys

NOT_WIRED_UP = (
    "idc: the command surface is not wired up yet (ticket T10.1). "
    "The package scaffold is in place; `make test` runs the suite."
)

EXIT_NOT_WIRED_UP = 2


def main(argv: list[str] | None = None) -> int:
    """Report that the command surface is not available yet.

    Args:
        argv: Command-line arguments, excluding the program name. Defaults to
            `sys.argv[1:]`. They are accepted so that the signature already
            matches the real entry point, and ignored until T10.1 lands.

    Returns:
        The process exit status: always `EXIT_NOT_WIRED_UP`.
    """
    del argv
    print(NOT_WIRED_UP, file=sys.stderr)
    return EXIT_NOT_WIRED_UP


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
