"""The evaluation command: `python -m idc.classify GOLD_DIR TRANSCRIPT_DIR`.

A development utility, not part of the tool's command surface (T10.1) — the same
arrangement as `python -m idc.schema`. It pairs every gold labelling of a
directory with the transcript of the same name, runs one adapter per transcript
and prints the agreement report of `idc.classify.evaluate`: per-chunk accuracy
and chain-level Rand index, per transcript and over the corpus.

```
python -m idc.classify tests/fixtures/gold tests/fixtures
python -m idc.classify GOLD_DIR TRANSCRIPT_DIR --decisions RUN_DIR
```

Without `--decisions` the adapter plays the **gold baseline** (`gold_script`),
which is right by construction and therefore reports 1.000 — the harness's own
check. With it, the recorded runs of that directory are replayed through the
fake adapter. Nothing here constructs a real backend: the LLM and S1M adapters
take their model as an injected dependency, so a recorded run is how a real
backend's decisions reach this command, and no invocation of it calls a model.

The argument parsing is a module of its own, and the harness a module the
package imports, so that running this command does not import the harness twice
— which is what `runpy` warns about when a package imports the module it was
asked to run.
"""

import argparse
import sys
from pathlib import Path

from idc.classify.errors import ClassifyError
from idc.classify.evaluate import (
    EXIT_INPUT_ERROR,
    EXIT_OK,
    evaluate_cases,
    format_report,
    load_cases,
)
from idc.schema import SchemaValidationError


def build_parser() -> argparse.ArgumentParser:
    """Return the command's argument parser.

    Returns:
        The parser; a function of its own so that the usage text can be
        inspected without running the command.
    """
    parser = argparse.ArgumentParser(
        prog="python -m idc.classify",
        description=(
            "Report how far a classifier run agrees with the gold chain labels: "
            "per-chunk accuracy and chain-level Rand index."
        ),
    )
    parser.add_argument("gold", metavar="GOLD_DIR", type=Path, help="directory of gold labellings")
    parser.add_argument(
        "transcripts",
        metavar="TRANSCRIPT_DIR",
        type=Path,
        help="directory holding the labelled transcripts, one <name>.json each",
    )
    parser.add_argument(
        "--decisions",
        metavar="RUN_DIR",
        type=Path,
        default=None,
        help=(
            "directory of recorded runs to replay, one <name>.json each; "
            "without it the gold baseline is run"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Evaluate a directory of labelled transcripts and print the report.

    Args:
        argv: Command-line arguments, excluding the program name. Defaults to
            `sys.argv[1:]`.

    Returns:
        `EXIT_OK` when a report was printed, `EXIT_INPUT_ERROR` when an input
        could not be used — in which case the reason is on stderr and nothing is
        on stdout.
    """
    arguments = build_parser().parse_args(argv)
    try:
        result = evaluate_cases(
            load_cases(arguments.gold, arguments.transcripts, arguments.decisions)
        )
    except (ClassifyError, SchemaValidationError, OSError) as error:
        print(f"idc: {error}", file=sys.stderr)
        return EXIT_INPUT_ERROR
    print(format_report(result))
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
