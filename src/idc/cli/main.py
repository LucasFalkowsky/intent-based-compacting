"""The command line itself: the parser, the command table and the one entry point.

`argparse` from the standard library, so the command surface costs no
dependency, and one subparser per command. Three things about the shape are
decisions rather than defaults:

- **`main` returns a status and never raises `SystemExit`.** `argparse` exits
  the process on `--help` and on a usage error; both are caught here and turned
  into `EXIT_OK` and `EXIT_USAGE_ERROR`. That is what lets the suite call the
  real command path in-process — `main(["run", ...], runtime=...)` — instead of
  testing a parser and a set of functions that nothing proves are connected.
- **The command line and the config file are merged before validation.** A flag
  is not a patch applied to a validated object; it is a key of the same
  document, checked exactly as a key in the file is. `configuration` does the
  merge and `idc.cli.config.validate_config` does the checking.
- **A value a command cannot do without is a `UsageError` naming its flag.**
  `argparse`'s own `required=True` cannot be used for these, because the config
  file is allowed to supply them; so the requirement is checked after the merge
  and reported with the same exit status `argparse` uses for a usage mistake.

Everything a command may fail on afterwards is one of the stage error families —
`CliError`, `SchemaValidationError`, `ClassifyError`, `ResolveError`,
`CompactError`, `ParseError`, `RenderError`, `MeasureError` — plus `OSError` and
`ValueError` for a file that cannot be written and a `ValueError` a stage draws
the line at. All of them become one line on stderr and `EXIT_INPUT_ERROR`. That
list is why every stage of this tool has a base error: `DIAGNOSTIC_ERRORS` is
the single `except` those docstrings are written against.
"""

import argparse
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from idc.classify import ClassifyError
from idc.cli.commands import (
    chunk_command,
    classify_command,
    compact_command,
    import_command,
    measure_command,
    render_command,
    resolve_command,
    run_command,
)
from idc.cli.config import ADAPTER_NAMES, PATH_KEYS, Config, read_config_document, validate_config
from idc.cli.errors import CliError, UsageError
from idc.cli.exit_codes import EXIT_INPUT_ERROR, EXIT_OK, EXIT_USAGE_ERROR
from idc.cli.runtime import Runtime
from idc.compact import CompactError
from idc.measure import MeasureError
from idc.parse import ParseError
from idc.render import RenderError
from idc.resolve import ResolveError
from idc.schema import Agent, SchemaValidationError

__all__ = [
    "COMMANDS",
    "COMMAND_NAMES",
    "DIAGNOSTIC_ERRORS",
    "PROGRAM",
    "Command",
    "build_parser",
    "configuration",
    "main",
]

PROGRAM = "idc"
"""The program name in every usage line, whichever way the module was started.

Set explicitly because `python -m idc` would otherwise print the path of
`__main__.py`, and `make run ARGS="compact --help"` is supposed to print
something a reader can type back.
"""

TRANSCRIPT = "--in"
"""The file a command reads. Every command has it, and every command needs it."""

OUTPUT = "--out"
"""The file a command writes."""

AGENT = "--agent"
"""The agent an input transcript comes from."""

TARGET = "--target"
"""The agent a summary is rendered for."""

COMPACTED = "--compacted"
"""`measure`'s second transcript: the compacted side of the comparison."""

ADAPTER = "--adapter"
"""Which classifier adapter the run asks for."""

REDUCE = "--reduce"
"""Whether to reduce the harness noise inside a chunk; `--no-reduce` turns it off."""

MANIFEST = "--manifest"
"""Where `run` writes its manifest."""

PREVIOUS_MANIFEST = "--previous-manifest"
"""The manifest of the run this one continues."""

DIAGNOSTIC_ERRORS = (
    CliError,
    SchemaValidationError,
    ClassifyError,
    ResolveError,
    CompactError,
    ParseError,
    RenderError,
    MeasureError,
    OSError,
    ValueError,
)
"""Everything a command may fail on that is a diagnostic rather than a defect.

One `except` over the base error of every stage that has one — five of the eight
stages do, and `idc.cli` itself — plus `OSError` and `ValueError`. `ValueError`
is last and deliberately broad, and it is what covers the two stages that raise
no family of their own: `idc.chunk` refuses a caller's bad range with a plain
`ValueError`, and `idc.importers.TranscriptImportError` *is* a `ValueError`, so
dropping it would put a traceback on the terminal for a transcript that cannot
be read.
"""


@dataclass(frozen=True, slots=True)
class Command:
    """One command of the surface: its name, its help, its flags and its body.

    Attributes:
        name: What the user types after `idc`.
        summary: The one-line description in the command list. Kept short
            enough to stay one line there.
        body: What runs once the configuration is valid and complete.
        options: The flags this command accepts beyond `--config` and `--in`.
        required: The flags whose value it cannot do without. `--in` is
            implicitly required for every command and is not listed.
    """

    name: str
    summary: str
    body: Callable[[Config, Runtime], int]
    options: tuple[str, ...] = ()
    required: tuple[str, ...] = ()


COMMANDS: tuple[Command, ...] = (
    Command(
        "import",
        "read an agent transcript into the canonical format",
        import_command,
        options=(AGENT, OUTPUT),
        required=(AGENT, OUTPUT),
    ),
    Command(
        "chunk",
        "split a canonical transcript into chunks",
        chunk_command,
        options=(OUTPUT, REDUCE),
        required=(OUTPUT,),
    ),
    Command(
        "classify",
        "assign each chunk to an intent chain",
        classify_command,
        options=(OUTPUT, ADAPTER),
        required=(OUTPUT,),
    ),
    Command(
        "resolve",
        "report the accepted solution of every chain",
        resolve_command,
    ),
    Command(
        "compact",
        "render new or extended chains into Markdown blocks",
        compact_command,
        options=(OUTPUT, TARGET),
        required=(OUTPUT, TARGET),
    ),
    Command(
        "render",
        "emit the summary in the target agent's native tags",
        render_command,
        options=(OUTPUT,),
        required=(OUTPUT,),
    ),
    Command(
        "measure",
        "count tokens before and after compaction",
        measure_command,
        options=(AGENT, COMPACTED),
        required=(AGENT, COMPACTED),
    ),
    Command(
        "run",
        "import, compact, render and measure in one pass",
        run_command,
        options=(AGENT, TARGET, ADAPTER, OUTPUT, REDUCE, MANIFEST, PREVIOUS_MANIFEST),
        required=(AGENT, OUTPUT),
    ),
)
"""The whole command surface, in pipeline order with `run` last.

A command is registered here and nowhere else, the way `idc.importers.IMPORTERS`
holds the importers: adding one is an entry here plus its body in
`idc.cli.commands`.
"""

COMMAND_NAMES = tuple(command.name for command in COMMANDS)
"""The command names, in the order the help lists them."""

COMMANDS_BY_NAME: Mapping[str, Command] = {command.name: command for command in COMMANDS}
"""The same table, keyed by name — what `main` dispatches on."""

AGENT_CHOICES = tuple(member.value for member in Agent)
"""The agent names `--agent` and `--target` accept, as the schema spells them."""

_SUPPLIED: Mapping[str, Callable[[Config], object | None]] = {
    TRANSCRIPT: lambda config: config.paths.transcript,
    OUTPUT: lambda config: config.paths.output,
    COMPACTED: lambda config: config.paths.compacted,
    AGENT: lambda config: config.agent,
    TARGET: lambda config: config.target_agent,
}
"""Where each requirable flag's value ends up, once the merge has run."""


def _add_options(parser: argparse.ArgumentParser, command: Command) -> None:
    """Add one command's flags to its subparser.

    Args:
        parser: The command's own parser.
        command: The command, whose `options` say which flags it has.
    """
    parser.add_argument(
        "--config",
        metavar="FILE",
        type=Path,
        default=None,
        help="TOML configuration file; a flag below overrides what it sets",
    )
    parser.add_argument(
        TRANSCRIPT, dest="transcript", metavar="PATH", default=None, help="the file to read"
    )
    if OUTPUT in command.options:
        parser.add_argument(
            OUTPUT, dest="output", metavar="PATH", default=None, help="the file to write"
        )
    if AGENT in command.options:
        parser.add_argument(
            AGENT,
            dest="agent",
            choices=AGENT_CHOICES,
            default=None,
            help="the agent the input transcript comes from",
        )
    if TARGET in command.options:
        parser.add_argument(
            TARGET,
            dest="target",
            choices=AGENT_CHOICES,
            default=None,
            help="the agent the summary is rendered for; defaults to --agent",
        )
    if COMPACTED in command.options:
        parser.add_argument(
            COMPACTED,
            dest="compacted",
            metavar="PATH",
            default=None,
            help="the compacted transcript to compare the input against",
        )
    if ADAPTER in command.options:
        parser.add_argument(
            ADAPTER,
            dest="adapter",
            choices=ADAPTER_NAMES,
            default=None,
            help="the classifier adapter the run asks for",
        )
    if REDUCE in command.options:
        parser.add_argument(
            REDUCE,
            dest="reduce",
            action=argparse.BooleanOptionalAction,
            default=None,
            help="reduce the harness noise inside a chunk (default: on)",
        )
    if MANIFEST in command.options:
        parser.add_argument(
            MANIFEST,
            dest="manifest",
            metavar="PATH",
            default=None,
            help="where to write the run manifest; defaults to <--out>.manifest.json",
        )
    if PREVIOUS_MANIFEST in command.options:
        parser.add_argument(
            PREVIOUS_MANIFEST,
            dest="previous_manifest",
            metavar="PATH",
            default=None,
            help="the manifest of the run this one continues",
        )


def build_parser() -> argparse.ArgumentParser:
    """Return the whole command surface's argument parser.

    A function rather than a module-level object, so that the usage texts can be
    inspected without running a command and so that two parses cannot share
    state.

    Returns:
        The parser, with one subparser per entry of `COMMANDS`.
    """
    parser = argparse.ArgumentParser(
        prog=PROGRAM,
        description=(
            "Compact a coding-agent transcript along the user intents it contains. "
            "One command per pipeline stage, with files in between, plus `run` for the "
            "whole pipeline in one pass."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")
    for command in COMMANDS:
        subparser = subparsers.add_parser(
            command.name, help=command.summary, description=command.summary
        )
        _add_options(subparser, command)
    return parser


def configuration(arguments: argparse.Namespace) -> Config:
    """Return the configuration one invocation resolves to.

    The config file is read first and the command line is applied on top, key by
    key, and the merged document is validated once. A flag whose value is `None`
    was not given and overrides nothing, which is why `--reduce/--no-reduce` is
    a `BooleanOptionalAction` with no default: a plain `store_false` would make
    "not given" indistinguishable from "turned off" and silently override the
    file.

    Args:
        arguments: The parsed command line.

    Returns:
        The validated configuration.

    Raises:
        ConfigError: If the config file cannot be read, is not TOML, or the
            merged document is not a valid configuration.
    """
    source = None if arguments.config is None else str(arguments.config)
    base = (
        Config()
        if source is None
        else validate_config(read_config_document(arguments.config), source=source)
    )
    overrides: dict[str, Any] = {
        key: getattr(arguments, key)
        for key in ("agent", "target", "adapter", "reduce")
        if getattr(arguments, key, None) is not None
    }
    paths = {
        key: getattr(arguments, key)
        for key in PATH_KEYS
        if getattr(arguments, key, None) is not None
    }
    if paths:
        overrides["paths"] = paths
    return base.with_overrides(overrides, source=source)


def _missing_option(command: Command, config: Config) -> str | None:
    """Return the first flag the command needs and nothing supplied, or `None`.

    Args:
        command: The command that is about to run.
        config: The merged configuration.

    Returns:
        The flag's name, in the order `(--in,) + command.required` lists them,
        or `None` when every one of them has a value.
    """
    for option in (TRANSCRIPT, *command.required):
        if _SUPPLIED[option](config) is None:
            return option
    return None


def main(argv: list[str] | None = None, *, runtime: Runtime | None = None) -> int:
    """Run one command and return the process exit status.

    Args:
        argv: Command-line arguments, excluding the program name. Defaults to
            `sys.argv[1:]`.
        runtime: The classifier adapter, the narrative backend, the tokenizers
            and the clock a command runs with. An empty `Runtime` when omitted,
            which is what a real invocation gets — and what makes a command
            that needs a model backend report that none is available. See
            `idc.cli.runtime`.

    Returns:
        `EXIT_OK`, `EXIT_INPUT_ERROR` or `EXIT_USAGE_ERROR`. Never raises
        `SystemExit`: `argparse`'s own exit on `--help` and on a usage error is
        caught and turned into a status, so a caller in the same process gets
        an answer instead of a dead interpreter.
    """
    parser = build_parser()
    try:
        arguments = parser.parse_args(argv)
    except SystemExit as requested:
        return EXIT_OK if requested.code in (0, None) else EXIT_USAGE_ERROR
    command = COMMANDS_BY_NAME[arguments.command]
    try:
        config = configuration(arguments)
        missing = _missing_option(command, config)
        if missing is not None:
            raise UsageError(missing, command.name)
        return command.body(config, runtime if runtime is not None else Runtime())
    except UsageError as error:
        print(f"{PROGRAM}: {error}", file=sys.stderr)
        return EXIT_USAGE_ERROR
    except DIAGNOSTIC_ERRORS as error:
        print(f"{PROGRAM}: {error}", file=sys.stderr)
        return EXIT_INPUT_ERROR
