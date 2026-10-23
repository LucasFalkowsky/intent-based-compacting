"""The failures of the command surface, as one catchable family.

The same shape every stage of this tool uses — `idc.classify.errors`,
`idc.resolve.errors`, `idc.compact.errors`, `idc.parse.errors`,
`idc.measure.errors` — and for the same reason: `idc.cli.main` catches the whole
surface with one `except` and turns it into a line on stderr and an exit status,
so no library traceback ever reaches a user of the tool. `pydantic`'s
`ValidationError` and `tomllib`'s `TOMLDecodeError` are both translated at the
boundary they cross; neither escapes `idc.cli.config`.

The line between the family and a plain `ValueError` is the one
`idc.measure.errors` draws, by **where the data comes from**:

- Everything the surface is *handed* is in the family: the config file, the
  pipeline artifacts a previous command wrote, the command line itself.
- A value only a caller's own code can produce is **not**. A `Config` built
  wrongly in Python raises a plain `ValueError`, exactly as a hand-built
  `RunManifest` does; see `idc.cli.config`.

`UsageError` is in the family but is the one member `main` maps to
`EXIT_USAGE_ERROR` rather than `EXIT_INPUT_ERROR`: "you did not say which file"
is a fact about the invocation, which is what `argparse` reports with that
status for the mistakes it catches itself.
"""

from collections.abc import Sequence

__all__ = [
    "ArtifactError",
    "BackendUnavailableError",
    "CliError",
    "ConfigError",
    "UsageError",
]


class CliError(Exception):
    """Base class of every failure the command surface raises itself."""


class UsageError(CliError):
    """The invocation is incomplete: a value the command needs was never given.

    Raised for what `argparse` cannot check, which is everything that may come
    either from a flag or from the config file. The message names the flag, so
    the repair is readable from the error alone.

    Attributes:
        option: The flag that would have supplied the value, e.g. `--in`.
        command: The command that needed it.
    """

    def __init__(self, option: str, command: str) -> None:
        """Build the error from the missing flag and the command that needed it.

        Args:
            option: The flag whose value is missing.
            command: The command's name.
        """
        self.option = option
        self.command = command
        super().__init__(
            f"{command}: {option} is required and was not given; pass it on the command line "
            f"or set it in the config file"
        )


class ConfigError(CliError):
    """The config file could not be read, parsed or validated.

    The file comes from outside the tool, so every problem with it is a
    diagnostic: a missing file, bytes that are not UTF-8, text that is not TOML,
    an unknown key, an agent that is not one of the study's. Built like
    `idc.measure.ManifestError` so that a validation failure lists one located
    problem per line.

    Attributes:
        source: The file the document came from, or `None` when it was
            validated from text or from an already-parsed mapping with no
            source named.
        errors: One `(location, message)` pair per problem, the location being
            the field name pydantic reported it under. A problem inside
            `[paths]`, `[models]` or `[options]` that pydantic locates itself
            comes out dotted (`models.classifier`); one that `Paths` refuses on
            its own behalf is listed under `paths` with its own lines
            underneath, because the nested model translates its construction
            error before `Config` sees it. That is the shape
            `idc.measure.ManifestError` produces, and the reason is the same.
            Empty for a failure that is not about a field.
    """

    def __init__(
        self,
        reason: str,
        *,
        source: str | None = None,
        errors: Sequence[tuple[str, str]] = (),
    ) -> None:
        """Build the error and render its message.

        Args:
            reason: Summary line, e.g. `2 problems` or `not valid TOML`.
            source: Where the document came from, named in the message when
                given.
            errors: The located problems, listed one per line after the
                summary.
        """
        self.source = source
        self.errors = tuple(errors)
        origin = f" from {source}" if source else ""
        lines = [f"invalid configuration{origin}: {reason}"]
        lines += [f"  {location}: {message}" for location, message in self.errors]
        super().__init__("\n".join(lines))


class ArtifactError(CliError):
    """A pipeline artifact could not be read, or does not hold what a stage needs.

    The pipeline is one command per stage with files in between, so a stage's
    input is a file some earlier command wrote — or a file a user wrote by
    hand, or the output of the wrong stage. All three are diagnostics. A
    document that parses but holds no chunks reports here too, and that case is
    the reason this error exists rather than a check inside each command:
    `idc.cli.commands.compacted_block` names the last chunk as the one kept
    verbatim, so an empty tuple reaches `chunks[-1]` and raises `IndexError` —
    which is not in `idc.cli.main.DIAGNOSTIC_ERRORS` and would reach the
    terminal as a traceback. "run `idc chunk` on it first" is the repair.

    Attributes:
        source: The file the artifact came from, or `None` when it was read
            from text with no source named.
    """

    def __init__(self, reason: str, *, source: str | None = None) -> None:
        """Build the error from what is wrong and where it came from.

        Args:
            reason: What is wrong with the artifact.
            source: The file it came from, named in the message when given.
        """
        self.source = source
        origin = f" {source}" if source else ""
        super().__init__(f"invalid pipeline artifact{origin}: {reason}")


class BackendUnavailableError(CliError):
    """A command needs a model backend and none was supplied.

    Not a bug and not a missing feature: this build ships no `ModelClient` for
    `idc.classify.LlmAdapter`, no `ContinuationScorer` for `S1MAdapter` and
    nothing that implements `idc.compact.NarrativeClient`. Every one of them is an injected
    dependency by design — determinism forbids a model call inside the test
    suite — so the backend reaches a command through `idc.cli.Runtime` or not at
    all. This error is the "or not at all" branch, and it names the seam to
    fill rather than failing somewhere inside the stage.

    Attributes:
        what: The dependency that is missing, in the words of the stage that
            wanted it.
    """

    def __init__(self, what: str, detail: str) -> None:
        """Build the error from the missing dependency and how to supply it.

        Args:
            what: The dependency that is missing.
            detail: What to pass, and where.
        """
        self.what = what
        super().__init__(f"{what} is not available: {detail}")
