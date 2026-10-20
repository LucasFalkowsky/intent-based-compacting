"""Access layer for the shared transcript fixture corpus in `tests/fixtures/`.

Every test that needs a transcript resolves it **by name** through this module
instead of building a path, so that moving or adding a fixture is a change in
one place:

```python
from fixture_corpus import CORPUS

session = CORPUS.session("interleaved_intents")   # validated on load
```

The corpus is discovered from disk on every call — a new `<name>.json` in
`tests/fixtures/` is picked up without touching this module or any test — and
the fixture name is the file name without its suffix. Anything that is not a
`*.json` file (the corpus `README.md`) is ignored.

Fixtures are loaded through `idc.schema.load`, so a malformed fixture fails as
one `SchemaValidationError` naming the file, and no test ever sees an
unvalidated transcript. The module has no dependency beyond the standard
library and the canonical schema itself.

The readers at the end of the module (`tool_calls`, `file_of`, `command_of`,
`touched_files`, `written_files`, `results_by_call_id`, `exec_runs`) are
mechanical: they only regroup the canonical fields a fixture already carries,
which is what a test needs to state what it expects. They are **not** the
pipeline's notions of a turn, a write or a verification — the chunker (T3) and
the resolver (T5) derive those themselves, and a test must not use these helpers
as their oracle.
"""

from collections.abc import Iterable
from pathlib import Path

from idc.schema import Message, Session, ToolCall, ToolCategory, ToolResult, load

TESTS_DIR = Path(__file__).parent.resolve()
"""Absolute path of the `tests/` directory."""

FIXTURES_DIR = TESTS_DIR / "fixtures"
"""Absolute path of the shared fixture corpus."""

SUFFIX = ".json"
"""Every fixture is one canonical `Session` document in a `*.json` file."""


class Corpus:
    """A directory of canonical transcript fixtures, addressed by name.

    Attributes:
        directory: The directory the fixtures are read from.
    """

    def __init__(self, directory: Path) -> None:
        """Bind the corpus to a directory without reading it yet.

        Args:
            directory: Directory holding the `*.json` fixtures. Tests use the
                shared `CORPUS`; a different directory is useful to exercise
                the discovery itself.
        """
        self.directory = Path(directory).resolve()

    def __repr__(self) -> str:
        """Return a representation naming the corpus directory."""
        return f"{type(self).__name__}({str(self.directory)!r})"

    def __contains__(self, name: object) -> bool:
        """Return whether a fixture of that name exists."""
        return isinstance(name, str) and name in self.names()

    def __len__(self) -> int:
        """Return how many fixtures the corpus holds."""
        return len(self.paths())

    def paths(self) -> list[Path]:
        """Return the path of every fixture, sorted by name.

        Discovery happens here, on every call, so a fixture added during a test
        run (or by a later ticket) needs no change to this module.
        """
        return sorted(self.directory.glob(f"*{SUFFIX}"))

    def names(self) -> list[str]:
        """Return every fixture name — the file name without its suffix."""
        return [path.stem for path in self.paths()]

    def path(self, name: str) -> Path:
        """Return the path of one fixture.

        Args:
            name: Fixture name, e.g. `interleaved_intents`.

        Returns:
            The absolute path of `<name>.json`.

        Raises:
            KeyError: If the corpus holds no fixture of that name; the message
                lists the names it does hold.
        """
        path = self.directory / f"{name}{SUFFIX}"
        if not path.is_file():
            known = ", ".join(self.names()) or "none"
            raise KeyError(f"no fixture {name!r} in {self.directory}; known fixtures: {known}")
        return path

    def text(self, name: str) -> str:
        """Return one fixture's file content verbatim, as UTF-8 text.

        Args:
            name: Fixture name.

        Returns:
            The unparsed JSON document, for tests that assert on the bytes.
        """
        return self.path(name).read_text(encoding="utf-8")

    def session(self, name: str) -> Session:
        """Return one fixture as a validated canonical session.

        Args:
            name: Fixture name.

        Returns:
            The `Session` the fixture holds.

        Raises:
            KeyError: If no fixture of that name exists.
            SchemaValidationError: If the fixture does not satisfy the schema.
        """
        return load(Session, self.path(name))

    def sessions(self) -> dict[str, Session]:
        """Return the whole corpus as validated sessions, keyed by name."""
        return {path.stem: load(Session, path) for path in self.paths()}


CORPUS = Corpus(FIXTURES_DIR)
"""The shared corpus; also available as the `corpus` pytest fixture."""


FILE_PATH_KEYS = ("file_path", "filePath", "absolute_path")
"""Argument keys the three agents use for the file a call works on.

Claude Code and Gemini CLI's `write_file`/`replace` use `file_path`, OpenCode
`filePath`, Gemini's `read_file` `absolute_path`. The canonical schema keeps a
call's arguments exactly as the agent sent them, so the corpus carries all three
spellings on purpose.

Every spelling listed here occurs in the corpus, and
`tests/test_fixture_access.py` keeps it that way: a key no agent sends would pin
behaviour that was never observed.
"""

COMMAND_KEYS = ("command",)
"""Argument keys an `exec` call may use for the command it runs.

All three agents send `command` — Claude Code's `Bash`, OpenCode's `bash` and
Gemini CLI's `run_shell_command` — and none of them sends anything else, so this
is a one-element tuple and stays one until a transcript shows otherwise. It is
written as a tuple because the lookup tolerates several spellings, the way
`FILE_PATH_KEYS` has to; it is not a place for prospective ones.
"""


def tool_calls(messages: Iterable[Message]) -> list[ToolCall]:
    """Return every tool call of `messages`, in transcript order."""
    return [call for message in messages for call in message.tool_calls]


def file_of(call: ToolCall) -> str | None:
    """Return the file a call works on, or `None` if its arguments name none."""
    for key in FILE_PATH_KEYS:
        value = call.args.get(key)
        if isinstance(value, str):
            return value
    return None


def command_of(call: ToolCall) -> str | None:
    """Return the command a call runs, or `None` if its arguments name none."""
    for key in COMMAND_KEYS:
        value = call.args.get(key)
        if isinstance(value, str):
            return value
    return None


def touched_files(messages: Iterable[Message]) -> set[str]:
    """Return every file the calls in `messages` read or wrote."""
    return {path for call in tool_calls(messages) if (path := file_of(call)) is not None}


def written_files(messages: Iterable[Message]) -> list[str]:
    """Return the file of every `write` call, in transcript order, with repeats."""
    return [
        path
        for call in tool_calls(messages)
        if call.category is ToolCategory.WRITE and (path := file_of(call)) is not None
    ]


def results_by_call_id(session: Session) -> dict[str, ToolResult]:
    """Return every tool result of the session, keyed by the call it answers."""
    return {
        message.tool_result.call_id: message.tool_result
        for message in session.messages
        if message.tool_result is not None
    }


def exec_runs(session: Session) -> list[tuple[str, ToolResult]]:
    """Return `(command, result)` for every `exec` call, in transcript order.

    Args:
        session: The session to read the `exec` calls of.

    Returns:
        One `(command, result)` pair per `exec` call, in transcript order.

    Raises:
        ValueError: If an `exec` call names its command under none of
            `COMMAND_KEYS` — a corpus problem, reported with the call it comes
            from instead of as a bare `KeyError` from a dict lookup.
    """
    results = results_by_call_id(session)
    runs: list[tuple[str, ToolResult]] = []
    for call in tool_calls(session.messages):
        if call.category is not ToolCategory.EXEC:
            continue
        command = command_of(call)
        if command is None:
            spellings = ", ".join(COMMAND_KEYS)
            plural = "s" if len(COMMAND_KEYS) > 1 else ""
            raise ValueError(
                f"exec call {call.id!r} ({call.name}) of session {session.id!r} names no "
                f"command; tried the argument key{plural} {spellings}"
            )
        runs.append((command, results[call.id]))
    return runs
