"""Access layer for the agent-native transcript fixtures in `tests/native/`.

The counterpart of `fixture_corpus` for the *input* side of the importers: those
fixtures are canonical `Session` documents, these are transcripts in an agent's
own on-disk format. An importer test resolves one **by agent and name** instead
of building a path, so adding or moving a fixture is a change in one place:

```python
from native_transcripts import NATIVE

path = NATIVE.path("claude-code", "golden_session")
```

Discovery happens on every call, so a fixture a later importer ticket adds
(`tests/native/opencode/…`, `tests/native/gemini-cli/…`) is picked up without
touching this module. See `tests/native/README.md` for what each fixture holds.
"""

from pathlib import Path

from idc.schema import Agent

TESTS_DIR = Path(__file__).parent.resolve()
"""Absolute path of the `tests/` directory."""

NATIVE_DIR = TESTS_DIR / "native"
"""Absolute path of the agent-native fixture directory."""

EXPECTED_SUFFIX = ".expected.json"
"""Suffix of the canonical golden file that may accompany a transcript."""


class NativeTranscripts:
    """A directory of agent-native transcript fixtures, addressed by name.

    Attributes:
        directory: The directory the per-agent sub-directories live in.
    """

    def __init__(self, directory: Path) -> None:
        """Bind to a directory without reading it yet.

        Args:
            directory: Directory holding one sub-directory per agent, named
                after the agent's wire string.
        """
        self.directory = Path(directory).resolve()

    def __repr__(self) -> str:
        """Return a representation naming the fixture directory."""
        return f"{type(self).__name__}({str(self.directory)!r})"

    def agent_directory(self, agent: Agent | str) -> Path:
        """Return the fixture directory of one agent.

        Args:
            agent: The agent, as an `Agent` or its wire string.

        Returns:
            The directory `tests/native/<agent>`, which need not exist yet: the
            two importers after T2.2 bring their own.
        """
        return self.directory / Agent(agent).value

    def paths(self, agent: Agent | str, suffix: str = ".jsonl") -> list[Path]:
        """Return every transcript fixture of one agent, sorted by name."""
        return sorted(
            path
            for path in self.agent_directory(agent).glob(f"*{suffix}")
            if not path.name.endswith(EXPECTED_SUFFIX)
        )

    def names(self, agent: Agent | str, suffix: str = ".jsonl") -> list[str]:
        """Return the fixture names of one agent — the file name without suffix."""
        return [path.name.removesuffix(suffix) for path in self.paths(agent, suffix)]

    def path(self, agent: Agent | str, name: str, suffix: str = ".jsonl") -> Path:
        """Return the path of one transcript fixture.

        Args:
            agent: The agent the fixture belongs to.
            name: Fixture name, e.g. `golden_session`.
            suffix: File suffix of the agent's transcript format.

        Returns:
            The absolute path of `tests/native/<agent>/<name><suffix>`.

        Raises:
            KeyError: If no such fixture exists; the message lists the ones that
                do.
        """
        path = self.agent_directory(agent) / f"{name}{suffix}"
        if not path.is_file():
            known = ", ".join(self.names(agent, suffix)) or "none"
            raise KeyError(
                f"no {Agent(agent).value} fixture {name!r} in {self.agent_directory(agent)}; "
                f"known fixtures: {known}"
            )
        return path

    def expected_path(self, agent: Agent | str, name: str) -> Path:
        """Return the path of a fixture's canonical golden file.

        Args:
            agent: The agent the fixture belongs to.
            name: Fixture name, e.g. `golden_session`.

        Returns:
            The absolute path of `<name>.expected.json`.

        Raises:
            KeyError: If the fixture carries no golden file.
        """
        path = self.agent_directory(agent) / f"{name}{EXPECTED_SUFFIX}"
        if not path.is_file():
            raise KeyError(f"no golden file for {Agent(agent).value} fixture {name!r}: {path}")
        return path

    def expected_text(self, agent: Agent | str, name: str) -> str:
        """Return a fixture's canonical golden file verbatim, as UTF-8 text."""
        return self.expected_path(agent, name).read_text(encoding="utf-8")


NATIVE = NativeTranscripts(NATIVE_DIR)
"""The shared agent-native fixture set; also available as the `native` pytest fixture."""
