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
"""

from pathlib import Path

from idc.schema import Session, load

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
