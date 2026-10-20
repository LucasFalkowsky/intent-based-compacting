"""Per-agent tool knowledge, read from a data file rather than written in code.

Two questions are answered here, and both are answered from `tool_registry.toml`
next to this module:

- **What does a tool call do to the workspace?** `category(agent, name)` maps an
  agent's own tool name to a `ToolCategory`; a tool the data file does not
  declare resolves to `other` and `knows()` reports it as unknown, which is what
  lets an importer log it (T2's parent criterion).
- **Is an `exec` command a verification run?** `match_verification(command)`
  matches the command line against the data file's verification patterns, so the
  resolver (T5.2) can keep the last passing verification of a chain.

Adding a tool or a pattern is therefore a change to the data file alone. The
file is validated when it is read: an unknown agent, an unknown category, an
unknown field or a missing section raises `ToolRegistryError` naming the file and
the offending path, instead of letting everything resolve to `other` quietly.

```python
from idc.importers import tool_registry

registry = tool_registry()                       # the packaged data file
registry.category("claude-code", "Edit")         # ToolCategory.WRITE
registry.is_verification("uv run pytest -q")     # True
```

**Matching is anchored.** A command line is split into segments at `&&`, `||`,
`;`, `|` and newlines (quote-aware, and stopping at a here-document body), each
segment is tokenized the way a shell would, leading `NAME=value` assignments and
the data file's `runner_prefixes` are dropped, and the command word is reduced to
its file name. A pattern then
has to match the *beginning* of what is left. That is deliberate: a false
positive would make the resolver present an unrelated command as the chain's
proof that its solution works, so `git commit -m "add pytest"` and
`echo "go test"` must not match, while a runner this file has never heard of is
merely missed.
"""

import re
import shlex
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path, PurePosixPath
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from idc.schema import Agent, ToolCategory

__all__ = [
    "REGISTRY_PATH",
    "REGISTRY_VERSION",
    "ToolRegistry",
    "ToolRegistryError",
    "ToolSpec",
    "VerificationMatch",
    "VerificationPattern",
    "load_tool_registry",
    "tool_registry",
]

REGISTRY_PATH = Path(__file__).with_name("tool_registry.toml")
"""The packaged registry data file — the default source of the registry."""

REGISTRY_VERSION = 1
"""The only `version` this build of the tool reads."""

_SEGMENT_CHARS = frozenset("&|;\n")
"""Characters that end a command segment when they are not inside quotes."""

_ENV_ASSIGNMENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
"""A leading `NAME=value` token, which precedes the command rather than being it."""


class ToolRegistryError(ValueError):
    """The tool registry could not be read, or was asked about an unknown agent.

    The one error type this module raises: the data file is part of the package,
    so a problem with it is a diagnostic for the CLI user, not a traceback.
    Pydantic's `ValidationError` is translated here exactly as the canonical
    schema translates it.
    """


def _as_registry_error(exc: ValidationError, source: Path | None) -> ToolRegistryError:
    """Translate a pydantic error into one located `ToolRegistryError`."""
    problems = tuple(
        (
            ".".join(str(part) for part in error["loc"]) or "<document>",
            error["msg"].removeprefix("Value error, "),
        )
        for error in exc.errors()
    )
    origin = f" {source}" if source is not None else ""
    count = f"{len(problems)} problem{'s' if len(problems) != 1 else ''}"
    lines = [f"invalid tool registry{origin}: {count}"]
    lines += [f"  {location}: {message}" for location, message in problems]
    return ToolRegistryError("\n".join(lines))


def _as_agent(value: Agent | str) -> Agent:
    """Return `value` as an `Agent`, or raise `ToolRegistryError`.

    Args:
        value: An `Agent` or its wire string, e.g. `claude-code`.

    Returns:
        The matching `Agent` member.

    Raises:
        ToolRegistryError: If the value names no agent of the study; the message
            lists the ones it could name.
    """
    try:
        return Agent(value)
    except ValueError:
        known = ", ".join(agent.value for agent in Agent)
        raise ToolRegistryError(f"unknown agent {value!r}; known agents: {known}") from None


def _segments(command: str) -> list[str]:
    """Split a command line into the commands it runs, quote-aware.

    Args:
        command: A command line as the agent recorded it.

    Returns:
        One entry per segment, in order, stripped and without the empty ones.
        `&&`, `||`, `;`, `|` and newlines separate segments; the same characters
        inside single or double quotes do not.

        Scanning stops at the newline after a here-document introducer (`<<`):
        what follows is the document's *body*, which the command writes or feeds
        to another program rather than runs. Treating it as commands made
        `python - <<EOF` with a `make test` line in its body look like a test
        run. Commands after the body's delimiter are given up with it, which
        only ever loses a match.
    """
    segments: list[str] = []
    current: list[str] = []
    quote: str | None = None
    heredoc = False
    index = 0
    while index < len(command):
        char = command[index]
        index += 1
        if quote is not None:
            current.append(char)
            if char == quote:
                quote = None
        elif char in ('"', "'"):
            quote = char
            current.append(char)
        elif char == "\\" and index < len(command):
            current.append(char)
            current.append(command[index])
            index += 1
        elif char == "<" and index < len(command) and command[index] == "<":
            heredoc = True
            current.append(char)
        elif char in _SEGMENT_CHARS:
            if char == "\n" and heredoc:
                break
            segments.append("".join(current))
            current = []
            while index < len(command) and command[index] in _SEGMENT_CHARS:
                index += 1
        else:
            current.append(char)
    segments.append("".join(current))
    return [segment.strip() for segment in segments if segment.strip()]


def _tokens(segment: str) -> tuple[str, ...]:
    """Split one command segment into tokens the way a POSIX shell would.

    Quoted text stays one token, which is what keeps `echo "go test"` from
    looking like a `go test` run. A segment whose quoting is unbalanced cannot be
    tokenized that way; it falls back to splitting on whitespace, which is safe
    because matching is anchored on the command word either way.
    """
    try:
        return tuple(shlex.split(segment, comments=False, posix=True))
    except ValueError:
        return tuple(segment.split())


class _RegistryModel(BaseModel):
    """Base of the registry's data types: no unknown fields, no mutation."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class ToolSpec(_RegistryModel):
    """What one tool of one agent does, as the data file declares it.

    Attributes:
        category: What a call to the tool does to the workspace.
        subagent: Whether the tool delegates work to a subagent, so that T3.3
            can lift that work into a chunk of its own. Orthogonal to the
            category: a delegating call itself neither reads nor writes.
    """

    category: ToolCategory = Field(description="What a call to this tool does to the workspace.")
    subagent: bool = Field(default=False, description="Whether the tool spawns a subagent.")


class VerificationPattern(_RegistryModel):
    """A command line that counts as a verification run when it is the command.

    Attributes:
        id: Stable identifier of the pattern, e.g. `make-test`.
        command: The pattern itself, written as the command line it stands for
            (`make test`); it is tokenized and normalized on load, and matched
            against the beginning of a command segment.
    """

    id: str = Field(min_length=1, description="Stable identifier of the pattern.")
    command: str = Field(min_length=1, description="The command line the pattern stands for.")


class _AgentTools(_RegistryModel):
    """The tool table of one agent."""

    tools: dict[str, ToolSpec] = Field(min_length=1, description="Tool name -> specification.")

    @model_validator(mode="after")
    def _reject_blank_names(self) -> Self:
        """Reject an empty tool name, which no agent can ever report."""
        if any(not name.strip() for name in self.tools):
            raise ValueError("a tool name must not be empty")
        return self


class _RegistryFile(_RegistryModel):
    """The data file's shape: version, prefixes, per-agent tools, patterns."""

    version: int = Field(description="Format version of the data file.")
    runner_prefixes: tuple[str, ...] = Field(
        default=(), description="Wrapper commands stripped before a pattern is matched."
    )
    agents: dict[Agent, _AgentTools] = Field(description="One tool table per agent of the study.")
    verification_patterns: tuple[VerificationPattern, ...] = Field(
        min_length=1, description="The command lines that count as a verification run."
    )

    @model_validator(mode="after")
    def _check_runner_prefixes(self) -> Self:
        """Reject a prefix that is no command word.

        A prefix that tokenizes to nothing would be stripped from every command
        line forever, so this is the one malformed entry that would hang the
        matcher rather than mis-match.
        """
        for prefix in self.runner_prefixes:
            if not _tokens(prefix):
                raise ValueError(f"runner prefix {prefix!r} names no command word")
        return self

    @model_validator(mode="after")
    def _check_completeness(self) -> Self:
        """Reject an unsupported version, a missing agent or a duplicate pattern id."""
        if self.version != REGISTRY_VERSION:
            raise ValueError(
                f"unsupported version {self.version}; this build reads version {REGISTRY_VERSION}"
            )
        missing = [agent.value for agent in Agent if agent not in self.agents]
        if missing:
            raise ValueError(f"no tools declared for {', '.join(missing)}")
        seen: set[str] = set()
        for pattern in self.verification_patterns:
            if pattern.id in seen:
                raise ValueError(f"duplicate verification pattern id {pattern.id!r}")
            seen.add(pattern.id)
        return self


@dataclass(frozen=True, slots=True)
class VerificationMatch:
    """A command line that a verification pattern matched.

    Attributes:
        pattern: The pattern that matched.
        command: The segment of the command line it matched, verbatim.
        tokens: That segment's tokens after the prefix stripping, so a caller
            can see what was actually compared.
    """

    pattern: VerificationPattern
    command: str
    tokens: tuple[str, ...]


class ToolRegistry:
    """The per-agent tool knowledge of one registry document.

    Built from a parsed data file; use `load_tool_registry` to read one from
    disk, or `tool_registry()` for the packaged one.
    """

    def __init__(self, document: Mapping[str, Any], *, source: Path | None = None) -> None:
        """Validate a parsed registry document and precompute its lookups.

        Args:
            document: The parsed data file, e.g. the result of `tomllib.load`.
            source: Where it came from, for error messages; `None` for a
                document built in memory.

        Raises:
            ToolRegistryError: If the document is not a mapping, or does not
                satisfy the registry's shape. The message names the source and
                one dotted path per problem.
        """
        self._source = Path(source) if source is not None else None
        if not isinstance(document, Mapping):
            raise ToolRegistryError(
                f"invalid tool registry: expected a mapping, got {type(document).__name__}"
            )
        try:
            data = _RegistryFile(**dict(document))
        except ValidationError as exc:
            raise _as_registry_error(exc, self._source) from None
        self._version = data.version
        self._tools = {agent: dict(table.tools) for agent, table in data.agents.items()}
        self._prefixes = tuple(
            tokens for prefix in data.runner_prefixes if (tokens := _tokens(prefix))
        )
        self._patterns = self._normalized_patterns(data.verification_patterns)

    def _normalized_patterns(
        self, patterns: tuple[VerificationPattern, ...]
    ) -> tuple[tuple[VerificationPattern, tuple[str, ...]], ...]:
        """Tokenize every pattern the way a command is tokenized.

        Args:
            patterns: The patterns as the data file declares them.

        Returns:
            One `(pattern, tokens)` pair per pattern, in file order.

        Raises:
            ToolRegistryError: If a pattern reduces to no command at all (it is
                only a runner prefix), or if two patterns reduce to the same
                tokens and would therefore match the same commands.
        """
        normalized: list[tuple[VerificationPattern, tuple[str, ...]]] = []
        seen: dict[tuple[str, ...], str] = {}
        for pattern in patterns:
            tokens = self._normalize(_tokens(pattern.command))
            if not tokens:
                raise ToolRegistryError(
                    f"invalid tool registry{self._origin}: verification pattern {pattern.id!r} "
                    f"({pattern.command!r}) names no command of its own"
                )
            if tokens in seen:
                raise ToolRegistryError(
                    f"invalid tool registry{self._origin}: verification pattern {pattern.id!r} "
                    f"matches the same commands as {seen[tokens]!r}"
                )
            seen[tokens] = pattern.id
            normalized.append((pattern, tokens))
        return tuple(normalized)

    @property
    def _origin(self) -> str:
        """Return ` <path>` when the registry came from a file, else the empty string."""
        return f" {self._source}" if self._source is not None else ""

    def _normalize(self, tokens: tuple[str, ...]) -> tuple[str, ...]:
        """Reduce a token sequence to the command it actually runs.

        Leading `NAME=value` assignments and the data file's runner prefixes are
        dropped (repeatedly, so `uv run poetry run pytest` reduces as well), and
        the command word is reduced to its file name, so that
        `./.venv/bin/pytest` and `pytest` are the same command. The arguments are
        left exactly as they were.
        """
        rest = list(tokens)
        changed = True
        while changed and rest:
            changed = False
            while rest and _ENV_ASSIGNMENT.match(rest[0]):
                del rest[0]
                changed = True
            for prefix in self._prefixes:
                if tuple(rest[: len(prefix)]) == prefix:
                    del rest[: len(prefix)]
                    changed = True
                    break
        if not rest:
            return ()
        return (PurePosixPath(rest[0]).name or rest[0], *rest[1:])

    def __repr__(self) -> str:
        """Return a representation naming the registry's source."""
        source = None if self._source is None else str(self._source)
        return f"{type(self).__name__}(source={source!r})"

    @property
    def source(self) -> Path | None:
        """The data file this registry was read from, or `None` if built in memory."""
        return self._source

    @property
    def version(self) -> int:
        """The data file's format version."""
        return self._version

    @property
    def agents(self) -> tuple[Agent, ...]:
        """Every agent the data file declares tools for, in schema order."""
        return tuple(agent for agent in Agent if agent in self._tools)

    def tool_names(self, agent: Agent | str) -> tuple[str, ...]:
        """Return the declared tool names of one agent, sorted.

        Args:
            agent: The agent, as an `Agent` or its wire string.

        Returns:
            The tool names the data file declares for it.

        Raises:
            ToolRegistryError: If the agent is not one of the study's.
        """
        return tuple(sorted(self._tools.get(_as_agent(agent), {})))

    def spec(self, agent: Agent | str, name: str) -> ToolSpec | None:
        """Return the declaration of one tool, or `None` if it is not declared.

        Args:
            agent: The agent that made the call.
            name: The tool name exactly as the agent reports it; lookup is
                case-sensitive, because `read` (OpenCode) and `Read` (Claude
                Code) are different tools of different agents.

        Returns:
            The `ToolSpec`, or `None` for a tool the registry does not know.

        Raises:
            ToolRegistryError: If the agent is not one of the study's.
        """
        return self._tools.get(_as_agent(agent), {}).get(name)

    def category(self, agent: Agent | str, name: str) -> ToolCategory:
        """Return what a call to that tool does to the workspace.

        Args:
            agent: The agent that made the call.
            name: The tool name as the agent reports it.

        Returns:
            The declared category, or `ToolCategory.OTHER` for a tool the data
            file does not declare — which is what the canonical schema's `other`
            is for. Use `knows` to tell the two apart, e.g. to log it.

        Raises:
            ToolRegistryError: If the agent is not one of the study's.
        """
        spec = self.spec(agent, name)
        return ToolCategory.OTHER if spec is None else spec.category

    def knows(self, agent: Agent | str, name: str) -> bool:
        """Return whether the data file declares that tool for that agent."""
        return self.spec(agent, name) is not None

    def is_subagent_tool(self, agent: Agent | str, name: str) -> bool:
        """Return whether that tool delegates its work to a subagent."""
        spec = self.spec(agent, name)
        return spec is not None and spec.subagent

    def subagent_tools(self, agent: Agent | str) -> tuple[str, ...]:
        """Return the agent's delegating tools, sorted — the input of T3.3."""
        return tuple(name for name in self.tool_names(agent) if self.is_subagent_tool(agent, name))

    @property
    def verification_patterns(self) -> tuple[VerificationPattern, ...]:
        """The verification patterns, in the order the data file lists them."""
        return tuple(pattern for pattern, _ in self._patterns)

    def match_verification(self, command: str) -> VerificationMatch | None:
        """Return the verification pattern a command line matches, if any.

        Args:
            command: The command line of an `exec` call, as the agent recorded
                it. Composed lines are handled: every segment is tried, in
                order, and the first match wins.

        Returns:
            The match, naming the pattern and the segment it matched, or `None`
            when the command line runs no verification. Matching is anchored on
            the command word, so a command that merely mentions a test runner in
            an argument never matches.
        """
        for segment in _segments(command):
            tokens = self._normalize(_tokens(segment))
            if not tokens:
                continue
            for pattern, expected in self._patterns:
                if tokens[: len(expected)] == expected:
                    return VerificationMatch(pattern=pattern, command=segment, tokens=tokens)
        return None

    def is_verification(self, command: str) -> bool:
        """Return whether a command line runs a verification."""
        return self.match_verification(command) is not None


def load_tool_registry(path: str | Path | None = None) -> ToolRegistry:
    """Read a registry data file and validate it.

    Args:
        path: The data file; the packaged `REGISTRY_PATH` when omitted.

    Returns:
        The validated registry.

    Raises:
        ToolRegistryError: If the file cannot be read, is not valid TOML, or
            does not satisfy the registry's shape. The message names the file.
    """
    source = REGISTRY_PATH if path is None else Path(path)
    try:
        text = source.read_bytes().decode("utf-8")
    except OSError as exc:
        raise ToolRegistryError(
            f"cannot read the tool registry {source}: {exc.strerror or exc}"
        ) from None
    except UnicodeDecodeError:
        raise ToolRegistryError(f"invalid tool registry {source}: not valid UTF-8") from None
    try:
        document = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise ToolRegistryError(f"invalid tool registry {source}: not valid TOML ({exc})") from None
    return ToolRegistry(document, source=source)


@lru_cache(maxsize=1)
def tool_registry() -> ToolRegistry:
    """Return the packaged registry, read once per process.

    Returns:
        The registry of `REGISTRY_PATH`. The result is immutable and shared, so
        every stage sees the same tool knowledge.

    Raises:
        ToolRegistryError: If the packaged data file is broken.
    """
    return load_tool_registry()
