"""Importers that read agent transcripts into the canonical format.

One importer per supported agent (Claude Code, Gemini CLI, OpenCode). An
importer is responsible for the agent's on-disk shape — JSON or JSONL, tool-call
encoding, role names — and for nothing else: everything downstream sees only
canonical objects.

```python
from idc.importers import import_transcript

imported = import_transcript("claude-code", "session.jsonl")
imported.session.messages[0].role    # Role.USER
imported.unknown_tools               # tools the registry does not declare
for note in imported.notes:
    print(note)                      # what the import could not represent
```

Three things are shared, so that an agent module is only its own format:

- `idc.importers.jsonl` reads a line-delimited transcript and reports a damaged
  line by its line number.
- `idc.importers.base` assembles and validates the canonical `Session`, resolves
  tool categories, logs the unknown tools and reads timestamps. Everything the
  importers raise themselves is a `TranscriptImportError` naming the file.
- `idc.importers.tool_registry` holds the agents' tool knowledge as a data file,
  not as code:

```python
from idc.importers import tool_registry

tool_registry().category("claude-code", "Edit")     # ToolCategory.WRITE
tool_registry().is_verification("uv run pytest -q") # True
```

See `idc.importers.tool_registry` for the lookup rules and for how a command
line is matched against the verification patterns. Of the three agents, only
Claude Code is implemented (T2.2); OpenCode (T2.3) and Gemini CLI (T2.4) follow,
and `import_transcript` says so rather than failing obscurely.
"""

from collections.abc import Callable
from pathlib import Path

from idc.importers.base import (
    NOTE_DROPPED_BLOCK,
    NOTE_SKIPPED_RECORD,
    NOTE_UNKNOWN_TOOL,
    ImportedSession,
    ImportNote,
    SessionBuilder,
    TranscriptImportError,
    block_message_id,
    joined_text,
    parse_timestamp,
)
from idc.importers.claude_code import import_claude_code
from idc.importers.jsonl import JsonLine, read_json_lines
from idc.importers.tool_registry import (
    REGISTRY_PATH,
    REGISTRY_VERSION,
    ToolRegistry,
    ToolRegistryError,
    ToolSpec,
    VerificationMatch,
    VerificationPattern,
    load_tool_registry,
    tool_registry,
)
from idc.schema import Agent

__all__ = [
    "IMPORTERS",
    "NOTE_DROPPED_BLOCK",
    "NOTE_SKIPPED_RECORD",
    "NOTE_UNKNOWN_TOOL",
    "PENDING_IMPORTERS",
    "REGISTRY_PATH",
    "REGISTRY_VERSION",
    "ImportNote",
    "ImportedSession",
    "JsonLine",
    "SessionBuilder",
    "ToolRegistry",
    "ToolRegistryError",
    "ToolSpec",
    "TranscriptImportError",
    "VerificationMatch",
    "VerificationPattern",
    "block_message_id",
    "import_claude_code",
    "import_transcript",
    "joined_text",
    "load_tool_registry",
    "parse_timestamp",
    "read_json_lines",
    "tool_registry",
]

Importer = Callable[..., ImportedSession]
"""Signature of an importer: `(path, *, registry=None) -> ImportedSession`."""

IMPORTERS: dict[Agent, Importer] = {
    Agent.CLAUDE_CODE: import_claude_code,
}
"""The implemented importers, by agent — what `import_transcript` dispatches on.

An importer is registered here and nowhere else, so adding one is a line in this
mapping plus its own module.
"""

PENDING_IMPORTERS: dict[Agent, str] = {
    Agent.OPENCODE: "T2.3",
    Agent.GEMINI_CLI: "T2.4",
}
"""The agents whose importer is still a ticket, so the error can name it."""


def import_transcript(
    agent: Agent | str, path: str | Path, *, registry: ToolRegistry | None = None
) -> ImportedSession:
    """Read one agent's transcript into a canonical session.

    Args:
        agent: The agent the file comes from, as an `Agent` or its wire string.
        path: The transcript file.
        registry: Tool knowledge to resolve the call categories with; the
            packaged registry when omitted.

    Returns:
        The canonical session together with the import's notes.

    Raises:
        TranscriptImportError: If the agent is not one of the study's, if its
            importer is not implemented yet (the message names the ticket), or
            if the transcript cannot be read.
        SchemaValidationError: If the transcript yields messages that do not
            satisfy the canonical schema.
    """
    try:
        known = Agent(agent)
    except ValueError:
        agents = ", ".join(member.value for member in Agent)
        raise TranscriptImportError(f"unknown agent {agent!r}; known agents: {agents}") from None
    importer = IMPORTERS.get(known)
    if importer is None:
        raise TranscriptImportError(
            f"no importer for {known.value} yet; it is ticket {PENDING_IMPORTERS.get(known, 'T2')}"
        )
    return importer(path, registry=registry)
