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

One function reads a session back rather than a file, and it is what makes the
three agents comparable at all — `idc.importers.provenance`:

```python
from idc.importers import visible_context

context = visible_context(imported)
context.message_ids          # the messages the agent is still being sent
context.hidden               # what its own compaction replaced, or the harness wrote
```

Four things are shared, so that an agent module is only its own format:

- `idc.importers.jsonl` reads a line-delimited transcript and reports a damaged
  line by its line number.
- `idc.importers.base` assembles and validates the canonical `Session`, resolves
  tool categories, logs the unknown tools, reads timestamps and records which
  message the agent's own compaction summary is — in the session itself, so the
  fact survives a dump to a file. Everything the importers raise themselves is a
  `TranscriptImportError` naming the file.
- `idc.importers.provenance` says which of the imported messages are the context
  the agent would see, by the same rule for all three agents.
- `idc.importers.tool_registry` holds the agents' tool knowledge as a data file,
  not as code:

```python
from idc.importers import tool_registry

tool_registry().category("claude-code", "Edit")     # ToolCategory.WRITE
tool_registry().is_verification("uv run pytest -q") # True
```

See `idc.importers.tool_registry` for the lookup rules and for how a command
line is matched against the verification patterns. Of the three agents, Claude
Code (T2.2), OpenCode (T2.3) and Gemini CLI (T2.4) are all implemented, and
`import_transcript` says so rather than failing obscurely.
"""

from collections.abc import Callable
from pathlib import Path

from idc.importers.base import (
    NO_TYPE,
    NOTE_DROPPED_BLOCK,
    NOTE_SKIPPED_RECORD,
    NOTE_UNKNOWN_TOOL,
    NULL_TYPE,
    ImportedSession,
    ImportNote,
    SessionBuilder,
    TranscriptImportError,
    block_message_id,
    joined_text,
    note_unreadable_content,
    parse_timestamp,
    shape,
    type_name,
)
from idc.importers.claude_code import import_claude_code
from idc.importers.gemini_cli import import_gemini_cli
from idc.importers.jsonl import JsonLine, read_json_lines
from idc.importers.opencode import import_opencode
from idc.importers.provenance import (
    HIDDEN_HARNESS_NOTICE,
    HIDDEN_REPLACED_BY_COMPACTION,
    NOTICE_MARKERS,
    HiddenMessage,
    VisibleContext,
    visible_context,
)
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
    "HIDDEN_HARNESS_NOTICE",
    "HIDDEN_REPLACED_BY_COMPACTION",
    "IMPORTERS",
    "NOTE_DROPPED_BLOCK",
    "NOTE_SKIPPED_RECORD",
    "NOTE_UNKNOWN_TOOL",
    "NOTICE_MARKERS",
    "NO_TYPE",
    "NULL_TYPE",
    "PENDING_IMPORTERS",
    "REGISTRY_PATH",
    "REGISTRY_VERSION",
    "HiddenMessage",
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
    "VisibleContext",
    "block_message_id",
    "import_claude_code",
    "import_gemini_cli",
    "import_opencode",
    "import_transcript",
    "joined_text",
    "load_tool_registry",
    "note_unreadable_content",
    "parse_timestamp",
    "read_json_lines",
    "shape",
    "tool_registry",
    "type_name",
    "visible_context",
]

Importer = Callable[..., ImportedSession]
"""Signature of an importer: `(path, *, registry=None) -> ImportedSession`."""

IMPORTERS: dict[Agent, Importer] = {
    Agent.CLAUDE_CODE: import_claude_code,
    Agent.GEMINI_CLI: import_gemini_cli,
    Agent.OPENCODE: import_opencode,
}
"""The implemented importers, by agent — what `import_transcript` dispatches on.

An importer is registered here and nowhere else, so adding one is a line in this
mapping plus its own module.
"""

PENDING_IMPORTERS: dict[Agent, str] = {
    # Empty: every agent has an importer. The dispatcher keeps the path, and
    # tests/test_claude_code_importer.py exercises it against a patched table,
    # because an empty parametrization would skip it silently.
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
