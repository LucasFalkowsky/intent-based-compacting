"""Sample canonical objects used by the schema tests.

The samples are deliberately *maximal*: every optional field is populated, every
enum member that a type can carry appears somewhere, and the strings contain
non-ASCII characters and newlines. A round trip that preserves these samples
preserves anything a real transcript can hold.

One pair of `Session` fields cannot be maximal in the same object:
`native_summary_id` is refused when `source` is `raw`, which the maximal session
is, so it is split in two. `session()` is the raw one and stays the sample every
type-generic test uses; `native_compacted_session()` is the other half of the
pair and is named explicitly by the tests that need it.

This module is test data only — the shared transcript fixture corpus on disk is
ticket T1.3.
"""

from datetime import UTC, datetime, timedelta, timezone

from idc.schema import (
    Agent,
    Chain,
    Chunk,
    ChunkOrigin,
    CompactedBlock,
    Message,
    Role,
    Session,
    SessionSource,
    ToolCall,
    ToolCategory,
    ToolResult,
)

CREATED_AT = datetime(2026, 10, 20, 9, 30, 15, 123456, tzinfo=UTC)
"""Fixed, timezone-aware instant with microseconds; no test reads the clock."""

OFFSET_AT = datetime(2026, 10, 20, 11, 30, 15, 123456, tzinfo=timezone(timedelta(hours=2)))
"""The same instant as `CREATED_AT`, written with a non-UTC offset."""


def tool_call() -> ToolCall:
    """Return a tool call with nested, mixed-type arguments."""
    return ToolCall(
        id="call-1",
        name="Edit",
        category=ToolCategory.WRITE,
        args={
            "file_path": "src/idc/schema/models.py",
            "replace_all": False,
            "limit": 42,
            "ratio": 0.5,
            "nested": {"list": [1, "two", None, True]},
        },
    )


def tool_result() -> ToolResult:
    """Return a failing tool result with an exit code."""
    return ToolResult(
        call_id="call-1",
        output="ruff: 1 error\n  E501 line too long — Ümlaut\n",
        exit_code=1,
        is_error=True,
    )


def message() -> Message:
    """Return an assistant message that carries a tool call."""
    return Message(
        id="m-2",
        role=Role.ASSISTANT,
        content="I will edit the schema module.\n\nFirst the models — then the tests.",
        timestamp=CREATED_AT,
        tool_calls=[tool_call()],
    )


def session() -> Session:
    """Return a session that exercises every role and both tool containers."""
    return Session(
        id="s-1",
        agent=Agent.CLAUDE_CODE,
        model="claude-opus-5",
        source=SessionSource.RAW,
        created_at=CREATED_AT,
        messages=[
            Message(
                id="m-0",
                role=Role.SYSTEM,
                content="<system-reminder>Keep it short.</system-reminder>",
                timestamp=CREATED_AT,
                is_system_injected=True,
            ),
            Message(
                id="m-1",
                role=Role.USER,
                content="/compact bitte die Schemadatei",
                timestamp=CREATED_AT,
                is_command=True,
            ),
            message(),
            Message(
                id="m-3",
                role=Role.TOOL,
                content="",
                timestamp=CREATED_AT,
                tool_result=tool_result(),
            ),
            Message(
                id="m-4",
                role=Role.ASSISTANT,
                content="The edit failed; retrying with a narrower match.",
                timestamp=None,
                tool_calls=[
                    ToolCall(id="call-2", name="Read", category=ToolCategory.READ, args={}),
                    ToolCall(id="call-3", name="osascript", category=ToolCategory.OTHER),
                ],
            ),
        ],
    )


def native_compacted_session() -> Session:
    """Return a session the agent compacted itself, naming its own summary."""
    return Session(
        id="s-native",
        agent=Agent.OPENCODE,
        model="claude-opus-5",
        source=SessionSource.NATIVE_COMPACTION,
        native_summary_id="c-0",
        created_at=CREATED_AT,
        messages=[
            Message(
                id="c-0",
                role=Role.SYSTEM,
                content="Zusammenfassung: der Exporter schreibt jetzt UTC.",
                timestamp=CREATED_AT,
                is_system_injected=True,
            ),
            Message(
                id="m-1",
                role=Role.USER,
                content="Now add the CLI flag.",
                timestamp=CREATED_AT,
            ),
        ],
    )


def chunk() -> Chunk:
    """Return the latest chunk of a transcript, originating from a subagent."""
    return Chunk(
        id="K3",
        start_index=2,
        end_index=5,
        is_latest=True,
        origin=ChunkOrigin.SUBAGENT,
    )


def chain() -> Chain:
    """Return an already compacted chain with two chunks."""
    return Chain(
        id="C2",
        label="Canonical schema for the pipeline",
        chunk_ids=["K1", "K3"],
        compacted=True,
        compacted_text="## [C2] Intent: Canonical schema\n### Solution\n- models.py\n",
    )


def compacted_block() -> CompactedBlock:
    """Return a compacted block holding one compacted and one open chain."""
    return CompactedBlock(
        chains=[
            chain(),
            Chain(id="C1", label="Repository scaffold", chunk_ids=["K0"]),
        ],
        latest_chunk_id="K3",
        rendered_for=Agent.OPENCODE,
    )


SAMPLES = {
    "ToolCall": tool_call,
    "ToolResult": tool_result,
    "Message": message,
    "Session": session,
    "Chunk": chunk,
    "Chain": chain,
    "CompactedBlock": compacted_block,
}
"""One maximal sample factory per canonical type, keyed by the type name."""
