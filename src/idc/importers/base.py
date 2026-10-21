"""What every importer needs, so that each agent module is only its own format.

An importer has two halves. The agent-specific half knows one on-disk shape —
which record carries a message, how a tool call is encoded, where the summary of
a native compaction sits. The other half is the same for all three agents and
lives here:

- **Assembling the canonical session.** `SessionBuilder` collects messages, asks
  the tool registry for every call's category, remembers the model and the
  provenance, and validates the result into a `Session` on `build()`.
- **Recording the agent's own summary.** `SessionBuilder.add_native_summary`
  appends it, sets the session's provenance and remembers which message it is,
  so that `idc.importers.provenance` can tell the context the agent still has
  from the history its compaction replaced.
- **Resolving a tool category and logging the unknown ones.** A tool the registry
  does not declare resolves to `other` *and* is reported — the parent criterion
  of T2 — both as an `ImportNote` on the result and as one log warning per
  distinct name.
- **Reading a timestamp.** The canonical schema requires timezone-aware
  instants; `parse_timestamp` attaches UTC when the agent records no offset.
- **Naming what could not be read.** `type_name` spells the `type` of a record or
  of one of its content blocks the way the notes do, `shape` names a value that
  has no kind at all by its JSON shape, and `note_unreadable_content` records a
  content that is neither text nor a block list. All three agents need exactly
  this and all three needed it identically.
- **One error type.** Everything the importers raise themselves is a
  `TranscriptImportError` naming the file, so the CLI never shows a traceback.
  The canonical schema's own `SchemaValidationError` is deliberately *not*
  wrapped: it already names the type and the offending field path.

What an importer deliberately does **not** do is repair a transcript. A
front-truncated file whose first `tool_result` has no `tool_use`, and a
`tool_use` whose result never arrived, both import as they are — the canonical
schema does not require referential completeness precisely so that they can
(see the decisions on #3), and the measurement needs to count them.

```python
builder = SessionBuilder(Agent.CLAUDE_CODE, source=path)
builder.add_message(message_id="u01", role=Role.USER, content="fix the exporter")
imported = builder.build(session_id="sess-1")
imported.session.messages[0].content      # 'fix the exporter'
imported.unknown_tools                    # ()
```
"""

import logging
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from idc.importers.tool_registry import ToolRegistry, tool_registry
from idc.schema import (
    Agent,
    Message,
    Role,
    Session,
    SessionSource,
    ToolCall,
    ToolCategory,
    ToolResult,
)

__all__ = [
    "NOTE_DROPPED_BLOCK",
    "NOTE_SKIPPED_RECORD",
    "NOTE_UNKNOWN_TOOL",
    "NO_TYPE",
    "NULL_TYPE",
    "ImportNote",
    "ImportedSession",
    "SessionBuilder",
    "TranscriptImportError",
    "block_message_id",
    "joined_text",
    "note_unreadable_content",
    "parse_timestamp",
    "shape",
    "type_name",
]

LOGGER = logging.getLogger("idc.importers")
"""Where the importers report what they could not interpret.

One logger for all three agents, so that a CLI command can show the whole
import's diagnostics by attaching a single handler.
"""

NOTE_UNKNOWN_TOOL = "unknown-tool"
"""A tool the registry does not declare; its calls resolve to `other`."""

NOTE_SKIPPED_RECORD = "skipped-record"
"""A transcript record that carries no canonical message, named by its kind."""

NOTE_DROPPED_BLOCK = "dropped-block"
"""A content block the canonical schema has no field for, named by its kind."""

NO_TYPE = "<no type>"
"""How a record, or one of its content blocks, that names no `type` is reported."""

NULL_TYPE = "<null type>"
"""How a record, or one of its content blocks, whose `type` is `null` is reported.

Distinct from `NO_TYPE`: a missing field is a shape the importer does not know,
a `null` one is damaged, and `str(None)` reads the same for both — and the same
as a kind genuinely spelled `"None"`.
"""


class TranscriptImportError(ValueError):
    """An agent transcript could not be read into a canonical session.

    The one error type the importers raise themselves: the file comes from
    outside the tool, so a problem with it is a diagnostic for the CLI user, not
    a traceback. Every message names the file, and a problem that belongs to one
    line names the line number as well.
    """


@dataclass(frozen=True, slots=True, order=True)
class ImportNote:
    """One thing an import could not represent, with how often it happened.

    Attributes:
        kind: What kind of note this is — one of `NOTE_UNKNOWN_TOOL`,
            `NOTE_SKIPPED_RECORD`, `NOTE_DROPPED_BLOCK`.
        detail: What it was about: a tool name, a record type, a block type.
        count: How many times it occurred in the transcript.
    """

    kind: str
    detail: str
    count: int = 1

    def __str__(self) -> str:
        """Return the note as one diagnostic line."""
        repeats = f" (x{self.count})" if self.count > 1 else ""
        return f"{self.kind}: {self.detail}{repeats}"


@dataclass(frozen=True, slots=True)
class ImportedSession:
    """A canonical session plus what the import could not represent.

    Attributes:
        session: The imported transcript, validated against the canonical
            schema.
        notes: One note per distinct problem, sorted by kind and detail so that
            importing the same file twice reports the same thing in the same
            order.

    Invariant, checked on construction: a session whose `source` is
    `native-compaction` names its summary. Provenance without an identifier is a
    transcript nothing can measure — `idc.importers.provenance` would count the
    history the agent dropped as if the agent still had it — and an importer that
    read a summary record always knows which message it was, so producing the
    half-stated form is a bug here rather than incomplete input. The converse —
    an identifier without the native provenance — is not checked here: a summary
    identifier is a legitimate fact under `intent-compaction` too, where it names
    the summary this tool wrote, and `Session` refuses the one combination that
    contradicts itself, `raw` plus an identifier (see `idc.schema.models`). No
    importer produces either form: an importer that read a summary record calls
    `add_native_summary`, which sets both fields together.
    """

    session: Session
    notes: tuple[ImportNote, ...] = ()

    @property
    def native_summary_id(self) -> str | None:
        """`Message.id` of the message the session's context starts at.

        Read off the canonical session rather than held a second time, so the
        two cannot drift apart: the identifier is a field of `Session` since
        T2.6, which is what lets a dumped transcript still say where the agent's
        context starts. `None` for a transcript the agent never compacted; a
        transcript it compacted twice names the **last** summary, because that
        is the one its context now starts at.
        """
        return self.session.native_summary_id

    def __post_init__(self) -> None:
        """Reject a compacted session that does not name its summary.

        Raises:
            TranscriptImportError: If the session was compacted by the agent
                itself but names no summary.
        """
        if (
            self.session.source is SessionSource.NATIVE_COMPACTION
            and self.session.native_summary_id is None
        ):
            raise TranscriptImportError(
                f"session {self.session.id!r} was compacted by "
                f"{self.session.agent.value} itself, but names no native summary; "
                "the message the agent continued from has to be recorded, or the "
                "history it dropped would be counted as context"
            )

    def notes_of(self, kind: str) -> tuple[ImportNote, ...]:
        """Return the notes of one kind, in the order they are held.

        Args:
            kind: One of `NOTE_UNKNOWN_TOOL`, `NOTE_SKIPPED_RECORD`,
                `NOTE_DROPPED_BLOCK`.

        Returns:
            The matching notes; empty when there are none.
        """
        return tuple(note for note in self.notes if note.kind == kind)

    @property
    def unknown_tools(self) -> tuple[str, ...]:
        """Return every tool name the registry does not declare, sorted."""
        return tuple(note.detail for note in self.notes_of(NOTE_UNKNOWN_TOOL))


def parse_timestamp(value: Any, *, source: str | Path | None = None) -> datetime | None:
    """Read a transcript timestamp as a timezone-aware instant.

    Args:
        value: The value the transcript records, normally an ISO-8601 string;
            `None` or an absent field yields `None`, because `Message.timestamp`
            is optional.
        source: Where the value came from, for the error message.

    Returns:
        The instant, with UTC attached when the agent recorded no offset — the
        canonical schema rejects a naive timestamp, and the alternative would be
        to drop an instant the transcript does record.

    Raises:
        TranscriptImportError: If the value is neither `None` nor a readable
            ISO-8601 string. A timestamp the importer cannot read is not guessed
            at: it would silently reorder a transcript.
    """
    if value is None:
        return None
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError:
            parsed = None
        if parsed is not None:
            return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)
    origin = f"{source}: " if source is not None else ""
    raise TranscriptImportError(f"{origin}cannot read the timestamp {value!r}")


def block_message_id(base: str, index: int, *, only: bool) -> str:
    """Return the canonical message ID for a message derived from a block.

    One transcript record can carry several canonical messages: the schema holds
    exactly one `tool_result` per message, so a record with two results becomes
    two. Their IDs are derived from the record's own identifier rather than
    renumbered, so that a canonical transcript can still be read against the
    file it came from.

    Args:
        base: The record's identifier.
        index: Index of the content block the message comes from.
        only: Whether this is the record's only canonical message.

    Returns:
        `base` when the record yields one message, else `base#<index>`.
    """
    return base if only else f"{base}#{index}"


def type_name(holder: Mapping[str, Any]) -> str:
    """Return the `type` of a record or of a content block as the notes name it.

    Args:
        holder: One transcript record, or one of its content blocks or parts.

    Returns:
        The `type` field as a string, or `NO_TYPE` when it names none and
        `NULL_TYPE` when it names `null`. Those two are different problems — a
        shape the importer does not know versus a damaged one — and `str(None)`
        would report them as the same one, which is also how a kind that is
        genuinely spelled `"None"` would become indistinguishable from both. A
        type that is present and not a string keeps its own spelling (`7`).

        One function for both levels on purpose: the collision was first fixed
        for records and then found again for blocks.
    """
    if "type" not in holder:
        return NO_TYPE
    kind = holder["type"]
    return NULL_TYPE if kind is None else str(kind)


def shape(value: Any) -> str:
    """Return the JSON shape of a value, to name something that has no kind.

    Args:
        value: The value a note is about.

    Returns:
        Its JSON type name — `object`, `array`, `string`, `number`, `boolean`,
        `null` — and otherwise the name of its Python type. A note about a block
        names the block's kind; a value that is not a block has none, so its
        shape is the only honest thing to report. `bool` is checked before the
        numbers because it is a subclass of `int`.
    """
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int | float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, Mapping):
        return "object"
    if isinstance(value, list):
        return "array"
    return type(value).__name__


class SessionBuilder:
    """Collects canonical messages and validates them into a `Session`.

    One builder per imported file. It is deliberately not reusable: `build()`
    reports what the whole transcript could not represent, which only makes
    sense once.
    """

    def __init__(
        self,
        agent: Agent | str,
        *,
        registry: ToolRegistry | None = None,
        source: str | Path | None = None,
        provenance: SessionSource = SessionSource.RAW,
    ) -> None:
        """Start an empty session for one agent.

        Args:
            agent: The agent whose transcript is being imported.
            registry: Tool knowledge to resolve categories with; the packaged
                registry when omitted. A test passes its own.
            source: The file being imported, named in every error message.
            provenance: Provenance to start from; an importer calls
                `set_provenance` when the transcript turns out to be a
                compacted one.
        """
        self._agent = Agent(agent)
        self._registry = tool_registry() if registry is None else registry
        self._source = source
        self._provenance = provenance
        self._messages: list[Message] = []
        self._counts: Counter[tuple[str, str]] = Counter()
        self._logged: set[tuple[str, str]] = set()
        self._model: str | None = None
        self._timestamps: list[datetime] = []
        self._native_summary_id: str | None = None

    def __repr__(self) -> str:
        """Return a representation naming the agent and the message count."""
        return f"{type(self).__name__}({self._agent.value!r}, messages={len(self._messages)})"

    @property
    def agent(self) -> Agent:
        """The agent whose transcript is being imported."""
        return self._agent

    @property
    def origin(self) -> str:
        """Return `<source>: ` for error messages, or the empty string."""
        return f"{self._source}: " if self._source is not None else ""

    def fail(self, reason: str, *, line: int | None = None) -> TranscriptImportError:
        """Build the error for a transcript this importer cannot read.

        Args:
            reason: What is wrong, as a sentence fragment.
            line: Line number the problem belongs to, when it belongs to one.

        Returns:
            The error to raise, naming the source file and the line.
        """
        location = f"line {line}: " if line is not None else ""
        return TranscriptImportError(f"{self.origin}{location}{reason}")

    def note(self, kind: str, detail: str, count: int = 1) -> None:
        """Record that something could not be represented canonically.

        Args:
            kind: One of the `NOTE_*` constants.
            detail: What it was about — a tool name, a record or block kind.
            count: How many occurrences to add.
        """
        self._counts[(kind, detail)] += count
        LOGGER.debug("%s%s: %s", self.origin, kind, detail)

    def set_model(self, model: Any) -> None:
        """Remember the model identifier, taking the first one the file names."""
        if self._model is None and isinstance(model, str) and model:
            self._model = model

    def set_provenance(self, provenance: SessionSource) -> None:
        """Set the session's provenance, e.g. on finding a native summary."""
        self._provenance = provenance

    def tool_call(self, name: str, *, call_id: str, args: dict[str, Any] | None = None) -> ToolCall:
        """Build a canonical tool call, with its category from the registry.

        Args:
            name: The tool name exactly as the agent reports it.
            call_id: The identifier the agent gave the call; it is kept, not
                reassigned, because the result references it.
            args: The call's arguments, kept as the agent sent them.

        Returns:
            The canonical call. A tool the registry does not declare resolves to
            `ToolCategory.OTHER`, is added to the notes and is logged once per
            distinct name.
        """
        if not self._registry.knows(self._agent, name):
            self.note(NOTE_UNKNOWN_TOOL, name)
            key = (NOTE_UNKNOWN_TOOL, name)
            if key not in self._logged:
                self._logged.add(key)
                LOGGER.warning(
                    "%stool %r is not declared for %s; its category resolves to %r",
                    self.origin,
                    name,
                    self._agent.value,
                    ToolCategory.OTHER.value,
                )
        return ToolCall(
            id=call_id,
            name=name,
            category=self._registry.category(self._agent, name),
            args=dict(args or {}),
        )

    def add_message(
        self,
        *,
        message_id: str,
        role: Role,
        content: str = "",
        timestamp: datetime | None = None,
        is_system_injected: bool = False,
        is_command: bool = False,
        tool_calls: Sequence[ToolCall] = (),
        tool_result: ToolResult | None = None,
    ) -> Message:
        """Append one canonical message.

        Args:
            message_id: Identifier of the message, derived from the transcript.
            role: Who authored it.
            content: Its text content.
            timestamp: When it was recorded, timezone-aware.
            is_system_injected: Whether the harness injected it.
            is_command: Whether it is a slash command or its echo.
            tool_calls: The calls the message issued.
            tool_result: The result it reports; required for `Role.TOOL`.

        Returns:
            The appended message.

        Raises:
            SchemaValidationError: If the message does not satisfy the canonical
                schema — a `role = tool` message without a result, or two calls
                with the same identifier in one message.
        """
        message = Message(
            id=message_id,
            role=role,
            content=content,
            timestamp=timestamp,
            is_system_injected=is_system_injected,
            is_command=is_command,
            tool_calls=list(tool_calls),
            tool_result=tool_result,
        )
        self._messages.append(message)
        if timestamp is not None:
            self._timestamps.append(timestamp)
        return message

    def add_native_summary(
        self,
        *,
        message_id: str,
        content: str,
        timestamp: datetime | None = None,
    ) -> Message:
        """Append the summary the agent's own compaction wrote.

        All three agents record that summary the same way once it is canonical —
        an injected `system` message — and all three mean the same thing by it:
        the agent dropped everything before it and continued from the summary.
        The two facts that follow from it are therefore set here rather than in
        each importer: the session's provenance becomes `native-compaction`, and
        the message is recorded as the session's `native_summary_id`. Both are
        canonical fields, so a dumped transcript still says where the agent's
        context starts — which is what `idc.importers.provenance` reads, in this
        process or two CLI invocations later.

        Args:
            message_id: Identifier of the summary message, from the transcript.
            content: The summary's text, as the agent wrote it.
            timestamp: When it was recorded, timezone-aware.

        Returns:
            The appended message.

        Raises:
            SchemaValidationError: If the message does not satisfy the canonical
                schema.
        """
        self.set_provenance(SessionSource.NATIVE_COMPACTION)
        message = self.add_message(
            message_id=message_id,
            role=Role.SYSTEM,
            content=content,
            timestamp=timestamp,
            is_system_injected=True,
        )
        self._native_summary_id = message.id
        return message

    def notes(self) -> tuple[ImportNote, ...]:
        """Return the notes collected so far, sorted by kind and detail."""
        return tuple(
            sorted(
                ImportNote(kind, detail, count) for (kind, detail), count in self._counts.items()
            )
        )

    def build(self, *, session_id: Any) -> ImportedSession:
        """Validate the collected messages into a canonical session.

        Args:
            session_id: The identifier the transcript records for the session.

        Returns:
            The session together with the import's notes.

        Raises:
            TranscriptImportError: If the transcript names no session, holds no
                message record, or records no timestamp at all — `Session.id`
                and `Session.created_at` are required and neither is invented
                here.
            SchemaValidationError: If the messages do not satisfy the canonical
                schema, e.g. because an identifier occurs twice.
        """
        if not isinstance(session_id, str) or not session_id:
            raise self.fail("the transcript names no session identifier")
        if not self._messages:
            raise self.fail("the transcript holds no message record")
        if not self._timestamps:
            raise self.fail("the transcript records no timestamp, so the session has no start")
        return ImportedSession(
            session=Session(
                id=session_id,
                agent=self._agent,
                model=self._model,
                source=self._provenance,
                native_summary_id=self._native_summary_id,
                created_at=min(self._timestamps),
                messages=self._messages,
            ),
            notes=self.notes(),
        )


def joined_text(parts: Iterable[str]) -> str:
    """Join the text blocks of one record into one canonical content string.

    Args:
        parts: The text of every text block of the record, in order.

    Returns:
        The non-empty parts joined by a blank line. `Message.content` is one
        string, so a record that carries its text in several blocks has to be
        joined somewhere; a blank line is what the agents' own renderers put
        between them.
    """
    return "\n\n".join(part for part in parts if part)


def note_unreadable_content(builder: SessionBuilder, content: Any, *, prefix: str = "") -> None:
    """Note a content value that is neither text nor a block list.

    Args:
        builder: The session being assembled, for the notes.
        content: The content value the record carried.
        prefix: What to put in front of the note's detail, naming where in the
            record the content sat.

    An absent content (`None`) is not noted: the record carried nothing, so
    nothing was lost. Anything else — a number, an object — did carry something
    the importer cannot read, and the note is what keeps that from being a
    silent loss in a tool whose output is a measurement.
    """
    if content is not None and not isinstance(content, str | list):
        builder.note(NOTE_DROPPED_BLOCK, f"{prefix}<content: {shape(content)}>")
