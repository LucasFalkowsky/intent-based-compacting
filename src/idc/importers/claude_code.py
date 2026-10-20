"""Claude Code: a session JSONL read into a canonical `Session`.

Claude Code appends one JSON object per line to
`~/.claude/projects/<slug>/<session-id>.jsonl`. Most of those objects are
bookkeeping; the ones that carry a conversation are `user`, `assistant` and
`system`, and the content of the first two is the Anthropic Messages API's own
block list. This module knows that shape and nothing else — reading the file,
resolving tool categories, logging unknown tools and validating the result are
`idc.importers.jsonl` and `idc.importers.base`.

**Which record becomes what.**

- `user`, content a string -> one `user` message.
- `user`, content a block list -> one `user` message carrying its text, plus
  one `tool` message per `tool_result` block.
- `user` with `isCompactSummary` -> an injected `system` message carrying the
  summary's text, and the session's `source` becomes `native-compaction`; any
  other block of that record is dropped and noted like anywhere else.
- `user` with `isMeta` -> a `user` message marked `is_system_injected`.
- `assistant` -> one `assistant` message with its text and every `tool_use`
  block it issued.
- `system` with a non-empty `content` -> one injected `system` message.
- anything else -> skipped, and noted by its record type.

A `user` record whose `content` is neither text nor a block list — a number, an
object, an absent `message` — yields **no** message either, and is noted twice:
by its record type and by the shape of the value that was lost. Importing it as
an empty `user` turn would not only drop the value, it would fabricate an intent
boundary, because that is what T3.1 splits chunks on and T4 classifies.

Two rules decide the rest.

**Identifiers come from the transcript.** A message keeps the record's `uuid`,
and a tool call keeps the `tool_use` id the result references. The canonical
schema holds one `tool_result` per message, so a record with two results becomes
two messages, `<uuid>#0` and `<uuid>#1` (see `block_message_id`). Nothing is
renumbered, so a canonical transcript can still be read against the file it came
from, and a duplicated identifier is reported by the schema instead of being
papered over.

**Nothing is repaired.** A `tool_result` whose `tool_use` is not in the file
(the transcript was cut off at the front) and a `tool_use` whose result never
arrived (the user interrupted the turn) are both imported as they are.

Known limitations, all of them deliberate:

- A `thinking` block is dropped and noted. `Message.content` is the text a later
  turn reads back, and the canonical schema has no field for reasoning —
  inventing one is a schema ticket, and folding a signed thinking block into
  plain text would make the renderer re-emit it as if the agent had said it.
- An `attachment` record is skipped and noted. Its payload shape is a different
  one per kind (`total_tokens_reminder`, `task_reminder`, `opened_file_in_ide`,
  …), and the ones that carry context are reminders the transcript also records
  as a message.
- `ToolResult.exit_code` stays `null`: Claude Code records no process status for
  a `Bash` call, only whether the tool reported an error. The resolver's
  verification check therefore reads `is_error`, not an exit code.
- `isSidechain` is ignored. A subagent's own records are imported as the
  messages they are; lifting them into a chunk of their own is T3.3's job, which
  works on the canonical session.
- A value this importer cannot read is dropped **and noted by its shape**: a
  content entry that is not an object (`<entry: string>`), a content that is
  neither text nor a block list (`<content: object>`), a `tool_use.input` that
  is not an object (`tool_use/Bash/<input: array>`). Nothing is guessed at, and
  nothing leaves the import without a line in the notes — for `Bash` the lost
  `input` would be the verification command T5.2 and T6 resolve.

```python
from idc.importers import import_claude_code

imported = import_claude_code("~/.claude/projects/repo/session.jsonl")
imported.session.source        # SessionSource.RAW
imported.unknown_tools         # ('mcp__server__thing',)
```
"""

from collections.abc import Callable, Mapping
from datetime import datetime
from pathlib import Path
from typing import Any, NamedTuple

from idc.importers.base import (
    NOTE_DROPPED_BLOCK,
    NOTE_SKIPPED_RECORD,
    ImportedSession,
    SessionBuilder,
    block_message_id,
    joined_text,
    parse_timestamp,
)
from idc.importers.jsonl import JsonLine, read_json_lines
from idc.importers.tool_registry import ToolRegistry
from idc.schema import Agent, Role, SessionSource, ToolCall, ToolResult

__all__ = ["COMMAND_MARKERS", "import_claude_code"]

RECORD_USER = "user"
RECORD_ASSISTANT = "assistant"
RECORD_SYSTEM = "system"

BLOCK_TEXT = "text"
BLOCK_TOOL_USE = "tool_use"
BLOCK_TOOL_RESULT = "tool_result"

SUMMARY_FLAG = "isCompactSummary"
"""Field Claude Code sets on the summary message its own compaction writes."""

INJECTED_FLAG = "isMeta"
"""Field Claude Code sets on a user record the harness wrote, not the user."""


class _UserPart(NamedTuple):
    """One canonical message a block-list `user` record yields.

    Attributes:
        message_id: Identifier derived from the record's own.
        role: `USER` for the record's text, `TOOL` for one of its results.
        content: The message's text content.
        result: The result it reports, for a `TOOL` part.
    """

    message_id: str
    role: Role
    content: str
    result: ToolResult | None


COMMAND_MARKERS = ("<command-name>", "<command-message>", "<local-command-stdout>")
"""How Claude Code records a slash command and its output in a user message.

The agent stores a slash command in the user role, wrapped in these tags, and
the canonical schema has `is_command` for exactly that: the chunker must not
read `/compact` as a user turn. Matched at the start of the message, so a user
who writes one of the tags *about* a transcript is not mistaken for one.
"""


def import_claude_code(
    path: str | Path, *, registry: ToolRegistry | None = None
) -> ImportedSession:
    """Read a Claude Code session JSONL into a canonical session.

    Args:
        path: The session file, one JSON object per line.
        registry: Tool knowledge to resolve the call categories with; the
            packaged registry when omitted.

    Returns:
        The canonical session together with the import's notes — the unknown
        tools, the records that carried no message and the blocks the schema has
        no field for.

    Raises:
        TranscriptImportError: If the file cannot be read, a line is not a JSON
            object, a record this importer has to read is malformed, or the
            transcript names no session, holds no message or records no
            timestamp.
        SchemaValidationError: If the messages the transcript yields do not
            satisfy the canonical schema, e.g. because a `tool_use` identifier
            occurs twice.
    """
    builder = SessionBuilder(Agent.CLAUDE_CODE, registry=registry, source=path)
    session_id: Any = None
    for line in read_json_lines(path):
        record = line.data
        if session_id is None:
            session_id = record.get("sessionId")
        kind = record.get("type")
        reader = READERS.get(str(kind))
        if reader is None or not reader(builder, line):
            builder.note(NOTE_SKIPPED_RECORD, str(kind))
    return builder.build(session_id=session_id)


def _record_id(line: JsonLine) -> str:
    """Return the record's own identifier, or one derived from its line.

    Args:
        line: The record and its line number.

    Returns:
        The record's `uuid`. A record without one falls back to `line-<n>`,
        which is unique within the file and points at the line it came from —
        the canonical schema requires an identifier, and the alternative would
        be to renumber every message of the session.
    """
    uuid = line.data.get("uuid")
    return uuid if isinstance(uuid, str) and uuid else f"line-{line.number}"


def _timestamp(builder: SessionBuilder, line: JsonLine) -> datetime | None:
    """Return the record's timestamp as a timezone-aware instant, or `None`."""
    return parse_timestamp(line.data.get("timestamp"), source=f"{builder.origin}line {line.number}")


def _blocks(builder: SessionBuilder, content: Any, *, prefix: str = "") -> list[Mapping[str, Any]]:
    """Return the content blocks of a message, noting the entries that are none.

    Args:
        builder: The session being assembled, for the notes.
        content: A message's content; anything but a list holds no blocks.
        prefix: What to put in front of a note's detail, as in `_noted_text`.

    Returns:
        The entries of the list that are objects, in order. An entry that is not
        one cannot be a content block and is dropped — but noted by its shape,
        because a filter that drops a value without a trace is how a measurement
        tool loses data.
    """
    if not isinstance(content, list):
        return []
    blocks: list[Mapping[str, Any]] = []
    for entry in content:
        if isinstance(entry, Mapping):
            blocks.append(entry)
        else:
            builder.note(NOTE_DROPPED_BLOCK, f"{prefix}<entry: {_shape(entry)}>")
    return blocks


def _shape(value: Any) -> str:
    """Return the JSON shape of a value, to name something that has no kind.

    Args:
        value: The value a note is about.

    Returns:
        Its JSON type name — `object`, `array`, `string`, `number`, `boolean`,
        `null`. A note about a block names the block's kind; a value that is not
        a block has none, so its shape is the only honest thing to report.
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


def _note_unreadable_content(builder: SessionBuilder, content: Any, *, prefix: str = "") -> None:
    """Note a content value that is neither text nor a block list.

    Args:
        builder: The session being assembled, for the notes.
        content: The content value the record carried.
        prefix: What to put in front of the note's detail, as in `_noted_text`.

    An absent content (`None`) is not noted: the record carried nothing, so
    nothing was lost. Anything else — a number, an object — did carry something
    this importer cannot read, and the note is what keeps that from being a
    silent loss in a tool whose output is a measurement.
    """
    if content is not None and not isinstance(content, str | list):
        builder.note(NOTE_DROPPED_BLOCK, f"{prefix}<content: {_shape(content)}>")


def _noted_text(builder: SessionBuilder, content: Any, *, prefix: str = "") -> str:
    """Return a content value as text, noting every block it drops.

    Args:
        builder: The session being assembled, for the notes.
        content: A message's content: a string, or the Messages API's block list.
        prefix: What to put in front of a note's detail, so that a block dropped
            inside a `tool_result` is reported as `tool_result/<kind>`.

    Returns:
        The text: a string as it is, a block list joined from its `text` blocks.
        Every other block of the list is noted as dropped rather than vanishing,
        which is what every path that reads content has to do — the schema has no
        field for a `thinking` block, and a reader must still be able to see that
        the transcript held one.
    """
    if isinstance(content, str):
        return content
    _note_unreadable_content(builder, content, prefix=prefix)
    texts: list[str] = []
    for block in _blocks(builder, content, prefix=prefix):
        if block.get("type") == BLOCK_TEXT:
            texts.append(str(block.get("text") or ""))
        else:
            builder.note(NOTE_DROPPED_BLOCK, f"{prefix}{block.get('type')}")
    return joined_text(texts)


def _is_command(content: str) -> bool:
    """Return whether a user message is a slash command or its output."""
    stripped = content.lstrip()
    return any(stripped.startswith(marker) for marker in COMMAND_MARKERS)


def _add_user_record(builder: SessionBuilder, line: JsonLine) -> bool:
    """Import a `user` record: a prompt, injected text, tool results, a summary.

    Args:
        builder: The session being assembled.
        line: The record and its line number.

    Returns:
        Whether the record yielded at least one canonical message. A record
        whose blocks hold neither text nor a result yields none, and so does one
        whose content is neither text nor a block list; both are then noted as
        skipped like any other record that carries no message, rather than
        becoming an empty user turn the user never took.

    Raises:
        TranscriptImportError: If a `tool_result` block names no `tool_use_id`,
            which would leave the result unattachable to any call.
    """
    record = line.data
    message = record.get("message")
    content = message.get("content") if isinstance(message, Mapping) else None
    timestamp = _timestamp(builder, line)
    if record.get(SUMMARY_FLAG) is True:
        builder.set_provenance(SessionSource.NATIVE_COMPACTION)
        builder.add_message(
            message_id=_record_id(line),
            role=Role.SYSTEM,
            content=_noted_text(builder, content),
            timestamp=timestamp,
            is_system_injected=True,
        )
        return True
    injected = record.get(INJECTED_FLAG) is True
    if not isinstance(content, list):
        if not isinstance(content, str):
            _note_unreadable_content(builder, content)
            return False
        text = content
        builder.add_message(
            message_id=_record_id(line),
            role=Role.USER,
            content=text,
            timestamp=timestamp,
            is_system_injected=injected,
            is_command=_is_command(text),
        )
        return True
    parts = _user_parts(builder, line, _blocks(builder, content))
    for part in parts:
        builder.add_message(
            message_id=part.message_id,
            role=part.role,
            content=part.content,
            timestamp=timestamp,
            is_system_injected=injected and part.role is Role.USER,
            is_command=_is_command(part.content),
            tool_result=part.result,
        )
    return bool(parts)


def _user_parts(
    builder: SessionBuilder, line: JsonLine, blocks: list[Mapping[str, Any]]
) -> list[_UserPart]:
    """Split a block-list `user` record into the canonical messages it holds.

    Args:
        builder: The session being assembled, for the notes.
        line: The record and its line number.
        blocks: The record's content blocks.

    Returns:
        One part per canonical message, in the order of the blocks they come
        from. All `text` blocks of the record are joined into one message,
        placed where the first of them sat; every `tool_result` block becomes a
        message of its own, because the schema holds one result per message.

    Raises:
        TranscriptImportError: If a `tool_result` block names no `tool_use_id`.
    """
    found: list[tuple[int, Role, str, ToolResult | None]] = []
    texts: list[str] = []
    text_index: int | None = None
    for index, block in enumerate(blocks):
        kind = block.get("type")
        if kind == BLOCK_TEXT:
            texts.append(str(block.get("text") or ""))
            text_index = index if text_index is None else text_index
        elif kind == BLOCK_TOOL_RESULT:
            found.append((index, Role.TOOL, "", _tool_result(builder, line, block)))
        else:
            builder.note(NOTE_DROPPED_BLOCK, str(kind))
    text = joined_text(texts)
    if text and text_index is not None:
        found.append((text_index, Role.USER, text, None))
    found.sort(key=lambda part: part[0])
    only = len(found) == 1
    return [
        _UserPart(block_message_id(_record_id(line), index, only=only), role, content, result)
        for index, role, content, result in found
    ]


def _tool_result(builder: SessionBuilder, line: JsonLine, block: Mapping[str, Any]) -> ToolResult:
    """Build the canonical result a `tool_result` block reports.

    Args:
        builder: The session being assembled, for the notes and the error.
        line: The record the block belongs to.
        block: The `tool_result` block.

    Returns:
        The canonical result. Its output is the block's content: a string as it
        is, a block list joined, with anything that is not text — another block
        kind, an entry that is no block, a content that is neither — noted and
        dropped. `exit_code` stays `None` — the transcript records none.

    Raises:
        TranscriptImportError: If the block names no `tool_use_id`. The
            reference is the only link between a result and its call, so a
            result without one is a damaged record, not a truncated transcript.
    """
    call_id = block.get("tool_use_id")
    if not isinstance(call_id, str) or not call_id:
        raise builder.fail("a tool_result block carries no tool_use_id", line=line.number)
    return ToolResult(
        call_id=call_id,
        output=_noted_text(builder, block.get("content"), prefix=f"{BLOCK_TOOL_RESULT}/"),
        is_error=block.get("is_error") is True,
    )


def _add_assistant_record(builder: SessionBuilder, line: JsonLine) -> bool:
    """Import an `assistant` record as one message with all its tool calls.

    Args:
        builder: The session being assembled.
        line: The record and its line number.

    Returns:
        `True`: an assistant record is always a turn, even when it carries
        neither text nor a call, because the agent did take that turn.

    Raises:
        TranscriptImportError: If a `tool_use` block names no identifier or no
            tool.
    """
    message = line.data.get("message")
    message = message if isinstance(message, Mapping) else {}
    builder.set_model(message.get("model"))
    content = message.get("content")
    _note_unreadable_content(builder, content)
    texts: list[str] = []
    calls: list[ToolCall] = []
    for block in _blocks(builder, content):
        kind = block.get("type")
        if kind == BLOCK_TEXT:
            texts.append(str(block.get("text") or ""))
        elif kind == BLOCK_TOOL_USE:
            calls.append(_tool_call(builder, line, block))
        else:
            builder.note(NOTE_DROPPED_BLOCK, str(kind))
    builder.add_message(
        message_id=_record_id(line),
        role=Role.ASSISTANT,
        content=content if isinstance(content, str) else joined_text(texts),
        timestamp=_timestamp(builder, line),
        tool_calls=calls,
    )
    return True


def _add_system_record(builder: SessionBuilder, line: JsonLine) -> bool:
    """Import a `system` record, which is a message only when it has content.

    Args:
        builder: The session being assembled.
        line: The record and its line number.

    Returns:
        Whether the record carried content. Most `system` records are pure
        bookkeeping (`turn_duration`, `agents_killed`) and hold no text at all;
        those are skipped like any other record without a message.
    """
    content = _noted_text(builder, line.data.get("content"))
    if not content:
        return False
    builder.add_message(
        message_id=_record_id(line),
        role=Role.SYSTEM,
        content=content,
        timestamp=_timestamp(builder, line),
        is_system_injected=True,
    )
    return True


def _tool_call(builder: SessionBuilder, line: JsonLine, block: Mapping[str, Any]) -> ToolCall:
    """Build the canonical call a `tool_use` block made.

    Args:
        builder: The session being assembled, which resolves the category.
        line: The record the block belongs to.
        block: The `tool_use` block.

    Returns:
        The canonical call, with the identifier and the arguments the agent
        sent and the category the registry gives its name. An `input` that is
        not an object yields no arguments and is noted with its tool: for a
        `Bash` call those arguments are the command T5.2 and T6 resolve, so
        losing them silently would be expensive.

    Raises:
        TranscriptImportError: If the block names no identifier or no tool. The
            identifier is what the result references and the name is what the
            category is resolved from, so neither can be guessed at.
    """
    call_id = block.get("id")
    name = block.get("name")
    if not isinstance(call_id, str) or not call_id:
        raise builder.fail("a tool_use block carries no id", line=line.number)
    if not isinstance(name, str) or not name:
        raise builder.fail(f"the tool_use block {call_id!r} carries no tool name", line=line.number)
    arguments = block.get("input")
    if isinstance(arguments, Mapping):
        args = dict(arguments)
    else:
        args = {}
        if arguments is not None:
            builder.note(
                NOTE_DROPPED_BLOCK, f"{BLOCK_TOOL_USE}/{name}/<input: {_shape(arguments)}>"
            )
    return builder.tool_call(name, call_id=call_id, args=args)


READERS: dict[str, Callable[[SessionBuilder, JsonLine], bool]] = {
    RECORD_USER: _add_user_record,
    RECORD_ASSISTANT: _add_assistant_record,
    RECORD_SYSTEM: _add_system_record,
}
"""The record types that can carry a message, and what reads them.

A reader returns whether the record yielded a canonical message; everything
else — a record type that is not in this table, and one that is but held no
message — is noted as skipped by its type. That is what keeps a new Claude Code
record type a line in the import's notes instead of a crash or a silent loss.
"""
