"""Gemini CLI: a chat recording read into a canonical `Session`.

Gemini CLI records a session under
`~/.gemini/tmp/<project_hash>/chats/session-<timestamp>-<short id>` in **two
shapes**, and reads both back. This module knows them and nothing else —
reading the file, resolving tool categories, logging unknown tools and
validating the result are `idc.importers.jsonl` and `idc.importers.base`.

- **One JSON document** (`.json`): an object holding `sessionId`,
  `projectHash`, `startTime`, `lastUpdated`, an optional `summary` and a
  `messages` array. Pretty-printed over many lines, so it is *not* JSONL and
  must not be read as such.
- **An append log** (`.jsonl`): the metadata as the first record, then one
  record per message, plus two operations that edit what came before.

A file that parses as one JSON object is read as that single record; only a file
that does not is read line by line, where a damaged line is fatal and named by
its number, as for every transcript.

**The log is replayed, not concatenated.** That is the difference to the other
two importers and it is load-bearing:

- A message record is identified by its `id`, and the agent appends the *whole*
  message again whenever it changes — a model turn is written once while its
  tool call is still running and once more when the result arrives. The later
  record is the message; it keeps the position of the first one. Taking every
  record would yield two canonical messages with one identifier, which the
  canonical schema rejects.
- `{"$rewindTo": "<id>"}` drops that message and everything recorded after it
  (`/rewind`). A message to rewind to that is not in the file clears the whole
  list.
- `{"$set": {…}}` updates the metadata, and `$set.messages` **replaces** the
  message list outright. That is how a native compaction lands in the file, so
  ignoring it would import the history the agent threw away and make the
  compaction look like it saved nothing.

This is exactly how Gemini CLI's own `loadConversationRecord` reads the file
back when it resumes a session, which is the only defensible reading: the
canonical session is the conversation the agent would resume, not the sum of
everything ever appended.

**Which record becomes what.**

- the metadata record (`sessionId` plus `projectHash`) -> no message; the
  session's identifier. Its `messages` array, which the pre-JSONL form of the
  format carried, is read if it is there.
- `user`, content a string or a list of parts -> one `user` message carrying its
  text, plus one `tool` message per `functionResponse` part. A part list that
  holds no text and no result — a pasted image — still yields the `user`
  message, with empty content and a note for the part that was dropped: the turn
  happened, and it is an intent boundary T3.1 splits on. A record that *does*
  report a result gets a `user` message only when it also carries text, so that
  no empty turn is invented in front of a tool result.
- `user` whose text holds `<state_snapshot>` -> an injected `system` message
  carrying the summary, and the session's `source` becomes `native-compaction`.
  The compression writes its summary back into the history as a user turn, and
  importing it as one would hand T3.1 a user intent the user never had.
- `user` whose text starts with `<session_context>` -> a `user` message marked
  `is_system_injected`: that is the environment context the harness puts in
  front of every session, not a prompt.
- `gemini` -> one `assistant` message with its text and every tool call it
  issued, plus one `tool` message per call that has a result.
- `info` or `error` with text -> one injected `system` message. Gemini CLI
  records its own notices under those two types — a model switch, a failed
  edit — and they are the harness talking, not a turn either side took.
- anything else -> skipped, and noted by its record type; `$set`, `$rewindTo`
  and the metadata record are noted by their own shape rather than collapsing
  into one `<no type>` line.

**Where a tool call comes from.** A completed call is a `toolCalls` entry of the
model's message: `id`, `name`, `args`, `status` and `result`, the result being
the `functionResponse` part the model was sent back. A model turn whose content
was rewritten from the live history carries the same call *again* as a
`functionCall` part; the recorded entry wins, and the part is not imported a
second time, because two canonical calls with one identifier do not validate.
A `functionCall` part whose identifier is not among the recorded entries — the
only shape a front-truncated or rewritten history leaves — is imported as the
call it is.

**Nothing is repaired.** A `functionResponse` whose call is not in the file (the
transcript was cut off at the front, or the kept remainder of a compaction
starts with one) and a call whose `result` is still `null` (the user interrupted
the turn or never approved it) are both imported as they are.

Known limitations, all of them deliberate:

- A thought is dropped and noted, both the `thoughts` field of a model message
  and a `{"text": …, "thought": true}` part. `Message.content` is the text a
  later turn reads back and the canonical schema has no field for reasoning;
  folding a thought into plain text would make the renderer re-emit it as if
  the agent had said it.
- `is_error` is "the agent saw this result as a failure", which for Gemini CLI
  is its recorded `status`: anything other than `success`, or a response that
  carries an `error` instead of an `output`. A shell command that exits
  non-zero is *not* that — the call itself succeeded — and is reported through
  `exit_code` instead (see `_exit_code`).
- `displayContent` is not read. It is what the UI showed the user before the
  prompt's `@file` references were expanded; `content` is what the model saw,
  and that is what the later stages count and re-render.
- `tokens`, `resultDisplay`, `displayName`, `renderOutputAsMarkdown`,
  `agentId`, `kind`, `directories`, `memoryScratchpad` and `startTime` are not
  read either. `startTime` is within a tick of the first message's timestamp,
  and `created_at` is the earliest message timestamp for all three agents; the
  rest is display or telemetry.
- `summary` is the session's title in the resume browser, not a compaction
  summary, and does not make a transcript `native-compaction`.
- Slash commands are not recorded: Gemini CLI handles them in the CLI, so
  nothing reaches the chat recording and `is_command` is never set. The loop
  recovery notice the agent injects as a user turn imports as an ordinary user
  message — a pointer for #10 (provenance normalisation), not a decision here.
- A value this importer cannot read is dropped **and noted by its shape**: a
  content that is neither text nor a part list (`<content: object>`), a list
  entry that is not an object (`<entry: string>`), a part whose kind it does not
  know (`inlineData`, `thought`, `<no part kind>`), a `text` that is not a
  string (`text/<text: object>`), arguments that are not an object
  (`toolCalls/run_shell_command/<args: array>`), a status that is not a string
  (`toolCalls/<status: number>`), a response it cannot read
  (`functionResponse/<response: object>`), an unusable `id` (`<id: number>`) and
  a second, different `sessionId` (`<sessionId: a second value>`). Nothing is
  guessed at and nothing is coerced. A field that is absent or `null` is not
  noted: it carried no value to lose.

**A record without a usable `id` is skipped**, and noted by its type as well as
by the shape of the `id` it did carry. Gemini CLI keys its messages by `id` and
drops such a record when it resumes the session, so importing it under an
invented identifier would add a turn the agent itself no longer has.

**Where the knowledge in this module comes from.** Two sources, and the
difference matters when a future version changes something:

- *Observed in real session files.* A structure-only survey of the 56 sessions
  on the machine this was written on (key names, type names, tool names and
  argument keys only, never content) established: the single-document shape and
  its five top-level keys; the message keys `id`, `timestamp`, `type`,
  `content`, `displayContent`, `model`, `thoughts`, `tokens`, `toolCalls`; the
  types `user`, `gemini`, `info` and `error`; content as a string or a list of
  `text` parts; the `toolCalls` keys `id`, `name`, `args`, `result`, `status`,
  `timestamp`, `resultDisplay`, `displayName`, `description`,
  `renderOutputAsMarkdown`; a result as exactly one
  `functionResponse: {id, name, response}` whose response carries `output` or
  `error`; the statuses `success` and `error`; and a `timestamp` on **every**
  message, which is what lets `created_at` be the earliest of them without
  inventing anything.
- *Read off the installed agent (0.46.0), not seen in those files.* The append
  log and its `$set`/`$rewindTo` operations and repeated records
  (`packages/core/src/services/chatRecordingService.ts`); the `functionCall`,
  `functionResponse` and `thought: true` content parts a rewritten turn carries
  (`convertToFunctionResponse`, `updateMessagesFromHistory`); the statuses
  `cancelled`, `awaiting_approval`, `validating`, `scheduled` and `executing`
  (`CoreToolCallStatus`); the `<state_snapshot>` a compaction writes
  (`chatCompressionService.ts` and the compression prompt); the
  `<session_context>` the harness prepends (`getEnvironmentContext`); and the
  shell tool's `Exit Code:` line (`getShellToolDescription` documents it as
  "only included if non-zero"). None of it is guessed: every one of these is a
  spelling the installed agent writes. They are kept because a transcript
  recorded with the installed version *will* carry them, and reading them
  wrongly would cost a tool result, a message or a provenance mark.

```python
from idc.importers import import_gemini_cli

imported = import_gemini_cli("~/.gemini/tmp/<hash>/chats/session-….json")
imported.session.source        # SessionSource.RAW
imported.unknown_tools         # ('teleport_file',)
```
"""

import json
from collections.abc import Callable, Mapping, Sequence
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

__all__ = ["ENVIRONMENT_MARKER", "SNAPSHOT_MARKER", "import_gemini_cli"]

RECORD_USER = "user"
RECORD_MODEL = "gemini"
RECORD_INFO = "info"
RECORD_ERROR = "error"

FIELD_SET = "$set"
"""Log operation that updates the metadata, and may replace the message list."""

FIELD_REWIND = "$rewindTo"
"""Log operation that drops a message and everything recorded after it."""

FIELD_MESSAGES = "messages"
FIELD_TOOL_CALLS = "toolCalls"
FIELD_THOUGHTS = "thoughts"
FIELD_SESSION = "sessionId"
FIELD_PROJECT = "projectHash"

PART_TEXT = "text"
PART_FUNCTION_CALL = "functionCall"
PART_FUNCTION_RESPONSE = "functionResponse"
PART_THOUGHT = "thought"

PART_KINDS = (PART_TEXT, PART_FUNCTION_CALL, PART_FUNCTION_RESPONSE, "inlineData", "fileData")
"""The part kinds this importer can name, in the order it looks for them.

A part has no `type` field: its kind *is* the key it carries. Only the keys the
agent's own code writes are listed, so a part this importer cannot name is
reported as `NO_PART_KIND` instead of under an invented kind.
"""

METADATA_RECORD = "<metadata>"
"""How the record that carries the session's metadata is named in the notes."""

NO_PART_KIND = "<no part kind>"
"""How a content part that carries none of the known keys is named."""

NO_TYPE = "<no type>"
"""How a record that names no `type` at all is reported in the notes."""

NULL_TYPE = "<null type>"
"""How a record whose `type` is `null` is reported in the notes.

Distinct from `NO_TYPE`: a missing field is a shape this importer does not know,
a `null` one is damaged, and `str(None)` reads the same for both — and the same
as a type genuinely spelled `"None"`.
"""

SECOND_SESSION = "<sessionId: a second value>"
"""How a record naming a session other than the first one is reported.

A session file is one session, so the identifier the first record names is the
one the canonical session keeps. A second, different one is a damaged or
concatenated file, and ignoring it silently would hide that.
"""

SNAPSHOT_MARKER = "<state_snapshot>"
"""The tag Gemini CLI's compression wraps its summary in.

Matched anywhere in the message, not at its start: the summarizer reasons in a
`<scratchpad>` first and both blocks are recorded together. That is also the
test the agent itself applies when it looks for an earlier snapshot in the
history, so a transcript this tool marks as compacted is the one the agent
treats as compacted. The cost is that a user who pastes the tag into a prompt
is taken for a snapshot — the same trade-off the agent makes.
"""

ENVIRONMENT_MARKER = "<session_context>"
"""The tag around the environment context the harness puts in front of a session.

Matched at the start of the message, where the agent writes it, so that a user
who mentions the tag is not mistaken for the harness.
"""

SUCCESS_STATUS = "success"
"""The one recorded tool-call status that is not a failure."""

SHELL_TOOL = "run_shell_command"
SHELL_OUTPUT_PREFIX = "Output: "
EXIT_CODE_PREFIX = "Exit Code: "
SHELL_TRAILING_FIELDS = (
    EXIT_CODE_PREFIX,
    "Signal: ",
    "Background PIDs: ",
    "Process Group PGID: ",
)
"""The fields Gemini CLI's shell tool appends after the command's output.

Its result is `Output: <output>`, optionally followed by `Error:` and by these,
each on its own line. The exit status is read only from this trailing block
(see `_exit_code`).
"""


class _Part(NamedTuple):
    """One canonical message a record's content yields.

    Attributes:
        index: Position the part sat at, which orders the messages and names
            them.
        role: `USER` for the record's text, `TOOL` for one of its results.
        content: The message's text content.
        result: The result it reports, for a `TOOL` part.
        timestamp: When it was recorded, when the part carries its own instant.
    """

    index: int
    role: Role
    content: str
    result: ToolResult | None
    timestamp: datetime | None = None


def import_gemini_cli(path: str | Path, *, registry: ToolRegistry | None = None) -> ImportedSession:
    """Read a Gemini CLI chat recording into a canonical session.

    Args:
        path: The session file: one JSON object per line, or the single JSON
            document the format used before that.
        registry: Tool knowledge to resolve the call categories with; the
            packaged registry when omitted.

    Returns:
        The canonical session together with the import's notes — the unknown
        tools, the records that carried no message and the parts the schema has
        no field for.

    Raises:
        TranscriptImportError: If the file cannot be read, a line is not a JSON
            object, a record this importer has to read is malformed, or the
            transcript names no session, holds no message or records no
            timestamp.
        SchemaValidationError: If the messages the transcript yields do not
            satisfy the canonical schema.
    """
    builder = SessionBuilder(Agent.GEMINI_CLI, registry=registry, source=path)
    log = _MessageLog()
    session_id: Any = None
    for line in _read_records(path):
        record = line.data
        session_id = _session_id(builder, record, session_id)
        rewind_to = record.get(FIELD_REWIND)
        if isinstance(rewind_to, str):
            log.rewind(rewind_to)
            builder.note(NOTE_SKIPPED_RECORD, FIELD_REWIND)
            continue
        message_id = _record_id(builder, record)
        if message_id is not None:
            log.put(message_id, line, record)
            continue
        updates = record.get(FIELD_SET)
        if isinstance(updates, Mapping):
            session_id = _session_id(builder, updates, session_id)
            replacement = updates.get(FIELD_MESSAGES)
            if isinstance(replacement, list):
                log.clear()
                log.extend(builder, line, replacement)
            builder.note(NOTE_SKIPPED_RECORD, FIELD_SET)
            continue
        if _is_metadata_record(record):
            embedded = record.get(FIELD_MESSAGES)
            if isinstance(embedded, list):
                log.extend(builder, line, embedded)
            builder.note(NOTE_SKIPPED_RECORD, METADATA_RECORD)
            continue
        builder.note(NOTE_SKIPPED_RECORD, _type_name(record))
    for message_id, line, record in log.messages():
        kind = _type_name(record)
        reader = READERS.get(kind)
        if reader is None or not reader(builder, line, record, message_id):
            builder.note(NOTE_SKIPPED_RECORD, kind)
    return builder.build(session_id=session_id)


def _read_records(path: str | Path) -> list[JsonLine]:
    """Return the transcript's records, whichever of its two shapes it has.

    Args:
        path: The session file.

    Returns:
        One `JsonLine` per record. A file that is one JSON object — the form the
        format had before it became a log, and the one the agent still falls
        back to — is that single record; anything else is read line by line.

    Raises:
        TranscriptImportError: If the file cannot be read or is not JSONL
            either; the line-by-line reader reports both, so a damaged
            transcript is still named by its line number.
    """
    document = _whole_document(Path(path))
    if document is not None:
        return [JsonLine(1, document)]
    return read_json_lines(path)


def _whole_document(source: Path) -> dict[str, Any] | None:
    """Return the file as one JSON object, or `None` when it is not one.

    Args:
        source: The session file.

    Returns:
        The parsed object, or `None` when the file cannot be read, is not JSON
        or is not a JSON object. Every reason to say `None` here is reported by
        `read_json_lines` afterwards, so this function never raises: deciding
        the shape must not change which error a damaged file produces.
    """
    try:
        text = source.read_bytes().decode("utf-8-sig")
    except (OSError, UnicodeDecodeError):
        return None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


class _MessageLog:
    """The messages of a chat recording, in the order the file establishes.

    A log of appended records, replayed: a record replaces the one with the same
    identifier *in place*, a rewind truncates, and a `$set.messages` replaces
    everything. That is what `loadConversationRecord` does when the agent
    resumes a session.
    """

    def __init__(self) -> None:
        """Start an empty log."""
        self._order: list[str] = []
        self._records: dict[str, tuple[JsonLine, Mapping[str, Any]]] = {}

    def __repr__(self) -> str:
        """Return a representation naming how many messages the log holds."""
        return f"{type(self).__name__}(messages={len(self._order)})"

    def put(self, message_id: str, line: JsonLine, record: Mapping[str, Any]) -> None:
        """Append a message, or replace an earlier one with the same identifier."""
        if message_id not in self._records:
            self._order.append(message_id)
        self._records[message_id] = (line, record)

    def clear(self) -> None:
        """Drop every message, for a `$set.messages` that replaces the list."""
        self._order.clear()
        self._records.clear()

    def rewind(self, message_id: str) -> None:
        """Drop the message with that identifier and everything after it.

        Args:
            message_id: The message to rewind to, which is dropped as well. One
                that is not in the log clears the whole list — the agent's own
                behaviour, and the only honest one: the log says the
                conversation no longer holds what came before that point.
        """
        if message_id not in self._records:
            self.clear()
            return
        cut = self._order.index(message_id)
        for dropped in self._order[cut:]:
            del self._records[dropped]
        del self._order[cut:]

    def extend(self, builder: SessionBuilder, line: JsonLine, entries: Sequence[Any]) -> None:
        """Add the messages an embedded `messages` array holds.

        Args:
            builder: The session being assembled, for the notes.
            line: The record the array came from, for the error messages of the
                messages inside it.
            entries: The array's entries. One that is not an object, and one
                without a usable identifier, is noted rather than dropped
                silently.
        """
        for entry in entries:
            if not isinstance(entry, Mapping):
                builder.note(NOTE_DROPPED_BLOCK, f"{FIELD_MESSAGES}/<entry: {_shape(entry)}>")
                continue
            message_id = _record_id(builder, entry)
            if message_id is None:
                builder.note(NOTE_SKIPPED_RECORD, _type_name(entry))
                continue
            self.put(message_id, line, entry)

    def messages(self) -> list[tuple[str, JsonLine, Mapping[str, Any]]]:
        """Return the surviving messages, in order, as `(id, line, record)`."""
        return [(message_id, *self._records[message_id]) for message_id in self._order]


def _is_metadata_record(record: Mapping[str, Any]) -> bool:
    """Return whether the record is the one carrying the session's metadata."""
    return isinstance(record.get(FIELD_SESSION), str) and isinstance(record.get(FIELD_PROJECT), str)


def _session_id(builder: SessionBuilder, holder: Mapping[str, Any], current: Any) -> Any:
    """Return the session identifier, keeping the first one the file names.

    Args:
        builder: The session being assembled, for the note.
        holder: A record, or the `$set` object of one — a resumed session
            records its identifier again as a metadata update.
        current: The identifier found so far, or `None`.

    Returns:
        `current` once it is set, else what this record names. A second,
        different identifier is noted; the note carries no identifier itself,
        so the diagnostic stays the same line however many records disagree.
    """
    named = holder.get(FIELD_SESSION)
    if current is None:
        return named
    if named is not None and named != current:
        builder.note(NOTE_DROPPED_BLOCK, SECOND_SESSION)
    return current


def _record_id(builder: SessionBuilder, record: Mapping[str, Any]) -> str | None:
    """Return the record's message identifier, or `None` when it has none usable.

    Args:
        builder: The session being assembled, for the note.
        record: One transcript record.

    Returns:
        The record's `id`, which is what the log keys its messages by. `None`
        when there is no usable one — and then an `id` that was *present* is
        noted by its shape, because "the record carried no identifier" and "I
        could not read the identifier it carried" are the two cases the notes
        exist to keep apart. No identifier is invented: the agent drops such a
        record when it resumes the session, so a canonical message under a
        made-up `line-<n>` would be a turn the agent itself no longer has.
    """
    value = record.get("id")
    if isinstance(value, str) and value:
        return value
    if value is not None:
        detail = "<id: empty string>" if value == "" else f"<id: {_shape(value)}>"
        builder.note(NOTE_DROPPED_BLOCK, detail)
    return None


def _type_name(holder: Mapping[str, Any]) -> str:
    """Return the `type` of a record as the notes name it.

    Args:
        holder: One transcript record.

    Returns:
        The `type` field as a string, or `NO_TYPE` when it names none and
        `NULL_TYPE` when it names `null`. A type that is present and not a
        string keeps its own spelling (`7`).
    """
    if "type" not in holder:
        return NO_TYPE
    kind = holder["type"]
    return NULL_TYPE if kind is None else str(kind)


def _shape(value: Any) -> str:
    """Return the JSON shape of a value, to name something that has no kind.

    Args:
        value: The value a note is about.

    Returns:
        Its JSON type name — `object`, `array`, `string`, `number`, `boolean`,
        `null`. A note about a part names the part's kind; a value that is not a
        part has none, so its shape is the only honest thing to report.
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


def _part_name(part: Mapping[str, Any]) -> str:
    """Return the kind of a content part, as the notes name it.

    Args:
        part: One entry of a message's content list.

    Returns:
        The key that makes the part what it is — a part carries no `type` field.
        A thought is named by its flag rather than by the `text` it also holds,
        so that a dropped thought is not reported as dropped text; a part with
        none of the known keys is `NO_PART_KIND`.
    """
    if part.get(PART_THOUGHT) is True:
        return PART_THOUGHT
    for kind in PART_KINDS:
        if kind in part:
            return kind
    return NO_PART_KIND


def _timestamp(builder: SessionBuilder, line: JsonLine, value: Any) -> datetime | None:
    """Return a recorded instant as a timezone-aware one, or `None`."""
    return parse_timestamp(value, source=f"{builder.origin}line {line.number}")


def _parts(builder: SessionBuilder, content: Any, *, prefix: str = "") -> list[Mapping[str, Any]]:
    """Return the content parts of a message, noting the entries that are none.

    Args:
        builder: The session being assembled, for the notes.
        content: A message's content; anything but a list holds no parts.
        prefix: What to put in front of a note's detail.

    Returns:
        The entries of the list that are objects, in order. An entry that is not
        one cannot be a content part and is dropped — but noted by its shape,
        because a filter that drops a value without a trace is how a measurement
        tool loses data.
    """
    if not isinstance(content, list):
        return []
    parts: list[Mapping[str, Any]] = []
    for entry in content:
        if isinstance(entry, Mapping):
            parts.append(entry)
        else:
            builder.note(NOTE_DROPPED_BLOCK, f"{prefix}<entry: {_shape(entry)}>")
    return parts


def _note_unreadable_content(builder: SessionBuilder, content: Any, *, prefix: str = "") -> None:
    """Note a content value that is neither text nor a part list.

    An absent content (`None`) is not noted: the record carried nothing, so
    nothing was lost. Anything else — a number, an object — did carry something
    this importer cannot read, and the note is what keeps that from being a
    silent loss in a tool whose output is a measurement.
    """
    if content is not None and not isinstance(content, str | list):
        builder.note(NOTE_DROPPED_BLOCK, f"{prefix}<content: {_shape(content)}>")


def _part_text(builder: SessionBuilder, part: Mapping[str, Any], *, prefix: str = "") -> str:
    """Return the text a `text` part carries, noting one that is not text.

    Returns:
        The part's `text` when it is a string — the empty one included, since
        the part did carry it — and the empty string otherwise. A `text` that is
        not a string is dropped and noted by its shape: coercing it would turn
        `0` and `[]` into `""` unnoticed and an object into a Python repr that
        then *is* the canonical content. Inventing content is worse than losing
        it.
    """
    value = part.get(PART_TEXT)
    if isinstance(value, str):
        return value
    if value is not None:
        builder.note(NOTE_DROPPED_BLOCK, f"{prefix}{PART_TEXT}/<text: {_shape(value)}>")
    return ""


def _text_content(builder: SessionBuilder, content: Any, *, prefix: str = "") -> str:
    """Return a content value as text, noting every part it drops.

    Args:
        builder: The session being assembled, for the notes.
        content: A message's content: a string, or a list of parts.
        prefix: What to put in front of a note's detail.

    Returns:
        The text: a string as it is, a part list joined from its `text` parts.
        Every other part is noted as dropped rather than vanishing — the schema
        has no field for a thought or an image, and a reader must still be able
        to see that the transcript held one.
    """
    if isinstance(content, str):
        return content
    _note_unreadable_content(builder, content, prefix=prefix)
    texts: list[str] = []
    for part in _parts(builder, content, prefix=prefix):
        if _part_name(part) == PART_TEXT:
            texts.append(_part_text(builder, part, prefix=prefix))
        else:
            builder.note(NOTE_DROPPED_BLOCK, f"{prefix}{_part_name(part)}")
    return joined_text(texts)


def _add_user_record(
    builder: SessionBuilder, line: JsonLine, record: Mapping[str, Any], message_id: str
) -> bool:
    """Import a `user` record: a prompt, injected text, results, a summary.

    Args:
        builder: The session being assembled.
        line: The record and its line number.
        record: The record itself, which may come from an embedded `messages`
            array rather than from the line directly.
        message_id: The record's identifier, which the messages derive theirs
            from.

    Returns:
        Whether the record yielded at least one canonical message. A record
        whose content is neither text nor a part list yields none and is noted
        as skipped, rather than becoming an empty user turn the user never took.
        A record whose content *is* a part list always yields one, because the
        turn happened whatever its parts held.

    Raises:
        TranscriptImportError: If a `functionResponse` part names no call, which
            would leave the result unattachable to any call.
    """
    content = record.get("content")
    timestamp = _timestamp(builder, line, record.get("timestamp"))
    if isinstance(content, str):
        text, text_index, found = content, 0, []
    else:
        _note_unreadable_content(builder, content)
        text, text_index, found = _user_parts(builder, line, content)
    if SNAPSHOT_MARKER in text:
        builder.set_provenance(SessionSource.NATIVE_COMPACTION)
        builder.add_message(
            message_id=message_id,
            role=Role.SYSTEM,
            content=text,
            timestamp=timestamp,
            is_system_injected=True,
        )
        return True
    if not isinstance(content, str | list):
        return False
    if text or not found:
        found.append(_Part(text_index if text_index is not None else 0, Role.USER, text, None))
    found.sort(key=lambda part: part.index)
    only = len(found) == 1
    for part in found:
        builder.add_message(
            message_id=block_message_id(message_id, part.index, only=only),
            role=part.role,
            content=part.content,
            timestamp=timestamp,
            is_system_injected=part.role is Role.USER
            and part.content.startswith(ENVIRONMENT_MARKER),
            tool_result=part.result,
        )
    return True


def _user_parts(
    builder: SessionBuilder, line: JsonLine, content: Any
) -> tuple[str, int | None, list[_Part]]:
    """Split a `user` record's content into its text and the results it reports.

    Args:
        builder: The session being assembled, for the notes.
        line: The record and its line number.
        content: The record's content.

    Returns:
        The joined text of every `text` part, the position of the first of them,
        and one part per `functionResponse`. All other parts are noted as
        dropped — a `functionCall` among them included: a call belongs to the
        model's turn, and importing one from a user record would attribute it to
        the user.

    Raises:
        TranscriptImportError: If a `functionResponse` part names no call.
    """
    texts: list[str] = []
    text_index: int | None = None
    found: list[_Part] = []
    for index, part in enumerate(_parts(builder, content)):
        kind = _part_name(part)
        if kind == PART_TEXT:
            texts.append(_part_text(builder, part))
            text_index = index if text_index is None else text_index
        elif kind == PART_FUNCTION_RESPONSE:
            found.append(_Part(index, Role.TOOL, "", _result_part(builder, line, part)))
        else:
            builder.note(NOTE_DROPPED_BLOCK, kind)
    return joined_text(texts), text_index, found


def _result_part(builder: SessionBuilder, line: JsonLine, part: Mapping[str, Any]) -> ToolResult:
    """Build the canonical result a `functionResponse` part reports.

    Args:
        builder: The session being assembled, for the notes and the error.
        line: The record the part belongs to.
        part: The part, whose `functionResponse` names the call it answers.

    Returns:
        The canonical result, with the output the response carries and the exit
        status a shell result reports (see `_exit_code`). `is_error` comes from
        the response alone here: a result the agent recorded on a user turn
        carries no status.

    Raises:
        TranscriptImportError: If the part names no call. The reference is the
            only link between a result and its call, so a result without one is
            a damaged record, not a truncated transcript.
    """
    response = part.get(PART_FUNCTION_RESPONSE)
    if not isinstance(response, Mapping):
        raise builder.fail(
            f"a {PART_FUNCTION_RESPONSE} part is a {_shape(response)}, not an object",
            line=line.number,
        )
    call_id = response.get("id")
    if not isinstance(call_id, str) or not call_id:
        raise builder.fail(f"a {PART_FUNCTION_RESPONSE} part carries no call id", line=line.number)
    name = response.get("name")
    output, failed = _response_output(builder, response)
    return ToolResult(
        call_id=call_id,
        output=output,
        exit_code=_exit_code(name, output),
        is_error=failed,
    )


def _response_output(builder: SessionBuilder, response: Mapping[str, Any]) -> tuple[str, bool]:
    """Return what a `functionResponse` reports, and whether it is a failure.

    Args:
        builder: The session being assembled, for the notes.
        response: The `functionResponse` object of a part.

    Returns:
        The response's `output`, or its `error` together with `True`. A response
        that carries neither, and is not empty, is noted by its shape and yields
        no output: a structured response — what an MCP tool returns — would
        otherwise become a Python repr in canonical content. An empty response
        is what a tool that reports nothing writes, and is not noted.
    """
    payload = response.get("response")
    if not isinstance(payload, Mapping):
        if payload is not None:
            builder.note(
                NOTE_DROPPED_BLOCK,
                f"{PART_FUNCTION_RESPONSE}/<response: {_shape(payload)}>",
            )
        return "", False
    for key, failure in (("output", False), ("error", True)):
        reported = payload.get(key)
        if isinstance(reported, str):
            return reported, failure
        if reported is not None:
            builder.note(
                NOTE_DROPPED_BLOCK, f"{PART_FUNCTION_RESPONSE}/<{key}: {_shape(reported)}>"
            )
            return "", failure
    if payload:
        builder.note(NOTE_DROPPED_BLOCK, f"{PART_FUNCTION_RESPONSE}/<response: object>")
    return "", False


def _exit_code(name: Any, output: str) -> int | None:
    """Return the exit status a shell result reports, or `None`.

    Args:
        name: The tool the result belongs to; only the shell tool reports an
            exit status.
        output: The result's output.

    Returns:
        The status, read from the `Exit Code: <n>` line the shell tool appends
        — and only from the trailing block of its own format, which starts after
        the output and holds nothing but the fields in `SHELL_TRAILING_FIELDS`.
        A command that *prints* such a line proves nothing about its own exit
        status, and a false reading here would make a failed verification look
        like a passing one in T6.

        `None` when there is none, which for Gemini CLI is every command that
        succeeded: the tool documents the field as "only included if non-zero",
        so `exit_code` is `null` exactly when the tool reports no status — which
        is what the canonical schema's `null` means. `is_error` stays what the
        agent recorded (the *call* succeeded), so the two fields answer two
        different questions.
    """
    if name != SHELL_TOOL or not output.startswith(SHELL_OUTPUT_PREFIX):
        return None
    for raw in reversed(output.split("\n")):
        if raw.startswith(EXIT_CODE_PREFIX):
            try:
                return int(raw.removeprefix(EXIT_CODE_PREFIX).strip())
            except ValueError:
                return None
        if not raw.startswith(SHELL_TRAILING_FIELDS):
            return None
    return None


def _add_model_record(
    builder: SessionBuilder, line: JsonLine, record: Mapping[str, Any], message_id: str
) -> bool:
    """Import a `gemini` record as one message with all its calls and results.

    Args:
        builder: The session being assembled.
        line: The record and its line number.
        record: The record itself.
        message_id: The record's identifier.

    Returns:
        `True`: a model record is always a turn, even when it carries neither
        text nor a call, because the agent did take that turn.

    Raises:
        TranscriptImportError: If a recorded call names no identifier or no
            tool.
    """
    builder.set_model(record.get("model"))
    if record.get(FIELD_THOUGHTS):
        builder.note(NOTE_DROPPED_BLOCK, FIELD_THOUGHTS)
    timestamp = _timestamp(builder, line, record.get("timestamp"))
    calls: list[ToolCall] = []
    results: list[_Part] = []
    for index, entry in enumerate(_call_entries(builder, record), start=1):
        call, result, completed = _tool_call(builder, line, entry)
        calls.append(call)
        if result is not None:
            results.append(_Part(index, Role.TOOL, "", result, completed))
    content = record.get("content")
    text = content if isinstance(content, str) else _model_text(builder, content, calls)
    builder.add_message(
        message_id=block_message_id(message_id, 0, only=not results),
        role=Role.ASSISTANT,
        content=text,
        timestamp=timestamp,
        tool_calls=calls,
    )
    for part in results:
        builder.add_message(
            message_id=block_message_id(message_id, part.index, only=False),
            role=Role.TOOL,
            content="",
            timestamp=part.timestamp or timestamp,
            tool_result=part.result,
        )
    return True


def _model_text(builder: SessionBuilder, content: Any, calls: list[ToolCall]) -> str:
    """Return a model record's text, reading the calls its parts still hold.

    Args:
        builder: The session being assembled, for the notes.
        content: The record's content as a list of parts.
        calls: The calls read from the record's `toolCalls`, extended in place
            with every `functionCall` part that is not already among them.

    Returns:
        The joined text of the record's `text` parts. A thought and anything
        else the schema has no field for is noted as dropped; a `functionCall`
        part repeating a recorded call is neither imported nor noted — nothing
        was lost, the call is already there.
    """
    _note_unreadable_content(builder, content)
    known = {call.id for call in calls}
    texts: list[str] = []
    for part in _parts(builder, content):
        kind = _part_name(part)
        if kind == PART_TEXT:
            texts.append(_part_text(builder, part))
        elif kind == PART_FUNCTION_CALL:
            call = _function_call_part(builder, part, known)
            if call is not None:
                calls.append(call)
                known.add(call.id)
        else:
            builder.note(NOTE_DROPPED_BLOCK, kind)
    return joined_text(texts)


def _function_call_part(
    builder: SessionBuilder, part: Mapping[str, Any], known: set[str]
) -> ToolCall | None:
    """Build the call a `functionCall` part made, unless it is already known.

    Args:
        builder: The session being assembled, which resolves the category.
        part: The part, whose `functionCall` names the call.
        known: Identifiers of the calls already read from `toolCalls`.

    Returns:
        The canonical call, or `None` when the part repeats a recorded call or
        names no identifier or tool. A part is the *weaker* record of a call —
        it carries no status and no result — so a malformed one is noted and
        dropped rather than fatal, which is the opposite of a `toolCalls` entry
        whose identifier is what a result references.
    """
    call = part.get(PART_FUNCTION_CALL)
    if not isinstance(call, Mapping):
        builder.note(NOTE_DROPPED_BLOCK, f"{PART_FUNCTION_CALL}/<call: {_shape(call)}>")
        return None
    call_id = call.get("id")
    name = call.get("name")
    if not isinstance(call_id, str) or not call_id or not isinstance(name, str) or not name:
        builder.note(NOTE_DROPPED_BLOCK, f"{PART_FUNCTION_CALL}/<no id or name>")
        return None
    if call_id in known:
        return None
    args = call.get("args")
    if not isinstance(args, Mapping):
        if args is not None:
            builder.note(NOTE_DROPPED_BLOCK, f"{PART_FUNCTION_CALL}/{name}/<args: {_shape(args)}>")
        args = {}
    return builder.tool_call(name, call_id=call_id, args=dict(args))


def _call_entries(builder: SessionBuilder, record: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Return the `toolCalls` entries of a model record, noting the rest.

    Args:
        builder: The session being assembled, for the notes.
        record: One `gemini` record.

    Returns:
        The entries that are objects, in order. A `toolCalls` that is present
        but not a list, and an entry that is not an object, are noted by their
        shape: a lost call is a lost solution step.
    """
    entries = record.get(FIELD_TOOL_CALLS)
    if entries is None:
        return []
    if not isinstance(entries, list):
        builder.note(NOTE_DROPPED_BLOCK, f"<{FIELD_TOOL_CALLS}: {_shape(entries)}>")
        return []
    calls: list[Mapping[str, Any]] = []
    for entry in entries:
        if isinstance(entry, Mapping):
            calls.append(entry)
        else:
            builder.note(NOTE_DROPPED_BLOCK, f"{FIELD_TOOL_CALLS}/<entry: {_shape(entry)}>")
    return calls


def _tool_call(
    builder: SessionBuilder, line: JsonLine, entry: Mapping[str, Any]
) -> tuple[ToolCall, ToolResult | None, datetime | None]:
    """Build the canonical call and result one `toolCalls` entry holds.

    Args:
        builder: The session being assembled, which resolves the category.
        line: The record the entry belongs to.
        entry: The `toolCalls` entry.

    Returns:
        The call, the result it already has — `None` while it is still running,
        was never approved, or the turn was interrupted — and the instant the
        entry records for itself, which is when the call finished rather than
        when the turn started.

    Raises:
        TranscriptImportError: If the entry names no identifier or no tool. The
            identifier is what the result references and the name is what the
            category is resolved from, so neither can be guessed at.
    """
    call_id = entry.get("id")
    name = entry.get("name")
    if not isinstance(call_id, str) or not call_id:
        raise builder.fail(f"a {FIELD_TOOL_CALLS} entry carries no id", line=line.number)
    if not isinstance(name, str) or not name:
        raise builder.fail(f"the tool call {call_id!r} carries no tool name", line=line.number)
    arguments = entry.get("args")
    if isinstance(arguments, Mapping):
        args = dict(arguments)
    else:
        args = {}
        if arguments is not None:
            builder.note(
                NOTE_DROPPED_BLOCK,
                f"{FIELD_TOOL_CALLS}/{name}/<args: {_shape(arguments)}>",
            )
    call = builder.tool_call(name, call_id=call_id, args=args)
    return (
        call,
        _tool_result(builder, line, entry, call_id=call_id, name=name),
        _timestamp(builder, line, entry.get("timestamp")),
    )


def _tool_result(
    builder: SessionBuilder,
    line: JsonLine,
    entry: Mapping[str, Any],
    *,
    call_id: str,
    name: str,
) -> ToolResult | None:
    """Build the result a `toolCalls` entry carries, or `None` when it has none.

    Args:
        builder: The session being assembled, for the notes.
        line: The record the entry belongs to.
        entry: The `toolCalls` entry.
        call_id: The call the result answers.
        name: The tool that produced it.

    Returns:
        The canonical result, or `None` when the entry's `result` is absent,
        `null` or empty — the call has not come back. Its output is what the
        response parts carry; a part that is not one is noted and dropped.
    """
    result = entry.get("result")
    if result is None:
        return None
    if not isinstance(result, list):
        builder.note(NOTE_DROPPED_BLOCK, f"{FIELD_TOOL_CALLS}/{name}/<result: {_shape(result)}>")
        return None
    parts = _parts(builder, result, prefix=f"{FIELD_TOOL_CALLS}/")
    if not parts:
        return None
    texts: list[str] = []
    failed = False
    for part in parts:
        kind = _part_name(part)
        if kind == PART_FUNCTION_RESPONSE:
            response = part[PART_FUNCTION_RESPONSE]
            if not isinstance(response, Mapping):
                builder.note(
                    NOTE_DROPPED_BLOCK,
                    f"{PART_FUNCTION_RESPONSE}/<response: {_shape(response)}>",
                )
                continue
            output, response_failed = _response_output(builder, response)
            texts.append(output)
            failed = failed or response_failed
        elif kind == PART_TEXT:
            texts.append(_part_text(builder, part, prefix=f"{FIELD_TOOL_CALLS}/"))
        else:
            builder.note(NOTE_DROPPED_BLOCK, f"{FIELD_TOOL_CALLS}/{kind}")
    output = joined_text(texts)
    return ToolResult(
        call_id=call_id,
        output=output,
        exit_code=_exit_code(name, output),
        is_error=failed or _is_error(builder, entry.get("status")),
    )


def _is_error(builder: SessionBuilder, status: Any) -> bool:
    """Return whether the agent recorded this call as a failure.

    Args:
        builder: The session being assembled, for the notes.
        status: The entry's `status`.

    Returns:
        Whether the status is anything other than `success`. A status that is
        absent or `null` is not a failure — a result the agent recorded on a
        user turn carries none. A status that is present but not a string is
        **taken as an error** and noted by its shape: the canonical schema has
        no third state, T6 resolves the last *passing* verification from this
        field, and the expensive direction is the wrong `False` — it would make
        a call the importer cannot certify as successful part of the accepted
        solution.
    """
    if isinstance(status, str):
        return status != SUCCESS_STATUS
    if status is not None:
        builder.note(NOTE_DROPPED_BLOCK, f"{FIELD_TOOL_CALLS}/<status: {_shape(status)}>")
        return True
    return False


def _add_notice_record(
    builder: SessionBuilder, line: JsonLine, record: Mapping[str, Any], message_id: str
) -> bool:
    """Import an `info` or `error` record, a message only when it has content.

    Args:
        builder: The session being assembled.
        line: The record and its line number.
        record: The record itself.
        message_id: The record's identifier.

    Returns:
        Whether the record carried text. Gemini CLI records its own notices
        under these two types — a model switch, a failed edit — and they are
        the harness talking, not a turn either side took; one without text is
        skipped like any other record without a message.
    """
    content = _text_content(builder, record.get("content"))
    if not content:
        return False
    builder.add_message(
        message_id=message_id,
        role=Role.SYSTEM,
        content=content,
        timestamp=_timestamp(builder, line, record.get("timestamp")),
        is_system_injected=True,
    )
    return True


Reader = Callable[[SessionBuilder, JsonLine, Mapping[str, Any], str], bool]
"""Signature of a record reader: it returns whether the record held a message."""

READERS: dict[str, Reader] = {
    RECORD_USER: _add_user_record,
    RECORD_MODEL: _add_model_record,
    RECORD_INFO: _add_notice_record,
    RECORD_ERROR: _add_notice_record,
}
"""The record types that can carry a message, and what reads them.

A reader returns whether the record yielded a canonical message; everything
else — a record type that is not in this table, and one that is but held no
message — is noted as skipped by its type, as `_type_name` spells it. That is
what keeps a new Gemini CLI record type a line in the import's notes instead of
a crash or a silent loss.
"""
