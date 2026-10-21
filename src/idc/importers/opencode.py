"""OpenCode: a session's message records read into a canonical `Session`.

OpenCode stores a session as one record per **message** (`session/message/<id>.json`)
with that message's **parts** beside it (`session/part/<message id>/<part id>.json`) —
its documented storage model, which could not be checked against a real session here (see
the first of the known limitations below). This importer reads that stream as JSONL — one
message record per line, in transcript order, with the record's parts under `parts` —
which is the same line-delimited shape `idc.importers.jsonl` reads for Claude Code. This
module knows that shape and nothing
else: reading the file, resolving tool categories, logging unknown tools and validating
the result are `idc.importers.jsonl` and `idc.importers.base`.

Two things are different enough from Claude Code to say out loud, because every rule
below follows from them:

- **A record names a `role`, not a `type`** — `user` or `assistant`. The one exception is
  the summary OpenCode's own compaction writes, which names `type = compaction`; a
  record is therefore identified by its `type` when it has one and by its `role`
  otherwise (see `_kind_name`).
- **A tool call carries its own result**, in the same part: `state.input` is what the
  agent sent, `state.output` (or `state.error`) is what came back. There is no separate
  result record, so one assistant record yields the assistant message *and* one `tool`
  message per part whose state has reached an end.

**Which record becomes what.**

- `role = user` -> exactly one `user` message, carrying the text of its `text` parts.
  Always one, whatever the parts held: a record whose only part is a pasted `file` still
  imports as the turn it was, with empty content and a note for the dropped part. The
  turn happened, and it is an intent boundary T3.1 splits on — losing it would fabricate
  the chain structure. A user record cannot carry a tool result in OpenCode, so unlike
  Claude Code's there is no case in which it yields something else instead. The same
  reason makes this a **deliberate divergence from the Claude Code importer**, which
  skips a record whose content it cannot read at all while this one keeps the turn with
  empty content and a note: there a `user` record may be a prompt or a tool result, so an
  unreadable content leaves the record's kind unknown, whereas here `role = user` already
  settles it — so keeping the turn invents nothing, and dropping it would fabricate the
  chain structure for T3.1.
- `role = user` whose `text` parts are all marked `synthetic` -> the same message, marked
  `is_system_injected`: `synthetic` is how OpenCode records text the harness wrote rather
  than the user. A record mixing typed and synthetic text stays the user's turn, and a
  `synthetic` that is not a boolean counts as unmarked and is noted by its shape.
- `role = assistant` -> one `assistant` message with the text of its `text` parts and
  every call its `tool` parts made, followed by one `tool` message per call whose result
  has arrived.
- `type = compaction` -> an injected `system` message carrying the summary, and the
  session's `source` becomes `native-compaction`.
- anything else -> skipped, and noted by its kind; a record that names neither a `role`
  nor a `type` is noted as `<no kind>`, one whose `role` is `null` as `<null role>` and
  one whose `type` is `null` as `<null type>`. A dropped *part* is named the same way, by
  `type_name`.

Two rules decide the rest.

**Identifiers come from the transcript.** A message keeps the record's `id` and a tool
call the `callID` its result references. An assistant record also yields one message per
tool result, and the canonical schema holds one result per message, so those are derived
from the record's own identifier as `<id>#<part index>` (see `block_message_id`) while
the assistant message keeps the record's identifier unsuffixed — the record *is* that
message, and the results are extra messages derived from its parts. Nothing is
renumbered, so a canonical transcript can still be read against the file it came from.

**Nothing is repaired.** A transcript cut off at the front imports as the session it is,
beginning mid-turn, with no user turn invented in front of it; a call whose result never
arrived — a `pending` or `running` state, which is how an interrupted turn is recorded —
imports as a call without a result.

Known limitations, all of them deliberate:

- **The field names here rest on OpenCode's documented storage model, not on a transcript
  anyone could read in this repository** — OpenCode is not installed in this environment,
  so there was no session to check them against. That covers `sessionID`, `role`,
  `time.created` in epoch milliseconds, `parts`, the part types `text` / `tool` /
  `reasoning` / `step-start` / `file`, a tool part's `state{status,input,output,error}`,
  **and the six tool names T2.3 added to the `opencode` table** of
  `tool_registry.toml`; only the lower-case spellings `read` / `edit` / `bash` with the
  argument keys `filePath`, `oldString` / `newString` and `command`, the JSONL input
  shape and the `compaction` record are corroborated from inside the repository. Before
  the study records its OpenCode condition, **one real session has to be checked against
  `tests/native/opencode/golden_session.jsonl`**; a different spelling is confined by
  construction to this module and `tests/native/opencode/`.
- A `reasoning` part is dropped and noted, for the reason a Claude Code `thinking` block
  is: `Message.content` is the text a later turn reads back, the canonical schema has no
  field for reasoning, and folding it into plain text would make the renderer re-emit it
  as if the agent had said it. The bookkeeping parts (`step-start`, `step-finish`,
  `snapshot`, `patch`, …) are dropped and noted by the same path, so a part kind this
  importer does not know is a line in the notes rather than a silent loss.
- `ToolResult.exit_code` stays `null`: the shell result OpenCode records in a tool state
  is its output, and no spelling of a process status is established by any transcript in
  this repository. The resolver's verification check therefore reads `is_error`, as it
  does for Claude Code.
- An `error` state is imported as a failed result whose output is the error text.
  A state whose status is neither an end (`completed`, `error`) nor a start (`pending`,
  `running`) yields **no** result and is noted by that status: a future status this
  importer cannot read must not become a result that silently claims to have passed.
- The fields this importer never reads carry no note, as for every importer: `providerID`
  (the model identifier is `modelID`), `time.completed`, a part's own `id`, `messageID`
  and `sessionID`, a tool state's `title`, `metadata` and `time`, the token and cost
  bookkeeping of an assistant record, its `path`, and the `system` array — that one is
  the system *prompt*, repeated verbatim on every assistant record, not a message of the
  transcript. `recent` on a compaction record is a list of the message identifiers the
  compaction kept; those messages are records of the file themselves, so the list is a
  reference like any other — but an entry that is *not* a reference is noted, because
  then it carried something this importer did not read.
- `Message.is_command` is never set. Claude Code wraps a slash command in
  `<command-name>` tags, and no equivalent marker is established for OpenCode; inventing
  one would make the chunker drop real user turns.
- The native summary is recognised by `type = compaction` only. An older OpenCode that
  marked the summary differently would import as a `raw` session; that is a miss the
  notes make visible (the record is skipped and noted by its kind), not a wrong
  provenance.

A value this importer cannot read is dropped **and noted by its shape** — a `parts` that
is not a list (`<parts: object>`), an entry of it that is not an object
(`<entry: string>`), the text of a `text` part that is not a string
(`text/<text: object>`), a tool state that is not an object (`tool/bash/<state: array>`),
an `input` that is not an object (`tool/bash/<input: array>`), an `output` or `error` that
is not a string (`tool/read/<output: object>`), a `synthetic` flag that is not a boolean
(`text/<synthetic: string>`), an unusable `id` the `line-<n>` fallback
stands in for (`<id: number>`), a second, different `sessionID`
(`<sessionID: a second value>`). A field that is absent or `null` carried nothing to lose
and is not noted; a field that carried something unreadable always is.

```python
from idc.importers import import_opencode

imported = import_opencode("session.jsonl")
imported.session.source        # SessionSource.RAW
imported.unknown_tools         # ('imaginary_tool',)
```
"""

from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from idc.importers.base import (
    NOTE_DROPPED_BLOCK,
    NOTE_SKIPPED_RECORD,
    ImportedSession,
    SessionBuilder,
    block_message_id,
    joined_text,
    shape,
    type_name,
)
from idc.importers.jsonl import JsonLine, read_json_lines
from idc.importers.tool_registry import ToolRegistry
from idc.schema import Agent, Role, ToolCall, ToolResult

__all__ = ["import_opencode"]

ROLE_USER = "user"
ROLE_ASSISTANT = "assistant"

RECORD_COMPACTION = "compaction"
"""Record type of the summary OpenCode's own compaction writes."""

PART_TEXT = "text"
PART_TOOL = "tool"

SYNTHETIC_FLAG = "synthetic"
"""Field OpenCode sets on a text part the harness wrote, not the user."""

STATE_COMPLETED = "completed"
STATE_ERROR = "error"
STATE_PENDING = "pending"
STATE_RUNNING = "running"

NULL_ROLE = "<null role>"
"""How a record whose `role` is `null` is reported, for the same reason."""

NO_KIND = "<no kind>"
"""How a record that names neither a `role` nor a `type` is reported.

OpenCode identifies a message by its role and the compaction summary by its
type, so a record with neither is not a shape this importer can place at all.
"""

NO_STATUS = "<no status>"
"""How a tool state that names no `status` is reported in the notes."""

NULL_STATUS = "<null status>"
"""How a tool state whose `status` is `null` is reported in the notes."""

SECOND_SESSION = "<sessionID: a second value>"
"""How a record naming a session other than the first one is reported.

A session file is one session, so the identifier the first record names is the
one the canonical session keeps. A second, different one is a damaged or
concatenated file, and ignoring it silently would hide that; the note carries no
identifier itself, so the diagnostic stays the same line however many records
disagree.
"""


def import_opencode(path: str | Path, *, registry: ToolRegistry | None = None) -> ImportedSession:
    """Read an OpenCode session into a canonical session.

    Args:
        path: The session file, one message record per line, with that
            message's parts nested under `parts`.
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
            satisfy the canonical schema, e.g. because a `callID` occurs twice.
    """
    builder = SessionBuilder(Agent.OPENCODE, registry=registry, source=path)
    session_id: Any = None
    for line in read_json_lines(path):
        named = line.data.get("sessionID")
        if session_id is None:
            session_id = named
        elif named is not None and named != session_id:
            builder.note(NOTE_DROPPED_BLOCK, SECOND_SESSION)
        kind = _kind_name(line.data)
        reader = READERS.get(kind)
        if reader is None or not reader(builder, line):
            builder.note(NOTE_SKIPPED_RECORD, kind)
    return builder.build(session_id=session_id)


def _kind_name(record: Mapping[str, Any]) -> str:
    """Return what kind of record this is, as the dispatch and the notes name it.

    Args:
        record: One transcript record.

    Returns:
        Its `type` when it names one — that is how the compaction summary is
        spelled — and otherwise its `role`, which is how every message record
        is. A record with neither is `NO_KIND`; a `null` role is `NULL_ROLE`,
        for the reason `type_name` keeps `NULL_TYPE` apart from `NO_TYPE`. A
        role that is present and not a string keeps its own spelling.
    """
    if "type" in record:
        return type_name(record)
    if "role" not in record:
        return NO_KIND
    role = record["role"]
    return NULL_ROLE if role is None else str(role)


def _record_id(builder: SessionBuilder, line: JsonLine) -> str:
    """Return the record's own identifier, or one derived from its line.

    Args:
        builder: The session being assembled, for the notes.
        line: The record and its line number.

    Returns:
        The record's `id`. A record without one falls back to `line-<n>`, which
        is unique within the file and points at the line it came from — the
        canonical schema requires an identifier, and the alternative would be to
        renumber every message of the session.

        An `id` that is *present* but unusable — a number, the empty string —
        takes the same fallback and is noted by its shape. That is a different
        thing from a record that names none: the transcript did record an
        identifier, and a canonical message carrying `line-7` instead would
        otherwise be indistinguishable from one whose record had no `id`.
    """
    identifier = line.data.get("id")
    if isinstance(identifier, str) and identifier:
        return identifier
    if identifier is not None:
        detail = "<id: empty string>" if identifier == "" else f"<id: {shape(identifier)}>"
        builder.note(NOTE_DROPPED_BLOCK, detail)
    return f"line-{line.number}"


def _timestamp(builder: SessionBuilder, line: JsonLine) -> datetime | None:
    """Return when the record was created, as a timezone-aware instant.

    Args:
        builder: The session being assembled, for the error message.
        line: The record and its line number.

    Returns:
        `time.created` as an instant in UTC, or `None` when the record names no
        `time` or no `created` — `Message.timestamp` is optional, and the
        session's own start is the earliest instant the file does record.

    Raises:
        TranscriptImportError: If `time` is not an object, or `created` is
            neither absent nor a number. OpenCode records an instant as epoch
            milliseconds (`Date.now()`), so there is no ISO-8601 string to fall
            back on, and a timestamp this importer cannot read is not guessed
            at: it would silently reorder a transcript.
    """
    recorded = line.data.get("time")
    if recorded is None:
        return None
    if not isinstance(recorded, Mapping):
        raise builder.fail(
            f"the record's time is a {shape(recorded)}, not an object", line=line.number
        )
    return _instant(builder, recorded.get("created"), line=line.number)


def _instant(builder: SessionBuilder, value: Any, *, line: int) -> datetime | None:
    """Read epoch milliseconds as a timezone-aware instant.

    Args:
        builder: The session being assembled, for the error message.
        value: The millisecond value the transcript records, or `None`.
        line: Line number the value came from.

    Returns:
        The instant in UTC, to the millisecond. Sub-millisecond precision is not
        recorded by OpenCode and is therefore not read.

    Raises:
        TranscriptImportError: If the value is neither `None` nor a number — a
        boolean included, which Python would otherwise accept as one.
    """
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise builder.fail(f"cannot read the timestamp {value!r}", line=line)
    seconds, milliseconds = divmod(int(value), 1000)
    try:
        return datetime.fromtimestamp(seconds, tz=UTC).replace(microsecond=milliseconds * 1000)
    except (OSError, OverflowError, ValueError):
        raise builder.fail(f"cannot read the timestamp {value!r}", line=line) from None


def _parts(builder: SessionBuilder, record: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Return the record's parts, noting the entries that are none.

    Args:
        builder: The session being assembled, for the notes.
        record: One transcript record.

    Returns:
        The entries of `parts` that are objects, in order. A `parts` that is not
        a list, and an entry of it that is not an object, cannot hold a part and
        are dropped — but noted by their shape, because a filter that drops a
        value without a trace is how a measurement tool loses data. An absent
        `parts` carried nothing and is not noted.
    """
    parts = record.get("parts")
    if parts is None:
        return []
    if not isinstance(parts, list):
        builder.note(NOTE_DROPPED_BLOCK, f"<parts: {shape(parts)}>")
        return []
    kept: list[Mapping[str, Any]] = []
    for entry in parts:
        if isinstance(entry, Mapping):
            kept.append(entry)
        else:
            builder.note(NOTE_DROPPED_BLOCK, f"<entry: {shape(entry)}>")
    return kept


def _part_text(builder: SessionBuilder, part: Mapping[str, Any]) -> str:
    """Return the text a `text` part carries, noting one that is not text.

    Args:
        builder: The session being assembled, for the notes.
        part: A part whose `type` is `text`.

    Returns:
        The part's `text` when it is a string — the empty one included, since
        the part did carry it — and the empty string otherwise. A `text` that is
        not a string is dropped and noted by its shape: coercing it would turn
        `0` and `[]` into `""` unnoticed and an object into a Python repr that
        then *is* the canonical content. Inventing content is worse than losing
        it, and a part without a `text` field carried nothing to lose.
    """
    value = part.get("text")
    if isinstance(value, str):
        return value
    if value is not None:
        builder.note(NOTE_DROPPED_BLOCK, f"{PART_TEXT}/<text: {shape(value)}>")
    return ""


def _texts(builder: SessionBuilder, parts: list[Mapping[str, Any]]) -> list[str]:
    """Return the text of every `text` part, noting every other part as dropped.

    Args:
        builder: The session being assembled, for the notes.
        parts: The record's parts.

    Returns:
        The text of the `text` parts, in order. Every other part is noted by its
        kind rather than vanishing — the schema has no field for a `reasoning`
        part, and a reader must still be able to see that the transcript held
        one. Used by the two records that carry no tool call.
    """
    texts: list[str] = []
    for part in parts:
        if part.get("type") == PART_TEXT:
            texts.append(_part_text(builder, part))
        else:
            builder.note(NOTE_DROPPED_BLOCK, type_name(part))
    return texts


def _is_synthetic(builder: SessionBuilder, part: Mapping[str, Any]) -> bool:
    """Return whether one `text` part is marked as written by the harness.

    Args:
        builder: The session being assembled, for the notes.
        part: A part whose `type` is `text`.

    Returns:
        Whether `synthetic` is exactly `True`. A `synthetic` that is neither a
        boolean nor absent — `"yes"`, `1` — is **not** read as a mark: guessing
        that it meant `True` would hand the chunker a user turn that is not one.
        It is noted by its shape all the same, because a field that carried
        something unreadable is always noted; `false`, `null` and an absent
        field carried nothing to lose and are not.
    """
    flag = part.get(SYNTHETIC_FLAG)
    if isinstance(flag, bool):
        return flag
    if flag is not None:
        builder.note(NOTE_DROPPED_BLOCK, f"{PART_TEXT}/<{SYNTHETIC_FLAG}: {shape(flag)}>")
    return False


def _is_injected(builder: SessionBuilder, parts: list[Mapping[str, Any]]) -> bool:
    """Return whether a user record's text was written by the harness.

    Args:
        builder: The session being assembled, for the notes.
        parts: The record's parts.

    Returns:
        Whether the record carries at least one `text` part and every one of
        them is marked `synthetic`. A record mixing typed and synthetic text is
        the user's turn: the chunker must not drop a prompt because the harness
        appended a reminder to it. An unreadable `synthetic` counts as unmarked
        and is noted — see `_is_synthetic`.
    """
    flags = [_is_synthetic(builder, part) for part in parts if part.get("type") == PART_TEXT]
    return bool(flags) and all(flags)


def _add_user_record(builder: SessionBuilder, line: JsonLine) -> bool:
    """Import a `user` record as the one turn it is.

    Args:
        builder: The session being assembled.
        line: The record and its line number.

    Returns:
        `True`, always. A user record whose parts hold no usable text — a pasted
        file, no part at all — still imports with empty content and a note:
        the turn happened, and T3.1 splits chunks on exactly that boundary, so
        losing it would fabricate the chain structure.
    """
    parts = _parts(builder, line.data)
    builder.add_message(
        message_id=_record_id(builder, line),
        role=Role.USER,
        content=joined_text(_texts(builder, parts)),
        timestamp=_timestamp(builder, line),
        is_system_injected=_is_injected(builder, parts),
    )
    return True


def _add_assistant_record(builder: SessionBuilder, line: JsonLine) -> bool:
    """Import an `assistant` record as its message plus the results it holds.

    Args:
        builder: The session being assembled.
        line: The record and its line number.

    Returns:
        `True`: an assistant record is always a turn, even when it carries
        neither text nor a call, because the agent did take that turn.

    Raises:
        TranscriptImportError: If a `tool` part names no `callID` or no tool.
    """
    record = line.data
    builder.set_model(record.get("modelID"))
    timestamp = _timestamp(builder, line)
    record_id = _record_id(builder, line)
    texts: list[str] = []
    calls: list[ToolCall] = []
    results: list[tuple[int, ToolResult]] = []
    for index, part in enumerate(_parts(builder, record)):
        kind = part.get("type")
        if kind == PART_TEXT:
            texts.append(_part_text(builder, part))
        elif kind == PART_TOOL:
            call, result = _tool_part(builder, line, part)
            calls.append(call)
            if result is not None:
                results.append((index, result))
        else:
            builder.note(NOTE_DROPPED_BLOCK, type_name(part))
    builder.add_message(
        message_id=record_id,
        role=Role.ASSISTANT,
        content=joined_text(texts),
        timestamp=timestamp,
        tool_calls=calls,
    )
    for index, result in results:
        builder.add_message(
            message_id=block_message_id(record_id, index, only=False),
            role=Role.TOOL,
            timestamp=timestamp,
            tool_result=result,
        )
    return True


def _add_compaction_record(builder: SessionBuilder, line: JsonLine) -> bool:
    """Import the summary OpenCode's own compaction wrote.

    Args:
        builder: The session being assembled.
        line: The record and its line number.

    Returns:
        `True`. The record always yields the injected `system` message that
        carries the summary, and the session's provenance becomes
        `native-compaction` — that provenance is what makes the agent's own
        compaction a measurable condition of the study, and it follows from the
        record existing, not from its text being readable.
    """
    record = line.data
    content = _summary(builder, record)
    _note_unread_references(builder, record.get("recent"))
    builder.add_native_summary(
        message_id=_record_id(builder, line),
        content=content,
        timestamp=_timestamp(builder, line),
    )
    return True


def _summary(builder: SessionBuilder, record: Mapping[str, Any]) -> str:
    """Return the text a compaction record carries as its summary.

    Args:
        builder: The session being assembled, for the notes.
        record: The `compaction` record.

    Returns:
        Its `summary` field, joined with the text of its parts when it carries
        parts as well — both are read, so that a summary stored as a field and
        one stored as the record's own text are each kept rather than one of
        them being dropped on the way. A `summary` that is present and not a
        string is noted by its shape and yields nothing, as every unreadable
        value does.
    """
    texts: list[str] = []
    summary = record.get("summary")
    if isinstance(summary, str):
        texts.append(summary)
    elif summary is not None:
        builder.note(NOTE_DROPPED_BLOCK, f"{RECORD_COMPACTION}/<summary: {shape(summary)}>")
    texts.extend(_texts(builder, _parts(builder, record)))
    return joined_text(texts)


def _note_unread_references(builder: SessionBuilder, recent: Any) -> None:
    """Note a `recent` field that holds more than message references.

    Args:
        builder: The session being assembled, for the notes.
        recent: The compaction record's `recent` field.

    `recent` names the messages the compaction kept verbatim, and those messages
    are records of the file themselves, so a list of identifiers is a reference
    like `parentUuid` is for Claude Code: nothing is lost by not reading it, and
    nothing is noted. An entry that is not an identifier did carry something this
    importer does not read, and that is noted by its shape.
    """
    if recent is None:
        return
    if not isinstance(recent, list):
        builder.note(NOTE_DROPPED_BLOCK, f"{RECORD_COMPACTION}/<recent: {shape(recent)}>")
        return
    for entry in recent:
        if not isinstance(entry, str):
            builder.note(NOTE_DROPPED_BLOCK, f"{RECORD_COMPACTION}/<recent entry: {shape(entry)}>")


def _tool_part(
    builder: SessionBuilder, line: JsonLine, part: Mapping[str, Any]
) -> tuple[ToolCall, ToolResult | None]:
    """Build the call a `tool` part made and the result it carries.

    Args:
        builder: The session being assembled, which resolves the category.
        line: The record the part belongs to.
        part: The `tool` part.

    Returns:
        The canonical call, with the identifier and the arguments the agent sent
        and the category the registry gives its name, and the result its state
        reports — `None` while the call has no result yet, which is how an
        interrupted turn is recorded.

    Raises:
        TranscriptImportError: If the part names no `callID` or no tool. The
            identifier is what the result references and the name is what the
            category is resolved from, so neither can be guessed at.
    """
    call_id = part.get("callID")
    name = part.get("tool")
    if not isinstance(call_id, str) or not call_id:
        raise builder.fail("a tool part carries no callID", line=line.number)
    if not isinstance(name, str) or not name:
        raise builder.fail(f"the tool part {call_id!r} names no tool", line=line.number)
    state = _state(builder, part, name)
    call = builder.tool_call(name, call_id=call_id, args=_arguments(builder, state, name))
    return call, _result(builder, state, call_id=call_id, name=name)


def _state(builder: SessionBuilder, part: Mapping[str, Any], name: str) -> Mapping[str, Any] | None:
    """Return a tool part's `state`, noting one that is not an object.

    Args:
        builder: The session being assembled, for the notes.
        part: The `tool` part.
        name: The tool the part called, for the note.

    Returns:
        The `state` object, or `None` when the part carries none. A `state` that
        is present but not an object is noted by its shape before `None` is
        returned: it holds both the call's arguments and its result, so treating
        it as absent without a note would lose the two things a tool part is
        read for.
    """
    state = part.get("state")
    if isinstance(state, Mapping):
        return state
    if state is not None:
        builder.note(NOTE_DROPPED_BLOCK, f"{PART_TOOL}/{name}/<state: {shape(state)}>")
    return None


def _arguments(
    builder: SessionBuilder, state: Mapping[str, Any] | None, name: str
) -> dict[str, Any]:
    """Return the arguments the call was made with, as the agent sent them.

    Args:
        builder: The session being assembled, for the notes.
        state: The tool part's state, or `None`.
        name: The tool the part called, for the note.

    Returns:
        `state.input` as a dictionary; empty when the state records none. An
        `input` that is not an object yields no arguments and is noted with its
        tool: for a `bash` call those arguments are the command T5.2 and T6
        resolve, so losing them silently would be expensive.
    """
    arguments = None if state is None else state.get("input")
    if isinstance(arguments, Mapping):
        return dict(arguments)
    if arguments is not None:
        builder.note(NOTE_DROPPED_BLOCK, f"{PART_TOOL}/{name}/<input: {shape(arguments)}>")
    return {}


def _result(
    builder: SessionBuilder, state: Mapping[str, Any] | None, *, call_id: str, name: str
) -> ToolResult | None:
    """Build the canonical result a tool state reports, if it reports one.

    Args:
        builder: The session being assembled, for the notes.
        state: The tool part's state, or `None`.
        call_id: The call the result answers.
        name: The tool that was called, for the notes.

    Returns:
        The result for a state that has reached an end: `completed` carries its
        `output`, `error` its `error` text and `is_error`. `None` for a state
        that has not — `pending` and `running` are how a call whose result never
        arrived is recorded, and the canonical schema does not require
        referential completeness precisely so that it can import as it is.
        `exit_code` stays `None`: OpenCode records no process status.

        A status this importer does not know yields no result either, and is
        noted by that status — an unreadable state must not become a result that
        claims to have passed, because T6 resolves the last *passing*
        verification command from `is_error`.
    """
    if state is None:
        return None
    status = state.get("status")
    if status == STATE_COMPLETED:
        return ToolResult(
            call_id=call_id,
            output=_output(builder, state.get("output"), name=name, field="output"),
        )
    if status == STATE_ERROR:
        return ToolResult(
            call_id=call_id,
            output=_output(builder, state.get("error"), name=name, field="error"),
            is_error=True,
        )
    if status not in (STATE_PENDING, STATE_RUNNING):
        builder.note(NOTE_DROPPED_BLOCK, f"{PART_TOOL}/{name}/{_status_detail(state)}")
    return None


def _status_detail(state: Mapping[str, Any]) -> str:
    """Return how a status this importer does not read is named in the notes.

    Args:
        state: The tool part's state.

    Returns:
        `<no status>` when the state names none and `<null status>` when it
        names `null` — the two are different problems, as for `type_name` —
        the status itself when it is a string, and its shape otherwise.
    """
    if "status" not in state:
        return NO_STATUS
    status = state["status"]
    if status is None:
        return NULL_STATUS
    return f"<status: {status}>" if isinstance(status, str) else f"<status: {shape(status)}>"


def _output(builder: SessionBuilder, value: Any, *, name: str, field: str) -> str:
    """Return a tool result's text, noting a value that is not text.

    Args:
        builder: The session being assembled, for the notes.
        value: The state's `output` or `error`.
        name: The tool that was called, for the note.
        field: Which of the two fields this is, for the note.

    Returns:
        The value when it is a string, and the empty string otherwise. A value
        that is not a string is dropped and noted by its shape rather than
        coerced into a Python repr that then *is* the canonical output.
    """
    if isinstance(value, str):
        return value
    if value is not None:
        builder.note(NOTE_DROPPED_BLOCK, f"{PART_TOOL}/{name}/<{field}: {shape(value)}>")
    return ""


READERS: dict[str, Callable[[SessionBuilder, JsonLine], bool]] = {
    ROLE_USER: _add_user_record,
    ROLE_ASSISTANT: _add_assistant_record,
    RECORD_COMPACTION: _add_compaction_record,
}
"""The record kinds that can carry a message, and what reads them.

Keyed by what `_kind_name` returns: the two roles a message record names, and
the type of the compaction summary. A reader returns whether the record yielded
a canonical message; everything else — a kind that is not in this table, and one
that is but held no message — is noted as skipped by that kind. That is what
keeps a new OpenCode record kind a line in the import's notes instead of a crash
or a silent loss.
"""
