"""Writing the assembled run out in the target agent's own transcript format.

The second half of the renderer (T7.2, #32). T7.1 decided *what* the output
holds — one summary text plus the latest chunk's messages; this module decides
what that looks like on disk for Claude Code, Gemini CLI and OpenCode, and
writes the file. Every shape below is corroborated inside this repository, in the
importer that reads it back: `idc.importers.claude_code`,
`idc.importers.gemini_cli` and `idc.importers.opencode` each document the
agent's records from the agent's own source, and the constants here are theirs.

**One writer per agent, behind a registry.** `WRITERS` maps `Agent` to a
`NativeWriter`, the way `idc.importers.IMPORTERS` maps it to an importer — the
established per-agent dispatch of this project. `render_transcript` looks the
writer up by the block's own `rendered_for`, so a later stage that holds a block
already (the measurement pass, the CLI) resolves a writer instead of branching
on the agent itself, and an agent the table does not hold is a
`MissingWriterError` rather than a silent fallback to one of the three. The three
writers are plain functions: they hold no state, and a class hierarchy would
only add one.

**How a summary is marked, per agent, and what is deliberately not set.**

- **Claude Code** marks it with `isCompactSummary` on a `user` record, which is
  the flag its importer reads as the native summary (`SUMMARY_FLAG`). `isMeta`
  is **not** set: that flag marks a *user turn the harness wrote* and would make
  the record an injected user message instead — the two paths are different
  branches of `idc.importers.claude_code._add_user_record`, and the summary path
  already yields an injected `system` message.
- **Gemini CLI** has no flag. Its compression writes the summary back into the
  history as a user turn and recognises it again by the `<state_snapshot>` tag
  in the text (`SNAPSHOT_MARKER`), so the summary is a `user` record whose text
  part is the block **wrapped in that tag and otherwise untouched**: the block's
  own bytes are the tag's content, so unwrapping restores them exactly.
- **OpenCode** has a record type of its own, `compaction` (`RECORD_COMPACTION`),
  with the text in `summary`. Its `recent` field names the messages the
  compaction kept, so it is written with the identifiers of the records this
  renderer kept — which is what the field means. No `role` is set: the record is
  identified by its `type`.

**What of a kept message survives, and what cannot.** The messages are written
back as records, and the next import re-derives the canonical form from them.
The statements below are about a session an **importer** built, which is the only
input this pipeline has, and in such a session the role already settles the
injected flag: `system` is always injected (all three importers set it on the
record they read a harness notice or a native summary from), `assistant` and
`tool` never are, and `user` may be either. Preserved for all three agents: the
role, the content byte for byte, every tool call's id, name and arguments, every
result's output **in full** and its `is_error`, the `is_system_injected` flag,
and the instant of a timestamp. Not preserved, and none of it fudged:

- **`ToolCall.category`** is not written, because no native record carries it.
  The importer resolves it from `tool_registry.toml` for the *target* agent, so
  a cross-agent render can give the same call another category.
- **`ToolResult.exit_code`** is not written either. Claude Code and OpenCode
  record no process status at all (both importers document `exit_code` as always
  `null`), and Gemini CLI re-derives one from the result's own output text — so
  it survives there exactly when the output still carries Gemini's own
  `Exit Code:` block, and is `null` otherwise.
- **`Message.is_command`** has no field in any of the three formats. Claude Code
  derives it from the `<command-name>` tags its harness wraps a command in, and
  those are part of the message's *text*; adding them would change what the
  agent reads back, so the flag is lost for a command whose text does not
  already carry them.
- **The role of an injected message** moves where the format has only one shape
  for harness text: a Gemini CLI `info` notice (role `system`) and an OpenCode
  `synthetic` text part (role `user`). Keeping the flag is what that choice is
  for, because a harness turn that came back as a plain user turn would hand
  T3.1 an intent boundary the user never set. Only a `user` or a `system`
  message is written in the harness shape, those being the two roles an importer
  marks; an injected *assistant* message — which no importer produces — is
  written as the assistant turn it is and loses the flag, in all three formats,
  because none of them marks a model turn as the harness's.
- **A `tool` message's own `content`** is not written: no native record carries
  text beside a result, and every importer in this repository produces a tool
  message with empty content (visible in the three `.expected.json` files under
  `tests/native/`), so there is nothing to lose for a session that came from one.
- **For OpenCode only**, a result's *message* identifier and its position. The
  format stores a result in the `state` of the tool part that made the call, so
  the identifier is re-derived from that part's index (`block_message_id`) and a
  result that arrived after other messages comes back at its call's position.
  A result whose call is not in the chunk has no record at all and is refused
  (`UnanchoredToolResultError`) rather than written with an invented call.
- Sub-millisecond precision, for OpenCode, whose `time.created` is epoch
  milliseconds.

**Determinism.** No clock, no randomness, no network: the output is a function of
the block, the session and the chunks. The only time value that belongs to no
message is the summary record's own, and it is `Session.created_at` — the one
timestamp the canonical schema requires, which is why there is always one to
write. For every session an importer built it is also the earliest message
timestamp (`idc.importers.base.SessionBuilder.build` takes the `min`), so the
summary record is ordered in front of the messages it summarizes. A message
without a timestamp is written without one. Timestamps are written in UTC with a
trailing `Z`, the spelling both Claude Code's and Gemini CLI's own files use; the
instant is preserved, the offset's spelling is not.

**What the tests prove, and what they do not.** The three agents cannot be run
here, so the evidence is a golden file per agent plus a round trip through the
matching importer (`tests/test_render_native.py`). That shows the file is
byte-stable and that *this repository's* importer reads it back; it does **not**
show that the agent itself would resume from it. For OpenCode it shows less
still: the exported file is the line-delimited shape `idc.importers.opencode`
reads, not the per-message/per-part file tree OpenCode stores, so the fields a
part file needs — a part's own `id`, for one — are absent rather than invented.

**Two measured properties of this stage, verified rather than reasoned about.**

*Rendering for a **different** agent than the transcript came from collapses every
tool category to `other`, loudly.* `ToolCall.category` is re-derived by the
importer from the tool **name**, and tool names are agent-specific: a
`gemini-cli` transcript rendered as `claude-code` carries `read_file`,
`write_file`, `run_shell_command` and `replace`, none of which the Claude Code
registry declares. Measured on `write_then_two_edits`: rendered for its own
agent the categories come back `['read','write','write','exec',…]` unchanged,
with one import note; rendered cross-agent they come back all `other`, with one
note **per** unknown name. It is therefore never silent — but the exploration
rate is computed from those categories, so a cross-agent render is not a
transcript to measure.

*The Gemini summary's block has to have its tag stripped before
`idc.parse.parse_chains` will read it.* The block's own bytes sit verbatim
inside `SNAPSHOT_OPEN`/`SNAPSHOT_CLOSE` — both exported for exactly this — and
handing the raw message content to `parse_chains` refuses with
`StrayContentError` rather than mis-parsing. A loud refusal, so a consumer that
forgets cannot read a corrupted chain; the two constants are what it strips.

**Not in scope, and a true statement for the ticket that is (#102):** a file
written here re-imports with `source` = `native-compaction`, because the summary
is written in the agent's own compaction shape and that shape is the only
provenance signal an importer reads. `SessionSource.INTENT_COMPACTION` is
produced nowhere in `src/` today.
"""

import json
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Protocol

from idc.importers.claude_code import (
    BLOCK_TEXT,
    BLOCK_TOOL_RESULT,
    BLOCK_TOOL_USE,
    INJECTED_FLAG,
    RECORD_ASSISTANT,
    RECORD_SYSTEM,
    RECORD_USER,
    SUMMARY_FLAG,
)
from idc.importers.gemini_cli import SNAPSHOT_MARKER
from idc.importers.opencode import PART_TEXT, PART_TOOL, RECORD_COMPACTION, SYNTHETIC_FLAG
from idc.render.assembly import AssembledTranscript, assemble
from idc.render.errors import MissingWriterError, UnanchoredToolResultError
from idc.schema import Agent, Chunk, CompactedBlock, Message, Role, Session

__all__ = [
    "SNAPSHOT_CLOSE",
    "SNAPSHOT_OPEN",
    "SUMMARY_MESSAGE_ID",
    "WRITERS",
    "NativeWriter",
    "render_transcript",
    "write_transcript",
]

SUMMARY_MESSAGE_ID = "idc-summary"
"""Identifier of the summary record — the one identifier this stage invents.

The summary is no message of the input, so it has no identifier to keep. A
constant rather than a derived name, so that the output is byte-stable for a
given input; a kept message carrying the same id makes the re-import fail on the
canonical schema's duplicate-message-id rule instead of two records quietly
sharing one identity.
"""

SNAPSHOT_OPEN = SNAPSHOT_MARKER
"""Opening tag Gemini CLI's compression wraps its summary in."""

SNAPSHOT_CLOSE = "</state_snapshot>"
"""Closing tag of a Gemini CLI state snapshot."""

GEMINI_PROJECT_HASH = ""
"""What the metadata record names as `projectHash`.

Gemini CLI's own path layout puts the project hash in the *directory* a session
file lives in (`idc.importers.gemini_cli`), and no canonical field carries it, so
there is nothing to write and nothing worth inventing. The key itself cannot be
omitted: `idc.importers.gemini_cli` recognises the metadata record by `sessionId`
*and* `projectHash` being strings, and a record holding `messages` without both
is a named import error. Its value is read by nothing.
"""

EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
"""Epoch OpenCode's `time.created` counts milliseconds from."""


class NativeWriter(Protocol):
    """How one agent's transcript file is written from an assembled run."""

    def __call__(self, transcript: AssembledTranscript, *, session: Session) -> str:
        """Return the file's whole text for one agent.

        Args:
            transcript: The assembled run — the summary text and the messages of
                the chunk kept verbatim, from `idc.render.assemble`.
            session: The session the messages come from, for the identifier, the
                model and `created_at`.

        Returns:
            The transcript file's text, ending in a newline.
        """


def _instant(stamp: datetime) -> str:
    """Return a timestamp as Claude Code and Gemini CLI spell one: UTC, `Z`."""
    return stamp.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _millis(stamp: datetime) -> int:
    """Return a timestamp as OpenCode's `time.created`: whole epoch milliseconds."""
    return (stamp - EPOCH) // timedelta(milliseconds=1)


def _lines(records: Iterable[Mapping[str, Any]]) -> str:
    """Return records as JSON Lines: one compact object per line.

    Args:
        records: The records in the order they are written.

    Returns:
        One line per record, each a JSON object with no inserted whitespace, and
        a trailing newline. `ensure_ascii` is off, so non-ASCII text is written
        as the UTF-8 it already is instead of as escape sequences, which would
        change the file's bytes without changing a character of what it says.
    """
    return "".join(
        json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n" for record in records
    )


def _text_block(content: str) -> list[dict[str, Any]]:
    """Return the one-element text block list of a content, or none if it is empty."""
    return [{"type": BLOCK_TEXT, "text": content}] if content else []


def _claude_code(transcript: AssembledTranscript, *, session: Session) -> str:
    """Write the run as a Claude Code session JSONL.

    One record per canonical message, which is why every identifier survives: a
    result gets a record of its own holding exactly one `tool_result` block, and
    `idc.importers.base.block_message_id` leaves a record's only message under
    the record's own identifier.

    Args:
        transcript: The assembled run.
        session: The session the messages come from.

    Returns:
        The file's text, the summary record first.
    """
    records = [
        {
            "type": RECORD_USER,
            SUMMARY_FLAG: True,
            "message": {"role": RECORD_USER, "content": transcript.summary},
            "uuid": SUMMARY_MESSAGE_ID,
            "timestamp": _instant(session.created_at),
            "sessionId": session.id,
        }
    ]
    for message in transcript.messages:
        records.append(_claude_record(message, session=session))
    return _lines(records)


def _claude_record(message: Message, *, session: Session) -> dict[str, Any]:
    """Return the Claude Code record one canonical message is written as.

    Args:
        message: The message to write.
        session: The session it belongs to, for the identifier and the model.

    Returns:
        A `system` record for a system message, a `user` record holding one
        `tool_result` block for a result, an `assistant` record with a text block
        and one `tool_use` block per call, and a `user` record otherwise —
        carrying `isMeta` when the harness wrote it.
    """
    record: dict[str, Any]
    if message.role is Role.SYSTEM:
        record = {"type": RECORD_SYSTEM, "content": message.content}
    elif message.role is Role.TOOL and message.tool_result is not None:
        result = message.tool_result
        block: dict[str, Any] = {"type": BLOCK_TOOL_RESULT, "tool_use_id": result.call_id}
        if result.is_error:
            block["is_error"] = True
        block["content"] = result.output
        record = {"type": RECORD_USER, "message": {"role": RECORD_USER, "content": [block]}}
    elif message.role is Role.ASSISTANT:
        body: dict[str, Any] = {"role": RECORD_ASSISTANT}
        if session.model is not None:
            body["model"] = session.model
        body["content"] = _text_block(message.content) + [
            {"type": BLOCK_TOOL_USE, "id": call.id, "name": call.name, "input": call.args}
            for call in message.tool_calls
        ]
        record = {"type": RECORD_ASSISTANT, "message": body}
    else:
        record = {"type": RECORD_USER}
        if message.is_system_injected:
            record[INJECTED_FLAG] = True
        record["message"] = {"role": RECORD_USER, "content": message.content}
    record["uuid"] = message.id
    if message.timestamp is not None:
        record["timestamp"] = _instant(message.timestamp)
    record["sessionId"] = session.id
    return record


def _gemini_cli(transcript: AssembledTranscript, *, session: Session) -> str:
    """Write the run as a Gemini CLI chat recording, in its append-log shape.

    Gemini CLI reads two shapes back and this is the line-delimited one: the
    metadata record first, then one record per message. Every identifier survives
    because every canonical message gets a record that yields exactly one
    message — a call is written as a `functionCall` part of the model's turn and
    its result as a `functionResponse` part of a user turn, which is the content
    shape the model is actually sent, rather than as the `toolCalls` bookkeeping
    entry that would make one record yield several messages.

    Args:
        transcript: The assembled run.
        session: The session the messages come from.

    Returns:
        The file's text, the metadata record first and the summary second.
    """
    stamps = [session.created_at] + [
        message.timestamp for message in transcript.messages if message.timestamp is not None
    ]
    records: list[dict[str, Any]] = [
        {
            "sessionId": session.id,
            "projectHash": GEMINI_PROJECT_HASH,
            "startTime": _instant(session.created_at),
            "lastUpdated": _instant(max(stamps)),
        },
        {
            "id": SUMMARY_MESSAGE_ID,
            "timestamp": _instant(session.created_at),
            "type": "user",
            "content": [
                {"text": f"{SNAPSHOT_OPEN}\n{transcript.summary}{SNAPSHOT_CLOSE}"},
            ],
        },
    ]
    names = {call.id: call.name for message in transcript.messages for call in message.tool_calls}
    for message in transcript.messages:
        records.append(_gemini_record(message, session=session, names=names))
    return _lines(records)


def _gemini_record(
    message: Message, *, session: Session, names: Mapping[str, str]
) -> dict[str, Any]:
    """Return the Gemini CLI record one canonical message is written as.

    Args:
        message: The message to write.
        session: The session it belongs to, for the model.
        names: Tool name per call id, read off the chunk's own calls. A
            `functionResponse` names the tool it answers in Gemini CLI, and
            `ToolResult` does not carry one, so the name comes from the call —
            and is left out for a result whose call is not in the chunk.

    Returns:
        An `info` notice for a message the harness injected, a `user` record with
        one `functionResponse` part for a result, a `gemini` record with the
        text and one `functionCall` part per call, and a `user` record otherwise.
    """
    record: dict[str, Any] = {"id": message.id}
    if message.timestamp is not None:
        record["timestamp"] = _instant(message.timestamp)
    if message.role is Role.TOOL and message.tool_result is not None:
        result = message.tool_result
        response: dict[str, Any] = {"id": result.call_id}
        if result.call_id in names:
            response["name"] = names[result.call_id]
        response["response"] = {"error" if result.is_error else "output": result.output}
        record["type"] = "user"
        record["content"] = [{"functionResponse": response}]
    elif message.is_system_injected and message.role is not Role.ASSISTANT:
        record["type"] = "info"
        record["content"] = [{"text": message.content}]
    elif message.role is Role.ASSISTANT:
        record["type"] = "gemini"
        if session.model is not None:
            record["model"] = session.model
        record["content"] = [{"text": message.content}] if message.content else []
        record["content"] += [
            {"functionCall": {"id": call.id, "name": call.name, "args": call.args}}
            for call in message.tool_calls
        ]
    else:
        record["type"] = "user"
        record["content"] = [{"text": message.content}]
    return record


def _opencode(transcript: AssembledTranscript, *, session: Session) -> str:
    """Write the run as an OpenCode message stream, one message record per line.

    The one writer that regroups: OpenCode has no record for a tool result, so a
    result is written into the `state` of the tool part that made its call. That
    is why a result's canonical identifier is re-derived on import and why a
    result whose call is not in the chunk is refused — see this module's
    docstring.

    Args:
        transcript: The assembled run.
        session: The session the messages come from.

    Returns:
        The file's text, the `compaction` record first.

    Raises:
        UnanchoredToolResultError: If a result's call was issued by no message of
            the chunk.
    """
    records: list[dict[str, Any]] = []
    states: dict[str, dict[str, Any]] = {}
    for message in transcript.messages:
        if message.role is Role.TOOL and message.tool_result is not None:
            result = message.tool_result
            state = states.get(result.call_id)
            if state is None:
                raise UnanchoredToolResultError(message.id, result.call_id, Agent.OPENCODE)
            state["status"] = "error" if result.is_error else "completed"
            state["error" if result.is_error else "output"] = result.output
            continue
        records.append(_opencode_record(message, session=session, states=states))
    summary = {
        "id": SUMMARY_MESSAGE_ID,
        "sessionID": session.id,
        "type": RECORD_COMPACTION,
        "time": {"created": _millis(session.created_at)},
        "summary": transcript.summary,
        "recent": [record["id"] for record in records],
    }
    return _lines([summary, *records])


def _opencode_record(
    message: Message, *, session: Session, states: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """Return the OpenCode record one canonical message is written as.

    Args:
        message: The message to write.
        session: The session it belongs to, for the identifier and the model.
        states: Tool-part states by call id, extended in place with this
            record's own calls so that the result arriving later can be written
            into the call that made it. A call whose result never arrives keeps
            the `pending` status, which is how OpenCode records an interrupted
            turn and how its importer reads a call without a result.

    Returns:
        An `assistant` record with the text and one tool part per call, or a
        `user` record whose text part carries `synthetic` when the harness wrote
        the message — the shape OpenCode uses for text that is not the user's,
        and the only one it has for a message that is neither side's turn.
    """
    record: dict[str, Any] = {"id": message.id, "sessionID": session.id}
    parts: list[dict[str, Any]] = []
    if message.role is Role.ASSISTANT:
        record["role"] = "assistant"
        if session.model is not None:
            record["modelID"] = session.model
        if message.content:
            parts.append({"type": PART_TEXT, "text": message.content})
        for call in message.tool_calls:
            state: dict[str, Any] = {"status": "pending", "input": call.args}
            states[call.id] = state
            parts.append({"type": PART_TOOL, "callID": call.id, "tool": call.name, "state": state})
    else:
        record["role"] = "user"
        text: dict[str, Any] = {"type": PART_TEXT, "text": message.content}
        if message.is_system_injected:
            text[SYNTHETIC_FLAG] = True
        parts.append(text)
    if message.timestamp is not None:
        record["time"] = {"created": _millis(message.timestamp)}
    record["parts"] = parts
    return record


WRITERS: dict[Agent, NativeWriter] = {
    Agent.CLAUDE_CODE: _claude_code,
    Agent.GEMINI_CLI: _gemini_cli,
    Agent.OPENCODE: _opencode,
}
"""The writer of each agent's transcript format, by agent.

A writer is registered here and nowhere else, so adding one is a line in this
mapping plus its own function — the shape `idc.importers.IMPORTERS` has for the
reading side.
"""


def render_transcript(block: CompactedBlock, *, session: Session, chunks: Sequence[Chunk]) -> str:
    """Render a run as the target agent's own transcript file, and return its text.

    The stage's entry point. Deterministic in the strict sense: it reads no file,
    calls no model, asks no clock and draws no random number.

    Args:
        block: The run's compacted block. Its `rendered_for` names the agent
            whose format is written — not `session.agent`, so that a session
            imported from one agent can be rendered for another.
        session: The session the chunks range over, and the source of the
            session identifier, the model and `created_at`.
        chunks: The session's chunks, as `idc.chunk.chunk_session` returned them.

    Returns:
        The transcript file's whole text, ending in a newline.

    Raises:
        RenderError: If the run cannot be rendered — `MissingWriterError` when
            no writer is registered for `block.rendered_for`,
            `UnanchoredToolResultError` when an OpenCode result's call is not in
            the chunk, plus the refusals `idc.render.assemble` raises about the
            block itself.
        ValueError: If `chunks` does not belong to `session`; see
            `idc.render.latest_messages`.
    """
    writer = WRITERS.get(block.rendered_for)
    if writer is None:
        raise MissingWriterError(block.rendered_for, WRITERS)
    return writer(assemble(block, session=session, chunks=chunks), session=session)


def write_transcript(
    block: CompactedBlock,
    *,
    session: Session,
    chunks: Sequence[Chunk],
    path: str | Path,
) -> Path:
    """Render a run and write it to a file the target agent's importer reads.

    Args:
        block: The run's compacted block; `rendered_for` picks the format.
        session: The session the chunks range over.
        chunks: The session's chunks.
        path: Where to write. The file is written as UTF-8 with LF line
            endings, so the bytes are the rendered text on every platform.

    Returns:
        The path written, as a `Path`.

    Raises:
        RenderError: Whatever `render_transcript` refuses; nothing is written
            then.
        OSError: If the file cannot be written.
    """
    text = render_transcript(block, session=session, chunks=chunks)
    destination = Path(path)
    destination.write_text(text, encoding="utf-8", newline="\n")
    return destination
