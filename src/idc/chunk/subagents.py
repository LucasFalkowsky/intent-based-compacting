"""Lifting the work of a subagent into a chunk of its own.

A coding agent can delegate a job to a subagent — Claude Code's `Task` — and
what comes back is a report, while the searching, reading and failing that
produced it was done in a context of its own. The importer flattens that
sidechain into transcript order (#4), so in a canonical session the delegated
work is the **span of messages between the subagent call and its result**. This
stage lifts that span out of the chunk that spawned it, which is what lets the
compactor summarize a subagent's exploration separately from the turn it
served.

**Detection is registry data, not a code path.** A tool spawns a subagent when
the tool registry says so (`subagent = true` on Claude Code's `Task`, T2.1,
#6); the category stays `other`. This module therefore takes the predicate
rather than hard-coding a tool name: pass
`tool_registry().is_subagent_tool` — `IsSubagentTool` is exactly that
method's shape — and the set of subagent tools stays in the registry's TOML
where the other per-agent tool data lives.

**What becomes a span.** A subagent call whose result arrives later in the
transcript, taken outermost-first: the span runs from the message *after* the
call up to and including the message that reports the result, so the report —
which is the subagent's own summary of its work — is lifted with it. Four
situations yield no span, each because the transcript does not say where the
delegated work ends or begins:

- the result never arrives (a truncated transcript, or an interrupted run);
- the result sits *before* the call it names (the call it answers was cut off
  at the front of the transcript);
- the call sits inside a span that was already taken (a subagent that delegates
  again, or two sidechains that cross) — its work is part of the outer span;
- a real user turn sits inside the span (the user wrote while the subagent was
  running), because a user turn opens a chunk of its own and must not be
  swallowed by one.

**What that does to the chunk list.** A `Chunk` is one half-open range, and the
ranges of a chunked session must partition the transcript — contiguous, no gap,
no overlap (T3.1, and `reduce_chunks` raises unless they do). A span in the
middle of its chunk therefore cannot be cut out and leave *one* chunk behind:
the chunk that spawned the subagent is **split**, into the part up to the call,
the subagent chunk, and the part that follows the report. The subagent chunk
sits immediately after the chunk holding the call, as #14 requires, and that
chunk no longer holds any message of the span — the price is the third chunk,
which is the main agent's work *on* the report and belongs to the same user
turn:

```python
from idc.chunk import chunk_with_subagents

chunks = chunk_with_subagents(session, registry.is_subagent_tool)
#   K1  origin=user      the user turn and the message that delegated
#   K2  origin=subagent  the subagent's work and the report it returned
#   K3  origin=user      what the main agent did with the report
```

Three consequences are decisions of this module, not accidents:

- **`is_latest` marks the last chunk that is not subagent work.** It names the
  chunk the renderer appends verbatim instead of compacting, and a subagent
  chunk is lifted out precisely so that it *can* be compacted; so when a span
  ends the transcript, the latest chunk is the chunk that spawned it, not the
  structurally last one. It is still set exactly once (only a list of nothing
  but subagent chunks falls back to the last chunk).
- **Chunk ids are re-assigned by position**, `K<n>` from 1, because that is
  what a chunk id is (T3.1) — inserting a chunk renumbers the ones behind it.
- **A chunk that already holds subagent work is passed through untouched**,
  which makes the stage idempotent: running it on its own output changes
  nothing, because no span lies inside a user chunk any more.

Nothing here reads a clock, a model or the filesystem, and no message is
copied: the result is a new list of ranges over the same `Session.messages`.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from idc.chunk.boundaries import chunk_id, chunk_session, is_user_turn
from idc.schema import Agent, Chunk, ChunkOrigin, Session

IsSubagentTool = Callable[[Agent, str], bool]
"""Decides whether a tool of an agent spawns a subagent.

The shape of `ToolRegistry.is_subagent_tool` (#6), which is what production
passes: the marker is registry data, so this stage asks instead of knowing.
"""


@dataclass(frozen=True, slots=True)
class SubagentSpan:
    """The messages one subagent invocation produced, as a message range.

    Attributes:
        call_id: `ToolCall.id` of the call that spawned the subagent.
        tool_name: Name of the tool that spawned it, as the agent reports it
            (`Task` for Claude Code) — kept so that a later stage can say which
            delegation a chunk came from without re-reading the call.
        call_index: Index of the message that issued the call. It is *not* part
            of the span: the main agent wrote it, and it stays in the chunk the
            subagent chunk is placed behind.
        result_index: Index of the message that reports the result. It *is* part
            of the span, because that message carries the subagent's report.
    """

    call_id: str
    tool_name: str
    call_index: int
    result_index: int

    def __post_init__(self) -> None:
        """Reject a result that does not come after its call.

        Raises:
            ValueError: If `result_index` is not greater than `call_index`,
                which would make the span empty or reversed — a result before
                its call answers a call that was cut off at the front of the
                transcript and is not the end of this span.
        """
        if self.result_index <= self.call_index:
            raise ValueError(
                f"result_index ({self.result_index}) must be greater than "
                f"call_index ({self.call_index}) for the subagent call {self.call_id!r}"
            )

    @property
    def start_index(self) -> int:
        """Return the index of the span's first message, included."""
        return self.call_index + 1

    @property
    def end_index(self) -> int:
        """Return the index after the span's last message, excluded."""
        return self.result_index + 1


def subagent_spans(session: Session, is_subagent_tool: IsSubagentTool) -> list[SubagentSpan]:
    """Return the subagent spans of `session`, in transcript order.

    See the module docstring for what becomes a span and what does not. The
    returned spans never overlap and never cover a real user turn, so they can
    be cut out of a chunk list without breaking its partition.

    Args:
        session: The canonical session to scan.
        is_subagent_tool: Whether a tool of an agent spawns a subagent — pass
            `tool_registry().is_subagent_tool` (#6).

    Returns:
        One span per subagent invocation whose delegated work the transcript
        delimits; empty when the session holds none.
    """
    results = _result_indices(session)
    spans: list[SubagentSpan] = []
    taken = 0
    for index, message in enumerate(session.messages):
        for call in message.tool_calls:
            if not is_subagent_tool(session.agent, call.name):
                continue
            result_index = _first_after(results.get(call.id, ()), index)
            if result_index is None:
                # No result: where the delegated work ends is not in the
                # transcript, so there is nothing to delimit.
                continue
            span = SubagentSpan(
                call_id=call.id,
                tool_name=call.name,
                call_index=index,
                result_index=result_index,
            )
            if span.start_index < taken or _covers_a_user_turn(session, span):
                continue
            spans.append(span)
            taken = span.end_index
    return spans


def extract_subagents(
    session: Session,
    chunks: Sequence[Chunk],
    is_subagent_tool: IsSubagentTool,
) -> list[Chunk]:
    """Lift the subagent spans inside `chunks` into chunks of their own.

    Every span that lies completely inside one chunk of origin `user` splits
    that chunk: the part in front of the span, the span itself with
    `origin = subagent`, and the part behind it, in that order — see the module
    docstring for why the third part exists and what happens to the ids and to
    `is_latest`. A span that is not contained in a single chunk is left alone,
    because lifting it would merge two chunks into one range. Chunks that hold
    subagent work already are passed through.

    Args:
        session: The canonical session the chunks range over.
        chunks: The session's chunks in transcript order, as `chunk_session`
            returns them.
        is_subagent_tool: Whether a tool of an agent spawns a subagent — pass
            `tool_registry().is_subagent_tool` (#6).

    Returns:
        The chunks in transcript order, the subagent chunks inserted, numbered
        `K<n>` from 1. The ranges partition `session.messages`.

    Raises:
        ValueError: If `chunks` does not partition `session.messages` — the
            stage cuts a span out of a chunk, so a gap or an overlap in the
            input would silently become a gap or an overlap in the output.
    """
    _require_partition(session, chunks)
    spans = subagent_spans(session, is_subagent_tool)
    pieces = [piece for chunk in chunks for piece in _split(chunk, spans)]
    return _numbered(pieces)


def chunk_with_subagents(session: Session, is_subagent_tool: IsSubagentTool) -> list[Chunk]:
    """Chunk a canonical session and lift its subagent spans out of the chunks.

    The one-call form of the stage: `chunk_session` followed by
    `extract_subagents`.

    Args:
        session: The canonical session to chunk.
        is_subagent_tool: Whether a tool of an agent spawns a subagent — pass
            `tool_registry().is_subagent_tool` (#6).

    Returns:
        The chunks in transcript order, subagent work in chunks of its own.
    """
    return extract_subagents(session, chunk_session(session), is_subagent_tool)


def _split(chunk: Chunk, spans: Sequence[SubagentSpan]) -> list[tuple[int, int, ChunkOrigin]]:
    """Return the pieces `chunk` falls into once its spans are lifted out.

    Args:
        chunk: The chunk to split.
        spans: Every span of the session, in transcript order; the ones outside
            this chunk are ignored.

    Returns:
        One `(start_index, end_index, origin)` piece per chunk the input chunk
        becomes, in transcript order — the chunk itself when it holds no span.
        Every piece is non-empty, and together they cover the chunk exactly.
    """
    if chunk.origin is not ChunkOrigin.USER:
        # Already a chunk of subagent work: there is nothing to lift out of it,
        # and its own span would have an empty part in front of it.
        return [(chunk.start_index, chunk.end_index, chunk.origin)]
    pieces: list[tuple[int, int, ChunkOrigin]] = []
    cursor = chunk.start_index
    for span in spans:
        if span.start_index < chunk.start_index or span.end_index > chunk.end_index:
            continue
        if cursor < span.start_index:
            pieces.append((cursor, span.start_index, ChunkOrigin.USER))
        pieces.append((span.start_index, span.end_index, ChunkOrigin.SUBAGENT))
        cursor = span.end_index
    if cursor < chunk.end_index:
        pieces.append((cursor, chunk.end_index, ChunkOrigin.USER))
    return pieces


def _numbered(pieces: Sequence[tuple[int, int, ChunkOrigin]]) -> list[Chunk]:
    """Return the pieces as chunks, numbered by position and one of them latest.

    Args:
        pieces: The `(start_index, end_index, origin)` pieces in transcript
            order.

    Returns:
        One chunk per piece, with the id `K<n>` counted from 1.
    """
    latest = _latest_position(pieces)
    return [
        Chunk(
            id=chunk_id(position + 1),
            start_index=start,
            end_index=end,
            is_latest=position == latest,
            origin=origin,
        )
        for position, (start, end, origin) in enumerate(pieces)
    ]


def _latest_position(pieces: Sequence[tuple[int, int, ChunkOrigin]]) -> int:
    """Return the position of the chunk that carries `is_latest`.

    The last piece that is not subagent work, see the module docstring. A list
    of nothing but subagent chunks — which `chunk_session` cannot produce, but
    an externally supplied chunk list can — falls back to its last piece, so
    that `is_latest` is set exactly once whenever there is a chunk at all.

    Args:
        pieces: The pieces in transcript order.

    Returns:
        The position of the latest chunk; `-1` for no piece at all, which is a
        position no chunk has.
    """
    for position in reversed(range(len(pieces))):
        if pieces[position][2] is ChunkOrigin.USER:
            return position
    return len(pieces) - 1


def _result_indices(session: Session) -> dict[str, list[int]]:
    """Return the indices of the messages reporting a result, by call id.

    A call id is unique per session, but the schema does not forbid two
    messages reporting a result for it (#3), so the indices are a list.

    Args:
        session: The session to scan.

    Returns:
        Per `ToolResult.call_id`, the indices reporting it, in transcript order.
    """
    indices: dict[str, list[int]] = {}
    for index, message in enumerate(session.messages):
        if message.tool_result is not None:
            indices.setdefault(message.tool_result.call_id, []).append(index)
    return indices


def _first_after(indices: Sequence[int], after: int) -> int | None:
    """Return the first index greater than `after`, or `None` if there is none.

    Args:
        indices: Indices in ascending order.
        after: The index the result has to come after.

    Returns:
        The first index behind `after`; `None` when every index is in front of
        it — a result that answers a call the transcript does not hold.
    """
    return next((index for index in indices if index > after), None)


def _covers_a_user_turn(session: Session, span: SubagentSpan) -> bool:
    """Return whether a real user turn sits inside `span`.

    Args:
        session: The session the span ranges over.
        span: The span to check.

    Returns:
        `True` if the span holds a message that opens a chunk of its own, so
        that lifting the span would swallow a user turn.
    """
    return any(
        is_user_turn(session.messages[index]) for index in range(span.start_index, span.end_index)
    )


def _require_partition(session: Session, chunks: Sequence[Chunk]) -> None:
    """Raise unless `chunks` covers every message of `session` exactly once.

    The same precondition `reduce_chunks` states, checked here because this
    stage rewrites the chunk list: a gap or an overlap in the input would be
    carried into the output instead of being reported.

    Args:
        session: The session the chunks are supposed to range over.
        chunks: The chunks to check, in transcript order.

    Raises:
        ValueError: If the ranges leave a gap, overlap, or do not end at the
            last message.
    """
    cursor = 0
    for chunk in chunks:
        if chunk.start_index != cursor:
            raise _not_a_partition(session, chunks)
        cursor = chunk.end_index
    if cursor != len(session.messages):
        raise _not_a_partition(session, chunks)


def _not_a_partition(session: Session, chunks: Sequence[Chunk]) -> ValueError:
    """Return the error for a chunk list that does not cover the session.

    Args:
        session: The session the chunks are supposed to range over.
        chunks: The offending chunks, in transcript order.

    Returns:
        The error to raise, naming every range so that the gap or the overlap
        is visible in the message.
    """
    ranges = ", ".join(f"{chunk.id}={chunk.start_index}:{chunk.end_index}" for chunk in chunks)
    return ValueError(
        f"the chunks must partition the {len(session.messages)} messages of session "
        f"{session.id!r} contiguously, but they are {ranges or 'no chunk'}"
    )
