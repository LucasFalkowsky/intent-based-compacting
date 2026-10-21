"""Reducing harness noise inside a chunk, with a note of what was reduced.

A transcript carries messages nobody wrote: the echo of a slash command and the
reminders a harness injects. T3.1 deliberately does not let them open a chunk
(see `idc.chunk.boundaries`); here they leave the chunk's content — **a
reducible message is replaced by its result text, or dropped when it has
none** — and the chunk keeps a `ReductionNote` of every removal, which is what
makes the reduction auditable later instead of a silent deletion.

Three questions decide the behaviour, and all three are answered from the
canonical fields alone — no model call, no agent-specific markup (that lives in
`idc.importers`), so the same transcript always reduces the same way:

- **What is reducible.** A message the harness wrote (`Message.is_command` or
  `Message.is_system_injected`) that carries nothing but text: no tool call and
  no tool result. A message that carries either is resolvable work, not noise —
  dropping it would leave the resolver (T5) with a result it cannot attribute to
  a call. A real user turn is never reducible, by `is_user_turn`'s own
  definition.
- **What its result is.** A command asks the harness something, so its result is
  what the harness answered: the run of injected messages that directly follows
  it, none of which is a command itself. Those are consumed — their text becomes
  the command message's content. An injected message answers nothing, so it has
  no result and is dropped; neither does a message the harness flagged as
  command *and* injected, which is an echo and therefore noise of its own.
- **What is exempt.** The preamble of a transcript that was compacted before
  (`Session.source` is not `raw`) is left alone: that is where a summary block
  sits — this tool's own, which the parser (T8) has to find again, or the
  agent's native one, which the measurement compares against. The block a
  natively compacted session names in `Session.native_summary_id` is exempt
  wherever it sits, which is not always the preamble: dropping it would leave
  the reduced document naming a message it no longer holds. The preamble rule
  therefore no longer carries that guarantee on its own, and it does not extend
  to a *superseded* block: in a session compacted twice, the first summary is
  neither named nor in the preamble and is reduced away — correctly, since it is
  history the agent dropped and the visible context hides it either way. And a
  chunk is never emptied: a `Chunk` spans at least one message, and losing one would
  change the chunk count.

A `Chunk` is a half-open range over `Session.messages` and copies no message,
so it cannot rewrite one in place — reduction therefore returns a
`ReducedSession`: a **new** transcript plus the chunks re-indexed onto it, with
the chunk ids, their order, `is_latest` and `origin` unchanged. Every message
the reduction did not touch is carried over **by identity**, and the new
transcript keeps the session's provenance — including the summary identifier a
later stage reads it from a file by — so the T3.1 properties still hold over
the result: the ranges partition the reduced transcript, and nothing downstream
reads a stale copy. The only new object is the one message that received a
result text — and because it is a `system` message with both flags cleared, it
is neither a user turn nor reducible again: reduction is idempotent, and
re-chunking the reduced transcript reproduces the same chunks.

```python
from idc.chunk import reduce_session
from idc.schema import Session, load

reduced = reduce_session(load(Session, "session.json"))
reduced.session      # the transcript without the harness noise
reduced.chunks       # the same chunks, re-indexed, each with its notes
reduced.notes        # every reduction, in transcript order
```
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum

from idc.chunk.boundaries import chunk_session, user_turn_indices
from idc.schema import Chunk, Message, Role, Session, SessionSource

RESULT_SEPARATOR = "\n\n"
"""How several answers to one command are joined into a single result text."""


class ReductionReason(StrEnum):
    """Why a message was reduced — which kind of harness noise it was.

    Attributes:
        COMMAND: A slash command or its echo (`Message.is_command`).
        SYSTEM_INJECTED: A message the harness injected
            (`Message.is_system_injected`).
    """

    COMMAND = "command"
    SYSTEM_INJECTED = "system-injected"


@dataclass(frozen=True, slots=True)
class ReductionNote:
    """The record that one message was replaced by its result or dropped.

    The note is what keeps the reduction auditable: it names the message that
    left the transcript, why it was noise, and which answers were consumed for
    it.

    Attributes:
        message_id: `Message.id` of the reduced message. The message is still in
            the transcript when it was replaced — carrying its result text
            instead of its own — and gone when it was dropped.
        reason: Which kind of harness noise the message was.
        result_from: `Message.id` of every answer that was consumed for this
            message, in transcript order; those messages are gone from the
            transcript. The result text is built from the consumed answers that
            carry text, so an answer that was blank is listed here although it
            contributed nothing — observed: a command answered by `"   "` and
            then by `"Model set."` is replaced by `"Model set."` and lists both
            ids. Listing every consumed message is what keeps the removal
            auditable. Empty exactly when the message was dropped for having no
            result.
    """

    message_id: str
    reason: ReductionReason
    result_from: tuple[str, ...] = ()

    @property
    def was_replaced(self) -> bool:
        """Return whether the message was replaced rather than dropped."""
        return bool(self.result_from)


@dataclass(frozen=True, slots=True)
class ReducedChunk:
    """One chunk of a reduced transcript, with what was reduced inside it.

    Attributes:
        chunk: The chunk, its range re-indexed onto the reduced transcript. Its
            id, position, `is_latest` and `origin` are the ones the chunker
            assigned.
        notes: One note per reduction inside this chunk, in transcript order;
            empty when the chunk held no harness noise.
    """

    chunk: Chunk
    notes: tuple[ReductionNote, ...] = ()


@dataclass(frozen=True, slots=True)
class ReducedSession:
    """A transcript with the harness noise reduced, and its chunks.

    Attributes:
        session: The reduced transcript: the same session with the same
            provenance, holding the messages that survived the reduction. Every
            untouched message is the session's own object, not a copy.
        chunks: The chunks in transcript order — as many as before, re-indexed
            onto `session` so that their ranges partition it again.
    """

    session: Session
    chunks: tuple[ReducedChunk, ...] = ()

    @property
    def notes(self) -> tuple[ReductionNote, ...]:
        """Return every note of every chunk, in transcript order."""
        return tuple(note for reduced_chunk in self.chunks for note in reduced_chunk.notes)


@dataclass(frozen=True, slots=True)
class _Plan:
    """What reduction decided about one chunk, before anything is rewritten.

    Attributes:
        notes: One note per reduction inside the chunk, in transcript order.
        removed: The indices that leave the transcript — dropped messages and
            the answers that were consumed as a result text.
        replacements: Per index whose message received a result text, the
            message that takes its place.
    """

    notes: tuple[ReductionNote, ...] = ()
    removed: tuple[int, ...] = ()
    replacements: dict[int, Message] = field(default_factory=dict)


def reduce_session(session: Session) -> ReducedSession:
    """Chunk a canonical session and reduce the harness noise inside its chunks.

    The one-call form of the stage: `chunk_session` followed by `reduce_chunks`.

    Args:
        session: The canonical session to chunk and reduce.

    Returns:
        The reduced transcript with its chunks and their notes.
    """
    return reduce_chunks(session, chunk_session(session))


def reduce_chunks(session: Session, chunks: Sequence[Chunk]) -> ReducedSession:
    """Reduce the harness noise inside `chunks` of `session`.

    Every reducible message is replaced by its result text or dropped, and each
    removal is recorded on the chunk it happened in; see the module docstring
    for what is reducible, what a result is, and what is exempt. The chunks are
    neither split nor merged nor dropped: the result holds as many chunks, with
    the same ids and in the same order, re-indexed onto the reduced transcript.

    Args:
        session: The canonical session the chunks range over.
        chunks: The session's chunks, in transcript order, as `chunk_session`
            returns them — including the chunks a later stage added, as long as
            they still partition the transcript.

    Returns:
        The reduced transcript with its re-indexed chunks and their notes.

    Raises:
        ValueError: If `chunks` does not partition `session.messages`, which
            would make the re-indexing meaningless — the reduction is defined
            over a transcript that is covered exactly once.
    """
    _require_partition(session, chunks)
    exempt = _exempt_indices(session)
    notes: list[tuple[ReductionNote, ...]] = []
    removed: set[int] = set()
    replacements: dict[int, Message] = {}
    for chunk in chunks:
        plan = _plan_chunk(session, chunk, exempt)
        if len(plan.removed) == chunk.message_count:
            # Reducing would leave the chunk without a message. That can only
            # happen to the single chunk of a transcript without a real user
            # turn (every other chunk opens at one), and a `Chunk` spans at
            # least one message — so this chunk is left as it is.
            plan = _Plan()
        notes.append(plan.notes)
        removed.update(plan.removed)
        replacements.update(plan.replacements)
    kept = [index for index in range(len(session.messages)) if index not in removed]
    messages = [
        replacements[index] if index in replacements else session.messages[index] for index in kept
    ]
    return ReducedSession(
        session=_with_messages(session, messages),
        chunks=_reindexed(chunks, notes, _kept_before(len(session.messages), removed)),
    )


def _reindexed(
    chunks: Sequence[Chunk],
    notes: Sequence[tuple[ReductionNote, ...]],
    kept_before: Sequence[int],
) -> tuple[ReducedChunk, ...]:
    """Return the chunks with their ranges moved onto the reduced transcript.

    Everything but the two indices is the chunker's: the id, the order,
    `is_latest` and `origin` are carried over, so the chunk count cannot change
    here.

    Args:
        chunks: The original chunks, in transcript order.
        notes: The notes of each chunk, in the same order.
        kept_before: Per original index, how many messages before it are kept,
            as `_kept_before` returns it.

    Returns:
        The reduced chunks, in transcript order.
    """
    return tuple(
        ReducedChunk(
            chunk=Chunk(
                id=chunk.id,
                start_index=kept_before[chunk.start_index],
                end_index=kept_before[chunk.end_index],
                is_latest=chunk.is_latest,
                origin=chunk.origin,
            ),
            notes=chunk_notes,
        )
        for chunk, chunk_notes in zip(chunks, notes, strict=True)
    )


def _is_reducible(message: Message) -> bool:
    """Return whether `message` is harness noise that reduction may remove.

    Args:
        message: The message to classify.

    Returns:
        `True` for a slash command or an injected message that carries nothing
        but text. A message with a tool call or a tool result is resolvable work
        and stays, and a real user turn is never reducible — `is_user_turn`
        excludes exactly the two flags tested here.
    """
    return (
        (message.is_command or message.is_system_injected)
        and not message.tool_calls
        and message.tool_result is None
    )


def _plan_chunk(
    session: Session,
    chunk: Chunk,
    exempt: frozenset[int],
) -> _Plan:
    """Decide what to reduce inside one chunk, without changing anything yet.

    Args:
        session: The session the chunk ranges over.
        chunk: The chunk to plan the reduction of.
        exempt: Indices reduction must not touch.

    Returns:
        The plan for this chunk; an empty one when it holds no harness noise.
    """
    notes: list[ReductionNote] = []
    removed: list[int] = []
    replacements: dict[int, Message] = {}
    index = chunk.start_index
    while index < chunk.end_index:
        message = session.messages[index]
        if index in exempt or not _is_reducible(message):
            index += 1
            continue
        if message.is_command:
            # A command asks the harness something, so it can have an answer.
            reason = ReductionReason.COMMAND
            answer = _answer(session, index, chunk, exempt)
        else:
            # An injected message answers nothing, so it has no result.
            reason = ReductionReason.SYSTEM_INJECTED
            answer = ()
        texts = [
            session.messages[answered].content
            for answered in answer
            if session.messages[answered].content.strip()
        ]
        if texts:
            replacements[index] = _result_message(message, RESULT_SEPARATOR.join(texts))
            removed.extend(answer)
            notes.append(
                ReductionNote(
                    message_id=message.id,
                    reason=reason,
                    result_from=tuple(session.messages[answered].id for answered in answer),
                )
            )
            index = answer[-1] + 1
            continue
        # No result: the message is dropped. An answer run that held no text is
        # not consumed here — each of its messages is injected noise of its own
        # and is reduced by the same rule on the next turn of this loop.
        removed.append(index)
        notes.append(ReductionNote(message_id=message.id, reason=reason))
        index += 1
    return _Plan(notes=tuple(notes), removed=tuple(removed), replacements=replacements)


def _answer(
    session: Session,
    index: int,
    chunk: Chunk,
    exempt: frozenset[int],
) -> tuple[int, ...]:
    """Return the indices of the harness's answer to the command at `index`.

    The answer is the run of reducible injected messages that directly follows
    the command, inside the same chunk. A message that is itself a command ends
    the run even when the harness also flagged it as injected: an echo is noise
    by this module's own definition, so taking its text as the answer would
    keep that noise and attribute it to a command it never answered. It is
    reduced in its own right on the next turn of `_plan_chunk`'s loop. The run
    can never reach beyond the chunk either: a chunk ends at the next real user
    turn, which is neither injected nor part of an answer.

    Args:
        session: The session the chunk ranges over.
        index: Index of the command message.
        chunk: The chunk the command sits in.
        exempt: Indices reduction must not touch.

    Returns:
        The indices of the answer in transcript order; empty when the harness
        said nothing and the command therefore has no result.
    """
    answer: list[int] = []
    following = index + 1
    while following < chunk.end_index:
        message = session.messages[following]
        if (
            following in exempt
            or not _is_reducible(message)
            or not message.is_system_injected
            or message.is_command
        ):
            break
        answer.append(following)
        following += 1
    return tuple(answer)


def _result_message(message: Message, result: str) -> Message:
    """Return the message that replaces `message`, carrying its result text.

    The reduced message keeps its id and its timestamp — it is the same message,
    with the result as its content — but it becomes a plain `system` message:
    both harness flags are cleared, so it is neither reducible a second time nor
    a user turn, whatever role the harness had put the command in.

    Args:
        message: The reduced message.
        result: The result text it is replaced by; never empty.

    Returns:
        The replacement message.
    """
    return Message(
        id=message.id,
        role=Role.SYSTEM,
        content=result,
        timestamp=message.timestamp,
    )


def _exempt_indices(session: Session) -> frozenset[int]:
    """Return the indices of `session` that reduction must not touch.

    A transcript that was compacted before opens with a summary block, written
    either by this tool (`intent-compaction`, the input the parser reads back)
    or by the agent itself (`native-compaction`, the baseline the measurement
    compares against). The block is an injected message, so the rule that would
    drop it has to stop at it: everything before the first real user turn of a
    transcript whose `source` is not `raw` is exempt.

    That is the block's **position**, and it is an approximation — the only one
    available until T2.6 (#69) made the location a field of the document. Where
    the session names its native summary, the message it names is exempt too,
    wherever it sits: the importers produce two shapes the position rule does
    not reach, a transcript that opens with the user turn the compaction
    replaced and one compacted twice, whose second block sits after a real user
    turn. Dropping that block would leave `Session.native_summary_id` naming a
    message the transcript no longer holds — a document the schema still
    accepts (#3) and provenance normalisation can only refuse.

    Args:
        session: The session to inspect.

    Returns:
        The exempt indices; empty for a raw transcript, which names no summary.
    """
    if session.source is SessionSource.RAW:
        return frozenset()
    turns = user_turn_indices(session)
    preamble = range(turns[0] if turns else len(session.messages))
    return frozenset(preamble) | _summary_indices(session)


def _summary_indices(session: Session) -> frozenset[int]:
    """Return the index of the message `session.native_summary_id` names.

    Args:
        session: The session to inspect.

    Returns:
        The one index, or no index when the session names no summary or names
        one it does not hold — a truncated transcript the schema accepts on
        purpose. Reduction neither repairs nor clears that reference: it is
        carried over unchanged, so the reduced document is refused by
        provenance normalisation for exactly the reason the original was.
    """
    if session.native_summary_id is None:
        return frozenset()
    return frozenset(
        index
        for index, message in enumerate(session.messages)
        if message.id == session.native_summary_id
    )


def _kept_before(total: int, removed: set[int]) -> list[int]:
    """Return, per original index, how many messages before it are kept.

    That count is the index the message gets in the reduced transcript, and
    taken at a chunk's `start_index` and `end_index` it is the chunk's new
    half-open range — which is why the list runs to `total` inclusive.

    Args:
        total: Number of messages in the original transcript.
        removed: The indices reduction removes.

    Returns:
        A list of `total + 1` counts.
    """
    counts = [0] * (total + 1)
    for index in range(total):
        counts[index + 1] = counts[index] + (0 if index in removed else 1)
    return counts


def _with_messages(session: Session, messages: list[Message]) -> Session:
    """Return `session` with `messages` in place of its own, provenance kept.

    Provenance includes `native_summary_id`: the reduced transcript is the same
    transcript, so it says the same thing about where the agent's context
    starts. The message that identifier names survives the reduction by being
    exempt from it (see `_exempt_indices`).

    Args:
        session: The session to rebuild.
        messages: The reduced messages, in transcript order; the untouched ones
            are `session`'s own objects, which pydantic does not re-validate and
            therefore does not copy.

    Returns:
        The reduced session.
    """
    return Session(
        id=session.id,
        agent=session.agent,
        model=session.model,
        source=session.source,
        native_summary_id=session.native_summary_id,
        created_at=session.created_at,
        messages=messages,
    )


def _require_partition(session: Session, chunks: Sequence[Chunk]) -> None:
    """Raise unless `chunks` covers every message of `session` exactly once.

    Args:
        session: The session the chunks are supposed to range over.
        chunks: The chunks to check, in transcript order.

    Raises:
        ValueError: If the ranges leave a gap, overlap, or do not end at the
            last message — the same property `chunk_session` guarantees.
    """
    covered = [index for chunk in chunks for index in range(chunk.start_index, chunk.end_index)]
    if covered != list(range(len(session.messages))):
        ranges = ", ".join(f"{chunk.id}={chunk.start_index}:{chunk.end_index}" for chunk in chunks)
        raise ValueError(
            f"the chunks must partition the {len(session.messages)} messages of session "
            f"{session.id!r}, but they cover {len(covered)} index positions "
            f"({ranges or 'no chunk'})"
        )
