"""Splitting a canonical transcript at its user-message boundaries.

One chunk is one user input plus everything the agent produced in response, so
the boundaries are the user turns — and the only hard question is what a *user
turn* is. Three message shapes carry the `user` role on the wire without being
one:

- a slash command and its echo (`Message.is_command`),
- a reminder the harness injected into the user role
  (`Message.is_system_injected`),
- anything the harness wrote before the first real turn.

The first two are skipped as boundaries here and reduced to their result inside
their chunk in T3.2; the third is the decision this module makes, see
`chunk_boundaries`.

A `Chunk` is a half-open range over `Session.messages` and copies no message
(see `idc.schema.models.Chunk`), so the split is lossless exactly when the
ranges partition the index range of the transcript: contiguous, no gap, no
overlap, union equal to the whole. Nothing in this module reads a clock, a
model or the filesystem — the same session always yields the same chunks.
"""

from idc.schema import Chunk, ChunkOrigin, Message, Role, Session

CHUNK_ID_PREFIX = "K"
"""Chunk ids are `K<n>`, counted from 1 — the `C<n>` of a chain, for a chunk."""


def chunk_id(position: int) -> str:
    """Return the id of the chunk at `position`.

    Args:
        position: The chunk's position in the transcript, counted from 1.

    Returns:
        The id `K<position>`. The number is the position and not a hash of the
        content, so re-running the tool on a transcript that only grew keeps
        the ids of the chunks before the new ones.
    """
    return f"{CHUNK_ID_PREFIX}{position}"


def is_user_turn(message: Message) -> bool:
    """Return whether `message` is a real user turn, i.e. a chunk boundary.

    A real user turn is a `user` message that the harness neither injected nor
    produced as the echo of a slash command. Both of those are `user` on the
    wire in at least one of the three agents, and treating them as a turn would
    cut a chunk in the middle of the agent's work.

    Args:
        message: The message to classify.

    Returns:
        `True` if a new chunk starts at this message.
    """
    return message.role is Role.USER and not message.is_system_injected and not message.is_command


def user_turn_indices(session: Session) -> list[int]:
    """Return the indices of `session`'s real user turns, in transcript order.

    Args:
        session: The canonical session to scan.

    Returns:
        The indices into `Session.messages` at which a user turn starts; empty
        when the transcript holds none.
    """
    return [index for index, message in enumerate(session.messages) if is_user_turn(message)]


def chunk_boundaries(session: Session) -> list[tuple[int, int]]:
    """Return the half-open message ranges of `session`'s chunks.

    The ranges are the user turns, with one deliberate exception: **what
    precedes the first user turn joins the first chunk** instead of becoming a
    chunk of its own, so the first range always starts at index 0.

    A transcript need not open with a user turn — it can open with a
    session-wide slash command, with an injected reminder, or (when this tool
    ran on it before) with a rendered summary block that the parser has to find
    again. Those messages have to stay inside some chunk, because dropping them
    would make the chunking lossy; and they cannot be a chunk of their own,
    because `ChunkOrigin` knows a chunk that starts at a user message and one
    that holds subagent work, and nothing else. They are the context the first
    turn was given, so that is the chunk they belong to.

    Taken to its end, the rule also answers the degenerate case: a transcript
    with no user turn at all is one chunk holding everything. A transcript with
    no message at all yields no chunk, since a `Chunk` spans at least one
    message.

    Args:
        session: The canonical session to split.

    Returns:
        `(start_index, end_index)` per chunk, in transcript order, `start_index`
        included and `end_index` excluded. The ranges partition
        `range(len(session.messages))`.
    """
    total = len(session.messages)
    if total == 0:
        return []
    turns = user_turn_indices(session)
    starts = [0, *turns[1:]]
    ends = [*starts[1:], total]
    return list(zip(starts, ends, strict=True))


def chunk_session(session: Session) -> list[Chunk]:
    """Split a canonical session into chunks at its user-message boundaries.

    One chunk is one user turn plus every message the agent produced in
    response, tool calls and results included; the ranges returned by
    `chunk_boundaries` partition the transcript, so the split neither loses nor
    duplicates a message. The last chunk carries `is_latest`, because it is the
    one the renderer appends verbatim instead of compacting.

    Args:
        session: The canonical session to split.

    Returns:
        The chunks in transcript order, each with the id `K<n>` counted from 1
        and `origin = user`. Subagent spans become chunks of their own in T3.3;
        until then they stay inside the chunk that spawned them. The list is
        empty exactly when the session holds no message.
    """
    boundaries = chunk_boundaries(session)
    last = len(boundaries) - 1
    return [
        Chunk(
            id=chunk_id(position + 1),
            start_index=start,
            end_index=end,
            is_latest=position == last,
            origin=ChunkOrigin.USER,
        )
        for position, (start, end) in enumerate(boundaries)
    ]
