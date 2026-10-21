"""The failures of the resolution stage, as one catchable family.

The resolver is handed three things that were produced elsewhere — a session, a
chunk list and a chain — and nothing checks that they belong together: a
`Chunk` carries no session id and a `Chain` carries chunk *ids* (#3). So the
two ways those three can disagree are the two refusals here, and both would
otherwise be silent: a chain naming a chunk nobody offered would resolve to
fewer writes than the chain made, and a chunk whose range reaches past the
transcript would slice short and drop the writes at its end. A solution that is
quietly incomplete is the one outcome this stage must not produce.

Every refusal is a `ResolveError`, so a caller (and T10's CLI) catches the whole
stage with one `except`, and each subclass carries the values its message is
built from, so a test asserts on an attribute instead of on prose — the shape
`idc.classify.errors` established.

What is *not* here: a value object a caller built wrongly. `SolutionFile`
raises a plain `ValueError` for an entry that states nothing, the way
`SubagentSpan` does in `idc.chunk.subagents`. The line is the one
`idc.classify.errors` draws — everything the stage is *handed* is in the
family, a bad literal in a caller's own construction is a programming error.

Nothing in this stage is in the family because a *transcript* is odd: a write
whose arguments name no file, a tool result that is missing, a command this
build has never seen are all data, and the resolver reports them (as a path
with no file, as `None`, as no verification) rather than refusing them.
"""

from collections.abc import Sequence

__all__ = [
    "ForeignChunkError",
    "ResolveError",
    "UnknownChunkError",
]


class ResolveError(Exception):
    """Base class of every failure the resolution stage raises."""


class UnknownChunkError(ResolveError):
    """A chain named a chunk that is not in the chunk list it was resolved against.

    `Chain.chunk_ids` are references (#3), so the chunk list is what they are
    resolved through; a missing one would otherwise make the chain's solution
    silently smaller than the work the chain did. The usual cause is a chain
    carried over from an earlier run, whose chunk ids are positions in its own
    transcript.

    Attributes:
        chunk_id: The id the chain named.
        chain_id: The chain that named it.
        known: The chunk ids the list holds, in the order it holds them.
    """

    def __init__(self, chunk_id: str, chain_id: str, known: Sequence[str]) -> None:
        """Build the error from the named chunk, its chain and the chunks on offer.

        Args:
            chunk_id: The id the chain named.
            chain_id: The chain that named it.
            known: The chunk ids the chunk list holds.
        """
        self.chunk_id = chunk_id
        self.chain_id = chain_id
        self.known = tuple(known)
        offered = ", ".join(self.known) if self.known else "none"
        super().__init__(
            f"chain {chain_id!r} names the chunk {chunk_id!r}, which is not in the chunk "
            f"list; it holds: {offered}"
        )


class ForeignChunkError(ResolveError):
    """A chunk's range reaches past the messages of the session it was resolved against.

    A chunk holds indices and no messages, so such a range would slice short
    instead of failing, and the writes at the end of the chunk would be missing
    from the solution without anything saying so.

    Like `idc.classify.errors.ForeignChunkError` this is an **out-of-range**
    check and not more: a chunk of a *different* transcript whose range happens
    to fit cannot be told apart from one of this transcript and is accepted.
    Keeping the artifacts of one run together is the caller's job.

    Attributes:
        chunk_id: The chunk that does not fit.
        session_id: The session it was resolved against.
    """

    def __init__(self, chunk_id: str, session_id: str, end_index: int, message_count: int) -> None:
        """Build the error from the chunk and the session it does not fit.

        Args:
            chunk_id: The chunk that does not fit.
            session_id: The session it was resolved against.
            end_index: The chunk's exclusive end index.
            message_count: How many messages the session holds.
        """
        self.chunk_id = chunk_id
        self.session_id = session_id
        self.end_index = end_index
        self.message_count = message_count
        super().__init__(
            f"chunk {chunk_id!r} ends at message {end_index}, but session {session_id!r} holds "
            f"{message_count} message(s); it is not a chunk of this transcript"
        )
