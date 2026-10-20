"""The failures of the classification stage, as one catchable family.

Classification is the one stage whose input comes from a model, so the stage's
job is as much to refuse a decision it cannot carry out as it is to carry one
out. Every refusal is a `ClassifyError`, so a caller (and the CLI) can catch the
whole stage with one `except`, and each subclass carries the values its message
is built from — the chain that was named, the chunk that was already assigned —
so that a test asserts on an attribute instead of on prose.

What is *not* here: an invalid value object. `Decision`, `ChainCandidate` and
the chain-id helpers raise `ValueError` the way `SubagentSpan` does in
`idc.chunk.subagents` — those are programming errors in the caller's own
construction, not failures of the stage.

The two families therefore meet in one place, and deliberately overlap rather
than split there: a backend's *answer* that is not usable is both a bad value
and a failure of the stage, so `AdapterContractError` is a `ValueError` **and** a
`ClassifyError`. The message still says which mistake was made — the id form
when the answer was a wrong-shaped chain id, the contract when it was not an
answer at all — but a caller never has to catch two families for one event.
"""

from collections.abc import Sequence

__all__ = [
    "AdapterContractError",
    "ChunkAlreadyAssignedError",
    "ChunkOrderError",
    "ClassifyError",
    "ForeignChunkError",
    "ScriptExhaustedError",
    "UnknownChainError",
]


class ClassifyError(Exception):
    """Base class of every failure the classification stage raises."""


class UnknownChainError(ClassifyError):
    """A decision named a chain the store does not hold.

    A model can name anything, including a chain that was never offered as a
    candidate; silently opening a new chain instead would hide a wrong
    classification, so the store refuses.

    Attributes:
        chain_id: The id the decision named.
        known: The chain ids the store holds, in the order they were offered.
    """

    def __init__(self, chain_id: str, known: Sequence[str]) -> None:
        """Build the error from the named chain and the chains on offer.

        Args:
            chain_id: The id the decision named.
            known: The chain ids the store holds.
        """
        self.chain_id = chain_id
        self.known = tuple(known)
        offered = ", ".join(self.known) if self.known else "none"
        super().__init__(f"no chain {chain_id!r} in the store; the candidates were: {offered}")


class ChunkAlreadyAssignedError(ClassifyError):
    """A chunk was about to be assigned to a second chain.

    Every chunk belongs to exactly one chain — that is what makes the chains a
    partition of the transcript and the summary complete.

    Attributes:
        chunk_id: The chunk that is already assigned.
        chain_id: The chain it belongs to.
    """

    def __init__(self, chunk_id: str, chain_id: str) -> None:
        """Build the error from the chunk and the chain that already holds it.

        Args:
            chunk_id: The chunk that is already assigned.
            chain_id: The chain it belongs to.
        """
        self.chunk_id = chunk_id
        self.chain_id = chain_id
        super().__init__(
            f"chunk {chunk_id!r} is already assigned to chain {chain_id!r}; "
            f"every chunk belongs to exactly one chain"
        )


class ChunkOrderError(ClassifyError):
    """A chunk was appended behind a chunk that comes later in the transcript.

    `Chain.chunk_ids` is documented to be in transcript order, and the resolver
    (T5) reads it as such — the last write per file is the last one in that
    order — so a chain may not be built out of order.

    Attributes:
        chunk_id: The chunk that was about to be appended.
        chain_id: The chain it was about to be appended to.
        start_index: Where the chunk starts in the transcript.
        last_start_index: Where the chain's current last chunk starts.
    """

    def __init__(
        self, chunk_id: str, chain_id: str, start_index: int, last_start_index: int
    ) -> None:
        """Build the error from the chunk, the chain and the two positions.

        Args:
            chunk_id: The chunk that was about to be appended.
            chain_id: The chain it was about to be appended to.
            start_index: Where the chunk starts in the transcript.
            last_start_index: Where the chain's current last chunk starts.
        """
        self.chunk_id = chunk_id
        self.chain_id = chain_id
        self.start_index = start_index
        self.last_start_index = last_start_index
        super().__init__(
            f"chunk {chunk_id!r} starts at message {start_index} and cannot be appended to "
            f"chain {chain_id!r}, whose last chunk starts at {last_start_index}; "
            f"a chain's chunks are in transcript order"
        )


class ForeignChunkError(ClassifyError):
    """A chunk is not a message range over the session the store was built on.

    A chunk carries indices and no messages, so a range that reaches past the
    transcript would otherwise resolve to no message at all — and the last user
    message of the chain would silently come out empty.

    Attributes:
        chunk_id: The chunk that does not fit.
        session_id: The session it was offered to.
    """

    def __init__(self, chunk_id: str, session_id: str, end_index: int, message_count: int) -> None:
        """Build the error from the chunk and the session it does not fit.

        Args:
            chunk_id: The chunk that does not fit.
            session_id: The session it was offered to.
            end_index: The chunk's exclusive end index.
            message_count: How many messages the session holds.
        """
        self.chunk_id = chunk_id
        self.session_id = session_id
        super().__init__(
            f"chunk {chunk_id!r} ends at message {end_index}, but session {session_id!r} holds "
            f"{message_count} message(s); it is not a chunk of this transcript"
        )


class AdapterContractError(ClassifyError, ValueError):
    """An adapter answered with something that cannot be read as a decision.

    The interface allows three forms — a `Decision`, `NEW`, or a chain id — and
    nothing else; guessing what a fourth one meant would be a classification
    nobody decided. A chain id of the wrong shape (`K1`, a label, prose) is the
    same event and the most likely one in practice, so it is this error too,
    with a message naming the `C<n>` form rather than the contract.

    A `ValueError` as well as a `ClassifyError`: the answer is a bad value, and
    it is a failure of the stage, so one `except ClassifyError` suffices for a
    caller (T10's CLI) without `except ValueError` becoming wrong for a caller
    that reads it as a value problem.
    """


class ScriptExhaustedError(ClassifyError):
    """The scripted adapter was asked for one decision more than it holds.

    A test-authoring mistake, and a loud one on purpose: a fake that repeated
    its last decision, or defaulted to opening a chain, would make the test pass
    for a reason the test never stated.

    Attributes:
        chunk_id: The chunk the decision was wanted for.
        script_length: How many decisions the script held.
    """

    def __init__(self, chunk_id: str, script_length: int) -> None:
        """Build the error from the chunk asked about and the script's length.

        Args:
            chunk_id: The chunk the decision was wanted for.
            script_length: How many decisions the script held.
        """
        self.chunk_id = chunk_id
        self.script_length = script_length
        super().__init__(
            f"the scripted adapter ran out of decisions at chunk {chunk_id!r}: "
            f"its script holds {script_length} decision(s)"
        )
