"""The failures of the classification stage, as one catchable family.

Classification is the one stage whose input comes from a model, so the stage's
job is as much to refuse a decision it cannot carry out as it is to carry one
out. Every refusal is a `ClassifyError`, so a caller (and the CLI) can catch the
whole stage with one `except`, and each subclass carries the values its message
is built from — the chain that was named, the chunk that was already assigned —
so that a test asserts on an attribute instead of on prose.

What is *not* here: a value object the caller built wrongly. `Decision`,
`ChainCandidate` and the chain-id helpers raise `ValueError` the way
`SubagentSpan` does in `idc.chunk.subagents` — a bad literal in a caller's own
construction is a programming error, not a failure of the stage. The line is
drawn by **where the data comes from**, not by which object notices: everything
the stage is *handed* is in the family, the chains a run is seeded with
included, because those are parsed out of a compacted block in a transcript
file (#33) and a hand-edited block must not reach T10's CLI as a traceback.

The evaluation harness (T4.4, #19) is in the family for the same reason: a gold
labelling and a recorded run are documents read from disk, so
`GoldLabellingError`, `GoldCoverageError` and `DecisionRecordError` are refusals
of input and not of a decision — but they are refusals of this stage, and one
`except ClassifyError` stays enough to catch it.

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
    "BlankChainLabelError",
    "ChainAlreadySeededError",
    "ChainOrderError",
    "ChunkAlreadyAssignedError",
    "ChunkOrderError",
    "ClassifyError",
    "DecisionRecordError",
    "ForeignChunkError",
    "GoldCoverageError",
    "GoldLabellingError",
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
    partition of the transcript and the summary complete. The clash has two
    causes, and the message names the one it is looking at: a chunk this run
    already classified, or a chunk id a *carried-over* chain claims — whether
    this run's chunk ran into it or a second seeded chain did. The second cause
    is the normal case once #33 seeds chains, because both runs number their
    chunks positionally, and "every chunk belongs to exactly one chain" would be
    a misleading thing to read there.

    Attributes:
        chunk_id: The chunk that is already assigned.
        chain_id: The chain it belongs to.
        carried_over: Whether that chain came from an earlier run, which makes
            a stale chunk id the likely cause.
    """

    def __init__(self, chunk_id: str, chain_id: str, *, carried_over: bool = False) -> None:
        """Build the error from the chunk and the chain that already holds it.

        Args:
            chunk_id: The chunk that is already assigned.
            chain_id: The chain it belongs to.
            carried_over: Whether that chain was seeded from an earlier run.
        """
        self.chunk_id = chunk_id
        self.chain_id = chain_id
        self.carried_over = carried_over
        cause = (
            "that chain was carried over from an earlier run, whose chunk ids are positions "
            "in its own transcript and have to be re-mapped before seeding"
            if carried_over
            else "every chunk belongs to exactly one chain"
        )
        super().__init__(f"chunk {chunk_id!r} is already assigned to chain {chain_id!r}; {cause}")


class ChainAlreadySeededError(ClassifyError):
    """Two chains a run was seeded with carry the same id.

    A chain id identifies exactly one chain — the compactor writes it into the
    summary block (T6.3) and the parser reads it back (T8) — so a block holding
    it twice says two different things about one chain, and the store cannot
    pick which. The sibling of `ChunkAlreadyAssignedError`, one level up: an
    identifier that is already taken.

    Attributes:
        chain_id: The id the second seeded chain repeats.
    """

    def __init__(self, chain_id: str) -> None:
        """Build the error from the id that arrived twice.

        Args:
            chain_id: The id the second seeded chain repeats.
        """
        self.chain_id = chain_id
        super().__init__(
            f"two seeded chains share the id {chain_id!r}; a chain id names exactly one chain"
        )


class BlankChainLabelError(ClassifyError):
    """A chain a run was seeded with carries a label that holds no word.

    `Chain.label` is only checked for being non-empty (#3), so a hand-edited
    summary block can name a chain `"   "` — and that string would then be what
    an adapter is shown as the chain's intent, and what the next summary block
    is written with. `ChainStore.open_chain` refuses the same label when a
    caller passes one explicitly; refusing it here keeps the two sides of the
    seam from disagreeing, and keeps the value object's own `ValueError` out of
    a path that carries input.

    Attributes:
        chain_id: The chain whose label is blank.
        label: The label as it arrived, so that the message shows which
            whitespace it was.
    """

    def __init__(self, chain_id: str, label: str) -> None:
        """Build the error from the chain and the label it carries.

        Args:
            chain_id: The chain whose label is blank.
            label: The label as it arrived.
        """
        self.chain_id = chain_id
        self.label = label
        super().__init__(
            f"seeded chain {chain_id!r} carries the label {label!r}, which holds no word; "
            f"a chain's label is what names its intent"
        )


class ChainOrderError(ClassifyError):
    """A seeded chain is numbered before the chain that was seeded ahead of it.

    `ChainStore.candidates` and `ChainStore.chains` are documented to be in
    opening order, and a chain id *is* its opening position — so a list whose
    numbers run backwards states two contradicting orders. What `ChunkOrderError`
    is for the chunks of one chain, this is for the chains of one store; the
    numbers still need not be contiguous.

    Attributes:
        chain_id: The chain that arrived out of order.
        previous_chain_id: The chain that was seeded before it and is numbered
            after it.
    """

    def __init__(self, chain_id: str, previous_chain_id: str) -> None:
        """Build the error from the chain and the one seeded ahead of it.

        Args:
            chain_id: The chain that arrived out of order.
            previous_chain_id: The chain seeded before it.
        """
        self.chain_id = chain_id
        self.previous_chain_id = previous_chain_id
        super().__init__(
            f"seeded chain {chain_id!r} is numbered before {previous_chain_id!r}, which was "
            f"seeded first; chains arrive in the order they were opened"
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
    """A chunk's range reaches past the messages of the session it was offered to.

    A chunk carries indices and no messages, so such a range would otherwise
    resolve to no message at all — and the last user message of the chain would
    silently come out empty.

    This is an **out-of-range** check and not more: a `Chunk` carries no session
    id (#3), so a chunk of a *different* transcript whose range happens to fit
    cannot be told apart from one of this transcript, and is accepted. Keeping
    the chunks of one run together is the caller's job; what cannot be checked
    here is documented rather than implied by the name.

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


class GoldLabellingError(ClassifyError):
    """A gold chain labelling is not usable as a reference.

    The labels the evaluation harness scores a run against are hand-written and
    read from a file (`idc.classify.gold`), so every way of getting them wrong
    is input and not a programming error: a document that is not JSON, a
    transcript name that does not match the file, a repeated chunk id, a chunk
    id that is no chunk id, a chain numbering the classifier could never
    produce, or — raised by the harness — a chunking it cannot perform. The
    message says which of those it is; there is no attribute to branch on,
    because a caller cannot repair a hand-written document anyway.
    """


class GoldCoverageError(ClassifyError):
    """A run and a gold labelling do not cover the same chunks.

    Both metrics are shares of the chunks, so comparing two labellings over
    different chunk sets would silently score a different question — the
    intersection, or the smaller of the two. The usual cause is a labelling
    written against another chunking, or a transcript that has grown since.

    It is a **library-level guard**, for a caller that scores a labelling it
    produced elsewhere (`agreement`, `per_chunk_accuracy`, `rand_index`). The
    command does not reach it: a recorded run of the wrong length is refused
    before anything is scored (`DecisionRecordError`), and a decision list that
    is too short runs out inside the classifier (`ScriptExhaustedError`).

    Attributes:
        missing: Chunk ids the gold labels and the run does not, in gold order.
        unexpected: Chunk ids the run labels and the gold does not, in run
            order.
    """

    def __init__(self, missing: Sequence[str], unexpected: Sequence[str]) -> None:
        """Build the error from the two sides of the difference.

        Args:
            missing: Chunk ids only the gold labelling holds.
            unexpected: Chunk ids only the run holds.
        """
        self.missing = tuple(missing)
        self.unexpected = tuple(unexpected)
        parts = []
        if self.missing:
            parts.append(f"not labelled by the run: {', '.join(self.missing)}")
        if self.unexpected:
            parts.append(f"not labelled by the gold: {', '.join(self.unexpected)}")
        super().__init__(
            "a run is scored against the gold labelling of the same chunks; "
            + "; ".join(parts or ["the two cover different chunks"])
        )


class DecisionRecordError(ClassifyError):
    """A recorded run the harness was asked to replay is not usable.

    A recorded run is the decision list an adapter produced, written to a file
    so that it can be scored offline — the only way the harness's command can
    run a real backend without calling a model. Like a gold labelling it is
    input: a missing document, an unknown key, an entry that is no decision, or
    a list whose length does not match the transcript's chunks is refused with
    the file named.
    """
