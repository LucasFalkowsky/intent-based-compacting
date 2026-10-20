"""The adapter contract: what a classification backend is asked, and may answer.

This is the interface the whole stage is built around, and it is deliberately
the smallest one that can carry a decision. An adapter is asked, per chunk,
**one** question — does this chunk continue one of the chains that are open, or
does it start a new one — and it answers with a chain id or with `NEW`, plus an
optional confidence:

```python
from collections.abc import Sequence

from idc.classify import NEW, ChainCandidate, ChunkUnderDecision, Decision

class MyAdapter:
    def decide(
        self, chunk: ChunkUnderDecision, candidates: Sequence[ChainCandidate]
    ) -> Decision:
        if not candidates or chunk.user_message not in candidates[0].last_user_message:
            return Decision.new()
        return Decision.extend(candidates[0].id, confidence=0.8)
```

Three properties of that shape are the point of it:

- **An adapter sees only what every backend can see, and it sees both sides of
  the question.** A `ChainCandidate` is an id, a label and the chain's last user
  message; a `ChunkUnderDecision` is the classified chunk's id and its own user
  message. Neither carries the transcript, a file list or a model handle, and
  both carry the same kind of text — a backend compares like with like.
  Switching the backend therefore changes no code outside the adapter (T4's
  acceptance criterion), and the fake adapter can produce everything a real one
  gets.
- **An adapter decides nothing else.** It does not name the new chain (the store
  assigns a provisional label), it does not order the chains, and it never sees
  the chains it was not offered. Everything that can be computed from the
  transcript is computed by code, not by a model.
- **An answer is either understood or refused.** `as_decision` reads the three
  allowed forms — a `Decision`, `NEW`, a chain id — and raises
  `AdapterContractError` on anything else, including a chain id of the wrong
  shape, so a backend that returns prose fails where it answered instead of
  somewhere downstream — and one `except ClassifyError` catches it.

Nothing in this module calls a model, and nothing in it may: the real backends
are T4.2 (#17) and T4.3 (#18), and every test of everything downstream runs on
`idc.classify.fake.FakeAdapter`.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, Protocol, Self, runtime_checkable

from idc.classify.errors import AdapterContractError
from idc.classify.ids import require_chain_id
from idc.schema import Chain, Chunk


@dataclass(frozen=True, slots=True)
class NewChain:
    """The answer "this chunk starts a new chain".

    A value of its own rather than `None` or an empty string, so that a decision
    can be read without knowing which sentinel the backend happened to use, and
    so that the type of a decision's target says what it is. The value has no
    fields: every `NewChain()` equals the exported `NEW`, which is what an
    adapter that builds its own sentinel relies on.
    """

    def __repr__(self) -> str:
        """Return `NEW`, the name this value is exported under."""
        return "NEW"


NEW: Final[NewChain] = NewChain()
"""The one decision that is not a chain id: open a new chain for this chunk."""


@dataclass(frozen=True, slots=True)
class ChunkUnderDecision:
    """The chunk being classified, as an adapter gets to see it.

    The counterpart of `ChainCandidate`, and there for the same reason: a
    `Chunk` is a half-open range over `Session.messages` and copies no message
    (#3), so an adapter handed the chunk itself could not read one word of what
    it is supposed to classify — while every chain it compares that chunk
    against arrives with its last user message. The store resolves the text and
    hands it over; the adapter never reaches into the transcript.

    A value object rather than a third parameter on `decide`, so that a later
    ticket can offer an adapter one more fact without changing the signature
    every backend is written against.

    Attributes:
        id: The chunk's id, `K<n>` — what the records and the error messages
            name it by.
        user_message: Content of the chunk's last real user turn, verbatim.
            Empty when the chunk holds no user turn at all, which is the case
            for a chunk of subagent work; the same absence that
            `ChainCandidate.last_user_message` spells as the empty string.
    """

    id: str
    user_message: str = ""

    def __post_init__(self) -> None:
        """Reject a chunk without an id.

        Raises:
            ValueError: If `id` is empty — `Chunk.id` is a non-empty
                identifier, and a decision has to be attributable to a chunk.
        """
        if not self.id:
            raise ValueError("the chunk under decision must carry its chunk id")

    @classmethod
    def from_chunk(cls, chunk: Chunk, user_message: str | None = None) -> Self:
        """Return the chunk as an adapter is shown it.

        Args:
            chunk: The chunk being classified.
            user_message: Content of its last real user turn, as the store
                resolved it; `None` when the chunk holds no user turn.

        Returns:
            The value carrying the chunk's id and that text.
        """
        return cls(id=chunk.id, user_message=user_message or "")


@dataclass(frozen=True, slots=True)
class ChainCandidate:
    """An open chain, as an adapter gets to see it.

    Deliberately not the `Chain` itself: a candidate carries what a backend can
    decide from and nothing that would let it reach into the transcript. The
    last user message is the one piece of the chain's content that is offered,
    because it is what says what the chain is currently about — the label alone
    is a provisional truncation until the compactor writes a real one.

    Attributes:
        id: The chain's stable id, `C<n>`.
        label: The chain's label; provisional until T6 replaces it.
        last_user_message: Content of the last real user turn of the chain's
            last chunk, verbatim. Empty when the chain has none — a chain
            carried over from an earlier run, whose chunks are not part of this
            transcript, or a chain whose last chunk holds subagent work.
    """

    id: str
    label: str
    last_user_message: str = ""

    def __post_init__(self) -> None:
        """Reject an id that is not a chain id and an empty label.

        Raises:
            ValueError: If `id` is not of the form `C<n>`, or `label` is empty —
                `Chain.label` is a non-empty identifier, so an empty label could
                never become one.
        """
        require_chain_id(self.id, what="a candidate's chain id")
        if not self.label:
            raise ValueError(f"candidate {self.id!r} must carry a non-empty label")

    @classmethod
    def from_chain(cls, chain: Chain, last_user_message: str = "") -> Self:
        """Return the candidate for an existing chain.

        Args:
            chain: The chain to offer.
            last_user_message: Content of the chain's last user turn, if the
                caller can resolve one.

        Returns:
            The candidate carrying the chain's id and label.
        """
        return cls(id=chain.id, label=chain.label, last_user_message=last_user_message)


@dataclass(frozen=True, slots=True)
class Decision:
    """One adapter answer: a chain to extend or `NEW`, with a confidence.

    An adapter that builds one from what a model wrote should read the id with
    `as_decision` (or check it with `is_chain_id`) first: `as_decision("K1")` is
    an `AdapterContractError` and so a failure of the stage, while
    `Decision.extend("K1")` is a plain `ValueError` — constructing a value
    object with a bad literal is the caller's own mistake, which is the one
    thing `except ClassifyError` does not cover.

    Attributes:
        target: The id of the chain the chunk continues, or `NEW` when it starts
            one.
        confidence: How sure the backend is, between 0 and 1, or `None` when the
            backend reports none. It is carried through to the assignment record
            for the evaluation harness (#19) and is never used to overrule the
            decision itself.
    """

    target: str | NewChain
    confidence: float | None = None

    def __post_init__(self) -> None:
        """Reject an unusable target or confidence.

        Raises:
            ValueError: If `target` is not a chain id and not a `NewChain`, or
                if `confidence` is not `None` and not a number in `[0, 1]`.
        """
        if not isinstance(self.target, NewChain):
            require_chain_id(self.target, what="a decision's target")
        if self.confidence is None:
            return
        if isinstance(self.confidence, bool) or not isinstance(self.confidence, int | float):
            raise ValueError(f"a decision's confidence must be a number; got {self.confidence!r}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"a decision's confidence must lie between 0 and 1; got {self.confidence!r}"
            )

    @property
    def is_new(self) -> bool:
        """Return whether this decision opens a new chain."""
        return isinstance(self.target, NewChain)

    @classmethod
    def new(cls, confidence: float | None = None) -> Self:
        """Return the decision to open a new chain.

        Args:
            confidence: Optional confidence between 0 and 1.

        Returns:
            The decision whose target is `NEW`.
        """
        return cls(target=NEW, confidence=confidence)

    @classmethod
    def extend(cls, chain_id: str, confidence: float | None = None) -> Self:
        """Return the decision to extend an existing chain.

        Args:
            chain_id: Id of the chain the chunk continues, `C<n>`.
            confidence: Optional confidence between 0 and 1.

        Returns:
            The decision whose target is that chain.
        """
        return cls(target=chain_id, confidence=confidence)


DecisionLike = Decision | str | NewChain
"""Everything an adapter may return: a decision, a chain id, or `NEW`.

The interface of #16 is written as `decide(chunk, candidates) -> ChainId | NEW`,
and a backend that answers in that shorter form is read by `as_decision` rather
than rejected. A `Decision` is the normal form, because it is the only one that
can carry a confidence.
"""


def as_decision(value: object) -> Decision:
    """Read an adapter's answer as a `Decision`.

    Args:
        value: What the adapter returned: a `Decision`, `NEW`, or a chain id.

    Returns:
        The decision; a `Decision` is returned unchanged, so the confidence it
        carries survives.

    Raises:
        AdapterContractError: If the value is none of the three allowed forms,
            or is a string that is not a chain id. The message differs — the
            second case names the `C<n>` form, because there the adapter did
            answer in an allowed form and got the id wrong — but the error is
            one, so a caller catches the stage with one `except`.
    """
    if isinstance(value, Decision):
        return value
    if isinstance(value, NewChain):
        return Decision.new()
    if isinstance(value, str):
        return Decision.extend(
            require_chain_id(value, what="an adapter's answer", error=AdapterContractError)
        )
    raise AdapterContractError(
        f"an adapter must answer with a Decision, NEW or a chain id; got {value!r} "
        f"of type {type(value).__name__}"
    )


@runtime_checkable
class IntentAdapter(Protocol):
    """The one method a classification backend has to provide.

    A protocol rather than a base class: an adapter is anything with a `decide`
    method of this shape, so #17 and #18 inherit nothing and the fake is not a
    special case. It is `runtime_checkable` so that a test can assert the shape
    without importing anything of the stage — but only the *presence* of a
    `decide` attribute is checkable at runtime, not its signature, so the
    classifier does not use `isinstance` as a guard and nothing here promises
    that it does.
    """

    def decide(
        self, chunk: ChunkUnderDecision, candidates: Sequence[ChainCandidate]
    ) -> DecisionLike:
        """Decide whether `chunk` continues one of `candidates` or starts a chain.

        Args:
            chunk: The chunk to classify — its id and its own last user message.
                The adapter is given neither the session nor the chunk's
                indices: what a backend needs from the transcript is resolved
                for it and handed over explicitly, which is what keeps a
                decision reproducible and a backend swappable.
            candidates: The open chains, in the order they were opened. All
                chains stay candidates until the end of the transcript, so this
                grows as the run proceeds. The classifier passes a list.

        Returns:
            The decision: a `Decision`, a chain id of one of the candidates, or
            `NEW`.
        """
        ...
