"""The adapter contract: what a classification backend is asked, and may answer.

This is the interface the whole stage is built around, and it is deliberately
the smallest one that can carry a decision. An adapter is asked, per chunk,
**one** question — does this chunk continue one of the chains that are open, or
does it start a new one — and it answers with a chain id or with `NEW`, plus an
optional confidence:

```python
from idc.classify import NEW, ChainCandidate, Decision

class MyAdapter:
    def decide(self, chunk: Chunk, candidates: list[ChainCandidate]) -> Decision:
        if not candidates:
            return Decision.new()
        return Decision.extend(candidates[0].id, confidence=0.8)
```

Three properties of that shape are the point of it:

- **An adapter sees only what every backend can see.** A `ChainCandidate` is an
  id, a label and the chain's last user message — no transcript, no file list,
  no model handle. Switching the backend therefore changes no code outside the
  adapter (T4's acceptance criterion), and the fake adapter can produce
  everything a real one gets.
- **An adapter decides nothing else.** It does not name the new chain (the store
  assigns a provisional label), it does not order the chains, and it never sees
  the chains it was not offered. Everything that can be computed from the
  transcript is computed by code, not by a model.
- **An answer is either understood or refused.** `as_decision` reads the three
  allowed forms — a `Decision`, `NEW`, a chain id — and raises
  `AdapterContractError` on anything else, so a backend that returns prose fails
  where it answered instead of somewhere downstream.

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
        AdapterContractError: If the value is none of the three allowed forms.
        ValueError: If the value is a string that is not a chain id — then the
            adapter did answer in the allowed form and got the id wrong, which
            is the more precise message.
    """
    if isinstance(value, Decision):
        return value
    if isinstance(value, NewChain):
        return Decision.new()
    if isinstance(value, str):
        return Decision.extend(value)
    raise AdapterContractError(
        f"an adapter must answer with a Decision, NEW or a chain id; got {value!r} "
        f"of type {type(value).__name__}"
    )


@runtime_checkable
class IntentAdapter(Protocol):
    """The one method a classification backend has to provide.

    A protocol rather than a base class: an adapter is anything with a `decide`
    method of this shape, so #17 and #18 inherit nothing and the fake is not a
    special case. `isinstance` works against it, which is what lets the
    classifier report a backend that is not an adapter.
    """

    def decide(self, chunk: Chunk, candidates: Sequence[ChainCandidate]) -> DecisionLike:
        """Decide whether `chunk` continues one of `candidates` or starts a chain.

        Args:
            chunk: The chunk to classify, as a range over the session's
                messages. The adapter is given the chunk, not the session: what
                a backend needs from the transcript it has to be handed
                explicitly, so that a decision stays reproducible.
            candidates: The open chains, in the order they were opened. All
                chains stay candidates until the end of the transcript, so this
                grows as the run proceeds. The classifier passes a list.

        Returns:
            The decision: a `Decision`, a chain id of one of the candidates, or
            `NEW`.
        """
        ...
