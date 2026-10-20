"""Sequential assignment: one decision per chunk, from the first to the last.

The stage's entry point. The chunks are walked in transcript order, and for each
one the adapter is asked whether it continues one of the chains that are open or
starts a new one; the chain store carries out the answer. Nothing else happens
here — which is what makes the result deterministic for a deterministic adapter
(T4's first acceptance criterion):

```python
from idc.classify import FakeAdapter, classify, classify_chunks

chains = classify(session, chunks, FakeAdapter([NEW, NEW, "C1"]))
result = classify_chunks(session, chunks, adapter)   # chains plus the record
result.assignments[2]                                 # which chunk went where
```

Three things are deliberate:

- **Every chunk is classified**, the latest one included. It is rendered
  verbatim instead of compacted (T7), but it still belongs to an intent, and the
  chain it belongs to is what the next run extends. A chunk of subagent work is
  classified like any other chunk, too.
- **Both halves of the question come from the store**: the candidates, read
  before every call so that a chain opened for chunk *n* is on offer for chunk
  *n+1*, and the `ChunkUnderDecision` the adapter decides about. All chains stay
  candidates until the transcript ends; none is ever closed.
- **The order of the answers is not trusted, the order of the chunks is.** The
  store refuses a chunk that would land in two chains or out of transcript order
  (`idc.classify.store`), so a model cannot produce a chain list that the
  resolver would then read wrongly.

`classify` returns the chains, which is the signature T4 states. `classify_chunks`
returns the same chains plus one `Assignment` per chunk — the decision and its
confidence, which is what the evaluation harness (#19) compares against a
reference labelling.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from idc.classify.adapters import IntentAdapter, NewChain, as_decision
from idc.classify.store import ChainStore
from idc.schema import Chain, Chunk, Session


@dataclass(frozen=True, slots=True)
class Assignment:
    """What was decided for one chunk.

    Attributes:
        chunk_id: `Chunk.id` the decision was made for.
        chain_id: `Chain.id` the chunk was assigned to.
        opened: Whether that chain was opened for this chunk, rather than
            extended by it.
        confidence: The adapter's confidence, or `None` when it reported none.
    """

    chunk_id: str
    chain_id: str
    opened: bool
    confidence: float | None = None


@dataclass(frozen=True, slots=True)
class Classification:
    """The result of classifying a transcript: its chains and the decisions.

    Attributes:
        chains: The chains, in the order they were opened.
        assignments: One assignment per classified chunk, in transcript order.
    """

    chains: tuple[Chain, ...]
    assignments: tuple[Assignment, ...]

    def chain_of(self, chunk_id: str) -> str | None:
        """Return the id of the chain a chunk was assigned to, or `None`.

        Args:
            chunk_id: `Chunk.id` to look up.

        Returns:
            The chain's id, or `None` when no assignment names that chunk.
        """
        for assignment in self.assignments:
            if assignment.chunk_id == chunk_id:
                return assignment.chain_id
        return None


def classify_chunks(
    session: Session,
    chunks: Sequence[Chunk],
    adapter: IntentAdapter,
    *,
    candidates: Iterable[Chain] = (),
) -> Classification:
    """Assign every chunk to an intent chain, one decision at a time.

    Args:
        session: The transcript the chunks range over. It is needed here, and
            not only in T4's stated signature, because a chunk copies no message
            while both halves of an adapter's question carry text — the chunk's
            own user message and each candidate's last one — and that text lives
            in `Session.messages`.
        chunks: The chunks in transcript order, as `idc.chunk` produced them.
        adapter: The decision backend; `FakeAdapter` in tests, an LLM (#17) or
            S1M (#18) adapter in production.
        candidates: Chains carried over from an earlier run, as T8 (#33) parsed
            them out of an existing summary block. They are offered as
            candidates from the first chunk on and keep their ids and summaries.

    Returns:
        The chains and one assignment per chunk.

    Raises:
        AdapterContractError: If the adapter answers with something that is
            neither a `Decision`, a chain id nor `NEW`.
        UnknownChainError: If it names a chain that was not a candidate.
        ClassifyError: If the decision cannot be carried out — see
            `idc.classify.store` for the chunk a chain cannot take.
    """
    store = ChainStore(session, candidates)
    assignments: list[Assignment] = []
    for chunk in chunks:
        decision = as_decision(adapter.decide(store.subject(chunk), store.candidates()))
        target = decision.target
        if isinstance(target, NewChain):
            chain = store.open_chain(chunk)
        else:
            chain = store.append(target, chunk)
        assignments.append(
            Assignment(
                chunk_id=chunk.id,
                chain_id=chain.id,
                opened=decision.is_new,
                confidence=decision.confidence,
            )
        )
    return Classification(chains=tuple(store.chains()), assignments=tuple(assignments))


def classify(
    session: Session,
    chunks: Sequence[Chunk],
    adapter: IntentAdapter,
    *,
    candidates: Iterable[Chain] = (),
) -> list[Chain]:
    """Assign every chunk to an intent chain and return the chains.

    The one-call form of the stage; `classify_chunks` is the same run with the
    per-chunk decisions kept.

    Args:
        session: The transcript the chunks range over.
        chunks: The chunks in transcript order.
        adapter: The decision backend.
        candidates: Chains carried over from an earlier run.

    Returns:
        The chains in the order they were opened, which is the transcript order
        of their first chunks. Empty exactly when `chunks` is empty and no chain
        was carried over.

    Raises:
        ClassifyError: If a decision cannot be read or carried out; see
            `classify_chunks`.
    """
    return list(classify_chunks(session, chunks, adapter, candidates=candidates).chains)
