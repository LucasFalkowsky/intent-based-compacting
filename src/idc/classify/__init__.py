"""Classification of chunks into intent chains, through a pluggable adapter.

A chain is a sequence of chunks that pursue the same user intent. The chunks are
walked in transcript order, and for each one a decision backend is asked the one
question the stage is about: does this chunk continue one of the chains that are
open, or does it start a new one. All chains stay candidates until the end of
the transcript, so an intent the user returns to later is *extended* rather than
summarized a second time — which is why chains exist here instead of a running
counter:

```python
from idc.chunk import chunk_session
from idc.classify import NEW, FakeAdapter, classify
from idc.schema import Session, load

session = load(Session, "session.json")
chunks = chunk_session(session)
chains = classify(session, chunks, FakeAdapter([NEW, NEW, "C1"]))
#   C1  K1, K3   the intent the third turn returns to
#   C2  K2       the unrelated one in between
```

The stage is split along what a model may decide and what it may not:

- `idc.classify.adapters` is the **contract**: `decide(chunk, candidates)`
  answers with a chain id or `NEW` plus an optional confidence. A
  `ChunkUnderDecision` shows an adapter the classified chunk's id and its own
  last user message, a `ChainCandidate` shows it each open chain's id, label and
  last user message — nothing else, and nothing that reaches into the
  transcript. Switching the backend therefore changes no code outside the
  adapter, and `FakeAdapter` can produce everything a real backend can. The real
  backends are T4.2 (#17) and T4.3 (#18); **nothing in this package calls a
  model**, and no test ever does.
- `idc.classify.store` is everything the transcript already answers: the chain
  ids (`C<n>` by position, so a re-run over a transcript that only grew keeps
  them), the provisional labels, the candidate list, and the refusal of a
  decision that would put a chunk into two chains or a chain out of transcript
  order.
- `idc.classify.classifier` is the sequential loop over the chunks, and
  `idc.classify.fake` the scripted adapter that keeps every downstream test —
  the end-to-end one above all (#42) — free of a model and of the network.

Every refusal the stage itself raises is a `ClassifyError`, so a caller catches
all of them with one `except` — an adapter answering a chain id of the wrong
shape included, which is why `AdapterContractError` is a `ValueError` as well,
and a chain seeded from an existing summary block included, because that chain
was read out of a transcript file (#33) and is input rather than a programmer's
literal. The message says which mistake was made; there is one family to catch.

Three things are deliberately *outside* that family, so a caller that has to
survive them handles them on their own:

- **A value object built with a bad literal** — `Decision(target="K1")`,
  `Decision.extend("K1")`, `ChainCandidate(id="K1", …)` — raises a plain
  `ValueError`, like `SubagentSpan` in `idc.chunk`: the mistake is in the
  caller's own construction, not in an answer the stage received. An adapter
  that turns what a model wrote into a decision should therefore hand the
  string to `as_decision`, or check it with `is_chain_id`, rather than
  construct a `Decision` from it — `as_decision("K1")` *is* an
  `AdapterContractError`, `Decision.extend("K1")` is not.
- **An exception an adapter raised itself** reaches the caller unwrapped. Only
  the backend knows what its own timeout or authentication failure means, and
  presenting it as a `ClassifyError` would read as a classification problem.
- **An object that is no adapter at all** fails with `AttributeError` on the
  missing `decide`. `IntentAdapter` is `runtime_checkable`, but that compares
  attribute names only, so nothing here checks the shape or claims to.
"""

from idc.classify.adapters import (
    NEW,
    ChainCandidate,
    ChunkUnderDecision,
    Decision,
    DecisionLike,
    IntentAdapter,
    NewChain,
    as_decision,
)
from idc.classify.classifier import Assignment, Classification, classify, classify_chunks
from idc.classify.errors import (
    AdapterContractError,
    BlankChainLabelError,
    ChainAlreadySeededError,
    ChainOrderError,
    ChunkAlreadyAssignedError,
    ChunkOrderError,
    ClassifyError,
    ForeignChunkError,
    ScriptExhaustedError,
    UnknownChainError,
)
from idc.classify.fake import FakeAdapter, FakeCall
from idc.classify.ids import CHAIN_ID_PREFIX, chain_id, is_chain_id
from idc.classify.store import (
    PROVISIONAL_LABEL_MAX_LENGTH,
    ChainStore,
    chunk_user_message,
    provisional_label,
)

__all__ = [
    "CHAIN_ID_PREFIX",
    "NEW",
    "PROVISIONAL_LABEL_MAX_LENGTH",
    "AdapterContractError",
    "Assignment",
    "BlankChainLabelError",
    "ChainAlreadySeededError",
    "ChainCandidate",
    "ChainOrderError",
    "ChainStore",
    "ChunkAlreadyAssignedError",
    "ChunkOrderError",
    "ChunkUnderDecision",
    "Classification",
    "ClassifyError",
    "Decision",
    "DecisionLike",
    "FakeAdapter",
    "FakeCall",
    "ForeignChunkError",
    "IntentAdapter",
    "NewChain",
    "ScriptExhaustedError",
    "UnknownChainError",
    "as_decision",
    "chain_id",
    "chunk_user_message",
    "classify",
    "classify_chunks",
    "is_chain_id",
    "provisional_label",
]
