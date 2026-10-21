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
  backends are T4.2 (#17) and T4.3 (#18); **nothing in it calls a model**, and
  no test ever calls one.
- `idc.classify.store` is everything the transcript already answers: the chain
  ids (`C<n>` by position, so a re-run over a transcript that only grew keeps
  them), the provisional labels, the candidate list, and the refusal of a
  decision that would put a chunk into two chains or a chain out of transcript
  order.
- `idc.classify.classifier` is the sequential loop over the chunks, and
  `idc.classify.fake` the scripted adapter that keeps every downstream test —
  the end-to-end one above all (#42) — free of a model and of the network.
- `idc.classify.llm` is the LLM backend (T4.2, #17): a fixed prompt, a validated
  JSON answer and a bounded retry, on top of a `ModelClient` **the caller
  injects**. It ships no client, opens no connection and reads no credential.
- `idc.classify.gold` and `idc.classify.evaluate` are the stage's measuring
  instrument (T4.4, #19): a hand-written gold labelling per transcript, and the
  per-chunk accuracy and chain-level Rand index of a run against it. The labels
  are read from an artifact nothing here writes, and both metrics are arithmetic
  over two labellings — so the harness neither scores itself nor calls a model.
  Every result carries the `RunSource` it was produced by, because the gold
  baseline scores 1.000 by construction and must not read like a measurement.
- `idc.classify.s1m` is the other real backend (T4.3, #18): a binary scorer
  asked once per candidate, thresholded and ranked into one decision. Its
  scorer is **injected** too, so that module holds no client, no transport and
  no credential either.

Both backends take their model as an injected dependency, which is why **no
test of this repository calls a model** even though two modules can ask one.

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
    DecisionRecordError,
    ForeignChunkError,
    GoldCoverageError,
    GoldLabellingError,
    ScriptExhaustedError,
    UnknownChainError,
)
from idc.classify.evaluate import (
    CHUNKINGS,
    EXIT_INPUT_ERROR,
    Agreement,
    CorpusAgreement,
    Disagreement,
    EvaluationCase,
    RunSource,
    agreement,
    chunks_for,
    evaluate_adapter,
    evaluate_cases,
    format_report,
    load_cases,
    load_decisions,
    per_chunk_accuracy,
    predicted_labels,
    rand_index,
)
from idc.classify.fake import FakeAdapter, FakeCall
from idc.classify.gold import (
    GOLD_CHUNKING,
    GoldChains,
    gold_script,
    load_gold,
    load_gold_directory,
)
from idc.classify.ids import CHAIN_ID_PREFIX, chain_id, is_chain_id
from idc.classify.llm import (
    DEFAULT_MAX_ATTEMPTS,
    LlmAdapter,
    LlmCall,
    ModelClient,
    UnreadableAnswerError,
    build_prompt,
)
from idc.classify.s1m import (
    DEFAULT_CONTINUATION_THRESHOLD,
    ContinuationScorer,
    S1MAdapter,
    S1MCall,
    ScorerContractError,
)
from idc.classify.store import (
    PROVISIONAL_LABEL_MAX_LENGTH,
    ChainStore,
    chunk_user_message,
    provisional_label,
)

__all__ = [
    "CHAIN_ID_PREFIX",
    "CHUNKINGS",
    "DEFAULT_MAX_ATTEMPTS",
    "EXIT_INPUT_ERROR",
    "GOLD_CHUNKING",
    "NEW",
    "PROVISIONAL_LABEL_MAX_LENGTH",
    "AdapterContractError",
    "Agreement",
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
    "CorpusAgreement",
    "Decision",
    "DecisionLike",
    "DecisionRecordError",
    "Disagreement",
    "EvaluationCase",
    "FakeAdapter",
    "FakeCall",
    "ForeignChunkError",
    "GoldChains",
    "GoldCoverageError",
    "GoldLabellingError",
    "IntentAdapter",
    "LlmAdapter",
    "LlmCall",
    "ModelClient",
    "NewChain",
    "RunSource",
    "ScriptExhaustedError",
    "UnknownChainError",
    "UnreadableAnswerError",
    "agreement",
    "as_decision",
    "build_prompt",
    "chain_id",
    "chunk_user_message",
    "chunks_for",
    "classify",
    "classify_chunks",
    "evaluate_adapter",
    "evaluate_cases",
    "format_report",
    "gold_script",
    "is_chain_id",
    "load_cases",
    "load_decisions",
    "load_gold",
    "load_gold_directory",
    "per_chunk_accuracy",
    "predicted_labels",
    "provisional_label",
    "rand_index",
]

# T4.3 (#18) — the S1M backend's own names, appended as a block of their own.
# #17 (the LLM adapter) and #18 extend this same `__all__` from two branches at
# once, and one sorted literal that both have to edit is a merge conflict by
# construction; the precedent for deriving around that is 786d431. Each block
# is self-contained, so keeping both is the whole resolution.
__all__ += [
    "DEFAULT_CONTINUATION_THRESHOLD",
    "ContinuationScorer",
    "S1MAdapter",
    "S1MCall",
    "ScorerContractError",
]
