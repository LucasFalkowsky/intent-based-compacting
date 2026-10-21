"""The S1M adapter: one binary judgment per candidate, thresholded and ranked.

An S1M backend is a *small* model that answers one closed question with a
probability, and the question this stage has for it is the narrowest one that
still decides anything: **does this chunk continue this chain?** That is asked
once per open chain — `k` questions for `k` candidates — and this module is what
turns the `k` answers into the one `Decision` the classifier needs:

```python
from idc.classify import S1MAdapter, classify

adapter = S1MAdapter(score_continuation, threshold=0.6)
chains = classify(session, chunks, adapter)
adapter.score_calls          # what the run cost: one call per open chain, per chunk
adapter.calls[2].scores      # every score the third decision was made on
```

The alternative — asking one model to pick among the candidates in a single
call — is T4.2 (#17). This one trades `k` cheap calls for a prompt that never
grows, which is the shape a small model can be accurate on; what it buys has to
be paid for per chunk, so the cost is part of the contract and not an accident
of the implementation. Four rules make it reproducible:

- **Exactly `k` calls, in the order the candidates were offered, with no
  short-circuit.** A run over a transcript costs the number of open chains at
  each chunk, summed — computable before the run and independent of the data.
  Stopping at the first candidate that passes would be cheaper and would make
  the cost depend on the candidate order *and* pick a candidate that is merely
  good enough over the best one; both are refused here.
- **A score at the threshold passes**: the comparison is `score >= threshold`,
  pinned on both sides. The threshold is configurable, defaults to
  `DEFAULT_CONTINUATION_THRESHOLD`, and is the only knob — nothing else about
  the decision changes with it.
- **Among the candidates that pass, the highest score wins; a tie goes to the
  one offered first.** The tie-break is positional and reads the order it is
  handed, which is what makes it deterministic for any caller.
  `ChainStore.candidates()` is documented to be in opening order, so on the
  classify path "offered first" means the oldest chain — the adapter relies on
  that documented order for its *meaning*, never for its determinism.
- **A score that is no probability stops the decision.** Outside `[0, 1]`,
  `NaN`, infinity, a bool or anything that is not a number is a
  `ScorerContractError` — a `ClassifyError`, so `except ClassifyError` covers
  it — raised at the candidate it came from, before the candidates behind it are
  asked and before anything is ranked. A scorer that raises an exception of its
  own is *not* wrapped: only the backend knows what its own timeout or
  authentication failure means, which is the carve-out `idc.classify`
  documents.

The confidence a decision carries is the scorer's own number for the answer that
was given: the winning score for an extension, `1 - highest score` for `NEW`.
Both are therefore on one scale and neither is scaled by the threshold. With no
candidate at all the best score is 0 and the confidence in `NEW` is 1.

**What a recorded run can be re-thresholded from — and the limits of it.**
Because the confidence is unscaled, a recorded decision can be re-evaluated at
another threshold without calling the backend again — but only from
`S1MCall.scores`, which holds every score exactly as it was given. The
confidence a `NEW` carries is *not* a substitute for them, in three ways that
the evaluation harness (#19) has to work around rather than around which it can
be written:

- `1 - highest` is **lossy**: `1 - (1 - x) != x` for 32 of the 101 two-decimal
  scores, so a sweep over a recorded confidence answers `NEW` where re-running
  the backend answers `EXTEND`.
- A recorded `NEW` **cannot say which chain** a lower threshold would have
  extended: it is one number for `k` candidates, and the per-candidate scores
  are the only place that information exists.
- **No recording supports a whole-transcript sweep at all**, because the
  candidate set is itself threshold-dependent: a lower threshold extends a
  chain where a higher one opens a second, so the run asks a different number of
  questions — including scores the recording simply does not contain.

**The scorer is injected and this module contains no backend.** It calls nothing
but the callable it was handed, holds no client, no transport and no credential,
and every test of it runs on a stub — which is what keeps the suite, the
end-to-end test (#42) included, free of the network.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from idc.classify.adapters import ChainCandidate, ChunkUnderDecision, Decision
from idc.classify.errors import ClassifyError

DEFAULT_CONTINUATION_THRESHOLD = 0.5
"""The score from which a candidate counts as continued, when none is given.

The midpoint of the scale, which is what a binary yes/no judgment reported as a
probability means by itself: at least as likely to continue the chain as not.
The study varies it; the default is the one value that needs no justification
beyond the scale.
"""


class ScorerContractError(ClassifyError, ValueError):
    """A continuation scorer answered with something that is no probability.

    The sibling of `AdapterContractError` one level down: there an *adapter*
    answered the classifier with something unusable, here a *scorer* answered
    the adapter. Both are a bad value and a failure of the stage, so both are a
    `ValueError` **and** a `ClassifyError` — a caller never has to catch two
    families for one event.

    Ranking what came before the broken answer would turn a backend that is
    malfunctioning into a classification nobody decided, so the decision is
    abandoned instead.

    Attributes:
        chunk_id: The chunk the decision was being made for.
        candidate_id: The candidate the scorer was asked about.
        score: What it answered, so that the message shows the value.
    """

    def __init__(self, chunk_id: str, candidate_id: str, score: object) -> None:
        """Build the error from the question asked and the answer given.

        Args:
            chunk_id: The chunk the decision was being made for.
            candidate_id: The candidate the scorer was asked about.
            score: What the scorer answered.
        """
        self.chunk_id = chunk_id
        self.candidate_id = candidate_id
        self.score = score
        super().__init__(
            f"the scorer answered {score!r} for chunk {chunk_id!r} and candidate "
            f"{candidate_id!r}; a continuation score must be a number between 0 and 1"
        )


class ContinuationScorer(Protocol):
    """The one question an S1M backend is asked, as a callable.

    A protocol rather than a base class, for the reason `IntentAdapter` is one:
    a scorer is anything callable of this shape — a bound method, a closure over
    a client, a stub in a test — and it inherits nothing. The two arguments are
    passed **positionally**, so a scorer may name them freely.
    """

    def __call__(self, chunk: ChunkUnderDecision, candidate: ChainCandidate) -> float:
        """Return how likely it is that `chunk` continues `candidate`.

        Args:
            chunk: The chunk being classified — its id and its own last user
                message.
            candidate: One open chain, with its id, label and last user message.

        Returns:
            A probability between 0 and 1 inclusive. Anything else — a value
            outside that range, `NaN`, infinity, a bool, a non-number — is a
            `ScorerContractError`; a scorer that cannot answer should raise its
            own exception instead, which reaches the caller unwrapped.
        """
        ...


@dataclass(frozen=True, slots=True)
class S1MCall:
    """One decision the adapter made, and every score it rests on.

    The counterpart of `FakeCall`, for a backend that is asked `k` questions
    instead of one. It is what makes a run auditable after the fact: the scores
    are exact, so the evaluation harness (#19) can re-threshold **this
    decision** from them — the one re-evaluation a recording supports, see the
    module docstring — and the measurement stage can read the cost off their
    number.

    Attributes:
        chunk_id: `Chunk.id` the decision was made for.
        scores: One `(candidate id, score)` pair per candidate, in the order
            they were scored — which is the order they were offered in. Empty
            when there was no candidate.
        decision: The decision the scores produced.
    """

    chunk_id: str
    scores: tuple[tuple[str, float], ...]
    decision: Decision

    @property
    def candidate_ids(self) -> tuple[str, ...]:
        """Return the ids of the candidates that were scored, in that order."""
        return tuple(candidate_id for candidate_id, _ in self.scores)


def _require_probability(value: object, *, what: str) -> float:
    """Return `value` as a probability, or raise naming what it must be.

    Args:
        value: The value to check.
        what: How the value is named in the message.

    Returns:
        The value as a float.

    Raises:
        ValueError: If the value is not a number in `[0, 1]`. `NaN` and the
            infinities are refused by the range comparison itself —
            `0.0 <= nan <= 1.0` is false — so they need no check of their own.
            A bool is refused rather than read as 1 or 0, which is the guard
            `Decision` applies to a confidence and for the same reason: a
            yes/no answer carrying no probability cannot be ranked. An `int`
            *is* accepted, because 1 and 0 are what a binary judgment reports
            most naturally, and comes back as a float.
    """
    if isinstance(value, bool) or not isinstance(value, int | float) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{what} must be a number between 0 and 1; got {value!r}")
    return float(value)


class S1MAdapter:
    """An `IntentAdapter` that asks a binary scorer once per candidate.

    Args:
        scorer: The backend, called as `scorer(chunk, candidate)` once per
            candidate. Everything that reaches a model is behind it; this class
            holds no client of its own.
        threshold: The score from which a candidate counts as continued, the
            comparison being `score >= threshold`. Defaults to
            `DEFAULT_CONTINUATION_THRESHOLD`.

    Raises:
        ValueError: If `threshold` is not a number between 0 and 1. A plain
            `ValueError` and deliberately not a `ClassifyError`: the threshold
            is the caller's own literal, which is the carve-out `idc.classify`
            documents for `Decision.extend` as well. A threshold that reaches
            the tool from a command line is T10's to validate before it gets
            here.
    """

    def __init__(
        self,
        scorer: ContinuationScorer,
        *,
        threshold: float = DEFAULT_CONTINUATION_THRESHOLD,
    ) -> None:
        """Bind the scorer and the threshold, refusing a threshold that is none."""
        self._scorer = scorer
        self._threshold = _require_probability(threshold, what="a continuation threshold")
        self._calls: list[S1MCall] = []
        self._score_calls = 0

    def __repr__(self) -> str:
        """Return a representation naming the scorer, the threshold and the cost."""
        return (
            f"{type(self).__name__}(scorer={self._scorer!r}, threshold={self._threshold!r}, "
            f"score_calls={self._score_calls})"
        )

    @property
    def scorer(self) -> ContinuationScorer:
        """Return the scorer the adapter was given."""
        return self._scorer

    @property
    def threshold(self) -> float:
        """Return the score from which a candidate counts as continued."""
        return self._threshold

    @property
    def calls(self) -> tuple[S1MCall, ...]:
        """Return every **completed** decision, in order.

        A decision the scorer broke off — a score that is no probability, an
        exception of the backend's own — leaves no entry: there is no decision
        to record. `score_calls` still counts what that attempt cost.
        """
        return tuple(self._calls)

    @property
    def score_calls(self) -> int:
        """Return how often the scorer was called, the broken-off attempts included.

        The cost of the run so far, in model calls: `k` per classified chunk,
        with `k` the number of chains that were open at it.
        """
        return self._score_calls

    def decide(self, chunk: ChunkUnderDecision, candidates: Sequence[ChainCandidate]) -> Decision:
        """Score every candidate and answer with the best one that passes.

        Args:
            chunk: The chunk to classify, with its own last user message.
            candidates: The open chains, in the order they were opened. Every
                one of them is scored, even once an earlier one has passed.

        Returns:
            The decision to extend the highest-scoring candidate at or above the
            threshold — the one offered first among equal scores — with that
            score as its confidence; or `Decision.new` when no candidate passes,
            with `1 - the highest score` as its confidence. With no candidate at
            all, the confidence in the new chain is 1.

        Raises:
            ScorerContractError: If the scorer answers something that is no
                probability. The candidates behind it are not asked and nothing
                is ranked.
        """
        scores = tuple(self._score(chunk, candidate) for candidate in candidates)
        decision = self._decide_from(scores)
        self._calls.append(S1MCall(chunk_id=chunk.id, scores=scores, decision=decision))
        return decision

    def _score(self, chunk: ChunkUnderDecision, candidate: ChainCandidate) -> tuple[str, float]:
        """Ask the scorer about one candidate and return its id with the score.

        Args:
            chunk: The chunk being classified.
            candidate: The candidate to score.

        Returns:
            The candidate's id and the score, as a pair.

        Raises:
            ScorerContractError: If the answer is no probability.
        """
        self._score_calls += 1
        answer = self._scorer(chunk, candidate)
        try:
            score = _require_probability(answer, what="a continuation score")
        except ValueError as error:
            raise ScorerContractError(chunk.id, candidate.id, answer) from error
        return candidate.id, score

    def _decide_from(self, scores: Sequence[tuple[str, float]]) -> Decision:
        """Return the decision the scores imply.

        Args:
            scores: One `(candidate id, score)` pair per candidate, in the order
                they were offered.

        Returns:
            The extension of the best passing candidate, or a new chain.
        """
        best: tuple[str, float] | None = None
        for candidate_id, score in scores:
            # `>=` is the boundary: a score *at* the threshold passes. `>` on the
            # running best is the tie-break: among equal scores the candidate
            # offered first — the oldest chain, on the classify path — wins.
            if score >= self._threshold and (best is None or score > best[1]):
                best = (candidate_id, score)
        if best is not None:
            return Decision.extend(best[0], confidence=best[1])
        highest = max((score for _, score in scores), default=0.0)
        return Decision.new(confidence=1.0 - highest)
