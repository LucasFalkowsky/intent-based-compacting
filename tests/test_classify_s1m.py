"""The S1M adapter: one binary judgment per candidate, ranked and thresholded.

The deliverable of T4.3 (#18). The backend here is **not** asked "which of these
chains does this chunk continue" — it is asked, once per open chain, the one
binary question a small model can answer well: *does this chunk continue this
chain?*, answered with a probability. The adapter is what turns `k` such answers
into the one decision the stage needs, and the three rules that does by are what
these tests pin:

- **how often the scorer is called** — exactly once per candidate, in the order
  the candidates were offered, with no short-circuit once one of them passes.
  That is the study's cost model (`k` calls per chunk, `k` = the number of open
  chains), so it is a pinned property and not an implementation detail;
- **where the threshold cuts** — a score *at* the threshold passes, and both
  sides of that boundary are pinned, because a threshold whose comparison
  nobody wrote down silently changes meaning between runs;
- **which of several passing candidates wins** — the highest score, and on
  equal scores the candidate that was offered first, deterministically.

The scorer is **injected**, and every scorer in this file is a stub: nothing in
this package calls a model and no test ever does (#42 depends on it). What a
real scorer would be — a model call, an HTTP client, an API key — is pinned to
stay out of the module by `test_the_module_holds_no_backend_of_its_own`.
"""

import importlib
import inspect
import math
from collections.abc import Mapping, Sequence

import pytest

from fixture_corpus import CORPUS
from idc.chunk import chunk_session
from idc.classify import (
    DEFAULT_CONTINUATION_THRESHOLD,
    NEW,
    ChainCandidate,
    ChunkUnderDecision,
    ClassifyError,
    Decision,
    IntentAdapter,
    S1MAdapter,
    S1MCall,
    ScorerContractError,
    as_decision,
    classify,
    classify_chunks,
)
from idc.schema import dumps

FORBIDDEN_TOKENS = [
    "httpx",
    "requests",
    "urllib",
    "socket",
    "subprocess",
    "openai",
    "anthropic",
    "os.environ",
    "getenv",
    "api_key",
]
"""Everything a real backend would need and this module may not hold.

The adapter takes its scorer as an argument; a model client, a transport or a
credential appearing in here would make the test suite one `pip install` away
from a network call.
"""


MESSAGE = "make the retry backoff configurable"
"""The chunk text the stub scorers are asked about; its content is never read."""


def subject(identifier: str = "K1", user_message: str = MESSAGE) -> ChunkUnderDecision:
    """Return the chunk under decision the stub scorers are asked about."""
    return ChunkUnderDecision(id=identifier, user_message=user_message)


def candidates(*identifiers: str) -> tuple[ChainCandidate, ...]:
    """Return one candidate per id, in the order given — the opening order."""
    return tuple(
        ChainCandidate(
            id=identifier,
            label=f"intent {identifier}",
            last_user_message=f"about {identifier}",
        )
        for identifier in identifiers
    )


class TableScorer:
    """A scorer that answers from a table, recording every question it was asked.

    Keyed by candidate id, or by `(chunk id, candidate id)` where a test needs
    the answer to depend on both. An id the table does not hold is answered with
    `default`, so a test writes down only the scores it means.
    """

    def __init__(self, table: Mapping[object, object] | None = None, default: object = 0.0) -> None:
        """Bind the table and the answer for a candidate it does not hold."""
        self.table = dict(table or {})
        self.default = default
        self.calls: list[tuple[str, str]] = []
        self.texts: list[tuple[str, str]] = []

    def __call__(self, chunk: ChunkUnderDecision, candidate: ChainCandidate) -> object:
        """Return the tabled answer, recording the question and both texts."""
        self.calls.append((chunk.id, candidate.id))
        self.texts.append((chunk.user_message, candidate.last_user_message))
        if (chunk.id, candidate.id) in self.table:
            return self.table[(chunk.id, candidate.id)]
        return self.table.get(candidate.id, self.default)


class ExplodingScorer:
    """A scorer that raises its own exception at the `at`-th candidate."""

    class BackendError(Exception):
        """What a real backend's timeout or authentication failure stands in as."""

    def __init__(self, at: int = 1, score: float = 0.9) -> None:
        """Bind which call raises and what the others answer."""
        self.at = at
        self.score = score
        self.calls: list[str] = []

    def __call__(self, chunk: ChunkUnderDecision, candidate: ChainCandidate) -> float:
        """Answer `score`, or raise the backend's own exception at call `at`."""
        self.calls.append(candidate.id)
        if len(self.calls) == self.at:
            raise self.BackendError("the backend is down")
        return self.score


class OverlapScorer:
    """A scorer that answers from the two texts alone, deterministically.

    The crude counterpart of a real S1M judgment: the Jaccard overlap of the
    content words of the chunk's user message and the candidate's. What it is
    here for is that a scorer *can* decide from nothing but `decide`'s two
    arguments — the property the whole adapter contract rests on.
    """

    def __init__(self) -> None:
        """Start with an empty record of the questions it was asked."""
        self.calls: list[tuple[str, str]] = []

    def __call__(self, chunk: ChunkUnderDecision, candidate: ChainCandidate) -> float:
        """Return the word overlap of the two texts, between 0 and 1."""
        self.calls.append((chunk.id, candidate.id))
        left = self._words(chunk.user_message)
        right = self._words(candidate.last_user_message)
        if not left or not right:
            return 0.0
        return len(left & right) / len(left | right)

    @staticmethod
    def _words(text: str) -> set[str]:
        token = ""
        words: set[str] = set()
        for character in f"{text.lower()} ":
            if character.isalnum():
                token += character
                continue
            if len(token) > 4:
                words.add(token)
            token = ""
        return words


# ---------------------------------------------------------------------------
# the cost model: k calls, in order, without a short-circuit
# ---------------------------------------------------------------------------


def test_the_scorer_is_asked_once_per_candidate_in_the_order_they_were_offered():
    scorer = TableScorer({"C2": 0.9})
    adapter = S1MAdapter(scorer)

    adapter.decide(subject(), candidates("C1", "C2", "C3"))

    assert scorer.calls == [("K1", "C1"), ("K1", "C2"), ("K1", "C3")]
    assert adapter.score_calls == 3


def test_every_candidate_is_scored_even_once_one_has_passed():
    # No short-circuit, for two reasons that are both pinned here: the cost of a
    # chunk is exactly `k` calls regardless of the data, and the ranking rule
    # needs the scores of the candidates *behind* the first one that passes.
    scorer = TableScorer({"C1": 0.8, "C2": 0.95, "C3": 0.6})
    adapter = S1MAdapter(scorer)

    decision = adapter.decide(subject(), candidates("C1", "C2", "C3"))

    assert scorer.calls == [("K1", "C1"), ("K1", "C2"), ("K1", "C3")]
    assert decision == Decision.extend("C2", confidence=0.95)


def test_the_cost_of_a_run_is_the_number_of_candidates_per_chunk():
    scorer = TableScorer()
    adapter = S1MAdapter(scorer)

    adapter.decide(subject("K1"), candidates())
    adapter.decide(subject("K2"), candidates("C1"))
    adapter.decide(subject("K3"), candidates("C1", "C2"))

    assert adapter.score_calls == 0 + 1 + 2
    assert [len(call.scores) for call in adapter.calls] == [0, 1, 2]


def test_the_scorer_is_handed_both_texts_and_nothing_else():
    scorer = TableScorer()
    adapter = S1MAdapter(scorer)

    adapter.decide(subject(user_message="the exporter writes naive timestamps"), candidates("C1"))

    assert scorer.texts == [("the exporter writes naive timestamps", "about C1")]


# ---------------------------------------------------------------------------
# the threshold, and both sides of its boundary
# ---------------------------------------------------------------------------


def test_a_score_exactly_at_the_threshold_passes():
    adapter = S1MAdapter(TableScorer({"C1": 0.5}), threshold=0.5)

    assert adapter.decide(subject(), candidates("C1")) == Decision.extend("C1", confidence=0.5)


def test_a_score_just_below_the_threshold_does_not_pass():
    just_below = math.nextafter(0.5, 0.0)
    adapter = S1MAdapter(TableScorer({"C1": just_below}), threshold=0.5)

    decision = adapter.decide(subject(), candidates("C1"))

    assert decision.is_new
    assert decision.confidence == pytest.approx(1.0 - just_below)


def test_the_threshold_is_configurable_and_moves_the_same_scores_both_ways():
    scores = {"C1": 0.4}

    offered = candidates("C1")

    assert S1MAdapter(TableScorer(scores), threshold=0.3).decide(subject(), offered).target == "C1"
    assert S1MAdapter(TableScorer(scores), threshold=0.5).decide(subject(), offered).is_new


def test_a_threshold_of_zero_lets_every_candidate_pass():
    # The degenerate end of a `>=` comparison, pinned so that it is read as the
    # consequence of the boundary rule and not as an accident: with no cut-off
    # at all even a score of 0.0 is an extension.
    adapter = S1MAdapter(TableScorer(default=0.0), threshold=0.0)

    assert adapter.decide(subject(), candidates("C1")) == Decision.extend("C1", confidence=0.0)


def test_a_threshold_of_one_passes_only_a_certain_candidate():
    adapter = S1MAdapter(TableScorer({"C1": math.nextafter(1.0, 0.0), "C2": 1.0}), threshold=1.0)

    assert adapter.decide(subject(), candidates("C1", "C2")) == Decision.extend(
        "C2", confidence=1.0
    )


def test_the_default_threshold_is_the_midpoint_of_the_scale():
    assert DEFAULT_CONTINUATION_THRESHOLD == 0.5
    assert S1MAdapter(TableScorer()).threshold == DEFAULT_CONTINUATION_THRESHOLD


@pytest.mark.parametrize("threshold", [-0.1, 1.1, float("nan"), float("inf"), "0.5", None, True])
def test_a_threshold_that_is_no_probability_is_refused_where_it_was_written(threshold):
    # A plain `ValueError` and deliberately *not* a `ClassifyError`: the
    # threshold is the caller's own literal, which is the carve-out
    # `idc.classify` documents for `Decision.extend` as well. A threshold that
    # reaches the tool from a command line is T10's to validate.
    with pytest.raises(ValueError, match="threshold") as raised:
        S1MAdapter(TableScorer(), threshold=threshold)

    assert not isinstance(raised.value, ClassifyError)


# ---------------------------------------------------------------------------
# the ranking rule, and its tie-break
# ---------------------------------------------------------------------------


def test_the_highest_scoring_candidate_wins_not_the_first_that_passed():
    adapter = S1MAdapter(TableScorer({"C1": 0.6, "C2": 0.7, "C3": 0.9}))

    assert adapter.decide(subject(), candidates("C1", "C2", "C3")).target == "C3"


def test_a_candidate_below_the_threshold_cannot_win_on_rank():
    adapter = S1MAdapter(TableScorer({"C1": 0.9, "C2": 0.95}), threshold=0.99)

    assert adapter.decide(subject(), candidates("C1", "C2")).is_new


def test_equal_scores_resolve_to_the_candidate_that_was_offered_first():
    adapter = S1MAdapter(TableScorer(default=0.8))

    assert adapter.decide(subject(), candidates("C1", "C2", "C3")).target == "C1"


def test_the_tie_break_follows_the_offered_order_and_not_the_chain_number():
    # The tie-break is positional: the *first* of the equally scored candidates
    # wins. `ChainStore.candidates()` is documented to be in opening order, so
    # on the classify path that is the oldest chain — but the rule itself reads
    # the order it is given, which is what makes it deterministic for any
    # caller. Offering the same two chains the other way round therefore flips
    # the winner, and that is the property, not a defect.
    adapter = S1MAdapter(TableScorer(default=0.8))

    assert adapter.decide(subject(), candidates("C2", "C1")).target == "C2"


def test_the_tie_break_is_stable_over_repeated_runs():
    for _ in range(5):
        adapter = S1MAdapter(TableScorer(default=0.75))

        assert adapter.decide(subject(), candidates("C1", "C2", "C3")).target == "C1"


# ---------------------------------------------------------------------------
# `NEW`, and the confidence a decision carries
# ---------------------------------------------------------------------------


def test_no_candidate_passing_opens_a_new_chain():
    adapter = S1MAdapter(TableScorer({"C1": 0.1, "C2": 0.3}))

    decision = adapter.decide(subject(), candidates("C1", "C2"))

    assert decision.target is NEW
    assert decision.is_new


def test_zero_candidates_open_a_new_chain_without_asking_the_scorer():
    scorer = TableScorer()
    adapter = S1MAdapter(scorer)

    decision = adapter.decide(subject(), ())

    assert decision == Decision.new(confidence=1.0)
    assert scorer.calls == []
    assert adapter.score_calls == 0


def test_the_confidence_of_an_extension_is_the_winning_score_itself():
    # The raw score and not a number derived from the threshold: the evaluation
    # harness (#19) has to be able to sweep the threshold over a recorded run,
    # and a confidence that already folded the threshold in would make that
    # impossible.
    adapter = S1MAdapter(TableScorer({"C1": 0.83}), threshold=0.5)

    assert adapter.decide(subject(), candidates("C1")).confidence == 0.83


def test_the_confidence_of_a_new_chain_complements_the_best_score():
    adapter = S1MAdapter(TableScorer({"C1": 0.2, "C2": 0.35}))

    decision = adapter.decide(subject(), candidates("C1", "C2"))

    assert decision.confidence == pytest.approx(0.65)


def test_a_new_chain_above_the_threshold_reports_how_close_it_was():
    # The honest reading of a high threshold: the scorer did think 0.9, the
    # threshold refused it, and the confidence in the `NEW` that came out is
    # correspondingly low. Both numbers stay comparable across runs because
    # neither of them is scaled by the threshold.
    adapter = S1MAdapter(TableScorer({"C1": 0.9}), threshold=0.95)

    decision = adapter.decide(subject(), candidates("C1"))

    assert decision.is_new
    assert decision.confidence == pytest.approx(0.1)


def test_every_answer_is_a_decision_in_the_normal_form():
    adapter = S1MAdapter(TableScorer({"C1": 0.9}))

    extension = adapter.decide(subject(), candidates("C1"))
    opening = adapter.decide(subject("K2"), candidates())

    for decision in (extension, opening):
        assert isinstance(decision, Decision)
        assert as_decision(decision) is decision
        assert decision.confidence is not None


def test_the_adapter_satisfies_the_adapter_interface():
    assert isinstance(S1MAdapter(TableScorer()), IntentAdapter)


# ---------------------------------------------------------------------------
# a scorer that breaks its own contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "answer",
    [1.5, -0.1, float("nan"), float("inf"), float("-inf"), "yes", None, True, False, object()],
    ids=repr,
)
def test_a_score_that_is_no_probability_is_a_failure_of_the_stage(answer):
    adapter = S1MAdapter(TableScorer({"C1": answer}))

    with pytest.raises(ScorerContractError) as raised:
        adapter.decide(subject(), candidates("C1"))

    assert isinstance(raised.value, ClassifyError)
    assert raised.value.candidate_id == "C1"
    assert raised.value.chunk_id == "K1"


def test_a_broken_score_is_a_bad_value_as_well_as_a_failure_of_the_stage():
    # The mirror of the assertion `AdapterContractError` carries one level up: a
    # scorer's unusable answer is a bad *value* and a failure of the *stage*, so
    # `except ValueError` and `except ClassifyError` each catch it on their own
    # and no caller has to catch two families for one event. Promised in the
    # class docstring, and only pinned if both halves are written down.
    assert issubclass(ScorerContractError, ClassifyError)
    assert issubclass(ScorerContractError, ValueError)

    adapter = S1MAdapter(TableScorer({"C1": "yes"}))

    with pytest.raises(ValueError):
        adapter.decide(subject(), candidates("C1"))


def test_an_integer_score_is_accepted_and_kept_as_a_float():
    # The one number a *binary* judgment reports most naturally: 1 or 0 rather
    # than 1.0 or 0.0. It is accepted and normalized, which is exactly the line
    # `True` falls on the other side of — a bool carries no probability to rank.
    adapter = S1MAdapter(TableScorer({"C1": 1, "C2": 0}))

    decision = adapter.decide(subject(), candidates("C1", "C2"))

    assert decision == Decision.extend("C1", confidence=1.0)
    assert [type(score) for _, score in adapter.calls[0].scores] == [float, float]


def test_a_broken_score_names_the_chunk_and_the_candidate_it_came_from():
    adapter = S1MAdapter(TableScorer({"C2": 2.0}))

    with pytest.raises(ScorerContractError, match="C2"):
        adapter.decide(subject("K7"), candidates("C1", "C2", "C3"))


def test_a_broken_score_stops_the_decision_instead_of_ranking_what_came_before():
    scorer = TableScorer({"C1": 0.9, "C2": "almost"})
    adapter = S1MAdapter(scorer)

    with pytest.raises(ScorerContractError):
        adapter.decide(subject(), candidates("C1", "C2", "C3"))

    # C3 is never asked, and no decision is recorded: a passing C1 must not
    # become the answer just because the scorer broke behind it.
    assert scorer.calls == [("K1", "C1"), ("K1", "C2")]
    assert adapter.calls == ()
    assert adapter.score_calls == 2


def test_an_exception_the_scorer_raised_reaches_the_caller_unwrapped():
    # The carve-out `idc.classify` documents: only the backend knows what its
    # own timeout or authentication failure means, and presenting it as a
    # `ClassifyError` would read as a classification problem.
    scorer = ExplodingScorer(at=2)
    adapter = S1MAdapter(scorer)

    with pytest.raises(ExplodingScorer.BackendError):
        adapter.decide(subject(), candidates("C1", "C2", "C3"))

    assert scorer.calls == ["C1", "C2"]
    assert adapter.calls == ()
    # The attempt still cost two calls. `score_calls` is the run's cost, so it
    # counts what was spent and not what came back.
    assert adapter.score_calls == 2


# ---------------------------------------------------------------------------
# the record the evaluation harness reads
# ---------------------------------------------------------------------------


def test_the_record_holds_every_score_the_decision_was_made_on():
    adapter = S1MAdapter(TableScorer({"C1": 0.2, "C2": 0.8}))

    decision = adapter.decide(subject(), candidates("C1", "C2"))

    assert adapter.calls == (
        S1MCall(chunk_id="K1", scores=(("C1", 0.2), ("C2", 0.8)), decision=decision),
    )
    assert adapter.calls[0].candidate_ids == ("C1", "C2")


def test_the_record_grows_by_one_entry_per_decision():
    adapter = S1MAdapter(TableScorer(default=0.1))

    adapter.decide(subject("K1"), candidates("C1"))
    adapter.decide(subject("K2"), candidates("C1", "C2"))

    assert [call.chunk_id for call in adapter.calls] == ["K1", "K2"]


def test_the_adapter_reports_its_configuration():
    adapter = S1MAdapter(TableScorer(), threshold=0.25)

    assert adapter.threshold == 0.25
    assert "0.25" in repr(adapter)


# ---------------------------------------------------------------------------
# through the pipeline
# ---------------------------------------------------------------------------


def test_the_adapter_classifies_a_transcript_through_the_stage():
    session = CORPUS.session("interleaved_intents")
    chunks = chunk_session(session)
    # A, B, A: the third chunk has to rejoin C1, and the table says so per
    # (chunk, candidate) pair — the growing candidate list is what the adapter
    # is handed, and C2 only exists from the third chunk on.
    scorer = TableScorer({("K2", "C1"): 0.1, ("K3", "C1"): 0.9, ("K3", "C2"): 0.2})
    adapter = S1MAdapter(scorer)

    chains = classify(session, chunks, adapter)

    assert [(chain.id, list(chain.chunk_ids)) for chain in chains] == [
        ("C1", ["K1", "K3"]),
        ("C2", ["K2"]),
    ]
    # 0 + 1 + 2 calls: the cost of this transcript is the number of open chains
    # at each chunk, and nothing else.
    assert scorer.calls == [("K2", "C1"), ("K3", "C1"), ("K3", "C2")]


def test_a_content_scorer_rejoins_the_earlier_intent_from_the_text_alone():
    session = CORPUS.session("interleaved_intents")
    chunks = chunk_session(session)
    adapter = S1MAdapter(OverlapScorer(), threshold=0.1)

    chains = classify(session, chunks, adapter)

    assert [(chain.id, list(chain.chunk_ids)) for chain in chains] == [
        ("C1", ["K1", "K3"]),
        ("C2", ["K2"]),
    ]


def test_the_confidence_reaches_the_assignment_record():
    session = CORPUS.session("interleaved_intents")
    chunks = chunk_session(session)
    adapter = S1MAdapter(TableScorer({("K2", "C1"): 0.1, ("K3", "C1"): 0.9, ("K3", "C2"): 0.2}))

    result = classify_chunks(session, chunks, adapter)

    assert [assignment.confidence for assignment in result.assignments] == [
        pytest.approx(1.0),
        pytest.approx(0.9),
        pytest.approx(0.9),
    ]


@pytest.mark.parametrize("name", CORPUS.names())
def test_the_same_scorer_classifies_a_transcript_the_same_way_twice(name):
    session = CORPUS.session(name)
    chunks = chunk_session(session)

    first = classify(session, chunks, S1MAdapter(OverlapScorer(), threshold=0.1))
    second = classify(session, chunks, S1MAdapter(OverlapScorer(), threshold=0.1))

    assert [dumps(chain) for chain in first] == [dumps(chain) for chain in second]


# ---------------------------------------------------------------------------
# no backend of its own
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("token", FORBIDDEN_TOKENS)
def test_the_module_holds_no_backend_of_its_own(token):
    source = inspect.getsource(importlib.import_module("idc.classify.s1m"))

    assert token not in source.lower()


def test_the_scorer_is_what_the_adapter_was_given():
    scorer = TableScorer()

    assert S1MAdapter(scorer).scorer is scorer


def test_the_candidates_may_be_any_sequence():
    adapter = S1MAdapter(TableScorer({"C1": 0.9}))
    offered: Sequence[ChainCandidate] = list(candidates("C1"))

    assert adapter.decide(subject(), offered).target == "C1"
