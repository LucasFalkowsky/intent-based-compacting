"""The public surface of `idc.classify`: what #17, #18, #33 and #42 build against.

A contract test, not a behaviour test — the precedent is
`tests/test_chunk_api.py`. The adapter interface is the deliverable of T4.1
(#16): the LLM adapter (#17) and the S1M adapter (#18) are written against
`IntentAdapter`, `ChainCandidate`, `Decision`, `NEW` and `as_decision` without
touching anything else, the parser (#33) seeds the store with the chains it read
back, and the end-to-end test (#42) uses `FakeAdapter` so that no test calls a
model. *What* these names compute is covered in
`tests/test_classify_adapters.py`, `tests/test_classify_store.py` and
`tests/test_classify_assignment.py`; what is pinned here is only that they stay
reachable from the package, so a refactor cannot drop one from `__all__` — or
out of the package — with the suite still green.
"""

import importlib

import pytest

from idc.classify import (
    CHAIN_ID_PREFIX,
    CHUNKINGS,
    DEFAULT_CONTINUATION_THRESHOLD,
    DEFAULT_MAX_ATTEMPTS,
    EXIT_INPUT_ERROR,
    GOLD_CHUNKING,
    NEW,
    PROVISIONAL_LABEL_MAX_LENGTH,
    AdapterContractError,
    Agreement,
    Assignment,
    BlankChainLabelError,
    ChainAlreadySeededError,
    ChainCandidate,
    ChainOrderError,
    ChainStore,
    ChunkAlreadyAssignedError,
    ChunkOrderError,
    ChunkUnderDecision,
    Classification,
    ClassifyError,
    ContinuationScorer,
    CorpusAgreement,
    Decision,
    DecisionLike,
    DecisionRecordError,
    Disagreement,
    EvaluationCase,
    FakeAdapter,
    FakeCall,
    ForeignChunkError,
    GoldChains,
    GoldCoverageError,
    GoldLabellingError,
    IntentAdapter,
    LlmAdapter,
    LlmCall,
    ModelClient,
    NewChain,
    RunSource,
    S1MAdapter,
    S1MCall,
    ScorerContractError,
    ScriptExhaustedError,
    UnknownChainError,
    UnreadableAnswerError,
    agreement,
    as_decision,
    build_prompt,
    chain_id,
    chunk_user_message,
    chunks_for,
    classify,
    classify_chunks,
    evaluate_adapter,
    evaluate_cases,
    format_report,
    gold_script,
    is_chain_id,
    load_cases,
    load_decisions,
    load_gold,
    load_gold_directory,
    per_chunk_accuracy,
    predicted_labels,
    provisional_label,
    rand_index,
)

EXPECTED_EXPORTS = {
    "CHAIN_ID_PREFIX",
    "DEFAULT_MAX_ATTEMPTS",
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
    "LlmAdapter",
    "LlmCall",
    "ModelClient",
    "NewChain",
    "ScriptExhaustedError",
    "UnknownChainError",
    "UnreadableAnswerError",
    "as_decision",
    "build_prompt",
    "chain_id",
    "chunk_user_message",
    "classify",
    "classify_chunks",
    "is_chain_id",
    "provisional_label",
}
"""The documented export surface: the stage's entry points, the adapter contract
the sibling tickets implement, and the failures a caller has to be able to catch."""

CALLABLE_EXPORTS = (
    as_decision,
    build_prompt,
    chain_id,
    chunk_user_message,
    classify,
    classify_chunks,
    is_chain_id,
    provisional_label,
)
"""Every exported name that is a function, imported from the package by name.

A name removed from `idc.classify` breaks the import above and takes the whole
module down with it, which is the point: the contract fails loudly.
"""

TYPE_EXPORTS = (
    AdapterContractError,
    Assignment,
    BlankChainLabelError,
    ChainAlreadySeededError,
    ChainCandidate,
    ChainOrderError,
    ChainStore,
    ChunkAlreadyAssignedError,
    ChunkOrderError,
    ChunkUnderDecision,
    Classification,
    ClassifyError,
    Decision,
    FakeAdapter,
    FakeCall,
    ForeignChunkError,
    IntentAdapter,
    LlmAdapter,
    LlmCall,
    ModelClient,
    NewChain,
    ScriptExhaustedError,
    UnknownChainError,
    UnreadableAnswerError,
)
"""Every exported name that is a type: the interface, the store, the fake, the
result objects and the error family."""

VALUE_EXPORTS = {
    "CHAIN_ID_PREFIX": CHAIN_ID_PREFIX,
    "DEFAULT_MAX_ATTEMPTS": DEFAULT_MAX_ATTEMPTS,
    "NEW": NEW,
    "PROVISIONAL_LABEL_MAX_LENGTH": PROVISIONAL_LABEL_MAX_LENGTH,
    "DecisionLike": DecisionLike,
}
"""Every exported name that is neither a function nor a class."""

ERROR_EXPORTS = (
    AdapterContractError,
    BlankChainLabelError,
    ChainAlreadySeededError,
    ChainOrderError,
    ChunkAlreadyAssignedError,
    ChunkOrderError,
    ForeignChunkError,
    ScriptExhaustedError,
    UnknownChainError,
    UnreadableAnswerError,
)
"""Every failure of the stage; all of them have to be one catchable family.

The seeding guards are in here for the reason the others are: a chain seeded
from an existing summary block comes out of a transcript file (#33), so its
refusal is a failure of the stage and not a programmer's bad literal.
"""


PROMISE_CARVE_OUTS = {
    "a value object built with a bad literal": "as_decision",
    "an exception the adapter raised itself": "unwrapped",
    "an object that is no adapter at all": "AttributeError",
}
"""What the package's one-`except` promise does **not** cover, and the word that
has to name it.

`except ClassifyError` is what T10's CLI is written against, so the sentence
stating it is part of the surface and not prose: an unqualified "every failure
of the stage is a `ClassifyError`" sent #17 at a trap — `Decision.extend` with
a wrong-shaped id — with nothing in the docs marking it. The three behaviours
are pinned in `tests/test_classify_assignment.py`; what is pinned here is that
the documentation still names them, so the sentence cannot quietly widen again.
"""


# --- T4.3 (#18) — the S1M backend's own names, appended as a block ----------
#
# Not merged into the sorted literals above on purpose. #17 (the LLM adapter)
# and #18 (this one) extend the very same export surface from two branches at
# once, and a sorted literal both of them have to edit is a merge conflict by
# construction — the precedent is 786d431, which derived a list in a test for
# exactly this reason. Each backend's block is self-contained and
# order-independent, so keeping both blocks is the whole resolution, and
# neither branch ever has to touch the other's names.

S1M_EXPORTS = {
    "DEFAULT_CONTINUATION_THRESHOLD",
    "ContinuationScorer",
    "S1MAdapter",
    "S1MCall",
    "ScorerContractError",
}
"""What T4.3 adds to the surface: the adapter, its scorer contract, its default
threshold, its per-decision record and the failure a scorer can cause."""

EXPECTED_EXPORTS |= S1M_EXPORTS
TYPE_EXPORTS += (ContinuationScorer, S1MAdapter, S1MCall, ScorerContractError)
ERROR_EXPORTS += (ScorerContractError,)
VALUE_EXPORTS["DEFAULT_CONTINUATION_THRESHOLD"] = DEFAULT_CONTINUATION_THRESHOLD


# --- T4.4 (#19) — the evaluation harness, appended as a block of its own ----
#
# Kept out of the sorted literals above for the reason the S1M block is: a
# ticket that extends this surface edits only its own block, so two branches
# doing it at once do not collide.

HARNESS_EXPORTS = {
    "CHUNKINGS",
    "EXIT_INPUT_ERROR",
    "GOLD_CHUNKING",
    "Agreement",
    "CorpusAgreement",
    "DecisionRecordError",
    "Disagreement",
    "EvaluationCase",
    "GoldChains",
    "GoldCoverageError",
    "GoldLabellingError",
    "RunSource",
    "agreement",
    "chunks_for",
    "evaluate_adapter",
    "evaluate_cases",
    "format_report",
    "gold_script",
    "load_cases",
    "load_decisions",
    "load_gold",
    "load_gold_directory",
    "per_chunk_accuracy",
    "predicted_labels",
    "rand_index",
}
"""What T4.4 adds: the gold labelling and its loaders, the two metrics, the
per-transcript and corpus-wide agreement, the source a result is reported under,
the report, and the three failures a hand-written evaluation input can cause."""

EXPECTED_EXPORTS |= HARNESS_EXPORTS
CALLABLE_EXPORTS += (
    agreement,
    chunks_for,
    evaluate_adapter,
    evaluate_cases,
    format_report,
    gold_script,
    load_cases,
    load_decisions,
    load_gold,
    load_gold_directory,
    per_chunk_accuracy,
    predicted_labels,
    rand_index,
)
TYPE_EXPORTS += (
    Agreement,
    CorpusAgreement,
    DecisionRecordError,
    Disagreement,
    EvaluationCase,
    GoldChains,
    GoldCoverageError,
    GoldLabellingError,
    RunSource,
)
ERROR_EXPORTS += (DecisionRecordError, GoldCoverageError, GoldLabellingError)
VALUE_EXPORTS["CHUNKINGS"] = CHUNKINGS
VALUE_EXPORTS["EXIT_INPUT_ERROR"] = EXIT_INPUT_ERROR
VALUE_EXPORTS["GOLD_CHUNKING"] = GOLD_CHUNKING


def test_the_documented_promise_names_the_carve_outs_it_rests_on():
    promise = importlib.import_module("idc.classify").__doc__ or ""

    missing = sorted(name for name, marker in PROMISE_CARVE_OUTS.items() if marker not in promise)

    assert missing == [], f"the promise in idc.classify does not name: {missing}"


def test_public_exports_are_declared_and_importable():
    module = importlib.import_module("idc.classify")

    assert set(module.__all__) == EXPECTED_EXPORTS
    for name in module.__all__:
        assert hasattr(module, name), name


def test_the_export_groups_cover_the_whole_surface():
    # Guards the parametrized tests below the same way `tests/test_chunk_api.py`
    # guards its own: a name dropped from a group would leave it with one case
    # less instead of with a failure.
    named = {function.__name__ for function in CALLABLE_EXPORTS}
    named |= {type_.__name__ for type_ in TYPE_EXPORTS}
    named |= set(VALUE_EXPORTS)

    assert named == EXPECTED_EXPORTS


@pytest.mark.parametrize("function", CALLABLE_EXPORTS, ids=lambda f: f.__name__)
def test_every_exported_function_is_callable(function):
    assert callable(function)


@pytest.mark.parametrize("type_", TYPE_EXPORTS, ids=lambda t: t.__name__)
def test_every_exported_type_is_a_class(type_):
    assert isinstance(type_, type)


@pytest.mark.parametrize("error", ERROR_EXPORTS, ids=lambda e: e.__name__)
def test_every_failure_of_the_stage_is_one_catchable_family(error):
    assert issubclass(error, ClassifyError)


def test_the_chain_id_prefix_is_exported_as_a_non_empty_string():
    assert isinstance(CHAIN_ID_PREFIX, str)
    assert CHAIN_ID_PREFIX


def test_the_provisional_label_length_is_exported_as_a_positive_integer():
    assert isinstance(PROVISIONAL_LABEL_MAX_LENGTH, int)
    assert PROVISIONAL_LABEL_MAX_LENGTH > 0


def test_the_new_sentinel_is_exported_as_a_new_chain_value():
    assert isinstance(NEW, NewChain)
