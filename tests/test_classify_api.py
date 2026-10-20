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
    NEW,
    PROVISIONAL_LABEL_MAX_LENGTH,
    AdapterContractError,
    Assignment,
    ChainCandidate,
    ChainStore,
    ChunkAlreadyAssignedError,
    ChunkOrderError,
    ChunkUnderDecision,
    Classification,
    ClassifyError,
    Decision,
    DecisionLike,
    FakeAdapter,
    FakeCall,
    ForeignChunkError,
    IntentAdapter,
    NewChain,
    ScriptExhaustedError,
    UnknownChainError,
    as_decision,
    chain_id,
    chunk_user_message,
    classify,
    classify_chunks,
    is_chain_id,
    provisional_label,
)

EXPECTED_EXPORTS = {
    "CHAIN_ID_PREFIX",
    "NEW",
    "PROVISIONAL_LABEL_MAX_LENGTH",
    "AdapterContractError",
    "Assignment",
    "ChainCandidate",
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
}
"""The documented export surface: the stage's entry points, the adapter contract
the sibling tickets implement, and the failures a caller has to be able to catch."""

CALLABLE_EXPORTS = (
    as_decision,
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
    ChainCandidate,
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
    NewChain,
    ScriptExhaustedError,
    UnknownChainError,
)
"""Every exported name that is a type: the interface, the store, the fake, the
result objects and the error family."""

VALUE_EXPORTS = {
    "CHAIN_ID_PREFIX": CHAIN_ID_PREFIX,
    "NEW": NEW,
    "PROVISIONAL_LABEL_MAX_LENGTH": PROVISIONAL_LABEL_MAX_LENGTH,
    "DecisionLike": DecisionLike,
}
"""Every exported name that is neither a function nor a class."""

ERROR_EXPORTS = (
    AdapterContractError,
    ChunkAlreadyAssignedError,
    ChunkOrderError,
    ForeignChunkError,
    ScriptExhaustedError,
    UnknownChainError,
)
"""Every failure of the stage; all of them have to be one catchable family."""


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
