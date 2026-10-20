"""The public surface of `idc.chunk`: the names T3.3 (#14) and the pipeline reuse.

A contract test, not a behaviour test. `is_user_turn`, `user_turn_indices`,
`chunk_boundaries` and `chunk_id` are exported next to `chunk_session` because
the sibling tickets need the same user-turn predicate and must not restate
it (decision on #12); `reduce_session`, `reduce_chunks` and the three result
types are the noise reduction T3.2 (#13) put on top of them, and
`subagent_spans`, `extract_subagents`, `chunk_with_subagents` and
`SubagentSpan` are the subagent extraction T3.3 (#14) put on top of those.
*What* they compute is covered transitively through `chunk_session` in
`tests/test_chunk_user_boundaries.py`, through `reduce_session` in
`tests/test_chunk_reduction.py` and through `chunk_with_subagents` in
`tests/test_chunk_subagents.py`; what is pinned here is only that they stay
reachable from the package, so a refactor cannot drop one from `__all__` — or
out of the package — with the suite still green.
"""

import importlib

import pytest

from idc.chunk import (
    CHUNK_ID_PREFIX,
    ReducedChunk,
    ReducedSession,
    ReductionNote,
    ReductionReason,
    SubagentSpan,
    chunk_boundaries,
    chunk_id,
    chunk_session,
    chunk_with_subagents,
    extract_subagents,
    is_user_turn,
    reduce_chunks,
    reduce_session,
    subagent_spans,
    user_turn_indices,
)

EXPECTED_EXPORTS = {
    "CHUNK_ID_PREFIX",
    "ReducedChunk",
    "ReducedSession",
    "ReductionNote",
    "ReductionReason",
    "SubagentSpan",
    "chunk_boundaries",
    "chunk_id",
    "chunk_session",
    "chunk_with_subagents",
    "extract_subagents",
    "is_user_turn",
    "reduce_chunks",
    "reduce_session",
    "subagent_spans",
    "user_turn_indices",
}
"""The documented export surface of the stage: the entry points plus the reuse contract."""

CALLABLE_EXPORTS = (
    chunk_boundaries,
    chunk_id,
    chunk_session,
    chunk_with_subagents,
    extract_subagents,
    is_user_turn,
    reduce_chunks,
    reduce_session,
    subagent_spans,
    user_turn_indices,
)
"""Every exported name that is a function, imported from the package by name.

A name removed from `idc.chunk` breaks the import above and takes the whole
module down with it, which is the point: the reuse contract fails loudly.
"""

TYPE_EXPORTS = (
    ReducedChunk,
    ReducedSession,
    ReductionNote,
    ReductionReason,
    SubagentSpan,
)
"""Every exported name that is a type: what the reduction and the subagent
extraction return."""


def test_public_exports_are_declared_and_importable():
    module = importlib.import_module("idc.chunk")

    assert set(module.__all__) == EXPECTED_EXPORTS
    for name in module.__all__:
        assert hasattr(module, name), name


def test_the_callable_exports_cover_the_whole_function_surface():
    # Guards the parametrized test below the same way the corpus guards its
    # own: a function dropped from `CALLABLE_EXPORTS` would leave it with one
    # case less instead of with a failure.
    named = {function.__name__ for function in CALLABLE_EXPORTS}
    named |= {type_.__name__ for type_ in TYPE_EXPORTS}

    assert named == EXPECTED_EXPORTS - {"CHUNK_ID_PREFIX"}


@pytest.mark.parametrize("function", CALLABLE_EXPORTS, ids=lambda f: f.__name__)
def test_every_exported_function_is_callable(function):
    assert callable(function)


def test_the_chunk_id_prefix_is_exported_as_a_non_empty_string():
    assert isinstance(CHUNK_ID_PREFIX, str)
    assert CHUNK_ID_PREFIX


@pytest.mark.parametrize("type_", TYPE_EXPORTS, ids=lambda t: t.__name__)
def test_every_exported_type_is_a_class(type_):
    assert isinstance(type_, type)
