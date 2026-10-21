"""The public surface of `idc.resolve`: what the sibling sub-tickets of T5 reuse.

A contract test, not a behaviour test. `solution_files` is the stage's entry
point; `chain_chunks`, `chain_calls` and `LocatedCall` are the chain walk T5.2
(#22) asks for the `exec` calls with and T5.3 (#23) for the path, and
`file_of`, `write_kind`, `content_of` and the three key tuples are the
argument-shape knowledge that must not be restated per sub-ticket. *What* they
compute is covered through `solution_files` in
`tests/test_resolve_writes.py`; what is pinned here is only that they stay
reachable from the package, so a refactor cannot drop one from `__all__` — or
out of the package — with the suite still green.

The module docstring is pinned too: the parent ticket (#20) accepts that an
edit made outside a tool call is not detected **on the condition that the
limitation is documented there**, so the sentence that documents it is part of
the stage's contract and not prose.
"""

import importlib

import pytest

from idc.resolve import (
    CONTENT_KEYS,
    FILE_PATH_KEYS,
    REPLACEMENT_KEY_PAIRS,
    ForeignChunkError,
    LocatedCall,
    ResolveError,
    SolutionFile,
    UnknownChunkError,
    WriteKind,
    chain_calls,
    chain_chunks,
    content_of,
    file_of,
    solution_files,
    write_calls,
    write_kind,
)

EXPECTED_EXPORTS = {
    "CONTENT_KEYS",
    "FILE_PATH_KEYS",
    "REPLACEMENT_KEY_PAIRS",
    "ForeignChunkError",
    "LocatedCall",
    "ResolveError",
    "SolutionFile",
    "UnknownChunkError",
    "WriteKind",
    "chain_calls",
    "chain_chunks",
    "content_of",
    "file_of",
    "solution_files",
    "write_calls",
    "write_kind",
}
"""The documented export surface of the stage after T5.1."""

CALLABLE_EXPORTS = (
    chain_calls,
    chain_chunks,
    content_of,
    file_of,
    solution_files,
    write_calls,
    write_kind,
)
"""Every exported name that is a function, imported from the package by name."""

TYPE_EXPORTS = (
    ForeignChunkError,
    LocatedCall,
    ResolveError,
    SolutionFile,
    UnknownChunkError,
    WriteKind,
)
"""Every exported name that is a type: the result types and the error family."""

KEY_TUPLE_EXPORTS = ("CONTENT_KEYS", "FILE_PATH_KEYS", "REPLACEMENT_KEY_PAIRS")
"""The exported names that are argument-key tables rather than code."""


def test_public_exports_are_declared_and_importable():
    module = importlib.import_module("idc.resolve")

    assert set(module.__all__) == EXPECTED_EXPORTS
    for name in module.__all__:
        assert hasattr(module, name), name


def test_the_declared_exports_cover_the_whole_surface():
    # Guards the parametrized tests below the same way `test_chunk_api.py` does:
    # a name dropped from one of the three tables would leave them with one case
    # less instead of with a failure.
    named = {function.__name__ for function in CALLABLE_EXPORTS}
    named |= {type_.__name__ for type_ in TYPE_EXPORTS}
    named |= set(KEY_TUPLE_EXPORTS)

    assert named == EXPECTED_EXPORTS


@pytest.mark.parametrize("function", CALLABLE_EXPORTS, ids=lambda f: f.__name__)
def test_every_exported_function_is_callable(function):
    assert callable(function)


@pytest.mark.parametrize("type_", TYPE_EXPORTS, ids=lambda t: t.__name__)
def test_every_exported_type_is_a_class(type_):
    assert isinstance(type_, type)


@pytest.mark.parametrize("name", KEY_TUPLE_EXPORTS)
def test_every_exported_key_table_is_a_non_empty_tuple(name):
    table = getattr(importlib.import_module("idc.resolve"), name)

    assert isinstance(table, tuple)
    assert table


def test_the_argument_keys_are_the_spellings_the_corpus_records():
    # The evidence rule of `tool_registry.toml`, restated for the keys this
    # stage reads: only an observed spelling is listed. A key added here
    # without a transcript showing it would pin behaviour nobody saw.
    assert FILE_PATH_KEYS == ("file_path", "filePath", "absolute_path")
    assert CONTENT_KEYS == ("content",)
    assert REPLACEMENT_KEY_PAIRS == (("old_string", "new_string"), ("oldString", "newString"))


def test_the_write_kinds_are_the_two_the_last_write_rule_distinguishes():
    assert [kind.value for kind in WriteKind] == ["full", "edit"]


def test_the_stage_documents_that_an_edit_outside_a_tool_call_is_not_detected():
    # The parent ticket's third acceptance criterion (#20): the limitation is
    # accepted *because* it is written down here.
    documentation = importlib.import_module("idc.resolve").__doc__ or ""

    assert "sed" in documentation
    assert "not detected" in documentation
