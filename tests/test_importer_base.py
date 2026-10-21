"""The helpers every importer shares, tested where they live.

`type_name`, `shape` and `note_unreadable_content` were written three times, once
per agent module, and were identical. They are now in `idc.importers.base` next
to `joined_text` and `block_message_id`, and this module covers them directly
rather than only through whichever agent happens to exercise them.
"""

import pytest

from idc.importers import (
    NO_TYPE,
    NOTE_DROPPED_BLOCK,
    NULL_TYPE,
    SessionBuilder,
    note_unreadable_content,
    shape,
    type_name,
)
from idc.schema import Agent


def builder() -> SessionBuilder:
    return SessionBuilder(Agent.CLAUDE_CODE, source="session.jsonl")


def test_a_holder_that_names_no_type_is_reported_as_such():
    assert type_name({"id": "u1"}) == NO_TYPE


def test_a_holder_whose_type_is_null_is_kept_apart_from_one_without_a_type():
    assert type_name({"type": None}) == NULL_TYPE
    assert NULL_TYPE != NO_TYPE


def test_a_type_spelled_none_is_neither_of_the_two_markers():
    assert type_name({"type": "None"}) == "None"


@pytest.mark.parametrize(
    ("holder", "expected"),
    [
        ({"type": "user"}, "user"),
        ({"type": 7}, "7"),
        ({"type": True}, "True"),
    ],
)
def test_a_type_that_is_present_keeps_its_own_spelling(holder, expected):
    assert type_name(holder) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, "null"),
        (True, "boolean"),
        (False, "boolean"),
        (7, "number"),
        (1.5, "number"),
        ("text", "string"),
        ({"a": 1}, "object"),
        ([1, 2], "array"),
    ],
)
def test_a_value_is_named_by_its_json_shape(value, expected):
    assert shape(value) == expected


def test_a_boolean_is_a_boolean_and_not_a_number():
    """`bool` is a subclass of `int`, so the order of the checks is the test."""
    assert shape(True) == "boolean"


def test_a_value_that_is_no_json_shape_at_all_is_named_by_its_type():
    """A value JSON cannot hold still gets a name rather than `object`."""
    assert shape({1, 2}) == "set"


def test_an_absent_content_is_not_noted():
    session = builder()

    note_unreadable_content(session, None)

    assert session.notes() == ()


@pytest.mark.parametrize("content", ["text", ["a block"], []])
def test_a_readable_content_is_not_noted(content):
    session = builder()

    note_unreadable_content(session, content)

    assert session.notes() == ()


@pytest.mark.parametrize(
    ("content", "detail"),
    [
        ({"text": "lost"}, "<content: object>"),
        (7, "<content: number>"),
        (True, "<content: boolean>"),
    ],
)
def test_a_content_that_is_neither_text_nor_a_block_list_is_noted_by_its_shape(content, detail):
    session = builder()

    note_unreadable_content(session, content)

    assert [(note.kind, note.detail) for note in session.notes()] == [(NOTE_DROPPED_BLOCK, detail)]


def test_the_note_carries_the_prefix_the_caller_passes():
    session = builder()

    note_unreadable_content(session, 7, prefix="tool_result/")

    assert [note.detail for note in session.notes()] == ["tool_result/<content: number>"]
