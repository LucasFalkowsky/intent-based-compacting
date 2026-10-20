"""Invalid canonical documents are rejected with a clear, located error."""

import json

import pytest

from idc.schema import (
    Chain,
    Chunk,
    CompactedBlock,
    Message,
    SchemaValidationError,
    Session,
    ToolCall,
    ToolResult,
    dumps,
    load,
    loads,
    validate_document,
)
from schema_samples import SAMPLES


def document(name: str) -> dict:
    """Return the valid sample document of `name` as plain JSON data."""
    return json.loads(dumps(SAMPLES[name]()))


def without(name: str, key: str) -> dict:
    data = document(name)
    del data[key]
    return data


def changed(name: str, **overrides) -> dict:
    return document(name) | overrides


# (case id, model, invalid document, substring the error message must contain)
INVALID_DOCUMENTS = [
    ("unknown role", Message, changed("Message", role="bot"), "role"),
    ("unknown category", ToolCall, changed("ToolCall", category="delete"), "category"),
    ("args is not an object", ToolCall, changed("ToolCall", args=["a", "b"]), "args"),
    ("empty tool-call id", ToolCall, changed("ToolCall", id=""), "id"),
    ("empty call_id", ToolResult, changed("ToolResult", call_id=""), "call_id"),
    ("unknown agent", Session, changed("Session", agent="cursor"), "agent"),
    ("unknown source", Session, changed("Session", source="compacted"), "source"),
    ("missing created_at", Session, without("Session", "created_at"), "created_at"),
    ("missing id", Session, without("Session", "id"), "id"),
    (
        "naive created_at",
        Session,
        changed("Session", created_at="2026-10-20T09:30:15.123456"),
        "created_at",
    ),
    ("unexpected field", Session, changed("Session", nickname="s1"), "nickname"),
    ("unknown origin", Chunk, changed("Chunk", origin="agent"), "origin"),
    ("negative start_index", Chunk, changed("Chunk", start_index=-1), "start_index"),
    ("empty message range", Chunk, changed("Chunk", start_index=5, end_index=5), "end_index"),
    ("reversed message range", Chunk, changed("Chunk", start_index=5, end_index=2), "end_index"),
    ("chain id without the C prefix", Chain, changed("Chain", id="X1"), "id"),
    ("chain id numbered from zero", Chain, changed("Chain", id="C0"), "id"),
    ("chain without chunks", Chain, changed("Chain", chunk_ids=[]), "chunk_ids"),
    ("chain with a duplicate chunk", Chain, changed("Chain", chunk_ids=["K1", "K1"]), "chunk_ids"),
    ("empty chain label", Chain, changed("Chain", label=""), "label"),
    (
        "compacted chain without text",
        Chain,
        changed("Chain", compacted=True, compacted_text=None),
        "compacted_text",
    ),
    (
        "compacted text on an open chain",
        Chain,
        changed("Chain", compacted=False, compacted_text="## [C2] Intent: x"),
        "compacted",
    ),
    (
        "unknown render target",
        CompactedBlock,
        changed("CompactedBlock", rendered_for="cursor"),
        "rendered_for",
    ),
    (
        "block without a latest chunk",
        CompactedBlock,
        changed("CompactedBlock", latest_chunk_id=""),
        "latest_chunk_id",
    ),
]


@pytest.mark.parametrize(
    ("model", "invalid", "expected"),
    [case[1:] for case in INVALID_DOCUMENTS],
    ids=[case[0] for case in INVALID_DOCUMENTS],
)
def test_invalid_document_is_rejected(model, invalid, expected):
    with pytest.raises(SchemaValidationError) as excinfo:
        loads(model, json.dumps(invalid))

    assert expected in str(excinfo.value)


def test_duplicate_message_ids_are_rejected():
    invalid = document("Session")
    invalid["messages"][1]["id"] = invalid["messages"][0]["id"]

    with pytest.raises(SchemaValidationError, match="duplicate message id"):
        loads(Session, json.dumps(invalid))


def test_duplicate_tool_call_ids_across_messages_are_rejected():
    invalid = document("Session")
    invalid["messages"][4]["tool_calls"][0]["id"] = "call-1"

    with pytest.raises(SchemaValidationError, match="duplicate tool-call id"):
        loads(Session, json.dumps(invalid))


def test_duplicate_tool_call_ids_within_one_message_are_rejected():
    invalid = document("Message")
    invalid["tool_calls"] = invalid["tool_calls"] * 2

    with pytest.raises(SchemaValidationError, match="duplicate tool-call id"):
        loads(Message, json.dumps(invalid))


def test_tool_message_without_a_result_is_rejected():
    invalid = changed("Message", role="tool", tool_calls=[], tool_result=None)

    with pytest.raises(SchemaValidationError, match="tool_result"):
        loads(Message, json.dumps(invalid))


def test_duplicate_chain_ids_in_a_block_are_rejected():
    invalid = document("CompactedBlock")
    invalid["chains"][1]["id"] = invalid["chains"][0]["id"]

    with pytest.raises(SchemaValidationError, match="duplicate chain id"):
        loads(CompactedBlock, json.dumps(invalid))


def test_error_names_the_model_and_locates_every_problem():
    invalid = document("Session")
    invalid["messages"][0]["role"] = "bot"
    invalid["agent"] = "cursor"

    with pytest.raises(SchemaValidationError) as excinfo:
        loads(Session, json.dumps(invalid))

    message = str(excinfo.value)
    assert "Session" in message
    assert "messages.0.role" in message
    assert "agent" in message
    assert len(excinfo.value.errors) == 2


def test_error_is_a_value_error_and_hides_the_pydantic_internals():
    with pytest.raises(ValueError, match="origin") as excinfo:
        loads(Chunk, json.dumps(changed("Chunk", origin="agent")))

    assert isinstance(excinfo.value, SchemaValidationError)
    assert "pydantic" not in str(excinfo.value).lower()


def test_malformed_json_is_rejected_with_the_same_error_type():
    with pytest.raises(SchemaValidationError, match="not valid JSON"):
        loads(Session, "{not json")


def test_a_json_document_that_is_not_an_object_is_rejected():
    with pytest.raises(SchemaValidationError, match="Session"):
        loads(Session, "[]")


def test_error_names_the_source_file(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text(json.dumps(changed("Session", agent="cursor")), encoding="utf-8")

    with pytest.raises(SchemaValidationError) as excinfo:
        load(Session, path)

    assert "broken.json" in str(excinfo.value)
    assert excinfo.value.source == str(path)


def test_missing_file_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        load(Session, tmp_path / "absent.json")


def test_validate_document_accepts_already_parsed_data():
    data = document("Chunk")

    assert validate_document(Chunk, data) == SAMPLES["Chunk"]()


def test_validate_document_rejects_invalid_data():
    with pytest.raises(SchemaValidationError, match="end_index"):
        validate_document(Chunk, changed("Chunk", start_index=9, end_index=1))


def test_assignment_is_validated_too():
    chunk = SAMPLES["Chunk"]()

    with pytest.raises(ValueError, match="origin"):
        chunk.origin = "agent"


def test_chunk_exposes_its_message_count():
    chunk = SAMPLES["Chunk"]()

    assert chunk.message_count == 3
