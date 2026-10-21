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
    SessionSource,
    ToolCall,
    ToolResult,
    dumps,
    load,
    loads,
    validate_document,
)
from schema_samples import CREATED_AT, SAMPLES, native_compacted_session


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
    (
        "summary named by a session that was never natively compacted",
        Session,
        changed("Session", native_summary_id="m-0"),
        "native_summary_id",
    ),
    (
        "empty native_summary_id",
        Session,
        changed("Session", source="native-compaction", native_summary_id=""),
        "native_summary_id",
    ),
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


def test_tool_result_whose_call_id_has_no_tool_call_is_accepted_on_purpose():
    """Referential completeness is deliberately not enforced (see the decision on #3).

    A transcript can be truncated at the front, so a result whose call was cut
    off is still valid data that the measurement (T9) has to count.
    """
    orphan = document("Session")
    orphan["messages"][3]["tool_result"]["call_id"] = "call-cut-off"

    restored = loads(Session, json.dumps(orphan))

    declared = {call.id for message in restored.messages for call in message.tool_calls}
    assert "call-cut-off" not in declared
    assert restored.messages[3].tool_result is not None
    assert restored.messages[3].tool_result.call_id == "call-cut-off"


def test_a_named_native_summary_requires_the_native_compaction_provenance():
    """The one half of the pair that is a contradiction inside a single document.

    `source` and `native_summary_id` say the same thing about the same
    transcript — *the agent compacted this and continued from that message* — so
    an identifier under any other provenance is not incomplete data, it is two
    fields disagreeing. The invariant used to sit on `ImportedSession`; with the
    identifier canonical it belongs to the type that carries both halves.
    """
    for source in ("raw", "intent-compaction"):
        invalid = changed("Session", source=source, native_summary_id="m-0")

        with pytest.raises(SchemaValidationError) as excinfo:
            loads(Session, json.dumps(invalid))

        assert "native_summary_id" in str(excinfo.value)
        assert "native-compaction" in str(excinfo.value)


def test_assigning_a_native_summary_to_a_raw_session_is_rejected_too():
    raw = SAMPLES["Session"]()

    with pytest.raises(SchemaValidationError, match="native_summary_id"):
        raw.native_summary_id = "m-0"


def test_constructing_a_session_that_names_a_summary_without_the_provenance_is_rejected():
    with pytest.raises(SchemaValidationError, match="native_summary_id"):
        Session(
            id="s-1",
            agent="claude-code",
            created_at=CREATED_AT,
            native_summary_id="m-0",
        )


def test_a_native_summary_identifier_that_names_no_message_is_accepted_on_purpose():
    """Referential completeness stays unenforced for this reference too (see #3).

    A `native-compaction` transcript truncated at the front loses the summary
    record before it loses anything else, and the document still records *that*
    the agent compacted it and *which* message it continued from — which is more
    than a document that dropped the identifier with it. `visible_context`
    refuses such a session, because it cannot locate the cut; the schema keeps
    it loadable, exactly as it does for a `ToolResult` whose call was cut off.
    """
    truncated = changed("Session", source="native-compaction", native_summary_id="c-cut-off")

    restored = loads(Session, json.dumps(truncated))

    assert restored.native_summary_id == "c-cut-off"
    assert "c-cut-off" not in {message.id for message in restored.messages}


def test_a_natively_compacted_session_without_an_identifier_is_a_valid_document():
    """The other half of the pair is deliberately *not* a document-validity rule.

    It is an obligation on whoever produced the document — the importers, where
    `ImportedSession` enforces it — and a precondition of the one computation
    that needs it, where `visible_context` refuses. Making it a schema rule
    would let a stage that rebuilds a session without the field turn a valid
    transcript into an unloadable one, and would be the single place the schema
    departs from "incomplete data stays valid, the stage refuses".
    """
    incomplete = changed("Session", source="native-compaction")

    restored = loads(Session, json.dumps(incomplete))

    assert restored.source is SessionSource.NATIVE_COMPACTION
    assert restored.native_summary_id is None


def test_a_session_may_name_a_summary_once_it_says_it_was_natively_compacted():
    compacted = native_compacted_session()

    assert compacted.native_summary_id == "c-0"
    assert loads(Session, dumps(compacted)) == compacted


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

    with pytest.raises(SchemaValidationError, match="origin"):
        chunk.origin = "agent"


def test_construction_raises_the_same_located_error_as_the_load_path():
    with pytest.raises(SchemaValidationError) as excinfo:
        Chunk(id="K1", start_index=5, end_index=2)

    assert "invalid Chunk document: 1 problem" in str(excinfo.value)
    assert [location for location, _ in excinfo.value.errors] == ["end_index"]
    assert excinfo.value.model == "Chunk"
    assert excinfo.value.source is None


def test_construction_reports_nested_fields_with_their_dotted_path():
    with pytest.raises(SchemaValidationError) as excinfo:
        Session(
            id="s-1",
            agent="claude-code",
            created_at=CREATED_AT,
            messages=[
                {
                    "id": "m-0",
                    "role": "assistant",
                    "tool_calls": [{"id": "call-1", "name": "", "category": "read"}],
                }
            ],
        )

    assert "messages.0.tool_calls.0.name" in str(excinfo.value)


@pytest.mark.parametrize(
    "construct",
    [
        pytest.param(lambda: Chunk(id="K1", start_index=5, end_index=2), id="construction"),
        pytest.param(lambda: setattr(SAMPLES["Chunk"](), "origin", "agent"), id="assignment"),
    ],
)
def test_construction_and_assignment_hide_the_pydantic_internals(construct):
    with pytest.raises(ValueError) as excinfo:  # noqa: PT011 - the subclass is asserted below
        construct()

    assert isinstance(excinfo.value, SchemaValidationError)
    message = str(excinfo.value)
    assert "pydantic" not in message.lower()
    assert "errors.pydantic.dev" not in message


def test_chunk_exposes_its_message_count():
    chunk = SAMPLES["Chunk"]()

    assert chunk.message_count == 3
