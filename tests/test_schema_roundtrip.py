"""Every canonical type survives a JSON round trip without loss."""

import json

import pytest

from idc.schema import (
    CANONICAL_MODELS,
    Agent,
    Message,
    Role,
    Session,
    SessionSource,
    dump,
    dumps,
    load,
    loads,
)
from schema_samples import CREATED_AT, OFFSET_AT, SAMPLES, native_compacted_session

MODELS_BY_NAME = {model.__name__: model for model in CANONICAL_MODELS}


def test_every_canonical_type_has_a_sample():
    assert set(MODELS_BY_NAME) == set(SAMPLES)


@pytest.mark.parametrize("name", sorted(SAMPLES))
def test_round_trip_through_json_text(name):
    original = SAMPLES[name]()

    restored = loads(MODELS_BY_NAME[name], dumps(original))

    assert restored == original
    assert type(restored) is type(original)


@pytest.mark.parametrize("name", sorted(SAMPLES))
def test_round_trip_is_idempotent_on_the_text(name):
    original = SAMPLES[name]()
    once = dumps(original)

    twice = dumps(loads(MODELS_BY_NAME[name], once))

    assert twice == once


@pytest.mark.parametrize("name", sorted(SAMPLES))
def test_round_trip_through_a_plain_json_document(name):
    """The serialized form is plain JSON — no pydantic-only types leak into it."""
    model = MODELS_BY_NAME[name]
    original = SAMPLES[name]()

    document = json.loads(dumps(original))
    restored = model.model_validate(document)

    assert restored == original
    assert json.loads(dumps(restored)) == document


@pytest.mark.parametrize("name", sorted(SAMPLES))
def test_round_trip_through_a_file(name, tmp_path):
    original = SAMPLES[name]()
    path = tmp_path / f"{name}.json"

    dump(original, path)

    assert path.read_text(encoding="utf-8").endswith("\n")
    assert load(MODELS_BY_NAME[name], path) == original


def test_canonical_session_round_trip_keeps_every_field():
    """The parent acceptance criterion: a canonical transcript, dumped and loaded back."""
    original = SAMPLES["Session"]()

    restored = loads(Session, dumps(original))

    assert restored == original
    assert restored.model_dump() == original.model_dump()
    assert len(restored.messages) == len(original.messages)
    assert [m.id for m in restored.messages] == [m.id for m in original.messages]
    assert [m.role for m in restored.messages] == [m.role for m in original.messages]
    assert restored.messages[2].tool_calls[0].args == original.messages[2].tool_calls[0].args
    assert restored.messages[3].tool_result == original.messages[3].tool_result
    assert restored.messages[4].timestamp is None
    assert restored.created_at == CREATED_AT
    assert restored.created_at.tzinfo is not None


def test_round_trip_preserves_microseconds_and_a_non_utc_offset():
    original = Session(
        id="s-offset",
        agent=Agent.GEMINI_CLI,
        source=SessionSource.NATIVE_COMPACTION,
        created_at=OFFSET_AT,
        messages=[Message(id="m-0", role=Role.USER, content="hi", timestamp=OFFSET_AT)],
    )

    restored = loads(Session, dumps(original))

    assert restored.created_at == OFFSET_AT
    assert restored.created_at.utcoffset() == OFFSET_AT.utcoffset()
    assert restored.created_at.microsecond == 123456
    assert restored == original


def test_round_trip_keeps_the_native_summary_identified():
    """The ticket's point: a natively compacted transcript stays self-describing.

    Without the identifier in the document, writing a `native-compaction`
    session to a file and reading it back in the next CLI invocation loses which
    message the agent continued from — and the baseline the study compares
    against cannot cross two commands.
    """
    original = native_compacted_session()

    restored = loads(Session, dumps(original))

    assert restored.native_summary_id == original.native_summary_id
    assert restored == original
    assert json.loads(dumps(restored))["native_summary_id"] == "c-0"


def test_a_raw_session_dumps_its_empty_summary_identifier():
    document = json.loads(dumps(SAMPLES["Session"]()))

    assert document["native_summary_id"] is None


def test_round_trip_preserves_non_ascii_content_as_text():
    original = SAMPLES["Session"]()

    text = dumps(original)

    assert "Ümlaut" in text, "non-ASCII content must not be escaped away"
    assert loads(Session, text) == original


def test_enums_serialize_as_their_wire_strings():
    document = json.loads(dumps(SAMPLES["Session"]()))

    assert document["agent"] == "claude-code"
    assert document["source"] == "raw"
    assert document["messages"][0]["role"] == "system"
    assert document["messages"][2]["tool_calls"][0]["category"] == "write"


def test_defaults_are_filled_in_when_absent_from_the_document():
    document = {
        "id": "s-min",
        "agent": "opencode",
        "created_at": "2026-10-20T09:30:15.123456Z",
    }

    session = loads(Session, json.dumps(document))

    assert session.source is SessionSource.RAW
    assert session.model is None
    assert session.native_summary_id is None
    assert session.messages == []
    assert loads(Session, dumps(session)) == session
