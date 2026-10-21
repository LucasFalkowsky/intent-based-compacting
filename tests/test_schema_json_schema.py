"""The committed JSON Schema document is generated from the types and stays in sync."""

import json

from idc.schema import (
    CANONICAL_MODELS,
    SCHEMA_DIALECT,
    SCHEMA_PATH,
    SCHEMA_TITLE,
    build_json_schema,
    load_json_schema,
    schema_document_text,
    write_json_schema,
)

MODEL_NAMES = {model.__name__ for model in CANONICAL_MODELS}
REGENERATE_HINT = "run `make schema` to regenerate src/idc/schema/canonical.schema.json"


def test_committed_schema_file_exists_and_is_valid_json():
    assert SCHEMA_PATH.is_file(), REGENERATE_HINT

    assert isinstance(json.loads(SCHEMA_PATH.read_text(encoding="utf-8")), dict)


def test_committed_schema_is_in_sync_with_the_types():
    assert SCHEMA_PATH.read_text(encoding="utf-8") == schema_document_text(), REGENERATE_HINT


def test_regenerating_the_schema_changes_nothing(tmp_path):
    """Writing the schema again must be a no-op — byte-identical to the commit."""
    regenerated = write_json_schema(tmp_path / "canonical.schema.json")

    assert regenerated.read_bytes() == SCHEMA_PATH.read_bytes(), REGENERATE_HINT


def test_generation_is_deterministic():
    assert schema_document_text() == schema_document_text()
    assert build_json_schema() == build_json_schema()


def test_schema_declares_the_dialect_and_the_title():
    schema = load_json_schema()

    assert schema["$schema"] == SCHEMA_DIALECT
    assert schema["title"] == SCHEMA_TITLE


def test_schema_defines_every_canonical_type():
    schema = load_json_schema()

    assert set(schema["$defs"]) >= MODEL_NAMES


def test_schema_carries_the_enum_values():
    defs = load_json_schema()["$defs"]

    assert set(defs["SessionSource"]["enum"]) == {"raw", "native-compaction", "intent-compaction"}
    assert set(defs["Role"]["enum"]) == {"user", "assistant", "system", "tool"}
    assert set(defs["ToolCategory"]["enum"]) == {"read", "write", "exec", "other"}
    assert set(defs["ChunkOrigin"]["enum"]) == {"user", "subagent"}
    assert set(defs["Agent"]["enum"]) == {"claude-code", "gemini-cli", "opencode"}


def test_schema_marks_required_fields_and_forbids_extras():
    session = load_json_schema()["$defs"]["Session"]

    assert session["additionalProperties"] is False
    assert set(session["required"]) == {"id", "agent", "created_at"}


def test_schema_declares_the_native_summary_identifier_as_optional():
    # The published contract has to carry the field, or a consumer outside this
    # repository cannot tell a reloaded `native-compaction` document from a
    # guessed one. It stays optional: `raw` and `intent-compaction` sessions
    # carry no summary, and a truncated one may name a message it no longer has.
    session = load_json_schema()["$defs"]["Session"]

    assert "native_summary_id" in session["properties"]
    assert "native_summary_id" not in session["required"]


def test_schema_file_is_shipped_inside_the_package():
    assert SCHEMA_PATH.parent.name == "schema"
    assert SCHEMA_PATH.name == "canonical.schema.json"
