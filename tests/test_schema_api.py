"""The public surface of `idc.schema`: the exported names, the enums, the model set."""

import importlib

import pytest

from idc.schema import (
    CANONICAL_MODELS,
    Agent,
    CanonicalModel,
    Chain,
    Chunk,
    ChunkOrigin,
    CompactedBlock,
    Message,
    Role,
    Session,
    SessionSource,
    ToolCall,
    ToolCategory,
    ToolResult,
)

EXPECTED_EXPORTS = {
    "Agent",
    "CANONICAL_MODELS",
    "CanonicalModel",
    "Chain",
    "Chunk",
    "ChunkOrigin",
    "CompactedBlock",
    "Message",
    "Role",
    "SCHEMA_DIALECT",
    "SCHEMA_PATH",
    "SCHEMA_TITLE",
    "SchemaValidationError",
    "Session",
    "SessionSource",
    "ToolCall",
    "ToolCategory",
    "ToolResult",
    "build_json_schema",
    "dump",
    "dumps",
    "load",
    "load_json_schema",
    "loads",
    "schema_document_text",
    "validate_document",
    "write_json_schema",
}


def test_public_exports_are_declared_and_importable():
    module = importlib.import_module("idc.schema")

    assert set(module.__all__) == EXPECTED_EXPORTS
    for name in module.__all__:
        assert hasattr(module, name), name


def test_canonical_models_are_the_seven_types_in_pipeline_order():
    assert (
        ToolCall,
        ToolResult,
        Message,
        Session,
        Chunk,
        Chain,
        CompactedBlock,
    ) == CANONICAL_MODELS


@pytest.mark.parametrize("model", CANONICAL_MODELS, ids=lambda m: m.__name__)
def test_every_canonical_model_is_documented_and_shares_the_base_config(model):
    assert issubclass(model, CanonicalModel)
    assert model.__doc__, f"{model.__name__} must document what it represents"
    assert model.model_config["extra"] == "forbid"
    assert model.model_config["validate_assignment"] is True


@pytest.mark.parametrize(
    ("member", "wire"),
    [
        (Agent.CLAUDE_CODE, "claude-code"),
        (Agent.GEMINI_CLI, "gemini-cli"),
        (Agent.OPENCODE, "opencode"),
        (SessionSource.RAW, "raw"),
        (SessionSource.NATIVE_COMPACTION, "native-compaction"),
        (SessionSource.INTENT_COMPACTION, "intent-compaction"),
        (Role.USER, "user"),
        (Role.ASSISTANT, "assistant"),
        (Role.SYSTEM, "system"),
        (Role.TOOL, "tool"),
        (ToolCategory.READ, "read"),
        (ToolCategory.WRITE, "write"),
        (ToolCategory.EXEC, "exec"),
        (ToolCategory.OTHER, "other"),
        (ChunkOrigin.USER, "user"),
        (ChunkOrigin.SUBAGENT, "subagent"),
    ],
    ids=lambda value: str(value),
)
def test_enum_members_carry_their_wire_value(member, wire):
    assert member == wire
    assert member.value == wire


@pytest.mark.parametrize(
    ("enum", "size"),
    [(Agent, 3), (SessionSource, 3), (Role, 4), (ToolCategory, 4), (ChunkOrigin, 2)],
    ids=lambda value: getattr(value, "__name__", value),
)
def test_enums_are_closed_sets(enum, size):
    assert len(list(enum)) == size
