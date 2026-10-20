"""The Claude Code importer: a session JSONL read into a canonical `Session`."""

import json
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from idc.importers import (
    NOTE_DROPPED_BLOCK,
    NOTE_SKIPPED_RECORD,
    NOTE_UNKNOWN_TOOL,
    IMPORTERS,
    PENDING_IMPORTERS,
    TranscriptImportError,
    import_claude_code,
    import_transcript,
    tool_registry,
)
from idc.schema import (
    Agent,
    Message,
    Role,
    SchemaValidationError,
    Session,
    SessionSource,
    ToolCategory,
    dumps,
    validate_document,
)
from native_transcripts import NATIVE

SESSION_ID = "7c1f0e2a-0000-4000-8000-000000000001"


def native(name: str) -> Path:
    return NATIVE.path("claude-code", name)


def write_jsonl(tmp_path: Path, records: list[dict], name: str = "session.jsonl") -> Path:
    path = tmp_path / name
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    return path


def user_record(uuid: str, content, **extra) -> dict:
    return {
        "type": "user",
        "uuid": uuid,
        "timestamp": "2026-09-20T12:00:00Z",
        "sessionId": "sess-tmp",
        "message": {"role": "user", "content": content},
        **extra,
    }


def assistant_record(uuid: str, content: list[dict], **extra) -> dict:
    return {
        "type": "assistant",
        "uuid": uuid,
        "timestamp": "2026-09-20T12:00:05Z",
        "sessionId": "sess-tmp",
        "message": {"role": "assistant", "model": "claude-opus-5", "content": content},
        **extra,
    }


def tool_use(call_id: str, name: str, **args) -> dict:
    return {"type": "tool_use", "id": call_id, "name": name, "input": dict(args)}


PASTED_IMAGE = {
    "type": "image",
    "source": {"type": "base64", "media_type": "image/png", "data": "iVBORw0KGgo="},
}
"""How Claude Code records a screenshot the user pasted into their prompt.

The canonical schema has no field for an image, so the block is dropped and
noted — but the user did take that turn, and T3.1 splits chunks on it.
"""


def messages_by_id(session: Session) -> dict[str, Message]:
    return {message.id: message for message in session.messages}


# --- the golden file --------------------------------------------------------


def test_the_sample_session_imports_to_the_golden_canonical_document():
    imported = import_claude_code(native("golden_session"))

    assert dumps(imported.session) + "\n" == NATIVE.expected_text("claude-code", "golden_session")


def test_the_imported_session_is_a_valid_canonical_document():
    imported = import_claude_code(native("golden_session"))

    revalidated = validate_document(Session, json.loads(dumps(imported.session)))

    assert revalidated == imported.session


def test_the_import_is_deterministic():
    first = import_claude_code(native("golden_session"))
    second = import_claude_code(native("golden_session"))

    assert dumps(first.session) == dumps(second.session)
    assert first.notes == second.notes


# --- session metadata -------------------------------------------------------


def test_the_session_takes_its_identifier_and_agent_from_the_transcript():
    session = import_claude_code(native("golden_session")).session

    assert session.id == SESSION_ID
    assert session.agent is Agent.CLAUDE_CODE


def test_the_model_is_taken_from_the_first_assistant_message():
    assert import_claude_code(native("golden_session")).session.model == "claude-opus-5"


def test_created_at_is_the_earliest_timestamp_of_the_transcript():
    session = import_claude_code(native("golden_session")).session

    assert session.created_at == datetime(2026, 9, 10, 9, 0, 0, tzinfo=UTC)
    assert session.created_at == min(
        message.timestamp for message in session.messages if message.timestamp is not None
    )


def test_a_transcript_without_a_session_identifier_is_rejected(tmp_path):
    path = write_jsonl(tmp_path, [{"type": "user", "uuid": "u1", "message": {"role": "user"}}])

    with pytest.raises(TranscriptImportError, match="session") as error:
        import_claude_code(path)

    assert str(path) in str(error.value)


def test_a_transcript_without_any_timestamp_is_rejected(tmp_path):
    record = user_record("u1", "hello")
    del record["timestamp"]
    path = write_jsonl(tmp_path, [record])

    with pytest.raises(TranscriptImportError, match="timestamp"):
        import_claude_code(path)


def test_a_transcript_without_a_single_message_is_rejected(tmp_path):
    path = write_jsonl(tmp_path, [{"type": "mode", "mode": "normal", "sessionId": "sess-tmp"}])

    with pytest.raises(TranscriptImportError, match="no message"):
        import_claude_code(path)


# --- messages ---------------------------------------------------------------


def test_the_messages_keep_the_identifiers_the_transcript_gave_them():
    session = import_claude_code(native("golden_session")).session

    assert [message.id for message in session.messages] == [
        "u01",
        "a01",
        "t01",
        "a02",
        "t02#0",
        "t02#1",
        "u02",
        "a03",
        "t03",
        "a04",
    ]


def test_the_roles_follow_the_block_kinds_not_the_wire_role():
    session = import_claude_code(native("golden_session")).session

    assert [message.role for message in session.messages] == [
        Role.USER,
        Role.ASSISTANT,
        Role.TOOL,
        Role.ASSISTANT,
        Role.TOOL,
        Role.TOOL,
        Role.USER,
        Role.ASSISTANT,
        Role.TOOL,
        Role.ASSISTANT,
    ]


def test_an_assistant_turn_keeps_its_text_and_its_calls_on_one_message():
    message = messages_by_id(import_claude_code(native("golden_session")).session)["a02"]

    assert message.content.startswith("`datetime.now()` is naive.")
    assert [call.name for call in message.tool_calls] == ["Edit", "Bash"]


def test_an_assistant_turn_that_only_calls_tools_has_empty_content():
    message = messages_by_id(import_claude_code(native("golden_session")).session)["a03"]

    assert message.content == ""
    assert [call.id for call in message.tool_calls] == ["toolu_04"]


def test_two_results_in_one_record_become_two_messages_with_distinct_ids():
    messages = messages_by_id(import_claude_code(native("golden_session")).session)

    assert messages["t02#0"].tool_result.call_id == "toolu_02"
    assert messages["t02#1"].tool_result.call_id == "toolu_03"
    assert messages["t02#0"].timestamp == messages["t02#1"].timestamp


def test_a_result_given_as_content_blocks_is_joined_into_the_output():
    result = messages_by_id(import_claude_code(native("golden_session")).session)[
        "t02#1"
    ].tool_result

    assert result.output == "1 failed, 3 passed\nE   assert '+00:00' in '2026-09-10T09:00:09'"
    assert result.is_error is True


def test_a_result_carries_no_exit_code_because_the_transcript_records_none():
    session = import_claude_code(native("golden_session")).session

    assert all(
        message.tool_result.exit_code is None
        for message in session.messages
        if message.tool_result is not None
    )


def test_an_injected_user_record_keeps_its_role_and_is_marked_as_injected():
    message = messages_by_id(import_claude_code(native("golden_session")).session)["u02"]

    assert message.role is Role.USER
    assert message.is_system_injected is True
    assert message.is_command is False


def test_a_typed_user_prompt_is_neither_injected_nor_a_command():
    message = messages_by_id(import_claude_code(native("golden_session")).session)["u01"]

    assert (message.is_system_injected, message.is_command) == (False, False)


def test_a_text_block_before_a_result_keeps_the_order_of_the_record(tmp_path):
    path = write_jsonl(
        tmp_path,
        [
            user_record(
                "u1",
                [
                    {"type": "text", "text": "and please also run the linter"},
                    {"type": "tool_result", "tool_use_id": "toolu_1", "content": "ok"},
                ],
            )
        ],
    )

    session = import_claude_code(path).session

    assert [(message.id, message.role) for message in session.messages] == [
        ("u1#0", Role.USER),
        ("u1#1", Role.TOOL),
    ]


def test_several_text_blocks_of_one_record_are_joined_into_one_message(tmp_path):
    path = write_jsonl(
        tmp_path,
        [
            user_record(
                "u1", [{"type": "text", "text": "first"}, {"type": "text", "text": "second"}]
            )
        ],
    )

    session = import_claude_code(path).session

    assert len(session.messages) == 1
    assert session.messages[0].content == "first\n\nsecond"


def test_a_system_record_with_content_becomes_an_injected_system_message(tmp_path):
    path = write_jsonl(
        tmp_path,
        [
            {
                "type": "system",
                "uuid": "s1",
                "subtype": "away_summary",
                "content": "You were away for 20 minutes.",
                "timestamp": "2026-09-20T12:00:00Z",
                "sessionId": "sess-tmp",
            }
        ],
    )

    message = import_claude_code(path).session.messages[0]

    assert (message.role, message.content, message.is_system_injected) == (
        Role.SYSTEM,
        "You were away for 20 minutes.",
        True,
    )


def test_a_line_break_character_in_a_message_does_not_break_the_import(tmp_path):
    text = f"const a=1;{chr(0x2028)}const b=2;{chr(0x2029)}const c=3;{chr(0x85)}done"
    path = tmp_path / "session.jsonl"
    path.write_text(
        json.dumps(user_record("u1", text), ensure_ascii=False) + "\n", encoding="utf-8"
    )

    assert import_claude_code(path).session.messages[0].content == text


def test_a_transcript_written_with_a_byte_order_mark_imports(tmp_path):
    path = tmp_path / "session.jsonl"
    path.write_bytes(
        b"\xef\xbb\xbf" + json.dumps(user_record("u1", "hello")).encode("utf-8") + b"\n"
    )

    assert import_claude_code(path).session.messages[0].content == "hello"


# --- timestamps -------------------------------------------------------------


def test_every_timestamp_is_timezone_aware_and_keeps_its_milliseconds():
    session = import_claude_code(native("truncated_front")).session

    assert session.messages[0].timestamp == datetime(2026, 9, 11, 14, 20, 0, 250000, tzinfo=UTC)
    assert all(
        message.timestamp.tzinfo is not None
        for message in session.messages
        if message.timestamp is not None
    )


def test_an_offset_timestamp_keeps_its_instant(tmp_path):
    record = user_record("u1", "hello")
    record["timestamp"] = "2026-09-20T14:00:00+02:00"
    path = write_jsonl(tmp_path, [record])

    timestamp = import_claude_code(path).session.messages[0].timestamp

    assert timestamp == datetime(2026, 9, 20, 12, 0, 0, tzinfo=UTC)
    assert timestamp.utcoffset() == timedelta(hours=2)


def test_a_naive_timestamp_is_read_as_utc(tmp_path):
    record = user_record("u1", "hello")
    record["timestamp"] = "2026-09-20T12:00:00"
    path = write_jsonl(tmp_path, [record])

    assert import_claude_code(path).session.messages[0].timestamp == datetime(
        2026, 9, 20, 12, 0, 0, tzinfo=UTC
    )


def test_a_timestamp_that_is_no_date_is_rejected(tmp_path):
    record = user_record("u1", "hello")
    record["timestamp"] = "yesterday"
    path = write_jsonl(tmp_path, [record])

    with pytest.raises(TranscriptImportError, match="timestamp"):
        import_claude_code(path)


# --- tool categories and unknown tools --------------------------------------


def test_every_call_of_the_sample_session_carries_the_registry_category():
    registry = tool_registry()
    session = import_claude_code(native("golden_session")).session

    calls = [call for message in session.messages for call in message.tool_calls]

    assert [(call.name, call.category) for call in calls] == [
        ("Read", ToolCategory.READ),
        ("Edit", ToolCategory.WRITE),
        ("Bash", ToolCategory.EXEC),
        ("Bash", ToolCategory.EXEC),
    ]
    assert all(registry.knows(Agent.CLAUDE_CODE, call.name) for call in calls)


def test_the_delegating_tool_of_the_real_transcripts_is_declared_as_a_subagent_tool():
    registry = tool_registry()

    assert registry.knows(Agent.CLAUDE_CODE, "Agent")
    assert registry.is_subagent_tool(Agent.CLAUDE_CODE, "Agent")
    assert registry.category(Agent.CLAUDE_CODE, "Agent") is ToolCategory.OTHER


def test_an_unknown_tool_resolves_to_other_and_is_noted():
    imported = import_claude_code(native("unknown_tool"))

    calls = {
        call.name: call for message in imported.session.messages for call in message.tool_calls
    }

    assert calls["Teleport"].category is ToolCategory.OTHER
    assert calls["mcp__example__getThing"].category is ToolCategory.OTHER
    assert imported.unknown_tools == ("Teleport", "mcp__example__getThing")


def test_a_declared_tool_is_not_reported_as_unknown():
    imported = import_claude_code(native("unknown_tool"))

    assert "Agent" not in imported.unknown_tools


def test_an_unknown_tool_is_logged_once_per_name(caplog):
    with caplog.at_level(logging.WARNING, logger="idc.importers"):
        import_claude_code(native("unknown_tool"))

    warnings = [
        record.getMessage() for record in caplog.records if record.levelno >= logging.WARNING
    ]

    assert sum("'Teleport'" in message for message in warnings) == 1
    assert sum("'mcp__example__getThing'" in message for message in warnings) == 1
    assert not any("'Agent'" in message for message in warnings)


def test_the_same_unknown_tool_called_twice_is_noted_with_its_count(tmp_path):
    path = write_jsonl(
        tmp_path,
        [
            user_record("u1", "do it"),
            assistant_record(
                "a1",
                [
                    tool_use("toolu_1", "Teleport", destination="/a"),
                    tool_use("toolu_2", "Teleport", destination="/b"),
                ],
            ),
        ],
    )

    imported = import_claude_code(path)

    notes = imported.notes_of(NOTE_UNKNOWN_TOOL)

    assert [(note.detail, note.count) for note in notes] == [("Teleport", 2)]


# --- records and blocks that carry no canonical message ---------------------


def test_bookkeeping_records_produce_no_message_and_are_noted():
    imported = import_claude_code(native("golden_session"))

    skipped = {note.detail: note.count for note in imported.notes_of(NOTE_SKIPPED_RECORD)}

    assert skipped == {
        "ai-title": 1,
        "attachment": 1,
        "file-history-snapshot": 1,
        "mode": 1,
        "permission-mode": 1,
        "system": 1,
    }


def test_a_thinking_block_is_dropped_and_noted():
    imported = import_claude_code(native("golden_session"))

    assert [(note.detail, note.count) for note in imported.notes_of(NOTE_DROPPED_BLOCK)] == [
        ("thinking", 1)
    ]
    assert "The exporter probably calls" not in dumps(imported.session)


def test_an_unknown_block_kind_is_dropped_and_noted_instead_of_failing(tmp_path):
    path = write_jsonl(
        tmp_path,
        [
            user_record("u1", "do it"),
            assistant_record(
                "a1", [{"type": "hologram", "data": "?"}, {"type": "text", "text": "ok"}]
            ),
        ],
    )

    imported = import_claude_code(path)

    assert imported.session.messages[1].content == "ok"
    assert [note.detail for note in imported.notes_of(NOTE_DROPPED_BLOCK)] == ["hologram"]


def test_an_unknown_record_type_is_skipped_and_noted_instead_of_failing(tmp_path):
    path = write_jsonl(
        tmp_path,
        [user_record("u1", "do it"), {"type": "hologram-link", "sessionId": "sess-tmp"}],
    )

    imported = import_claude_code(path)

    assert len(imported.session.messages) == 1
    assert [note.detail for note in imported.notes_of(NOTE_SKIPPED_RECORD)] == ["hologram-link"]


@pytest.mark.parametrize("content", [7, 3.5, True, {"role": "user", "text": "lost"}, None])
# A list is deliberately not among them any more: a list *is* the block list this
# importer reads, so a record carrying one is a turn whose blocks could not be
# represented, not a content the importer cannot read. The case it used to cover
# is pinned by `test_a_user_record_whose_only_entry_is_no_block_is_still_the_turn_it_was`.
def test_a_user_record_whose_content_is_not_readable_invents_no_turn(tmp_path, content):
    path = write_jsonl(
        tmp_path, [user_record("u1", "fix the exporter"), user_record("u2", content)]
    )

    imported = import_claude_code(path)

    assert [message.id for message in imported.session.messages] == ["u1"]
    assert ("user", 1) in [
        (note.detail, note.count) for note in imported.notes_of(NOTE_SKIPPED_RECORD)
    ]


def test_a_user_record_without_a_message_invents_no_turn(tmp_path):
    record = user_record("u2", "")
    del record["message"]
    path = write_jsonl(tmp_path, [user_record("u1", "fix the exporter"), record])

    imported = import_claude_code(path)

    assert [message.id for message in imported.session.messages] == ["u1"]
    assert [(note.detail, note.count) for note in imported.notes_of(NOTE_SKIPPED_RECORD)] == [
        ("user", 1)
    ]


def test_the_content_a_user_record_could_not_be_read_from_is_noted(tmp_path):
    path = write_jsonl(
        tmp_path,
        [user_record("u1", "fix the exporter"), user_record("u2", {"text": "lost"})],
    )

    imported = import_claude_code(path)

    assert [note.detail for note in imported.notes_of(NOTE_DROPPED_BLOCK)] == ["<content: object>"]


def test_a_user_record_whose_text_is_empty_is_still_the_turn_it_was(tmp_path):
    path = write_jsonl(tmp_path, [user_record("u1", "")])

    session = import_claude_code(path).session

    assert [(message.id, message.role, message.content) for message in session.messages] == [
        ("u1", Role.USER, "")
    ]


# --- a block-list user record whose blocks carry no text --------------------
#
# The scalar path imports `""` as the turn it was, because the transcript did
# record a user message there. These pin the same answer for a block list: a
# pasted screenshot is a turn whose content the schema cannot hold, not a turn
# that never happened — and T3.1 splits chunks on exactly that boundary.


def test_a_pasted_screenshot_is_still_the_turn_the_user_took(tmp_path):
    path = write_jsonl(tmp_path, [user_record("u1", [PASTED_IMAGE])])

    imported = import_claude_code(path)

    assert [
        (message.id, message.role, message.content) for message in imported.session.messages
    ] == [("u1", Role.USER, "")]
    assert [note.detail for note in imported.notes_of(NOTE_DROPPED_BLOCK)] == ["image"]
    assert imported.notes_of(NOTE_SKIPPED_RECORD) == ()


def test_a_user_record_whose_only_text_block_is_empty_is_still_the_turn_it_was(tmp_path):
    path = write_jsonl(tmp_path, [user_record("u1", [{"type": "text", "text": ""}])])

    imported = import_claude_code(path)

    assert [
        (message.id, message.role, message.content) for message in imported.session.messages
    ] == [("u1", Role.USER, "")]
    assert imported.notes == ()


def test_a_user_record_whose_block_list_is_empty_is_still_the_turn_it_was(tmp_path):
    path = write_jsonl(tmp_path, [user_record("u1", [])])

    imported = import_claude_code(path)

    assert [
        (message.id, message.role, message.content) for message in imported.session.messages
    ] == [("u1", Role.USER, "")]
    assert imported.notes == ()


def test_a_user_record_whose_only_entry_is_no_block_is_still_the_turn_it_was(tmp_path):
    path = write_jsonl(tmp_path, [user_record("u1", ["just a string"])])

    imported = import_claude_code(path)

    assert [
        (message.id, message.role, message.content) for message in imported.session.messages
    ] == [("u1", Role.USER, "")]
    assert [note.detail for note in imported.notes_of(NOTE_DROPPED_BLOCK)] == ["<entry: string>"]
    assert imported.notes_of(NOTE_SKIPPED_RECORD) == ()


def test_a_screenshot_next_to_a_typed_prompt_keeps_the_prompt(tmp_path):
    path = write_jsonl(
        tmp_path,
        [user_record("u1", [{"type": "text", "text": "what is wrong here?"}, PASTED_IMAGE])],
    )

    imported = import_claude_code(path)

    assert [
        (message.id, message.role, message.content) for message in imported.session.messages
    ] == [("u1", Role.USER, "what is wrong here?")]
    assert [note.detail for note in imported.notes_of(NOTE_DROPPED_BLOCK)] == ["image"]


def test_a_text_block_of_blanks_keeps_its_blanks(tmp_path):
    path = write_jsonl(tmp_path, [user_record("u1", [{"type": "text", "text": "   "}])])

    imported = import_claude_code(path)

    assert [message.content for message in imported.session.messages] == ["   "]
    assert imported.notes == ()


def test_a_record_holding_only_a_result_is_a_tool_message_and_no_user_turn(tmp_path):
    path = write_jsonl(
        tmp_path,
        [user_record("u1", [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "ok"}])],
    )

    imported = import_claude_code(path)

    assert [(message.id, message.role) for message in imported.session.messages] == [
        ("u1", Role.TOOL)
    ]
    assert imported.notes == ()


def test_a_dropped_block_next_to_a_result_adds_no_empty_user_turn(tmp_path):
    path = write_jsonl(
        tmp_path,
        [
            user_record(
                "u1",
                [
                    PASTED_IMAGE,
                    {"type": "tool_result", "tool_use_id": "toolu_1", "content": "ok"},
                ],
            )
        ],
    )

    imported = import_claude_code(path)

    assert [(message.id, message.role) for message in imported.session.messages] == [
        ("u1", Role.TOOL)
    ]
    assert [note.detail for note in imported.notes_of(NOTE_DROPPED_BLOCK)] == ["image"]


def test_an_empty_text_block_next_to_a_result_adds_no_empty_user_turn(tmp_path):
    path = write_jsonl(
        tmp_path,
        [
            user_record(
                "u1",
                [
                    {"type": "text", "text": ""},
                    {"type": "tool_result", "tool_use_id": "toolu_1", "content": "ok"},
                ],
            )
        ],
    )

    imported = import_claude_code(path)

    assert [(message.id, message.role) for message in imported.session.messages] == [
        ("u1", Role.TOOL)
    ]
    assert imported.notes == ()


# --- a text block whose text is not a string --------------------------------
#
# Coercing it was worse than dropping it: `0`, `false` and `[]` became `""`
# with no note, and an object or a list became a Python repr that landed in
# canonical content. A measurement tool may lose a value it cannot read; it may
# not invent one.


@pytest.mark.parametrize(
    ("value", "shape"),
    [(0, "number"), (False, "boolean"), ([], "array"), ({"a": 1}, "object")],
)
def test_a_text_block_whose_text_is_no_string_is_dropped_and_noted(tmp_path, value, shape):
    path = write_jsonl(tmp_path, [user_record("u1", [{"type": "text", "text": value}])])

    imported = import_claude_code(path)

    assert [message.content for message in imported.session.messages] == [""]
    assert [note.detail for note in imported.notes_of(NOTE_DROPPED_BLOCK)] == [
        f"text/<text: {shape}>"
    ]


def test_an_object_text_never_reaches_the_canonical_content(tmp_path):
    path = write_jsonl(tmp_path, [user_record("u1", [{"type": "text", "text": {"a": 1}}])])

    imported = import_claude_code(path)

    assert "{'a': 1}" not in dumps(imported.session)
    assert "a" not in imported.session.messages[0].content


def test_a_text_block_whose_text_is_empty_keeps_the_empty_string_unnoted(tmp_path):
    path = write_jsonl(
        tmp_path,
        [user_record("u1", [{"type": "text", "text": ""}, {"type": "text", "text": "hi"}])],
    )

    imported = import_claude_code(path)

    assert [message.content for message in imported.session.messages] == ["hi"]
    assert imported.notes == ()


def test_a_text_block_without_a_text_field_is_not_noted(tmp_path):
    path = write_jsonl(tmp_path, [user_record("u1", [{"type": "text"}])])

    imported = import_claude_code(path)

    assert [message.content for message in imported.session.messages] == [""]
    assert imported.notes == ()


def test_an_assistant_text_block_whose_text_is_no_string_is_dropped_and_noted(tmp_path):
    path = write_jsonl(
        tmp_path,
        [
            user_record("u1", "fix the exporter"),
            assistant_record("a1", [{"type": "text", "text": ["lost"]}]),
        ],
    )

    imported = import_claude_code(path)

    assert imported.session.messages[1].content == ""
    assert [note.detail for note in imported.notes_of(NOTE_DROPPED_BLOCK)] == ["text/<text: array>"]


def test_a_result_text_block_whose_text_is_no_string_is_noted_inside_its_result(tmp_path):
    path = write_jsonl(
        tmp_path,
        [
            user_record(
                "u1",
                [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_1",
                        "content": [{"type": "text", "text": {"stdout": "1 passed"}}],
                    }
                ],
            )
        ],
    )

    imported = import_claude_code(path)

    assert imported.session.messages[0].tool_result.output == ""
    assert [note.detail for note in imported.notes_of(NOTE_DROPPED_BLOCK)] == [
        "tool_result/text/<text: object>"
    ]


def test_a_summary_text_block_whose_text_is_no_string_is_noted(tmp_path):
    path = write_jsonl(
        tmp_path, [user_record("c1", [{"type": "text", "text": 0}], isCompactSummary=True)]
    )

    imported = import_claude_code(path)

    assert imported.session.messages[0].content == ""
    assert [note.detail for note in imported.notes_of(NOTE_DROPPED_BLOCK)] == [
        "text/<text: number>"
    ]


def test_a_content_entry_that_is_not_an_object_is_dropped_and_noted(tmp_path):
    path = write_jsonl(
        tmp_path,
        [user_record("u1", ["just a string", {"type": "text", "text": "fix the exporter"}])],
    )

    imported = import_claude_code(path)

    assert imported.session.messages[0].content == "fix the exporter"
    assert [note.detail for note in imported.notes_of(NOTE_DROPPED_BLOCK)] == ["<entry: string>"]


def test_a_tool_use_whose_arguments_are_not_an_object_is_noted_with_its_tool(tmp_path):
    path = write_jsonl(
        tmp_path,
        [
            user_record("u1", "run the tests"),
            assistant_record(
                "a1",
                [
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "Bash",
                        "input": ["uv", "run", "pytest", "-q"],
                    }
                ],
            ),
        ],
    )

    imported = import_claude_code(path)

    assert imported.session.messages[1].tool_calls[0].args == {}
    assert [note.detail for note in imported.notes_of(NOTE_DROPPED_BLOCK)] == [
        "tool_use/Bash/<input: array>"
    ]


def test_a_tool_use_without_arguments_is_not_noted(tmp_path):
    path = write_jsonl(
        tmp_path,
        [
            user_record("u1", "run the tests"),
            assistant_record("a1", [{"type": "tool_use", "id": "toolu_1", "name": "Bash"}]),
        ],
    )

    imported = import_claude_code(path)

    assert imported.session.messages[1].tool_calls[0].args == {}
    assert imported.notes_of(NOTE_DROPPED_BLOCK) == ()


def test_a_tool_result_whose_content_is_an_object_is_noted(tmp_path):
    path = write_jsonl(
        tmp_path,
        [
            user_record(
                "u1",
                [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_1",
                        "content": {"stdout": "1 passed"},
                    }
                ],
            )
        ],
    )

    imported = import_claude_code(path)

    assert imported.session.messages[0].tool_result.output == ""
    assert [note.detail for note in imported.notes_of(NOTE_DROPPED_BLOCK)] == [
        "tool_result/<content: object>"
    ]


def test_an_assistant_record_whose_content_is_an_object_is_noted(tmp_path):
    path = write_jsonl(
        tmp_path,
        [
            user_record("u1", "fix the exporter"),
            {
                "type": "assistant",
                "uuid": "a1",
                "timestamp": "2026-09-20T12:00:05Z",
                "sessionId": "sess-tmp",
                "message": {"role": "assistant", "content": {"text": "lost"}},
            },
        ],
    )

    imported = import_claude_code(path)

    assert imported.session.messages[1].content == ""
    assert [note.detail for note in imported.notes_of(NOTE_DROPPED_BLOCK)] == ["<content: object>"]


def test_a_system_record_whose_content_is_an_object_is_noted(tmp_path):
    path = write_jsonl(
        tmp_path,
        [
            user_record("u1", "fix the exporter"),
            {
                "type": "system",
                "uuid": "s1",
                "content": {"text": "lost"},
                "timestamp": "2026-09-20T12:00:05Z",
                "sessionId": "sess-tmp",
            },
        ],
    )

    imported = import_claude_code(path)

    assert [message.id for message in imported.session.messages] == ["u1"]
    assert [note.detail for note in imported.notes_of(NOTE_DROPPED_BLOCK)] == ["<content: object>"]


def test_a_record_without_a_type_and_one_whose_type_is_null_are_noted_apart(tmp_path):
    path = write_jsonl(
        tmp_path,
        [
            user_record("u1", "fix the exporter"),
            {"uuid": "x1", "sessionId": "sess-tmp"},
            {"type": None, "uuid": "x2", "sessionId": "sess-tmp"},
        ],
    )

    imported = import_claude_code(path)

    assert [(note.detail, note.count) for note in imported.notes_of(NOTE_SKIPPED_RECORD)] == [
        ("<no type>", 1),
        ("<null type>", 1),
    ]


def test_a_record_whose_type_is_a_number_keeps_the_number_in_the_note(tmp_path):
    path = write_jsonl(
        tmp_path,
        [user_record("u1", "fix the exporter"), {"type": 7, "sessionId": "sess-tmp"}],
    )

    imported = import_claude_code(path)

    assert [note.detail for note in imported.notes_of(NOTE_SKIPPED_RECORD)] == ["7"]


# --- a block whose type is missing, null, or the string "None" --------------
#
# The same collision the record level was fixed for, one level down: all three
# read as `dropped-block: None`, so a block kind a future Claude Code version
# spells `None` could not be told from a damaged one.


def test_a_block_without_a_type_and_one_whose_type_is_null_are_noted_apart(tmp_path):
    path = write_jsonl(
        tmp_path,
        [
            user_record("u1", "fix the exporter"),
            assistant_record("a1", [{"data": "?"}, {"type": None}]),
        ],
    )

    imported = import_claude_code(path)

    assert [(note.detail, note.count) for note in imported.notes_of(NOTE_DROPPED_BLOCK)] == [
        ("<no type>", 1),
        ("<null type>", 1),
    ]


def test_a_block_whose_type_is_the_string_none_keeps_its_own_spelling(tmp_path):
    path = write_jsonl(
        tmp_path,
        [user_record("u1", "fix the exporter"), assistant_record("a1", [{"type": "None"}])],
    )

    imported = import_claude_code(path)

    assert [note.detail for note in imported.notes_of(NOTE_DROPPED_BLOCK)] == ["None"]


def test_a_block_whose_type_is_a_number_keeps_the_number_in_the_note(tmp_path):
    path = write_jsonl(
        tmp_path, [user_record("u1", "fix the exporter"), assistant_record("a1", [{"type": 7}])]
    )

    imported = import_claude_code(path)

    assert [note.detail for note in imported.notes_of(NOTE_DROPPED_BLOCK)] == ["7"]


def test_a_user_block_without_a_type_and_one_whose_type_is_null_are_noted_apart(tmp_path):
    path = write_jsonl(
        tmp_path,
        [user_record("u1", [{"data": "?"}, {"type": None}, {"type": "text", "text": "hi"}])],
    )

    imported = import_claude_code(path)

    assert imported.session.messages[0].content == "hi"
    assert [note.detail for note in imported.notes_of(NOTE_DROPPED_BLOCK)] == [
        "<no type>",
        "<null type>",
    ]


def test_a_result_block_without_a_type_is_noted_inside_its_result(tmp_path):
    path = write_jsonl(
        tmp_path,
        [
            user_record(
                "u1",
                [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_1",
                        "content": [{"data": "?"}, {"type": None}],
                    }
                ],
            )
        ],
    )

    imported = import_claude_code(path)

    assert [note.detail for note in imported.notes_of(NOTE_DROPPED_BLOCK)] == [
        "tool_result/<no type>",
        "tool_result/<null type>",
    ]


def test_a_summary_block_without_a_type_is_noted_apart_from_a_null_one(tmp_path):
    path = write_jsonl(
        tmp_path,
        [
            user_record(
                "c1",
                [{"data": "?"}, {"type": None}, {"type": "text", "text": "continued"}],
                isCompactSummary=True,
            )
        ],
    )

    imported = import_claude_code(path)

    assert imported.session.messages[0].content == "continued"
    assert [note.detail for note in imported.notes_of(NOTE_DROPPED_BLOCK)] == [
        "<no type>",
        "<null type>",
    ]


# --- record fields the importer cannot use ----------------------------------
#
# The identifier it falls back from, the session it keeps and the `message` it
# reads are the three record-level values that were replaced, ignored or
# treated as absent without a word. The docstring documents the `line-<n>`
# fallback for a record *without* a uuid; a record that has an unusable one is
# a different thing, and so is "a value I cannot read" versus "no value".


def test_a_record_whose_identifier_is_no_string_falls_back_and_is_noted(tmp_path):
    record = user_record("u1", "fix the exporter")
    record["uuid"] = 7
    path = write_jsonl(tmp_path, [record])

    imported = import_claude_code(path)

    assert [message.id for message in imported.session.messages] == ["line-1"]
    assert [note.detail for note in imported.notes_of(NOTE_DROPPED_BLOCK)] == ["<uuid: number>"]


def test_a_record_whose_identifier_is_empty_falls_back_and_is_noted(tmp_path):
    record = user_record("u1", "fix the exporter")
    record["uuid"] = ""
    path = write_jsonl(tmp_path, [record])

    imported = import_claude_code(path)

    assert [message.id for message in imported.session.messages] == ["line-1"]
    assert [note.detail for note in imported.notes_of(NOTE_DROPPED_BLOCK)] == [
        "<uuid: empty string>"
    ]


def test_a_record_without_an_identifier_falls_back_unnoted(tmp_path):
    record = user_record("u1", "fix the exporter")
    del record["uuid"]
    path = write_jsonl(tmp_path, [record])

    imported = import_claude_code(path)

    assert [message.id for message in imported.session.messages] == ["line-1"]
    assert imported.notes == ()


def test_an_unusable_identifier_is_noted_once_per_record_not_per_message(tmp_path):
    record = user_record(
        "u1",
        [
            {"type": "tool_result", "tool_use_id": "toolu_1", "content": "a"},
            {"type": "tool_result", "tool_use_id": "toolu_2", "content": "b"},
        ],
    )
    record["uuid"] = 7
    path = write_jsonl(tmp_path, [record])

    imported = import_claude_code(path)

    assert [message.id for message in imported.session.messages] == ["line-1#0", "line-1#1"]
    assert [(note.detail, note.count) for note in imported.notes_of(NOTE_DROPPED_BLOCK)] == [
        ("<uuid: number>", 1)
    ]


def test_a_second_session_identifier_is_ignored_but_noted(tmp_path):
    second = user_record("u2", "and the importer")
    second["sessionId"] = "sess-other"
    path = write_jsonl(tmp_path, [user_record("u1", "fix the exporter"), second])

    imported = import_claude_code(path)

    assert imported.session.id == "sess-tmp"
    assert [(note.detail, note.count) for note in imported.notes_of(NOTE_DROPPED_BLOCK)] == [
        ("<sessionId: a second value>", 1)
    ]


def test_the_same_session_identifier_on_every_record_is_not_noted(tmp_path):
    path = write_jsonl(
        tmp_path, [user_record("u1", "fix the exporter"), user_record("u2", "and the importer")]
    )

    imported = import_claude_code(path)

    assert imported.session.id == "sess-tmp"
    assert imported.notes == ()


def test_a_record_that_names_no_session_is_not_noted(tmp_path):
    second = user_record("u2", "and the importer")
    del second["sessionId"]
    path = write_jsonl(tmp_path, [user_record("u1", "fix the exporter"), second])

    imported = import_claude_code(path)

    assert imported.session.id == "sess-tmp"
    assert imported.notes == ()


@pytest.mark.parametrize(
    ("message", "shape"), [(["lost"], "array"), ("lost", "string"), (7, "number")]
)
def test_a_user_record_whose_message_is_no_object_is_noted_by_its_shape(tmp_path, message, shape):
    second = user_record("u2", "lost")
    second["message"] = message
    path = write_jsonl(tmp_path, [user_record("u1", "fix the exporter"), second])

    imported = import_claude_code(path)

    assert [note.detail for note in imported.notes_of(NOTE_DROPPED_BLOCK)] == [
        f"<message: {shape}>"
    ]
    assert [note.detail for note in imported.notes_of(NOTE_SKIPPED_RECORD)] == ["user"]


def test_a_user_record_whose_message_is_null_is_only_noted_as_skipped(tmp_path):
    second = user_record("u2", "lost")
    second["message"] = None
    path = write_jsonl(tmp_path, [user_record("u1", "fix the exporter"), second])

    imported = import_claude_code(path)

    assert imported.notes_of(NOTE_DROPPED_BLOCK) == ()
    assert [note.detail for note in imported.notes_of(NOTE_SKIPPED_RECORD)] == ["user"]


def test_an_assistant_record_whose_message_is_no_object_is_noted_by_its_shape(tmp_path):
    second = assistant_record("a1", [])
    second["message"] = ["lost"]
    path = write_jsonl(tmp_path, [user_record("u1", "fix the exporter"), second])

    imported = import_claude_code(path)

    assert [message.id for message in imported.session.messages] == ["u1", "a1"]
    assert [note.detail for note in imported.notes_of(NOTE_DROPPED_BLOCK)] == ["<message: array>"]


# --- the error flag a result carries ----------------------------------------
#
# T6 resolves the last *passing* verification command from this flag, so a
# wrong `False` is the expensive direction: it would make a run that failed the
# accepted solution. `is True` silently produced exactly that for a
# non-boolean, so an unreadable flag now counts as an error and is noted.


def result_record(uuid: str, **flag) -> dict:
    return user_record(
        uuid, [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "out", **flag}]
    )


@pytest.mark.parametrize(
    ("value", "shape"),
    [("true", "string"), ("false", "string"), (1, "number"), ([], "array"), ({}, "object")],
)
def test_a_result_whose_error_flag_is_no_boolean_counts_as_an_error_and_is_noted(
    tmp_path, value, shape
):
    path = write_jsonl(tmp_path, [result_record("u1", is_error=value)])

    imported = import_claude_code(path)

    assert imported.session.messages[0].tool_result.is_error is True
    assert [note.detail for note in imported.notes_of(NOTE_DROPPED_BLOCK)] == [
        f"tool_result/<is_error: {shape}>"
    ]


def test_a_result_flagged_as_an_error_stays_one_unnoted(tmp_path):
    path = write_jsonl(tmp_path, [result_record("u1", is_error=True)])

    imported = import_claude_code(path)

    assert imported.session.messages[0].tool_result.is_error is True
    assert imported.notes == ()


def test_a_result_flagged_as_no_error_stays_none_unnoted(tmp_path):
    path = write_jsonl(tmp_path, [result_record("u1", is_error=False)])

    imported = import_claude_code(path)

    assert imported.session.messages[0].tool_result.is_error is False
    assert imported.notes == ()


def test_a_result_without_an_error_flag_is_no_error_and_unnoted(tmp_path):
    path = write_jsonl(tmp_path, [result_record("u1")])

    imported = import_claude_code(path)

    assert imported.session.messages[0].tool_result.is_error is False
    assert imported.notes == ()


def test_a_result_whose_error_flag_is_null_is_no_error_and_unnoted(tmp_path):
    path = write_jsonl(tmp_path, [result_record("u1", is_error=None)])

    imported = import_claude_code(path)

    assert imported.session.messages[0].tool_result.is_error is False
    assert imported.notes == ()


# --- a transcript that is cut off or interrupted -----------------------------


def test_a_front_truncated_transcript_imports_as_a_valid_session():
    session = import_claude_code(native("truncated_front")).session

    first = session.messages[0]

    assert first.role is Role.TOOL
    assert first.tool_result.call_id == "toolu_00"
    assert [call.id for message in session.messages for call in message.tool_calls] == []


def test_a_call_whose_result_never_arrives_is_kept(tmp_path):
    session = import_claude_code(native("interrupted_turn")).session

    calls = [call.id for message in session.messages for call in message.tool_calls]
    results = [
        message.tool_result.call_id
        for message in session.messages
        if message.tool_result is not None
    ]

    assert calls == ["toolu_20"]
    assert results == []


def test_the_interruption_notice_is_imported_as_a_user_message():
    session = import_claude_code(native("interrupted_turn")).session

    last = session.messages[-1]

    assert (last.role, last.is_system_injected) == (Role.USER, False)
    assert "interrupted" in last.content


def test_a_result_that_references_no_call_is_rejected_with_its_line(tmp_path):
    path = write_jsonl(
        tmp_path,
        [user_record("u1", "do it"), user_record("t1", [{"type": "tool_result", "content": "ok"}])],
    )

    with pytest.raises(TranscriptImportError, match="line 2") as error:
        import_claude_code(path)

    assert "tool_use_id" in str(error.value)


def test_a_call_without_a_name_or_an_identifier_is_rejected_with_its_line(tmp_path):
    path = write_jsonl(
        tmp_path,
        [user_record("u1", "do it"), assistant_record("a1", [{"type": "tool_use", "input": {}}])],
    )

    with pytest.raises(TranscriptImportError, match="line 2"):
        import_claude_code(path)


def test_a_tool_call_identifier_used_twice_is_reported_by_the_schema(tmp_path):
    path = write_jsonl(
        tmp_path,
        [
            user_record("u1", "do it"),
            assistant_record("a1", [tool_use("toolu_1", "Read", file_path="/a")]),
            assistant_record("a2", [tool_use("toolu_1", "Read", file_path="/b")]),
        ],
    )

    with pytest.raises(SchemaValidationError, match="toolu_1"):
        import_claude_code(path)


# --- native compaction ------------------------------------------------------


def test_a_native_compaction_summary_sets_the_session_provenance():
    session = import_claude_code(native("native_compaction")).session

    assert session.source is SessionSource.NATIVE_COMPACTION


def test_the_summary_is_imported_as_an_injected_system_message():
    session = import_claude_code(native("native_compaction")).session

    summary = messages_by_id(session)["c40"]

    assert (summary.role, summary.is_system_injected) == (Role.SYSTEM, True)
    assert summary.content.startswith("This session is being continued")


def test_the_compact_command_is_marked_as_a_command():
    session = import_claude_code(native("native_compaction")).session

    command = messages_by_id(session)["u40"]

    assert command.is_command is True
    assert "/compact" in command.content


def test_the_turn_after_the_summary_is_an_ordinary_user_turn():
    session = import_claude_code(native("native_compaction")).session

    turn = messages_by_id(session)["u41"]

    assert (turn.is_command, turn.is_system_injected) == (False, False)


def test_a_transcript_without_a_summary_stays_raw():
    assert import_claude_code(native("golden_session")).session.source is SessionSource.RAW


def test_the_echo_of_a_slash_command_is_marked_as_a_command(tmp_path):
    path = write_jsonl(
        tmp_path,
        [user_record("u1", "<local-command-stdout>branch is clean</local-command-stdout>")],
    )

    assert import_claude_code(path).session.messages[0].is_command is True


def test_a_thinking_block_in_a_summary_record_is_dropped_and_noted(tmp_path):
    path = write_jsonl(
        tmp_path,
        [
            user_record(
                "c1",
                [
                    {"type": "thinking", "thinking": "the user was fixing the exporter"},
                    {"type": "text", "text": "This session is being continued"},
                ],
                isCompactSummary=True,
            )
        ],
    )

    imported = import_claude_code(path)

    assert imported.session.messages[0].content == "This session is being continued"
    assert [(note.detail, note.count) for note in imported.notes_of(NOTE_DROPPED_BLOCK)] == [
        ("thinking", 1)
    ]
    assert "the user was fixing" not in dumps(imported.session)


def test_a_tool_result_block_in_a_summary_record_is_dropped_and_noted(tmp_path):
    path = write_jsonl(
        tmp_path,
        [
            user_record(
                "c1",
                [
                    {"type": "text", "text": "This session is being continued"},
                    {"type": "tool_result", "tool_use_id": "toolu_1", "content": "lost output"},
                ],
                isCompactSummary=True,
            )
        ],
    )

    imported = import_claude_code(path)

    assert [note.detail for note in imported.notes_of(NOTE_DROPPED_BLOCK)] == ["tool_result"]
    assert [message.id for message in imported.session.messages] == ["c1"]


# --- the shared entry point -------------------------------------------------


def test_the_dispatcher_reaches_the_claude_code_importer():
    path = native("golden_session")

    assert import_transcript(Agent.CLAUDE_CODE, path) == import_claude_code(path)


def test_the_dispatcher_accepts_the_agents_wire_string():
    path = native("golden_session")

    assert import_transcript("claude-code", path).session.agent is Agent.CLAUDE_CODE


# Every agent now has an importer, so `PENDING_IMPORTERS` is empty and
# parametrizing over it would make this test vanish into a silent "empty
# parameter set" skip -- taking the dispatcher's pending-agent path out of the
# suite unnoticed. The path still exists and still has to work, so it is
# exercised against a monkeypatched table instead of against the live one.
def test_an_agent_without_an_importer_is_reported_with_its_ticket(monkeypatch, tmp_path):
    agent = Agent.OPENCODE
    monkeypatch.setitem(PENDING_IMPORTERS, agent, "T2.3")
    monkeypatch.delitem(IMPORTERS, agent)
    path = write_jsonl(tmp_path, [user_record("u1", "hello")])

    with pytest.raises(TranscriptImportError, match=agent.value) as error:
        import_transcript(agent, path)

    assert "T2." in str(error.value)


def test_no_agent_is_left_without_an_importer() -> None:
    """The reason the test above needs a monkeypatched table: nothing is pending."""
    assert PENDING_IMPORTERS == {}
    assert set(IMPORTERS) == set(Agent)


def test_an_agent_the_study_does_not_cover_is_rejected(tmp_path):
    path = write_jsonl(tmp_path, [user_record("u1", "hello")])

    with pytest.raises(TranscriptImportError, match="cursor"):
        import_transcript("cursor", path)
