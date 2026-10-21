"""The OpenCode importer: a session of message records read into a `Session`."""

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import pytest

from idc.importers import (
    NOTE_DROPPED_BLOCK,
    NOTE_SKIPPED_RECORD,
    NOTE_UNKNOWN_TOOL,
    TranscriptImportError,
    import_opencode,
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

SESSION_ID = "ses_6kq4w2m0x9t1"

CREATED = 1789120800000
"""`2026-09-11T10:00:00Z` as OpenCode records an instant: epoch milliseconds."""


def native(name: str) -> Path:
    return NATIVE.path("opencode", name)


def write_jsonl(tmp_path: Path, records: list[dict], name: str = "session.jsonl") -> Path:
    path = tmp_path / name
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    return path


def text_part(text, part_id: str = "prt_x", **extra) -> dict:
    return {"id": part_id, "type": "text", "text": text, **extra}


def tool_part(call_id: str, tool: str, state: dict | None = None, **extra) -> dict:
    part = {"id": f"prt_{call_id}", "type": "tool", "callID": call_id, "tool": tool, **extra}
    if state is not None:
        part["state"] = state
    return part


def completed(output: str = "ok", **input_args) -> dict:
    return {"status": "completed", "input": dict(input_args), "output": output}


def user_record(record_id: str, parts, **extra) -> dict:
    return {
        "id": record_id,
        "sessionID": "ses_tmp",
        "role": "user",
        "time": {"created": CREATED},
        "parts": parts,
        **extra,
    }


def assistant_record(record_id: str, parts, **extra) -> dict:
    return {
        "id": record_id,
        "sessionID": "ses_tmp",
        "role": "assistant",
        "providerID": "anthropic",
        "modelID": "claude-sonnet-4-5",
        "time": {"created": CREATED, "completed": CREATED + 2000},
        "parts": parts,
        **extra,
    }


def compaction_record(record_id: str, **extra) -> dict:
    return {
        "id": record_id,
        "sessionID": "ses_tmp",
        "type": "compaction",
        "time": {"created": CREATED},
        **extra,
    }


PASTED_FILE = {
    "id": "prt_file",
    "type": "file",
    "mime": "image/png",
    "filename": "screenshot.png",
    "url": "data:image/png;base64,iVBORw0KGgo=",
}
"""How OpenCode records a file the user attached to their prompt.

The canonical schema has no field for it, so the part is dropped and noted —
but the user did take that turn, and T3.1 splits chunks on it.
"""


def messages_by_id(session: Session) -> dict[str, Message]:
    return {message.id: message for message in session.messages}


def notes(imported, kind: str) -> list[str]:
    return [note.detail for note in imported.notes_of(kind)]


# --- the golden file --------------------------------------------------------


def test_the_sample_session_imports_to_the_golden_canonical_document():
    imported = import_opencode(native("golden_session"))

    assert dumps(imported.session) + "\n" == NATIVE.expected_text("opencode", "golden_session")


def test_the_imported_session_is_a_valid_canonical_document():
    imported = import_opencode(native("golden_session"))

    revalidated = validate_document(Session, json.loads(dumps(imported.session)))

    assert revalidated == imported.session


def test_the_import_is_deterministic():
    first = import_opencode(native("golden_session"))
    second = import_opencode(native("golden_session"))

    assert dumps(first.session) == dumps(second.session)
    assert first.notes == second.notes


def test_the_golden_session_notes_every_part_the_schema_cannot_hold():
    imported = import_opencode(native("golden_session"))

    assert notes(imported, NOTE_DROPPED_BLOCK) == [
        "file",
        "reasoning",
        "step-finish",
        "step-start",
    ]
    assert imported.unknown_tools == ()


# --- session metadata -------------------------------------------------------


def test_the_session_takes_its_identifier_and_agent_from_the_transcript():
    session = import_opencode(native("golden_session")).session

    assert session.id == SESSION_ID
    assert session.agent is Agent.OPENCODE


def test_the_model_is_taken_from_the_first_assistant_record():
    assert import_opencode(native("golden_session")).session.model == "claude-sonnet-4-5"


def test_created_at_is_the_earliest_timestamp_of_the_transcript():
    session = import_opencode(native("golden_session")).session

    assert session.created_at == datetime(2026, 9, 11, 10, 0, 0, tzinfo=UTC)
    assert session.created_at == min(
        message.timestamp for message in session.messages if message.timestamp is not None
    )


def test_an_epoch_millisecond_instant_keeps_its_milliseconds():
    session = import_opencode(native("truncated_front")).session

    assert session.created_at == datetime(2026, 9, 11, 11, 30, 0, 123_000, tzinfo=UTC)


def test_a_transcript_without_a_session_identifier_is_rejected(tmp_path):
    record = user_record("msg_1", [text_part("hello")])
    del record["sessionID"]
    path = write_jsonl(tmp_path, [record])

    with pytest.raises(TranscriptImportError, match="session") as error:
        import_opencode(path)

    assert str(path) in str(error.value)


def test_a_transcript_without_any_timestamp_is_rejected(tmp_path):
    record = user_record("msg_1", [text_part("hello")])
    del record["time"]
    path = write_jsonl(tmp_path, [record])

    with pytest.raises(TranscriptImportError, match="timestamp"):
        import_opencode(path)


def test_a_transcript_without_a_single_message_is_rejected(tmp_path):
    path = write_jsonl(tmp_path, [{"id": "x", "sessionID": "ses_tmp", "role": "bookkeeping"}])

    with pytest.raises(TranscriptImportError, match="no message"):
        import_opencode(path)


def test_a_second_session_identifier_is_noted_rather_than_taken(tmp_path):
    path = write_jsonl(
        tmp_path,
        [
            user_record("msg_1", [text_part("hello")]),
            user_record("msg_2", [text_part("again")], sessionID="ses_other"),
        ],
    )

    imported = import_opencode(path)

    assert imported.session.id == "ses_tmp"
    assert notes(imported, NOTE_DROPPED_BLOCK) == ["<sessionID: a second value>"]


def test_a_record_time_that_is_not_an_object_is_rejected(tmp_path):
    path = write_jsonl(tmp_path, [user_record("msg_1", [text_part("hi")], time=CREATED)])

    with pytest.raises(TranscriptImportError, match="line 1"):
        import_opencode(path)


def test_a_created_instant_that_is_not_a_number_is_rejected(tmp_path):
    record = user_record("msg_1", [text_part("hi")])
    record["time"] = {"created": "2026-09-11T10:00:00Z"}
    path = write_jsonl(tmp_path, [record])

    with pytest.raises(TranscriptImportError, match="timestamp"):
        import_opencode(path)


def test_a_record_without_a_time_keeps_the_message_untimed(tmp_path):
    untimed = user_record("msg_2", [text_part("later")])
    del untimed["time"]
    path = write_jsonl(tmp_path, [user_record("msg_1", [text_part("hello")]), untimed])

    session = import_opencode(path).session

    assert messages_by_id(session)["msg_2"].timestamp is None


# --- messages ---------------------------------------------------------------


def test_the_messages_keep_the_identifiers_the_transcript_gave_them():
    session = import_opencode(native("golden_session")).session

    assert [message.id for message in session.messages] == [
        "msg_01u",
        "msg_02a",
        "msg_02a#3",
        "msg_03a",
        "msg_03a#1",
        "msg_03a#2",
        "msg_04u",
        "msg_05a",
        "msg_05a#0",
        "msg_06a",
    ]


def test_a_tool_message_is_named_after_the_part_it_comes_from():
    session = import_opencode(native("golden_session")).session
    messages = messages_by_id(session)

    assert messages["msg_03a#1"].tool_result.call_id == "call_02"
    assert messages["msg_03a#2"].tool_result.call_id == "call_03"


def test_the_assistant_message_keeps_the_record_identifier_unsuffixed():
    session = import_opencode(native("golden_session")).session

    assert messages_by_id(session)["msg_02a"].role is Role.ASSISTANT


def test_a_record_without_an_identifier_falls_back_to_its_line(tmp_path):
    record = user_record("msg_1", [text_part("hello")])
    del record["id"]
    path = write_jsonl(tmp_path, [record])

    imported = import_opencode(path)

    assert [message.id for message in imported.session.messages] == ["line-1"]
    assert notes(imported, NOTE_DROPPED_BLOCK) == []


def test_an_unusable_identifier_takes_the_same_fallback_and_is_noted(tmp_path):
    path = write_jsonl(tmp_path, [user_record(7, [text_part("hello")])])

    imported = import_opencode(path)

    assert [message.id for message in imported.session.messages] == ["line-1"]
    assert notes(imported, NOTE_DROPPED_BLOCK) == ["<id: number>"]


def test_several_text_parts_are_joined_by_a_blank_line(tmp_path):
    path = write_jsonl(
        tmp_path,
        [user_record("msg_1", [text_part("first", "prt_1"), text_part("second", "prt_2")])],
    )

    session = import_opencode(path).session

    assert session.messages[0].content == "first\n\nsecond"


def test_a_user_turn_survives_a_part_the_schema_cannot_hold(tmp_path):
    # The hard-won rule: a pasted screenshot must not make the user's turn
    # vanish, because T3.1 splits chunks on user turns and losing one
    # fabricates the chain structure.
    path = write_jsonl(tmp_path, [user_record("msg_1", [PASTED_FILE])])

    imported = import_opencode(path)

    assert [(m.id, m.role, m.content) for m in imported.session.messages] == [
        ("msg_1", Role.USER, "")
    ]
    assert notes(imported, NOTE_DROPPED_BLOCK) == ["file"]


def test_a_user_record_without_any_part_still_imports_as_the_turn_it_was(tmp_path):
    record = user_record("msg_1", [])
    path = write_jsonl(tmp_path, [record])

    imported = import_opencode(path)

    assert [(m.id, m.role, m.content) for m in imported.session.messages] == [
        ("msg_1", Role.USER, "")
    ]
    assert notes(imported, NOTE_DROPPED_BLOCK) == []


def test_a_parts_field_that_is_not_a_list_is_noted_and_the_turn_is_kept(tmp_path):
    path = write_jsonl(tmp_path, [user_record("msg_1", {"type": "text", "text": "hi"})])

    imported = import_opencode(path)

    assert [m.id for m in imported.session.messages] == ["msg_1"]
    assert notes(imported, NOTE_DROPPED_BLOCK) == ["<parts: object>"]


def test_a_part_entry_that_is_not_an_object_is_noted_by_its_shape(tmp_path):
    path = write_jsonl(tmp_path, [user_record("msg_1", ["hello"])])

    imported = import_opencode(path)

    assert notes(imported, NOTE_DROPPED_BLOCK) == ["<entry: string>"]


def test_a_text_that_is_not_a_string_is_dropped_rather_than_coerced(tmp_path):
    path = write_jsonl(tmp_path, [user_record("msg_1", [text_part({"a": 1})])])

    imported = import_opencode(path)

    assert imported.session.messages[0].content == ""
    assert notes(imported, NOTE_DROPPED_BLOCK) == ["text/<text: object>"]


def test_a_part_without_a_type_is_noted_as_such(tmp_path):
    path = write_jsonl(tmp_path, [user_record("msg_1", [{"id": "prt_1", "text": "hi"}])])

    imported = import_opencode(path)

    assert notes(imported, NOTE_DROPPED_BLOCK) == ["<no type>"]


def test_a_part_whose_type_is_null_is_a_different_note_from_a_missing_one(tmp_path):
    path = write_jsonl(tmp_path, [user_record("msg_1", [{"id": "prt_1", "type": None}])])

    imported = import_opencode(path)

    assert notes(imported, NOTE_DROPPED_BLOCK) == ["<null type>"]


# --- injected text ----------------------------------------------------------


def test_a_synthetic_user_record_is_marked_as_injected():
    session = import_opencode(native("golden_session")).session

    assert messages_by_id(session)["msg_04u"].is_system_injected is True


def test_a_user_record_the_user_typed_is_not_marked_as_injected():
    session = import_opencode(native("golden_session")).session

    assert messages_by_id(session)["msg_01u"].is_system_injected is False


def test_a_record_mixing_typed_and_synthetic_text_is_the_users_turn(tmp_path):
    path = write_jsonl(
        tmp_path,
        [
            user_record(
                "msg_1",
                [text_part("fix it", "prt_1"), text_part("reminder", "prt_2", synthetic=True)],
            )
        ],
    )

    session = import_opencode(path).session

    assert session.messages[0].is_system_injected is False


def test_an_assistant_record_is_never_marked_as_injected(tmp_path):
    path = write_jsonl(tmp_path, [assistant_record("msg_1", [text_part("done", synthetic=True)])])

    session = import_opencode(path).session

    assert session.messages[0].is_system_injected is False


def test_an_unreadable_synthetic_flag_keeps_the_turn_and_is_noted_by_its_shape(tmp_path):
    """`synthetic: "yes"` fails safe *and* is noted, as every unreadable field is."""
    path = write_jsonl(
        tmp_path, [user_record("msg_1", [text_part("fix it", "prt_1", synthetic="yes")])]
    )

    imported = import_opencode(path)

    assert imported.session.messages[0].is_system_injected is False
    assert imported.session.messages[0].content == "fix it"
    assert notes(imported, NOTE_DROPPED_BLOCK) == ["text/<synthetic: string>"]


def test_a_synthetic_flag_that_is_false_or_null_is_not_noted(tmp_path):
    """Absent, `null` and `false` all carried nothing to lose — no note."""
    path = write_jsonl(
        tmp_path,
        [
            user_record(
                "msg_1",
                [
                    text_part("a", "prt_1", synthetic=False),
                    text_part("b", "prt_2", synthetic=None),
                    text_part("c", "prt_3"),
                ],
            )
        ],
    )

    imported = import_opencode(path)

    assert imported.session.messages[0].is_system_injected is False
    assert notes(imported, NOTE_DROPPED_BLOCK) == []


# --- tool calls and their results -------------------------------------------


def test_a_tool_part_yields_the_call_and_the_result_it_carries():
    session = import_opencode(native("golden_session")).session
    messages = messages_by_id(session)

    call = messages["msg_02a"].tool_calls[0]
    assert (call.id, call.name, call.category) == ("call_01", "read", ToolCategory.READ)
    assert call.args == {"filePath": "/repo/src/billing/export.py"}
    assert messages["msg_02a#3"].tool_result.output.startswith("     1\timport datetime")
    assert messages["msg_02a#3"].tool_result.is_error is False


def test_an_error_state_is_imported_as_a_failed_result():
    session = import_opencode(native("golden_session")).session

    result = messages_by_id(session)["msg_03a#2"].tool_result
    assert result.is_error is True
    assert result.output.startswith("1 failed, 3 passed")


def test_the_exit_code_stays_unset_because_opencode_records_none():
    session = import_opencode(native("golden_session")).session

    assert all(
        message.tool_result.exit_code is None
        for message in session.messages
        if message.tool_result is not None
    )


def test_a_call_whose_result_never_arrived_is_imported_without_one():
    imported = import_opencode(native("interrupted_turn"))
    session = imported.session

    calls = [call for message in session.messages for call in message.tool_calls]
    assert [call.id for call in calls] == ["call_31"]
    assert calls[0].args["command"] == "uv run pytest -q"
    assert [message.tool_result for message in session.messages if message.role is Role.TOOL] == []
    assert notes(imported, NOTE_DROPPED_BLOCK) == []


def test_a_pending_call_is_imported_without_a_result(tmp_path):
    path = write_jsonl(
        tmp_path,
        [assistant_record("msg_1", [tool_part("call_1", "bash", {"status": "pending"})])],
    )

    imported = import_opencode(path)

    assert [message.role for message in imported.session.messages] == [Role.ASSISTANT]
    assert notes(imported, NOTE_DROPPED_BLOCK) == []


def test_a_state_whose_status_is_unknown_is_noted_rather_than_read(tmp_path):
    path = write_jsonl(
        tmp_path,
        [
            assistant_record(
                "msg_1",
                [tool_part("call_1", "bash", {"status": "queued", "output": "ignored"})],
            )
        ],
    )

    imported = import_opencode(path)

    assert [message.role for message in imported.session.messages] == [Role.ASSISTANT]
    assert notes(imported, NOTE_DROPPED_BLOCK) == ["tool/bash/<status: queued>"]


def test_a_state_without_a_status_is_noted_as_such(tmp_path):
    path = write_jsonl(
        tmp_path, [assistant_record("msg_1", [tool_part("call_1", "bash", {"output": "ignored"})])]
    )

    imported = import_opencode(path)

    assert notes(imported, NOTE_DROPPED_BLOCK) == ["tool/bash/<no status>"]


def test_a_tool_part_without_a_state_yields_the_call_alone(tmp_path):
    path = write_jsonl(tmp_path, [assistant_record("msg_1", [tool_part("call_1", "read")])])

    imported = import_opencode(path)

    assert imported.session.messages[0].tool_calls[0].args == {}
    assert notes(imported, NOTE_DROPPED_BLOCK) == []


def test_a_state_that_is_not_an_object_is_noted_by_its_shape(tmp_path):
    path = write_jsonl(
        tmp_path, [assistant_record("msg_1", [tool_part("call_1", "read", state=["completed"])])]
    )

    imported = import_opencode(path)

    assert notes(imported, NOTE_DROPPED_BLOCK) == ["tool/read/<state: array>"]


def test_an_input_that_is_not_an_object_is_noted_with_its_tool(tmp_path):
    path = write_jsonl(
        tmp_path,
        [
            assistant_record(
                "msg_1",
                [tool_part("call_1", "bash", {"status": "completed", "input": ["pytest"]})],
            )
        ],
    )

    imported = import_opencode(path)

    assert imported.session.messages[0].tool_calls[0].args == {}
    assert notes(imported, NOTE_DROPPED_BLOCK) == ["tool/bash/<input: array>"]


def test_an_output_that_is_not_a_string_is_noted_rather_than_coerced(tmp_path):
    path = write_jsonl(
        tmp_path,
        [
            assistant_record(
                "msg_1",
                [tool_part("call_1", "read", {"status": "completed", "output": {"lines": 3}})],
            )
        ],
    )

    imported = import_opencode(path)

    results = [message.tool_result for message in imported.session.messages if message.tool_result]
    assert [result.output for result in results] == [""]
    assert notes(imported, NOTE_DROPPED_BLOCK) == ["tool/read/<output: object>"]


def test_an_error_that_is_not_a_string_is_noted_but_still_fails(tmp_path):
    path = write_jsonl(
        tmp_path,
        [
            assistant_record(
                "msg_1",
                [tool_part("call_1", "bash", {"status": "error", "error": {"code": 1}})],
            )
        ],
    )

    imported = import_opencode(path)

    results = [message.tool_result for message in imported.session.messages if message.tool_result]
    assert [(result.output, result.is_error) for result in results] == [("", True)]
    assert notes(imported, NOTE_DROPPED_BLOCK) == ["tool/bash/<error: object>"]


def test_a_tool_part_without_a_call_identifier_is_rejected(tmp_path):
    part = tool_part("call_1", "read", completed())
    del part["callID"]
    path = write_jsonl(tmp_path, [assistant_record("msg_1", [part])])

    with pytest.raises(TranscriptImportError, match="callID") as error:
        import_opencode(path)

    assert "line 1" in str(error.value)


def test_a_tool_part_without_a_tool_name_is_rejected(tmp_path):
    part = tool_part("call_1", "read", completed())
    del part["tool"]
    path = write_jsonl(tmp_path, [assistant_record("msg_1", [part])])

    with pytest.raises(TranscriptImportError, match="call_1"):
        import_opencode(path)


def test_a_repeated_call_identifier_is_reported_by_the_schema(tmp_path):
    path = write_jsonl(
        tmp_path,
        [
            assistant_record("msg_1", [tool_part("call_1", "read", completed())]),
            assistant_record("msg_2", [tool_part("call_1", "read", completed())]),
        ],
    )

    with pytest.raises(SchemaValidationError, match="tool-call id"):
        import_opencode(path)


# --- unknown tools (the parent criterion) -----------------------------------


def test_an_undeclared_tool_resolves_to_other_and_is_noted():
    imported = import_opencode(native("unknown_tool"))

    assert imported.unknown_tools == ("imaginary_tool",)
    calls = {
        call.name: call for message in imported.session.messages for call in message.tool_calls
    }
    assert calls["imaginary_tool"].category is ToolCategory.OTHER


def test_an_undeclared_tool_is_logged_once_with_its_agent(tmp_path, caplog):
    """Three calls to one undeclared tool are **one** warning, not three.

    The native `unknown_tool` fixture calls `imaginary_tool` exactly once, so it
    cannot tell per-name logging from per-call logging. This record calls it
    three times across two records — the shape that fails under per-call
    logging — next to a declared tool that must stay out of the warnings.
    """
    path = write_jsonl(
        tmp_path,
        [
            assistant_record(
                "msg_1",
                [
                    tool_part("call_1", "imaginary_tool", completed()),
                    tool_part("call_2", "imaginary_tool", completed()),
                    tool_part("call_3", "task", completed()),
                ],
            ),
            assistant_record("msg_2", [tool_part("call_4", "imaginary_tool", completed())]),
        ],
    )

    with caplog.at_level(logging.WARNING, logger="idc.importers"):
        imported = import_opencode(path)

    warnings = [
        record.getMessage() for record in caplog.records if record.levelno >= logging.WARNING
    ]

    assert sum("'imaginary_tool'" in message for message in warnings) == 1
    assert all("opencode" in message for message in warnings)
    assert not any("'task'" in message for message in warnings)
    assert imported.unknown_tools == ("imaginary_tool",)
    assert [note.count for note in imported.notes_of(NOTE_UNKNOWN_TOOL)] == [3]


def test_a_declared_tool_is_not_logged(caplog):
    with caplog.at_level(logging.WARNING, logger="idc.importers"):
        import_opencode(native("golden_session"))

    assert [record for record in caplog.records if record.levelno >= logging.WARNING] == []


def test_the_delegating_tool_the_transcript_shows_is_declared():
    registry = tool_registry()

    assert registry.knows(Agent.OPENCODE, "task")
    assert registry.is_subagent_tool(Agent.OPENCODE, "task")
    assert notes(import_opencode(native("unknown_tool")), NOTE_UNKNOWN_TOOL) == ["imaginary_tool"]


# --- the native compaction summary ------------------------------------------


def test_a_compaction_record_marks_the_session_as_natively_compacted():
    imported = import_opencode(native("native_compaction"))

    assert imported.session.source is SessionSource.NATIVE_COMPACTION


def test_the_summary_becomes_an_injected_system_message():
    session = import_opencode(native("native_compaction")).session
    summary = messages_by_id(session)["msg_22c"]

    assert summary.role is Role.SYSTEM
    assert summary.is_system_injected is True
    assert summary.content.startswith("The exporter stamps aware UTC")
    assert [message.id for message in session.messages] == ["msg_21u", "msg_22c", "msg_23u"]


def test_the_kept_message_references_of_a_compaction_are_not_noted():
    imported = import_opencode(native("native_compaction"))

    assert imported.notes == ()


def test_a_session_without_a_compaction_record_stays_raw():
    assert import_opencode(native("golden_session")).session.source is SessionSource.RAW


def test_a_recent_entry_that_is_not_a_reference_is_noted(tmp_path):
    path = write_jsonl(
        tmp_path,
        [compaction_record("msg_1", summary="done", recent=[{"id": "msg_0"}])],
    )

    imported = import_opencode(path)

    assert notes(imported, NOTE_DROPPED_BLOCK) == ["compaction/<recent entry: object>"]


def test_a_recent_field_that_is_not_a_list_is_noted(tmp_path):
    path = write_jsonl(tmp_path, [compaction_record("msg_1", summary="done", recent=3)])

    imported = import_opencode(path)

    assert notes(imported, NOTE_DROPPED_BLOCK) == ["compaction/<recent: number>"]


def test_a_summary_that_cannot_be_read_is_noted_and_the_provenance_still_flips(tmp_path):
    path = write_jsonl(tmp_path, [compaction_record("msg_1", summary={"text": "done"})])

    imported = import_opencode(path)

    assert imported.session.source is SessionSource.NATIVE_COMPACTION
    assert imported.session.messages[0].content == ""
    assert notes(imported, NOTE_DROPPED_BLOCK) == ["compaction/<summary: object>"]


def test_a_summary_carried_in_text_parts_is_read_from_them(tmp_path):
    path = write_jsonl(
        tmp_path,
        [compaction_record("msg_1", parts=[text_part("the summary"), PASTED_FILE])],
    )

    imported = import_opencode(path)

    assert imported.session.messages[0].content == "the summary"
    assert notes(imported, NOTE_DROPPED_BLOCK) == ["file"]


# --- records that carry no message ------------------------------------------


def test_a_record_whose_role_is_unknown_is_skipped_and_noted(tmp_path):
    path = write_jsonl(
        tmp_path,
        [
            user_record("msg_1", [text_part("hello")]),
            {"id": "msg_2", "sessionID": "ses_tmp", "role": "system"},
        ],
    )

    imported = import_opencode(path)

    assert [message.id for message in imported.session.messages] == ["msg_1"]
    assert notes(imported, NOTE_SKIPPED_RECORD) == ["system"]


def test_a_record_naming_neither_a_role_nor_a_type_is_noted_as_such(tmp_path):
    path = write_jsonl(
        tmp_path,
        [user_record("msg_1", [text_part("hello")]), {"id": "msg_2", "sessionID": "ses_tmp"}],
    )

    imported = import_opencode(path)

    assert notes(imported, NOTE_SKIPPED_RECORD) == ["<no kind>"]


def test_a_null_role_is_a_different_note_from_a_missing_one(tmp_path):
    path = write_jsonl(
        tmp_path,
        [
            user_record("msg_1", [text_part("hello")]),
            {"id": "msg_2", "sessionID": "ses_tmp", "role": None},
        ],
    )

    imported = import_opencode(path)

    assert notes(imported, NOTE_SKIPPED_RECORD) == ["<null role>"]


def test_a_null_type_is_noted_by_its_own_spelling(tmp_path):
    path = write_jsonl(
        tmp_path,
        [
            user_record("msg_1", [text_part("hello")]),
            {"id": "msg_2", "sessionID": "ses_tmp", "type": None, "role": "user"},
        ],
    )

    imported = import_opencode(path)

    assert notes(imported, NOTE_SKIPPED_RECORD) == ["<null type>"]
    assert [message.id for message in imported.session.messages] == ["msg_1"]


def test_the_same_skipped_kind_is_counted_once(tmp_path):
    path = write_jsonl(
        tmp_path,
        [
            user_record("msg_1", [text_part("hello")]),
            {"id": "msg_2", "sessionID": "ses_tmp", "role": "system"},
            {"id": "msg_3", "sessionID": "ses_tmp", "role": "system"},
        ],
    )

    imported = import_opencode(path)

    assert [(note.detail, note.count) for note in imported.notes_of(NOTE_SKIPPED_RECORD)] == [
        ("system", 2)
    ]


# --- nothing is repaired ----------------------------------------------------


def test_a_front_truncated_transcript_imports_as_the_session_it_is():
    imported = import_opencode(native("truncated_front"))
    session = imported.session

    assert [message.role for message in session.messages] == [
        Role.ASSISTANT,
        Role.TOOL,
        Role.USER,
    ]
    assert session.messages[1].tool_result.call_id == "call_41"
    assert validate_document(Session, json.loads(dumps(session))) == session


# --- the shared entry point -------------------------------------------------


def test_the_dispatcher_reaches_the_opencode_importer():
    path = native("golden_session")

    assert import_transcript(Agent.OPENCODE, path) == import_opencode(path)


def test_the_dispatcher_accepts_the_agents_wire_string():
    path = native("golden_session")

    assert import_transcript("opencode", path).session.agent is Agent.OPENCODE
