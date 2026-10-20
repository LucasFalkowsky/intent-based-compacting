"""The Gemini CLI importer: a session JSONL read into a canonical `Session`."""

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
    import_gemini_cli,
    import_transcript,
    tool_registry,
)
from idc.schema import (
    Agent,
    Message,
    Role,
    Session,
    SessionSource,
    ToolCategory,
    dumps,
    validate_document,
)
from native_transcripts import NATIVE

SESSION_ID = "3f8c1d2e-0000-4000-8000-000000000011"
PROJECT_HASH = "a1b2c3d4e5f6a7b8"


def native(name: str) -> Path:
    return NATIVE.path("gemini-cli", name)


def write_jsonl(tmp_path: Path, records: list[dict], name: str = "session-tmp.jsonl") -> Path:
    path = tmp_path / name
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    return path


def metadata_record(**extra) -> dict:
    """The first record of a session file: its metadata, no message."""
    return {
        "sessionId": "sess-tmp",
        "projectHash": PROJECT_HASH,
        "startTime": "2026-09-20T12:00:00.000Z",
        "lastUpdated": "2026-09-20T12:00:30.000Z",
        "kind": "main",
        **extra,
    }


def user_record(message_id: str, content, **extra) -> dict:
    return {
        "id": message_id,
        "timestamp": "2026-09-20T12:00:02.000Z",
        "type": "user",
        "content": content,
        **extra,
    }


def gemini_record(message_id: str, content, **extra) -> dict:
    return {
        "id": message_id,
        "timestamp": "2026-09-20T12:00:08.000Z",
        "type": "gemini",
        "content": content,
        "model": "gemini-2.5-pro",
        **extra,
    }


def tool_call(call_id: str, name: str, args: dict, **extra) -> dict:
    """One entry of a `gemini` record's `toolCalls` list."""
    return {
        "id": call_id,
        "name": name,
        "args": dict(args),
        "result": None,
        "status": "executing",
        "timestamp": "2026-09-20T12:00:09.000Z",
        "displayName": name,
        "description": "",
        "renderOutputAsMarkdown": False,
        **extra,
    }


def function_response(call_id: str, name: str, response: dict) -> dict:
    """The part Gemini CLI records as a tool call's result."""
    return {"functionResponse": {"id": call_id, "name": name, "response": response}}


def output(call_id: str, name: str, text: str) -> list[dict]:
    return [function_response(call_id, name, {"output": text})]


PASTED_IMAGE = {"inlineData": {"mimeType": "image/png", "data": "iVBORw0KGgo="}}
"""How Gemini CLI records a screenshot the user pasted into their prompt.

The canonical schema has no field for an image, so the part is dropped and
noted — but the user did take that turn, and T3.1 splits chunks on it.
"""


def messages_by_id(session: Session) -> dict[str, Message]:
    return {message.id: message for message in session.messages}


# --- the golden file --------------------------------------------------------


def test_the_sample_session_imports_to_the_golden_canonical_document():
    imported = import_gemini_cli(native("golden_session"))

    assert dumps(imported.session) + "\n" == NATIVE.expected_text("gemini-cli", "golden_session")


def test_the_imported_session_is_a_valid_canonical_document():
    imported = import_gemini_cli(native("golden_session"))

    revalidated = validate_document(Session, json.loads(dumps(imported.session)))

    assert revalidated == imported.session


def test_the_import_is_deterministic():
    first = import_gemini_cli(native("golden_session"))
    second = import_gemini_cli(native("golden_session"))

    assert dumps(first.session) == dumps(second.session)
    assert first.notes == second.notes


# --- session metadata -------------------------------------------------------


def test_the_session_takes_its_identifier_and_agent_from_the_transcript():
    session = import_gemini_cli(native("golden_session")).session

    assert session.id == SESSION_ID
    assert session.agent is Agent.GEMINI_CLI


def test_the_model_is_taken_from_the_first_model_message():
    assert import_gemini_cli(native("golden_session")).session.model == "gemini-2.5-pro"


def test_created_at_is_the_earliest_timestamp_of_the_transcript():
    session = import_gemini_cli(native("golden_session")).session

    assert session.created_at == datetime(2026, 9, 12, 8, 30, 0, 112000, tzinfo=UTC)
    assert session.created_at == min(
        message.timestamp for message in session.messages if message.timestamp is not None
    )


def test_a_naive_timestamp_is_read_as_utc(tmp_path):
    path = write_jsonl(
        tmp_path,
        [metadata_record(), user_record("u1", [{"text": "hi"}], timestamp="2026-09-20T12:00:02")],
    )

    session = import_gemini_cli(path).session

    assert session.created_at == datetime(2026, 9, 20, 12, 0, 2, tzinfo=UTC)


def test_the_metadata_record_carries_no_message_and_is_noted(tmp_path):
    path = write_jsonl(tmp_path, [metadata_record(), user_record("u1", [{"text": "hi"}])])

    imported = import_gemini_cli(path)

    assert [message.id for message in imported.session.messages] == ["u1"]
    assert [note.detail for note in imported.notes_of(NOTE_SKIPPED_RECORD)] == ["<metadata>"]


def test_a_transcript_without_a_session_identifier_is_rejected(tmp_path):
    path = write_jsonl(tmp_path, [user_record("u1", [{"text": "hi"}])])

    with pytest.raises(TranscriptImportError, match="session") as error:
        import_gemini_cli(path)

    assert str(path) in str(error.value)


def test_a_second_session_identifier_is_noted(tmp_path):
    path = write_jsonl(
        tmp_path,
        [
            metadata_record(),
            user_record("u1", [{"text": "hi"}]),
            metadata_record(sessionId="sess-other"),
        ],
    )

    imported = import_gemini_cli(path)

    assert imported.session.id == "sess-tmp"
    assert any("sessionId" in note.detail for note in imported.notes)


def test_the_session_identifier_may_arrive_in_a_metadata_update(tmp_path):
    path = write_jsonl(
        tmp_path,
        [{"$set": {"sessionId": "sess-set"}}, user_record("u1", [{"text": "hi"}])],
    )

    assert import_gemini_cli(path).session.id == "sess-set"


def test_a_transcript_without_a_message_is_rejected(tmp_path):
    path = write_jsonl(tmp_path, [metadata_record()])

    with pytest.raises(TranscriptImportError, match="no message record"):
        import_gemini_cli(path)


def test_a_transcript_without_a_timestamp_is_rejected(tmp_path):
    record = user_record("u1", [{"text": "hi"}])
    del record["timestamp"]
    path = write_jsonl(tmp_path, [metadata_record(), record])

    with pytest.raises(TranscriptImportError, match="no timestamp"):
        import_gemini_cli(path)


def test_an_unreadable_timestamp_is_rejected_by_its_line(tmp_path):
    path = write_jsonl(
        tmp_path,
        [metadata_record(), user_record("u1", [{"text": "hi"}], timestamp="yesterday")],
    )

    with pytest.raises(TranscriptImportError, match="line 2") as error:
        import_gemini_cli(path)

    assert "yesterday" in str(error.value)


# --- a single JSON document instead of one record per line ------------------


def test_a_legacy_single_json_document_imports_as_the_session_it_holds(tmp_path):
    path = tmp_path / "session-legacy.json"
    path.write_text(
        json.dumps(
            metadata_record(messages=[user_record("u1", [{"text": "hi"}])]),
            indent=2,
        ),
        encoding="utf-8",
    )

    imported = import_gemini_cli(path)

    assert imported.session.id == "sess-tmp"
    assert [message.content for message in imported.session.messages] == ["hi"]


def test_a_damaged_jsonl_line_is_fatal_and_names_its_line(tmp_path):
    path = tmp_path / "session-broken.jsonl"
    path.write_text(
        json.dumps(metadata_record()) + "\nnot json\n",
        encoding="utf-8",
    )

    with pytest.raises(TranscriptImportError, match="line 2"):
        import_gemini_cli(path)


# --- user records -----------------------------------------------------------


def test_a_user_prompt_becomes_a_user_message():
    message = messages_by_id(import_gemini_cli(native("golden_session")).session)["u01"]

    assert message.role is Role.USER
    assert message.content.startswith("The CSV importer drops every row")
    assert message.is_system_injected is False


def test_a_string_content_is_taken_as_the_message_text(tmp_path):
    path = write_jsonl(tmp_path, [metadata_record(), user_record("u1", "just text")])

    assert import_gemini_cli(path).session.messages[0].content == "just text"


def test_several_text_parts_are_joined_by_a_blank_line(tmp_path):
    path = write_jsonl(
        tmp_path,
        [metadata_record(), user_record("u1", [{"text": "first"}, {"text": "second"}])],
    )

    assert import_gemini_cli(path).session.messages[0].content == "first\n\nsecond"


def test_the_environment_context_is_marked_as_system_injected():
    message = messages_by_id(import_gemini_cli(native("golden_session")).session)["env01"]

    assert message.role is Role.USER
    assert message.is_system_injected is True
    assert message.content.startswith("<session_context>")


def test_a_user_turn_whose_parts_hold_no_text_still_imports_as_that_turn():
    # A pasted image is a turn the user took, and T3.1 splits chunks on exactly
    # that boundary: losing it would fabricate the chain structure.
    imported = import_gemini_cli(native("golden_session"))

    message = messages_by_id(imported.session)["u04"]
    assert message.role is Role.USER
    assert message.content == ""
    assert [note.detail for note in imported.notes_of(NOTE_DROPPED_BLOCK)] == ["inlineData"]


def test_a_pasted_image_alone_still_yields_the_user_turn(tmp_path):
    path = write_jsonl(tmp_path, [metadata_record(), user_record("u1", [PASTED_IMAGE])])

    imported = import_gemini_cli(path)

    assert [(m.id, m.role) for m in imported.session.messages] == [("u1", Role.USER)]
    assert [note.detail for note in imported.notes_of(NOTE_DROPPED_BLOCK)] == ["inlineData"]


def test_a_user_record_that_only_reports_a_result_yields_no_user_turn():
    # The opposite guard: an empty user turn in front of every tool result
    # would be an invented intent boundary.
    session = import_gemini_cli(native("golden_session")).session

    message = messages_by_id(session)["u03"]
    assert message.role is Role.TOOL
    assert message.tool_result is not None
    assert message.tool_result.call_id == "call-04"


def test_a_user_record_with_a_result_and_text_yields_both(tmp_path):
    path = write_jsonl(
        tmp_path,
        [
            metadata_record(),
            user_record(
                "u1",
                [
                    function_response("c1", "read_file", {"output": "file body"}),
                    {"text": "and now the next thing"},
                ],
            ),
        ],
    )

    session = import_gemini_cli(path).session

    assert [(m.id, m.role) for m in session.messages] == [
        ("u1#0", Role.TOOL),
        ("u1#1", Role.USER),
    ]


def test_a_content_that_is_neither_text_nor_parts_yields_no_message(tmp_path):
    path = write_jsonl(tmp_path, [metadata_record(), user_record("u1", {"text": "object"})])

    with pytest.raises(TranscriptImportError, match="no message record"):
        import_gemini_cli(path)


def test_an_unreadable_content_is_noted_by_its_shape(tmp_path):
    path = write_jsonl(
        tmp_path,
        [metadata_record(), user_record("u1", {"text": "object"}), user_record("u2", "text")],
    )

    imported = import_gemini_cli(path)

    assert [note.detail for note in imported.notes_of(NOTE_DROPPED_BLOCK)] == [
        "<content: object>"
    ]


def test_a_part_that_is_not_an_object_is_noted_by_its_shape(tmp_path):
    path = write_jsonl(tmp_path, [metadata_record(), user_record("u1", ["bare string"])])

    imported = import_gemini_cli(path)

    assert [note.detail for note in imported.notes_of(NOTE_DROPPED_BLOCK)] == ["<entry: string>"]


def test_an_absent_content_is_not_noted(tmp_path):
    record = user_record("u1", None)
    path = write_jsonl(tmp_path, [metadata_record(), record, user_record("u2", "text")])

    imported = import_gemini_cli(path)

    assert imported.notes_of(NOTE_DROPPED_BLOCK) == ()


# --- model records ----------------------------------------------------------


def test_a_model_record_becomes_an_assistant_message_with_its_calls():
    session = import_gemini_cli(native("golden_session")).session

    message = messages_by_id(session)["g01#0"]
    assert message.role is Role.ASSISTANT
    assert message.content == "Reading the importer."
    assert [(call.id, call.name, call.category) for call in message.tool_calls] == [
        ("call-01", "read_file", ToolCategory.READ)
    ]


def test_the_arguments_are_kept_exactly_as_the_agent_sent_them():
    session = import_gemini_cli(native("golden_session")).session

    call = messages_by_id(session)["g02#0"].tool_calls[0]
    assert call.args == {
        "file_path": "/repo/src/imports/csv.py",
        "old_string": "    if not amount:\n        continue",
        "new_string": '    amount = amount or "0"',
    }


def test_each_result_of_one_record_becomes_a_tool_message_of_its_own():
    session = import_gemini_cli(native("golden_session")).session

    assert [m.id for m in session.messages if m.id.startswith("g02")] == [
        "g02#0",
        "g02#1",
        "g02#2",
    ]
    assert messages_by_id(session)["g02#1"].tool_result.call_id == "call-02"
    assert messages_by_id(session)["g02#2"].tool_result.call_id == "call-03"


def test_a_result_takes_the_timestamp_the_call_records(tmp_path):
    path = write_jsonl(
        tmp_path,
        [
            metadata_record(),
            gemini_record(
                "g1",
                "running",
                toolCalls=[
                    tool_call(
                        "c1",
                        "read_file",
                        {"file_path": "/repo/a.py"},
                        result=output("c1", "read_file", "body"),
                        status="success",
                        timestamp="2026-09-20T12:00:11.500Z",
                    )
                ],
            ),
        ],
    )

    session = import_gemini_cli(path).session

    assert messages_by_id(session)["g1#1"].timestamp == datetime(
        2026, 9, 20, 12, 0, 11, 500000, tzinfo=UTC
    )


def test_a_thought_part_and_the_thoughts_field_are_dropped_and_noted():
    imported = import_gemini_cli(native("golden_session"))

    assert [note.detail for note in imported.notes_of(NOTE_DROPPED_BLOCK)] == [
        "inlineData",
        "thought",
        "thoughts",
    ]
    assert messages_by_id(imported.session)["g03"].content == (
        "Checking which tests touch the importer."
    )


def test_a_function_call_part_is_read_as_a_tool_call():
    session = import_gemini_cli(native("golden_session")).session

    call = messages_by_id(session)["g03"].tool_calls[0]
    assert (call.id, call.name, call.args) == ("call-04", "glob", {"pattern": "tests/**/*csv*.py"})


def test_a_function_call_part_repeating_a_recorded_call_is_not_imported_twice(tmp_path):
    # `updateMessagesFromHistory` rewrites a model turn's content from the live
    # history while keeping its `toolCalls`, so the same call is in the record
    # twice; two canonical calls with one identifier would not validate.
    path = write_jsonl(
        tmp_path,
        [
            metadata_record(),
            gemini_record(
                "g1",
                [
                    {"text": "reading"},
                    {"functionCall": {"id": "c1", "name": "read_file", "args": {}}},
                ],
                toolCalls=[tool_call("c1", "read_file", {"file_path": "/repo/a.py"})],
            ),
        ],
    )

    session = import_gemini_cli(path).session

    calls = messages_by_id(session)["g1"].tool_calls
    assert [call.id for call in calls] == ["c1"]
    assert calls[0].args == {"file_path": "/repo/a.py"}


def test_a_call_without_an_identifier_is_rejected_by_its_line(tmp_path):
    call = tool_call("c1", "read_file", {})
    del call["id"]
    path = write_jsonl(tmp_path, [metadata_record(), gemini_record("g1", "x", toolCalls=[call])])

    with pytest.raises(TranscriptImportError, match="line 2") as error:
        import_gemini_cli(path)

    assert "no id" in str(error.value)


def test_a_call_without_a_tool_name_is_rejected_by_its_line(tmp_path):
    call = tool_call("c1", "read_file", {})
    del call["name"]
    path = write_jsonl(tmp_path, [metadata_record(), gemini_record("g1", "x", toolCalls=[call])])

    with pytest.raises(TranscriptImportError, match="line 2") as error:
        import_gemini_cli(path)

    assert "c1" in str(error.value)


def test_arguments_that_are_not_an_object_are_dropped_and_noted(tmp_path):
    path = write_jsonl(
        tmp_path,
        [
            metadata_record(),
            gemini_record(
                "g1", "x", toolCalls=[tool_call("c1", "run_shell_command", {}, args=["ls"])]
            ),
        ],
    )

    imported = import_gemini_cli(path)

    assert messages_by_id(imported.session)["g1"].tool_calls[0].args == {}
    assert [note.detail for note in imported.notes_of(NOTE_DROPPED_BLOCK)] == [
        "toolCalls/run_shell_command/<args: array>"
    ]


# --- an `info` record -------------------------------------------------------


def test_an_info_record_becomes_an_injected_system_message():
    message = messages_by_id(import_gemini_cli(native("golden_session")).session)["info01"]

    assert message.role is Role.SYSTEM
    assert message.is_system_injected is True
    assert message.content == "Binary content received. Proceeding with analysis."


def test_a_record_type_this_importer_does_not_know_is_skipped_and_noted(tmp_path):
    path = write_jsonl(
        tmp_path,
        [metadata_record(), user_record("u1", "hi"), {"id": "x1", "type": "whatever"}],
    )

    imported = import_gemini_cli(path)

    assert [m.id for m in imported.session.messages] == ["u1"]
    assert [note.detail for note in imported.notes_of(NOTE_SKIPPED_RECORD)] == [
        "<metadata>",
        "whatever",
    ]


def test_a_record_without_a_usable_identifier_is_skipped_and_noted(tmp_path):
    # Gemini CLI keys its messages by `id`; a record without a usable one is
    # dropped by the agent's own loader, so importing it would add a turn the
    # agent itself no longer has.
    path = write_jsonl(
        tmp_path,
        [metadata_record(), user_record("u1", "hi"), user_record("", "lost"), {"type": "user"}],
    )

    imported = import_gemini_cli(path)

    assert [m.id for m in imported.session.messages] == ["u1"]
    assert [note.detail for note in imported.notes_of(NOTE_DROPPED_BLOCK)] == [
        "<id: empty string>"
    ]
    assert [(note.detail, note.count) for note in imported.notes_of(NOTE_SKIPPED_RECORD)] == [
        ("<metadata>", 1),
        ("user", 2),
    ]


# --- results, errors and exit codes -----------------------------------------


def test_a_successful_result_is_not_an_error():
    session = import_gemini_cli(native("golden_session")).session

    result = messages_by_id(session)["g01#1"].tool_result
    assert result.is_error is False
    assert result.exit_code is None
    assert result.output.startswith("    11\tfor row in reader:")


def test_a_non_zero_exit_status_is_read_out_of_a_shell_result():
    # Gemini CLI appends `Exit Code: <n>` to a shell result only when the
    # command failed, and the call itself still counts as a success.
    session = import_gemini_cli(native("golden_session")).session

    result = messages_by_id(session)["g02#2"].tool_result
    assert result.exit_code == 1
    assert result.is_error is False


def test_a_shell_result_without_an_exit_status_reports_none(tmp_path):
    path = write_jsonl(
        tmp_path,
        [
            metadata_record(),
            gemini_record(
                "g1",
                "x",
                toolCalls=[
                    tool_call(
                        "c1",
                        "run_shell_command",
                        {"command": "uv run pytest -q"},
                        result=output("c1", "run_shell_command", "Output: 4 passed in 0.31s"),
                        status="success",
                    )
                ],
            ),
        ],
    )

    result = messages_by_id(import_gemini_cli(path).session)["g1#1"].tool_result
    assert result.exit_code is None
    assert result.is_error is False


def test_an_exit_status_in_the_middle_of_the_output_is_not_read_as_one(tmp_path):
    # The exit status is one of the trailing fields of the shell tool's own
    # format; a command that prints the same line proves nothing about its own
    # exit status.
    path = write_jsonl(
        tmp_path,
        [
            metadata_record(),
            gemini_record(
                "g1",
                "x",
                toolCalls=[
                    tool_call(
                        "c1",
                        "run_shell_command",
                        {"command": "cat log.txt"},
                        result=output(
                            "c1", "run_shell_command", "Output: Exit Code: 3\nlast line of the log"
                        ),
                        status="success",
                    )
                ],
            ),
        ],
    )

    assert messages_by_id(import_gemini_cli(path).session)["g1#1"].tool_result.exit_code is None


def test_a_call_the_agent_reports_as_failed_is_an_error(tmp_path):
    path = write_jsonl(
        tmp_path,
        [
            metadata_record(),
            gemini_record(
                "g1",
                "x",
                toolCalls=[
                    tool_call(
                        "c1",
                        "replace",
                        {"file_path": "/repo/a.py"},
                        result=[
                            function_response(
                                "c1", "replace", {"error": "Failed to edit, 0 occurrences found"}
                            )
                        ],
                        status="error",
                    )
                ],
            ),
        ],
    )

    result = messages_by_id(import_gemini_cli(path).session)["g1#1"].tool_result
    assert result.is_error is True
    assert result.output == "Failed to edit, 0 occurrences found"


def test_a_cancelled_call_is_an_error(tmp_path):
    path = write_jsonl(
        tmp_path,
        [
            metadata_record(),
            gemini_record(
                "g1",
                "x",
                toolCalls=[
                    tool_call(
                        "c1",
                        "run_shell_command",
                        {"command": "rm -rf /"},
                        result=[
                            function_response(
                                "c1", "run_shell_command", {"error": "[Operation Cancelled]"}
                            )
                        ],
                        status="cancelled",
                    )
                ],
            ),
        ],
    )

    assert messages_by_id(import_gemini_cli(path).session)["g1#1"].tool_result.is_error is True


def test_a_status_that_is_not_a_string_counts_as_an_error_and_is_noted(tmp_path):
    # A result that cannot be read as a successful one does not pass: T6
    # resolves the last passing verification from this field.
    path = write_jsonl(
        tmp_path,
        [
            metadata_record(),
            gemini_record(
                "g1",
                "x",
                toolCalls=[
                    tool_call(
                        "c1",
                        "read_file",
                        {"file_path": "/repo/a.py"},
                        result=output("c1", "read_file", "body"),
                        status=7,
                    )
                ],
            ),
        ],
    )

    imported = import_gemini_cli(path)

    assert messages_by_id(imported.session)["g1#1"].tool_result.is_error is True
    assert [note.detail for note in imported.notes_of(NOTE_DROPPED_BLOCK)] == [
        "toolCalls/<status: number>"
    ]


def test_a_response_this_importer_cannot_read_is_noted_by_its_shape(tmp_path):
    path = write_jsonl(
        tmp_path,
        [
            metadata_record(),
            gemini_record(
                "g1",
                "x",
                toolCalls=[
                    tool_call(
                        "c1",
                        "read_file",
                        {},
                        result=[function_response("c1", "read_file", {"rows": [1, 2]})],
                        status="success",
                    )
                ],
            ),
        ],
    )

    imported = import_gemini_cli(path)

    assert messages_by_id(imported.session)["g1#1"].tool_result.output == ""
    assert [note.detail for note in imported.notes_of(NOTE_DROPPED_BLOCK)] == [
        "functionResponse/<response: object>"
    ]


def test_an_empty_response_is_not_noted(tmp_path):
    path = write_jsonl(
        tmp_path,
        [
            metadata_record(),
            gemini_record(
                "g1",
                "x",
                toolCalls=[
                    tool_call(
                        "c1",
                        "read_file",
                        {},
                        result=[function_response("c1", "read_file", {})],
                        status="success",
                    )
                ],
            ),
        ],
    )

    imported = import_gemini_cli(path)

    assert messages_by_id(imported.session)["g1#1"].tool_result.output == ""
    assert imported.notes_of(NOTE_DROPPED_BLOCK) == ()


def test_a_result_part_without_a_call_reference_is_rejected(tmp_path):
    path = write_jsonl(
        tmp_path,
        [metadata_record(), user_record("u1", [{"functionResponse": {"name": "read_file"}}])],
    )

    with pytest.raises(TranscriptImportError, match="line 2") as error:
        import_gemini_cli(path)

    assert "functionResponse" in str(error.value)


# --- nothing is repaired ----------------------------------------------------


def test_a_front_truncated_transcript_imports_as_the_session_it_is():
    imported = import_gemini_cli(native("truncated_front"))

    session = imported.session
    assert session.messages[0].role is Role.TOOL
    assert session.messages[0].tool_result.call_id == "call-10"
    assert [call.id for message in session.messages for call in message.tool_calls] == []


def test_a_call_whose_result_never_arrives_imports_without_one():
    session = import_gemini_cli(native("interrupted_turn")).session

    message = messages_by_id(session)["g20"]
    assert [call.id for call in message.tool_calls] == ["call-20"]
    assert [m.id for m in session.messages if m.role is Role.TOOL] == []


def test_a_rewind_drops_the_turn_it_names_and_everything_after_it():
    session = import_gemini_cli(native("rewound_turn")).session

    assert [message.id for message in session.messages] == ["u50", "g51"]


def test_a_rewind_to_an_unknown_message_drops_every_message_before_it(tmp_path):
    # The agent's own loader clears the whole list when it cannot find the
    # message to rewind to; a session that kept them would not be the one the
    # agent resumes.
    path = write_jsonl(
        tmp_path,
        [
            metadata_record(),
            user_record("u1", "first"),
            {"$rewindTo": "nowhere"},
            user_record("u2", "second"),
        ],
    )

    session = import_gemini_cli(path).session

    assert [message.id for message in session.messages] == ["u2"]
    assert session.messages[0].content == "second"


def test_a_repeated_record_supersedes_the_earlier_one_in_place():
    # Gemini CLI appends the whole message again when its tool call completes,
    # so the same identifier occurs twice and the later record is the message.
    session = import_gemini_cli(native("golden_session")).session

    assert [message.id for message in session.messages][:4] == [
        "env01",
        "u01",
        "g01#0",
        "g01#1",
    ]
    assert messages_by_id(session)["g01#1"].tool_result is not None


def test_a_metadata_update_replaces_the_message_list(tmp_path):
    path = write_jsonl(
        tmp_path,
        [
            metadata_record(),
            user_record("u1", "first"),
            {"$set": {"messages": [user_record("u2", "kept")]}},
        ],
    )

    session = import_gemini_cli(path).session

    assert [(m.id, m.content) for m in session.messages] == [("u2", "kept")]


# --- the native compaction summary ------------------------------------------


def test_a_natively_compacted_transcript_is_marked_as_such():
    imported = import_gemini_cli(native("native_compaction"))

    assert imported.session.source is SessionSource.NATIVE_COMPACTION


def test_the_state_snapshot_is_an_injected_system_message_not_a_user_turn():
    session = import_gemini_cli(native("native_compaction")).session

    message = messages_by_id(session)["c40"]
    assert message.role is Role.SYSTEM
    assert message.is_system_injected is True
    assert "<state_snapshot>" in message.content


def test_the_compaction_keeps_only_the_messages_the_agent_kept():
    session = import_gemini_cli(native("native_compaction")).session

    assert [message.id for message in session.messages] == ["c40", "c41", "u41"]


def test_an_uncompacted_transcript_stays_raw():
    assert import_gemini_cli(native("golden_session")).session.source is SessionSource.RAW


def test_a_snapshot_is_recognised_wherever_it_sits_in_the_message(tmp_path):
    # The summarizer reasons in a `<scratchpad>` before it writes the snapshot,
    # so the marker is not at the start of the text; this is the same substring
    # test Gemini CLI itself uses to find a previous snapshot.
    path = write_jsonl(
        tmp_path,
        [
            metadata_record(),
            user_record("c1", [{"text": "<scratchpad>…</scratchpad>\n<state_snapshot>…"}]),
        ],
    )

    imported = import_gemini_cli(path)

    assert imported.session.source is SessionSource.NATIVE_COMPACTION
    assert imported.session.messages[0].role is Role.SYSTEM


# --- tool categories and the unknown tools ----------------------------------


def test_every_call_of_the_golden_session_carries_a_category():
    session = import_gemini_cli(native("golden_session")).session

    categories = {
        call.name: call.category for message in session.messages for call in message.tool_calls
    }
    assert categories == {
        "read_file": ToolCategory.READ,
        "glob": ToolCategory.READ,
        "replace": ToolCategory.WRITE,
        "run_shell_command": ToolCategory.EXEC,
    }


def test_an_undeclared_tool_resolves_to_other_and_is_noted():
    imported = import_gemini_cli(native("unknown_tool"))

    assert imported.unknown_tools == ("teleport_file",)
    calls = {
        call.name: call.category
        for message in imported.session.messages
        for call in message.tool_calls
    }
    assert calls["teleport_file"] is ToolCategory.OTHER


def test_an_undeclared_tool_is_logged_once_per_name(caplog):
    with caplog.at_level(logging.WARNING, logger="idc.importers"):
        import_gemini_cli(native("unknown_tool"))

    warnings = [record for record in caplog.records if "teleport_file" in record.getMessage()]
    assert len(warnings) == 1
    assert "other" in warnings[0].getMessage()


def test_a_declared_tool_is_not_logged(caplog):
    with caplog.at_level(logging.WARNING, logger="idc.importers"):
        import_gemini_cli(native("golden_session"))

    assert [record for record in caplog.records if record.levelno >= logging.WARNING] == []


def test_the_subagent_tool_is_declared_and_marked():
    registry = tool_registry()

    assert registry.is_subagent_tool(Agent.GEMINI_CLI, "invoke_agent")
    assert registry.category(Agent.GEMINI_CLI, "invoke_agent") is ToolCategory.OTHER


def test_the_tools_the_corpus_and_the_fixtures_use_are_all_declared():
    registry = tool_registry()
    names = ("read_file", "write_file", "replace", "run_shell_command", "glob", "invoke_agent")

    assert [name for name in names if not registry.knows(Agent.GEMINI_CLI, name)] == []


# --- the dispatcher ---------------------------------------------------------


def test_import_transcript_dispatches_to_this_importer():
    through_dispatcher = import_transcript(Agent.GEMINI_CLI, native("golden_session"))
    directly = import_gemini_cli(native("golden_session"))

    assert dumps(through_dispatcher.session) == dumps(directly.session)


def test_import_transcript_accepts_the_agents_wire_string():
    assert import_transcript("gemini-cli", native("golden_session")).session.agent is (
        Agent.GEMINI_CLI
    )


def test_the_importer_no_longer_reports_itself_as_pending():
    from idc.importers import IMPORTERS, PENDING_IMPORTERS

    assert Agent.GEMINI_CLI in IMPORTERS
    assert Agent.GEMINI_CLI not in PENDING_IMPORTERS


# --- a registry of its own --------------------------------------------------


def test_a_caller_may_pass_its_own_registry(tmp_path):
    from idc.importers import load_tool_registry

    source = tmp_path / "registry.toml"
    source.write_text(
        'version = 1\nrunner_prefixes = []\n'
        '[agents.claude-code.tools]\nRead = { category = "read" }\n'
        '[agents.gemini-cli.tools]\nread_file = { category = "exec" }\n'
        '[agents.opencode.tools]\nread = { category = "read" }\n'
        '[[verification_patterns]]\nid = "pytest"\ncommand = "pytest"\n',
        encoding="utf-8",
    )
    path = write_jsonl(
        tmp_path,
        [
            metadata_record(),
            gemini_record("g1", "x", toolCalls=[tool_call("c1", "read_file", {})]),
        ],
    )

    imported = import_gemini_cli(path, registry=load_tool_registry(source))

    assert imported.session.messages[0].tool_calls[0].category is ToolCategory.EXEC
