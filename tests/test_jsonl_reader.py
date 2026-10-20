"""The shared JSONL reader: the one place a line-delimited transcript is read."""

import json
from pathlib import Path

import pytest

from idc.importers import JsonLine, TranscriptImportError, read_json_lines


def write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_every_line_is_returned_with_its_line_number(tmp_path):
    path = write(tmp_path / "s.jsonl", '{"type":"a"}\n{"type":"b"}\n')

    lines = read_json_lines(path)

    assert lines == [JsonLine(1, {"type": "a"}), JsonLine(2, {"type": "b"})]


def test_blank_lines_are_skipped_without_shifting_the_numbers(tmp_path):
    path = write(tmp_path / "s.jsonl", '{"type":"a"}\n\n   \n{"type":"b"}\n')

    lines = read_json_lines(path)

    assert [line.number for line in lines] == [1, 4]


def test_a_missing_trailing_newline_is_no_problem(tmp_path):
    path = write(tmp_path / "s.jsonl", '{"type":"a"}')

    assert read_json_lines(path) == [JsonLine(1, {"type": "a"})]


def test_non_ascii_content_survives_the_read(tmp_path):
    path = write(tmp_path / "s.jsonl", json.dumps({"text": "Größe — ✅"}) + "\n")

    assert read_json_lines(path)[0].data == {"text": "Größe — ✅"}


def test_a_missing_file_is_reported_with_its_path(tmp_path):
    missing = tmp_path / "absent.jsonl"

    with pytest.raises(TranscriptImportError, match=str(missing)):
        read_json_lines(missing)


def test_a_line_that_is_not_json_is_reported_with_the_line_number(tmp_path):
    path = write(tmp_path / "s.jsonl", '{"type":"a"}\nnot json\n')

    with pytest.raises(TranscriptImportError, match="line 2") as error:
        read_json_lines(path)

    assert str(path) in str(error.value)


def test_a_line_that_is_json_but_no_object_is_rejected(tmp_path):
    path = write(tmp_path / "s.jsonl", '{"type":"a"}\n[1, 2]\n')

    with pytest.raises(TranscriptImportError, match="line 2") as error:
        read_json_lines(path)

    assert "object" in str(error.value)


def test_a_file_without_records_is_rejected(tmp_path):
    path = write(tmp_path / "s.jsonl", "\n\n")

    with pytest.raises(TranscriptImportError, match="no records"):
        read_json_lines(path)


def test_a_file_that_is_not_utf8_is_rejected(tmp_path):
    path = tmp_path / "s.jsonl"
    path.write_bytes(b'{"text":"\xff"}\n')

    with pytest.raises(TranscriptImportError, match="UTF-8"):
        read_json_lines(path)


def test_a_directory_is_reported_as_unreadable(tmp_path):
    with pytest.raises(TranscriptImportError, match=str(tmp_path)):
        read_json_lines(tmp_path)


# --- what counts as a line break -------------------------------------------

LINE_BREAK_CHARACTERS = [
    ("line separator", chr(0x2028)),
    ("paragraph separator", chr(0x2029)),
    ("next line", chr(0x85)),
]
# The characters `str.splitlines()` breaks on and JSON does not escape, written
# as code points because they are invisible: a record whose content holds one of
# them is a valid line of a valid transcript.


@pytest.mark.parametrize(("name", "character"), LINE_BREAK_CHARACTERS)
def test_a_line_break_character_inside_a_record_does_not_split_the_line(tmp_path, name, character):
    text = f"minified{character}javascript"
    path = write(tmp_path / "s.jsonl", json.dumps({"text": text}, ensure_ascii=False) + "\n")

    assert read_json_lines(path) == [JsonLine(1, {"text": text})]


@pytest.mark.parametrize(("name", "character"), LINE_BREAK_CHARACTERS)
def test_a_line_break_character_does_not_shift_the_following_line_numbers(
    tmp_path, name, character
):
    first = json.dumps({"text": f"a{character}b"}, ensure_ascii=False)
    path = write(tmp_path / "s.jsonl", f"{first}\nnot json\n")

    with pytest.raises(TranscriptImportError, match="line 2"):
        read_json_lines(path)


def test_windows_line_endings_are_read_as_line_endings(tmp_path):
    path = write(tmp_path / "s.jsonl", '{"type":"a"}\r\n{"type":"b"}\r\n')

    assert read_json_lines(path) == [JsonLine(1, {"type": "a"}), JsonLine(2, {"type": "b"})]


# --- a byte-order mark ------------------------------------------------------

BOM = b"\xef\xbb\xbf"
# What an editor on Windows prepends to a UTF-8 file. It is not part of the
# first record, so the first line is still JSON.


def test_a_byte_order_mark_is_not_part_of_the_first_record(tmp_path):
    path = tmp_path / "s.jsonl"
    path.write_bytes(BOM + b'{"type":"a"}\n{"type":"b"}\n')

    assert read_json_lines(path) == [JsonLine(1, {"type": "a"}), JsonLine(2, {"type": "b"})]


def test_a_file_without_a_byte_order_mark_is_read_the_same_way(tmp_path):
    path = tmp_path / "s.jsonl"
    path.write_bytes(b'{"type":"a"}\n')

    assert read_json_lines(path) == [JsonLine(1, {"type": "a"})]
