"""Corpus-wide invariants of the shared transcript fixture corpus.

The nine cases the corpus has to cover are listed in `REQUIRED_CASES`; every
`*.json` file in `tests/fixtures/` is *discovered*, not listed, so a fixture
added later cannot silently skip validation.

This module resolves and loads the directory itself instead of going through
`fixture_corpus`: it is what guarantees that the files on disk are valid, so it
must not depend on the access layer that reads them. The layer's own discovery
is checked against this one in `test_fixture_access.py`.
"""

import re
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from idc.schema import Agent, Session, ToolCategory, dumps, load, loads

TESTS_DIR = Path(__file__).parent.resolve()
FIXTURES_DIR = TESTS_DIR / "fixtures"
README_PATH = FIXTURES_DIR / "README.md"

REQUIRED_CASES = (
    "single_intent",
    "two_sequential_intents",
    "interleaved_intents",
    "slash_command",
    "system_injected_message",
    "subagent_result",
    "failed_then_passing_command",
    "write_then_two_edits",
    "already_compacted_block",
)
"""The nine cases of the ticket's scope list, one fixture file each."""

FIXTURE_PATHS = sorted(FIXTURES_DIR.glob("*.json"))
"""Every fixture on disk, discovered at collection time."""

FIXTURE_NAMES = [path.stem for path in FIXTURE_PATHS]

MESSAGE_ID_PATTERN = re.compile(r"^m[0-9]{2,}$")
"""Message ids are `m01…` — see the conventions section of the corpus README."""

TOOL_CALL_ID_PATTERN = re.compile(r"^c[0-9]{2,}$")
"""Tool-call ids are `c01…` — see the conventions section of the corpus README."""

UTC_OFFSET = timedelta(0)
"""The offset most of the corpus is recorded in; two fixtures use another one."""


def test_fixtures_dir_exists_and_is_a_directory():
    assert FIXTURES_DIR.is_dir()


def test_fixtures_dir_fixture_points_at_the_corpus(fixtures_dir):
    assert fixtures_dir == FIXTURES_DIR
    assert fixtures_dir.is_dir()


def test_fixtures_dir_is_absolute_and_survives_a_chdir(fixtures_dir, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    assert fixtures_dir.is_absolute()
    assert fixtures_dir.is_dir()


def test_the_corpus_is_not_empty():
    # Guards the parametrized tests below: an empty corpus would turn them into
    # zero test cases instead of into failures.
    assert FIXTURE_PATHS, f"no *.json fixtures found in {FIXTURES_DIR}"


def test_every_required_case_has_a_fixture():
    assert set(REQUIRED_CASES) <= set(FIXTURE_NAMES)


def test_the_corpus_holds_nothing_but_fixtures_and_its_readme():
    unexpected = [
        path.name
        for path in sorted(FIXTURES_DIR.iterdir())
        if path.name != README_PATH.name and path.suffix != ".json"
    ]

    assert unexpected == []


@pytest.mark.parametrize("path", FIXTURE_PATHS, ids=FIXTURE_NAMES)
def test_every_fixture_validates_against_the_schema(path):
    # `load` validates and raises SchemaValidationError; nothing else to assert.
    assert isinstance(load(Session, path), Session)


@pytest.mark.parametrize("path", FIXTURE_PATHS, ids=FIXTURE_NAMES)
def test_every_fixture_round_trips_through_the_schema(path):
    session = load(Session, path)

    assert loads(Session, dumps(session)) == session


@pytest.mark.parametrize("path", FIXTURE_PATHS, ids=FIXTURE_NAMES)
def test_every_fixture_is_stored_in_canonical_dump_format(path):
    # The corpus is study material that is read in diffs, so the files are kept
    # byte-identical to what the tool itself writes.
    assert path.read_text(encoding="utf-8") == dumps(load(Session, path)) + "\n"


@pytest.mark.parametrize("path", FIXTURE_PATHS, ids=FIXTURE_NAMES)
def test_every_fixture_has_a_readme_line(path):
    lines = [
        line
        for line in README_PATH.read_text(encoding="utf-8").splitlines()
        if f"`{path.name}`" in line
    ]

    assert len(lines) == 1, f"{path.name} needs exactly one README line, found {len(lines)}"
    description = lines[0].split("—", 1)[-1].strip()
    assert len(description) >= 30, f"README line for {path.name} does not say what it exercises"


@pytest.mark.parametrize("path", FIXTURE_PATHS, ids=FIXTURE_NAMES)
def test_every_fixture_has_a_non_trivial_conversation(path):
    session = load(Session, path)

    assert len(session.messages) >= 5
    assert any(message.tool_calls for message in session.messages)


@pytest.mark.parametrize("path", FIXTURE_PATHS, ids=FIXTURE_NAMES)
def test_every_fixture_names_itself_after_its_file(path):
    session = load(Session, path)

    assert session.id == "sess-" + path.stem.replace("_", "-")


def test_session_ids_are_unique_across_the_corpus():
    ids = [load(Session, path).id for path in FIXTURE_PATHS]

    assert len(set(ids)) == len(ids)


@pytest.mark.parametrize("path", FIXTURE_PATHS, ids=FIXTURE_NAMES)
def test_every_tool_call_has_exactly_one_later_result(path):
    session = load(Session, path)
    calls = {
        call.id: index
        for index, message in enumerate(session.messages)
        for call in message.tool_calls
    }
    results = [
        (index, message.tool_result.call_id)
        for index, message in enumerate(session.messages)
        if message.tool_result is not None
    ]
    answered = [call_id for _, call_id in results]

    assert len(set(answered)) == len(answered), "a call is answered twice"
    assert set(answered) == set(calls), "every call is answered, every result answers a call"
    for index, call_id in results:
        assert index > calls[call_id], f"result of {call_id} precedes its call"


@pytest.mark.parametrize("path", FIXTURE_PATHS, ids=FIXTURE_NAMES)
def test_every_message_is_timestamped_in_transcript_order(path):
    session = load(Session, path)
    timestamps: list[datetime] = []
    for message in session.messages:
        assert message.timestamp is not None, f"{message.id} has no timestamp"
        timestamps.append(message.timestamp)

    assert timestamps == sorted(timestamps)
    assert session.created_at <= timestamps[0]


@pytest.mark.parametrize("path", FIXTURE_PATHS, ids=FIXTURE_NAMES)
def test_every_fixture_numbers_its_messages_and_calls_in_the_corpus_shape(path):
    # The README promises the `m01…` / `c01…` shape, and downstream fixtures of
    # later tickets are read in diffs next to these: the numbering is part of
    # the contract, not an accident of how the first nine were written.
    session = load(Session, path)
    message_ids = [message.id for message in session.messages]
    call_ids = [call.id for message in session.messages for call in message.tool_calls]

    assert all(MESSAGE_ID_PATTERN.match(identifier) for identifier in message_ids), message_ids
    assert all(TOOL_CALL_ID_PATTERN.match(identifier) for identifier in call_ids), call_ids
    assert message_ids == [f"m{number:02d}" for number in range(1, len(message_ids) + 1)], (
        "message ids are numbered from m01 in transcript order"
    )
    assert call_ids == [f"c{number:02d}" for number in range(1, len(call_ids) + 1)], (
        "tool-call ids are numbered from c01 in the order the calls are made"
    )


def test_the_corpus_mixes_utc_with_another_timestamp_offset():
    # Deliberate: every downstream stage that orders or compares timestamps has
    # to survive a transcript recorded in a local offset, so the corpus keeps
    # both kinds. Which fixtures carry the offset is free; that both occur is not.
    offsets = {
        message.timestamp.utcoffset()
        for path in FIXTURE_PATHS
        for message in load(Session, path).messages
        if message.timestamp is not None
    }

    assert UTC_OFFSET in offsets, "the bulk of the corpus is recorded in UTC"
    assert any(offset != UTC_OFFSET for offset in offsets), (
        "at least one fixture must stay recorded in a non-UTC offset"
    )


@pytest.mark.parametrize("path", FIXTURE_PATHS, ids=FIXTURE_NAMES)
def test_exit_code_is_set_for_exec_calls_only(path):
    # The invariant the exploration and tool statistics (T9) key off: a result
    # carries an exit code exactly when it answers an `exec` call.
    session = load(Session, path)
    category = {
        call.id: call.category for message in session.messages for call in message.tool_calls
    }
    results = [
        message.tool_result for message in session.messages if message.tool_result is not None
    ]

    for result in results:
        runs_a_command = category[result.call_id] is ToolCategory.EXEC
        assert (result.exit_code is not None) is runs_a_command, (
            f"{result.call_id} answers a {category[result.call_id]} call "
            f"with exit_code {result.exit_code!r}; exit_code is for exec calls only"
        )


def test_the_corpus_covers_every_agent():
    agents = {load(Session, path).agent for path in FIXTURE_PATHS}

    assert agents == set(Agent)
