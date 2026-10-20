"""System and command reduction: harness noise leaves the chunk, the note stays.

A reducible message is one the harness wrote and that carries nothing but text
(a slash command or an injected reminder, with no tool call and no tool result).
It is replaced by its result text where it has one and dropped where it has not,
and every removal is recorded as a `ReductionNote` on the chunk it happened in —
which is what makes the reduction auditable instead of a silent deletion.

Three kinds of test live here, following `tests/test_chunk_user_boundaries.py`:

- **The expectation table.** `EXPECTED_NOTES` is read off the fixtures by hand,
  not computed, so the per-fixture tests have an oracle the implementation
  cannot agree with by accident. It is pinned to the discovered corpus in both
  directions, so a fixture added later fails loudly until its reductions have
  been read off by hand.
- **The corpus-wide invariants.** Everything parametrized over `FIXTURE_NAMES`
  is discovered from `tests/fixtures/` at collection time. They state what
  reduction must *not* break: the chunk count, the partition and no-copy
  properties T3.1 established, and idempotence — a second reduction of a
  reduced transcript changes nothing and has nothing left to report.
- **The synthetic cases**, for the shapes the corpus does not hold: a
  transcript of nothing but noise, a command that carries a tool call, the
  preamble of a natively compacted transcript.

What a reducible message is, and what its result is, is restated locally on
purpose — a test that imported the implementation's own predicate would only
check that the code agrees with itself.
"""

import json
from itertools import pairwise

import pytest

from fixture_corpus import CORPUS
from idc.chunk import (
    ReducedChunk,
    ReducedSession,
    ReductionNote,
    ReductionReason,
    chunk_session,
    reduce_chunks,
    reduce_session,
)
from idc.schema import Chunk, Role, Session, SessionSource, dumps, loads

FIXTURE_NAMES = CORPUS.names()
"""Every fixture of the shared corpus, discovered at collection time."""

EXPECTED_NOTES: dict[str, list[tuple[str, str, str, tuple[str, ...]]]] = {
    # No harness noise at all: nothing to reduce, and the transcript must come
    # out of the stage unchanged.
    "single_intent": [],
    "two_sequential_intents": [],
    "interleaved_intents": [],
    "subagent_result": [],
    "failed_then_passing_command": [],
    "write_then_two_edits": [],
    # `/model claude-opus-5` (m01) is answered by the injected m02, so it is
    # replaced by that text; `/test tests/api` (m08) is answered by the agent
    # working, not by the harness, so it has no result and is dropped.
    "slash_command": [
        ("K1", "m01", "command", ("m02",)),
        ("K1", "m08", "command", ()),
    ],
    # Three injected reminders, none of them the answer to anything: dropped.
    "system_injected_message": [
        ("K1", "m01", "system-injected", ()),
        ("K1", "m05", "system-injected", ()),
        ("K1", "m10", "system-injected", ()),
    ],
    # The injected first message carries the rendered summary block of a
    # previous compaction, which the parser (T8) has to find again — the
    # preamble of an already compacted transcript is exempt from reduction.
    "already_compacted_block": [],
}
"""The reductions each fixture must produce, read off the corpus by hand.

One `(chunk id, message id, reason, result source ids)` tuple per note, in
transcript order. The table has to name *every* fixture of the corpus: a
fixture added later without its hand-read reductions would quietly leave the
table-driven tests below with one case less, and
`test_the_expectation_table_covers_the_discovered_fixtures` turns that into a
failure instead.
"""

CREATED_AT = "2026-10-20T09:00:00+00:00"
"""Fixed instant for the synthetic sessions; no test reads the clock."""

BLANK_LINE = "\n\n"
"""How several result texts are expected to be joined into one."""


def synthetic_session(messages: list[dict[str, object]], **session: object) -> Session:
    """Return a session built from message stubs, validated through `loads`.

    Each stub needs a `role` only; the id and the timestamp are filled in.
    Keyword arguments override the session's own fields (`source`, `agent`, …).
    The document goes through `loads` rather than `Session.model_validate`,
    which is the load path every stage uses (#3).
    """
    document = {
        "id": "sess-synthetic",
        "agent": "claude-code",
        "created_at": CREATED_AT,
        **session,
        "messages": [
            {"id": f"m{index:02d}", "timestamp": CREATED_AT, **message}
            for index, message in enumerate(messages)
        ],
    }
    return loads(Session, json.dumps(document))


def flat_notes(reduced: ReducedSession) -> list[tuple[str, str, str, tuple[str, ...]]]:
    """Return every note of a reduction as a comparable tuple, in order."""
    return [
        (reduced_chunk.chunk.id, note.message_id, str(note.reason), note.result_from)
        for reduced_chunk in reduced.chunks
        for note in reduced_chunk.notes
    ]


def removed_ids(name: str) -> set[str]:
    """Return the message ids the expectation table removes from a fixture.

    A dropped message and the messages whose text was taken as a result are
    gone from the reduced transcript; a replaced message stays, with the result
    as its content.
    """
    removed: set[str] = set()
    for _chunk_id, message_id, _reason, result_from in EXPECTED_NOTES[name]:
        removed.update(result_from)
        if not result_from:
            removed.add(message_id)
    return removed


def replaced_ids(name: str) -> set[str]:
    """Return the message ids the expectation table replaces by their result."""
    return {
        message_id
        for _chunk_id, message_id, _reason, result_from in EXPECTED_NOTES[name]
        if result_from
    }


def test_the_corpus_was_discovered():
    # Guards every parametrized test below: an empty corpus would silently turn
    # them into zero test cases instead of into failures.
    assert FIXTURE_NAMES


def test_the_expectation_table_covers_the_discovered_fixtures():
    # Guards the table-driven tests below, the only ones here whose oracle is
    # not derived from the code under test. The table is pinned to the corpus in
    # both directions: every discovered fixture needs its hand-read reductions,
    # and a name the table invents has no fixture to read them from.
    assert EXPECTED_NOTES, "the hand-read expectation table must not be empty"
    missing = sorted(set(FIXTURE_NAMES) - set(EXPECTED_NOTES))
    unknown = sorted(set(EXPECTED_NOTES) - set(FIXTURE_NAMES))

    assert missing == [], f"fixtures without hand-read expected reductions: {missing}"
    assert unknown == [], f"expected reductions for fixtures that do not exist: {unknown}"


# --- the expectation table -------------------------------------------------


@pytest.mark.parametrize("name", list(EXPECTED_NOTES))
def test_the_notes_are_the_expected_reductions(name):
    reduced = reduce_session(CORPUS.session(name))

    assert flat_notes(reduced) == EXPECTED_NOTES[name]


@pytest.mark.parametrize("name", list(EXPECTED_NOTES))
def test_the_reduced_transcript_holds_the_expected_messages(name):
    session = CORPUS.session(name)
    expected = [message.id for message in session.messages if message.id not in removed_ids(name)]

    reduced = reduce_session(session)

    assert [message.id for message in reduced.session.messages] == expected


@pytest.mark.parametrize("name", list(EXPECTED_NOTES))
def test_every_note_names_a_message_of_the_chunk_it_sits_on(name):
    session = CORPUS.session(name)
    spans = {
        chunk.id: {message.id for message in session.messages[chunk.start_index : chunk.end_index]}
        for chunk in chunk_session(session)
    }

    for reduced_chunk in reduce_session(session).chunks:
        for note in reduced_chunk.notes:
            assert note.message_id in spans[reduced_chunk.chunk.id], note


# --- the cases the ticket exists for ---------------------------------------


def test_a_slash_command_is_replaced_by_its_result_text():
    session = CORPUS.session("slash_command")
    command, result = session.messages[0], session.messages[1]

    reduced = reduce_session(session)

    assert command.is_command, "fixture precondition: the session opens with a slash command"
    assert command.content == "/model claude-opus-5"
    assert result.is_system_injected, "fixture precondition: the harness answers the command"
    contents = [message.content for message in reduced.session.messages]
    assert result.content in contents, "the result text must be present"
    assert command.content not in contents, "the command must be absent"
    # The result is carried by the command's own message, not by a second one.
    assert [message.id for message in reduced.session.messages].count(result.id) == 0
    reduced_message = reduced.session.messages[0]
    assert reduced_message.id == command.id
    assert reduced_message.content == result.content


def test_a_slash_command_without_a_result_is_dropped():
    session = CORPUS.session("slash_command")
    command = session.messages[7]

    reduced = reduce_session(session)

    assert command.is_command, "fixture precondition: a slash command inside the turn"
    assert command.content == "/test tests/api"
    assert command.id not in [message.id for message in reduced.session.messages]
    assert ("K1", command.id, "command", ()) in flat_notes(reduced)


def test_the_chunk_count_is_unchanged_by_a_slash_command_reduction():
    session = CORPUS.session("slash_command")

    assert len(reduce_session(session).chunks) == len(chunk_session(session))


def test_injected_messages_are_dropped_with_a_note_each():
    session = CORPUS.session("system_injected_message")
    injected = [message for message in session.messages if message.is_system_injected]

    reduced = reduce_session(session)

    assert len(injected) == 3, "fixture precondition: three harness-injected messages"
    surviving = [message.id for message in reduced.session.messages]
    contents = [message.content for message in reduced.session.messages]
    for message in injected:
        assert message.id not in surviving, f"{message.id} must be gone"
        assert message.content not in contents, f"the text of {message.id} must be gone"
    assert [note.message_id for note in reduced.notes] == [message.id for message in injected]
    assert all(note.reason is ReductionReason.SYSTEM_INJECTED for note in reduced.notes)
    assert all(note.was_replaced is False for note in reduced.notes)


def test_the_chunk_count_is_unchanged_by_an_injected_message_reduction():
    session = CORPUS.session("system_injected_message")

    assert len(reduce_session(session).chunks) == len(chunk_session(session))


def test_an_injected_user_message_is_dropped_without_opening_a_chunk():
    # The subtle case: the injected message sits in the *user* role in the
    # middle of the agent's work. T3.1 does not let it open a chunk, and the
    # reduction must not turn it into one either — it leaves the transcript.
    session = CORPUS.session("system_injected_message")
    injected_user = [
        message
        for message in session.messages
        if message.role is Role.USER and message.is_system_injected
    ]

    reduced = reduce_session(session)

    assert injected_user, "fixture precondition: an injected message in the user role"
    assert len(reduced.chunks) == 1
    assert all(
        message.role is not Role.USER or not message.is_system_injected
        for message in reduced.session.messages
    )


def test_a_rendered_summary_block_survives_reduction():
    # `already_compacted_block` opens with an injected message that carries this
    # tool's own summary block — the input the parser (T8) has to find again. A
    # transcript that was compacted before therefore keeps its preamble.
    session = CORPUS.session("already_compacted_block")
    block = session.messages[0]

    reduced = reduce_session(session)

    assert session.source is SessionSource.INTENT_COMPACTION
    assert block.is_system_injected, "fixture precondition: the block is an injected message"
    assert "[C1]" in block.content
    assert reduced.notes == ()
    assert reduced.session.messages[0] is block
    assert "[C1]" in reduced.session.messages[0].content
    assert "[C2]" in reduced.session.messages[0].content


def test_the_preamble_of_a_natively_compacted_transcript_is_exempt_too():
    # Same reason, other condition of the study: the agent's own summary opens
    # the transcript as an injected message, and the measurement compares it
    # against ours. Dropping it would change what is being measured.
    session = synthetic_session(
        [
            {
                "role": "system",
                "content": "Summary of the conversation so far: the exporter was rewritten.",
                "is_system_injected": True,
            },
            {"role": "user", "content": "Now add the CLI flag."},
            {
                "role": "assistant",
                "content": "Done.",
            },
        ],
        source="native-compaction",
    )

    reduced = reduce_session(session)

    assert reduced.notes == ()
    assert [message.id for message in reduced.session.messages] == ["m00", "m01", "m02"]


@pytest.mark.parametrize(
    "source", [SessionSource.INTENT_COMPACTION, SessionSource.NATIVE_COMPACTION]
)
def test_only_the_preamble_of_a_compacted_transcript_is_exempt(source):
    # The upper bound of the exemption, for both conditions of the study: what
    # is protected is the summary block the transcript opens with, not the
    # transcript. Noise *after* the first real user turn is harness noise like
    # any other and is reduced. Without this case the suite cannot tell
    # "preamble exempt" from "everything exempt" — and the second would make
    # reduction a no-op for exactly the inputs T8 (#33) and T9 (#36) measure,
    # with the block surviving either way.
    session = synthetic_session(
        [
            {
                "role": "system",
                "content": "## [C1] Rewrite the exporter\n\nThe exporter was rewritten.",
                "is_system_injected": True,
            },
            {"role": "user", "content": "Now add the CLI flag."},
            {
                "role": "system",
                "content": "<system-reminder>Context left: 42%.</system-reminder>",
                "is_system_injected": True,
            },
            {"role": "assistant", "content": "Done."},
        ],
        source=str(source),
    )

    reduced = reduce_session(session)

    assert session.source is source, "precondition: the transcript was compacted before"
    assert [(note.message_id, str(note.reason)) for note in reduced.notes] == [
        ("m02", "system-injected")
    ]
    assert [message.id for message in reduced.session.messages] == ["m00", "m01", "m03"]
    # The preamble is still the session's own message, carrying the block.
    assert reduced.session.messages[0] is session.messages[0]
    assert "[C1]" in reduced.session.messages[0].content


def test_the_preamble_of_a_raw_transcript_is_not_exempt():
    session = synthetic_session(
        [
            {
                "role": "system",
                "content": "<system-reminder>Context left: 42%.</system-reminder>",
                "is_system_injected": True,
            },
            {"role": "user", "content": "Add the CLI flag."},
        ]
    )

    reduced = reduce_session(session)

    assert [note.message_id for note in reduced.notes] == ["m00"]
    assert [message.id for message in reduced.session.messages] == ["m01"]


# --- corpus-wide invariants ------------------------------------------------


@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_reduction_keeps_the_chunk_count(name):
    session = CORPUS.session(name)

    assert len(reduce_session(session).chunks) == len(chunk_session(session))


@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_reduction_keeps_the_chunk_identities(name):
    session = CORPUS.session(name)
    chunks = chunk_session(session)

    reduced = reduce_session(session)

    assert [reduced_chunk.chunk.id for reduced_chunk in reduced.chunks] == [
        chunk.id for chunk in chunks
    ]
    assert [reduced_chunk.chunk.is_latest for reduced_chunk in reduced.chunks] == [
        chunk.is_latest for chunk in chunks
    ]
    assert [reduced_chunk.chunk.origin for reduced_chunk in reduced.chunks] == [
        chunk.origin for chunk in chunks
    ]


@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_the_reduced_ranges_partition_the_reduced_messages(name):
    reduced = reduce_session(CORPUS.session(name))

    covered = [
        index
        for reduced_chunk in reduced.chunks
        for index in range(reduced_chunk.chunk.start_index, reduced_chunk.chunk.end_index)
    ]

    # The T3.1 property, restated over the reduced transcript: the covered
    # indices in the order the chunks list them are exactly 0 … n-1, which holds
    # only if the ranges start at the first message, end at the last, follow
    # each other without a gap and never cover an index twice.
    assert covered == list(range(len(reduced.session.messages)))


@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_the_reduced_ranges_are_contiguous(name):
    reduced = reduce_session(CORPUS.session(name))

    chunks = [reduced_chunk.chunk for reduced_chunk in reduced.chunks]

    assert all(earlier.end_index == later.start_index for earlier, later in pairwise(chunks))


@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_re_chunking_the_reduced_transcript_reproduces_the_reduced_chunks(name):
    # Reduction never removes a user turn and never creates one, so chunking the
    # reduced transcript from scratch has to land on exactly the same ranges.
    reduced = reduce_session(CORPUS.session(name))

    assert chunk_session(reduced.session) == [
        reduced_chunk.chunk for reduced_chunk in reduced.chunks
    ]


@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_an_untouched_message_is_carried_by_identity(name):
    session = CORPUS.session(name)
    touched = removed_ids(name) | replaced_ids(name)
    originals = {message.id: message for message in session.messages if message.id not in touched}

    reduced = reduce_session(session)

    carried = [message for message in reduced.session.messages if message.id in originals]
    assert len(carried) == len(originals)
    # Identity, not equality: reduction rewrites the one message it reduces and
    # references every other one, so nothing downstream reads a stale copy.
    assert all(message is originals[message.id] for message in carried)


@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_reduction_invents_no_message(name):
    session = CORPUS.session(name)

    reduced = reduce_session(session)

    ids = [message.id for message in reduced.session.messages]
    assert len(set(ids)) == len(ids)
    assert set(ids) <= {message.id for message in session.messages}
    assert len(ids) <= len(session.messages)


@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_reduction_keeps_the_provenance_of_the_transcript(name):
    session = CORPUS.session(name)

    reduced = reduce_session(session)

    assert (reduced.session.id, reduced.session.agent, reduced.session.model) == (
        session.id,
        session.agent,
        session.model,
    )
    assert (reduced.session.source, reduced.session.created_at) == (
        session.source,
        session.created_at,
    )


@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_reduction_keeps_every_real_user_turn(name):
    session = CORPUS.session(name)
    turns = [
        message.id
        for message in session.messages
        if message.role is Role.USER and not message.is_system_injected and not message.is_command
    ]

    reduced = reduce_session(session)

    assert [
        message.id
        for message in reduced.session.messages
        if message.role is Role.USER and not message.is_system_injected and not message.is_command
    ] == turns


@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_reduction_keeps_every_tool_call_and_result(name):
    session = CORPUS.session(name)
    calls = [call.id for message in session.messages for call in message.tool_calls]
    results = [
        message.tool_result.call_id
        for message in session.messages
        if message.tool_result is not None
    ]

    reduced = reduce_session(session)

    assert [call.id for message in reduced.session.messages for call in message.tool_calls] == calls
    assert [
        message.tool_result.call_id
        for message in reduced.session.messages
        if message.tool_result is not None
    ] == results


@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_reducing_a_reduced_transcript_changes_nothing(name):
    reduced = reduce_session(CORPUS.session(name))

    again = reduce_session(reduced.session)

    # The notes of the first pass are not repeated — there is nothing left to
    # report — so idempotence is about the transcript and the chunks.
    assert again.notes == (), "a reduced transcript has nothing left to reduce"
    assert again.session == reduced.session
    assert [chunk.chunk for chunk in again.chunks] == [chunk.chunk for chunk in reduced.chunks]
    assert all(
        a is b for a, b in zip(again.session.messages, reduced.session.messages, strict=True)
    ), "a second pass must not copy a message either"


@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_reducing_the_same_session_twice_yields_the_same_reduction(name):
    assert reduce_session(CORPUS.session(name)) == reduce_session(CORPUS.session(name))


@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_the_reduced_transcript_is_a_valid_canonical_document(name):
    reduced = reduce_session(CORPUS.session(name))

    assert loads(Session, dumps(reduced.session)) == reduced.session


# --- the shapes the corpus does not hold -----------------------------------


def test_a_transcript_without_messages_reduces_to_nothing():
    session = synthetic_session([])

    reduced = reduce_session(session)

    assert reduced.session.messages == []
    assert reduced.chunks == ()
    assert reduced.notes == ()


def test_a_chunk_is_never_emptied_by_reduction():
    # A chunk that holds nothing but noise can only be the single chunk of a
    # transcript without a real user turn (every other chunk opens at one). It
    # is left unreduced rather than removed: a `Chunk` spans at least one
    # message, and dropping the chunk would change the chunk count.
    session = synthetic_session(
        [
            {
                "role": "system",
                "content": "<system-reminder>The project targets Python 3.13.</system-reminder>",
                "is_system_injected": True,
            },
            {
                "role": "system",
                "content": "<system-reminder>Context left: 42%.</system-reminder>",
                "is_system_injected": True,
            },
        ]
    )

    reduced = reduce_session(session)

    assert len(reduced.chunks) == 1
    assert reduced.notes == ()
    assert [message.id for message in reduced.session.messages] == ["m00", "m01"]
    assert all(a is b for a, b in zip(reduced.session.messages, session.messages, strict=True))


def test_a_command_that_carries_a_tool_call_is_kept():
    # Reduction removes text the harness wrote, never resolvable work: a call
    # whose message disappeared would leave the resolver (T5) with a result it
    # cannot attribute.
    session = synthetic_session(
        [
            {"role": "user", "content": "Fix the exporter."},
            {
                "role": "user",
                "content": "/test tests/api",
                "is_command": True,
                "tool_calls": [
                    {"id": "c01", "name": "Bash", "category": "exec", "args": {"command": "pytest"}}
                ],
            },
        ]
    )

    reduced = reduce_session(session)

    assert reduced.notes == ()
    assert [message.id for message in reduced.session.messages] == ["m00", "m01"]


def test_an_injected_message_that_carries_a_tool_result_is_kept():
    session = synthetic_session(
        [
            {"role": "user", "content": "Fix the exporter."},
            {
                "role": "tool",
                "is_system_injected": True,
                "tool_result": {"call_id": "c01", "output": "7 passed", "exit_code": 0},
            },
        ]
    )

    reduced = reduce_session(session)

    assert reduced.notes == ()
    assert [message.id for message in reduced.session.messages] == ["m00", "m01"]


def test_several_injected_answers_to_one_command_are_joined_into_its_result():
    session = synthetic_session(
        [
            {"role": "user", "content": "Set the model up."},
            {"role": "user", "content": "/model claude-opus-5", "is_command": True},
            {
                "role": "system",
                "content": "Model set to claude-opus-5.",
                "is_system_injected": True,
            },
            {"role": "system", "content": "Thinking budget reset.", "is_system_injected": True},
            {"role": "assistant", "content": "Ready."},
        ]
    )

    reduced = reduce_session(session)

    assert [note.result_from for note in reduced.notes] == [("m02", "m03")]
    assert [message.id for message in reduced.session.messages] == ["m00", "m01", "m04"]
    assert reduced.session.messages[1].content == BLANK_LINE.join(
        ["Model set to claude-opus-5.", "Thinking budget reset."]
    )


def test_the_result_of_a_command_is_not_a_user_turn():
    # The command sat in the user role, and T3.1 did not let it open a chunk.
    # The message that carries its result must not do so either, or reducing a
    # transcript would change how it chunks.
    session = synthetic_session(
        [
            {"role": "user", "content": "Set the model up."},
            {"role": "user", "content": "/model claude-opus-5", "is_command": True},
            {
                "role": "system",
                "content": "Model set to claude-opus-5.",
                "is_system_injected": True,
            },
        ]
    )

    reduced = reduce_session(session)

    result = reduced.session.messages[1]
    assert result.role is Role.SYSTEM
    assert result.is_command is False
    assert result.is_system_injected is False
    assert result.timestamp == session.messages[1].timestamp
    assert len(reduced.chunks) == 1


def test_an_injected_message_does_not_take_over_the_next_ones_text():
    # Only a command has an answer. An injected reminder followed by another one
    # is two reminders, not a request and its result — both are dropped.
    session = synthetic_session(
        [
            {"role": "user", "content": "Fix the exporter."},
            {
                "role": "system",
                "content": "<system-reminder>A.</system-reminder>",
                "is_system_injected": True,
            },
            {
                "role": "system",
                "content": "<system-reminder>B.</system-reminder>",
                "is_system_injected": True,
            },
        ]
    )

    reduced = reduce_session(session)

    assert [note.result_from for note in reduced.notes] == [(), ()]
    assert [message.id for message in reduced.session.messages] == ["m00"]


def test_a_message_that_is_command_and_injected_is_not_another_commands_result():
    # The adversarial shape: the harness marked one message as *both* a slash
    # command and injected. A command echo is noise by this ticket's own
    # definition, so taking its text as the preceding command's answer would
    # keep that noise *and* label it as the harness's answer to a command it
    # never answered. The answer run therefore stops at it, and it is reduced in
    # its own right — it has no answer of its own, so it is dropped.
    session = synthetic_session(
        [
            {"role": "user", "content": "Fix the exporter."},
            {"role": "user", "content": "/model claude-opus-5", "is_command": True},
            {
                "role": "user",
                "content": "/test tests/api",
                "is_command": True,
                "is_system_injected": True,
            },
            {"role": "assistant", "content": "Done."},
        ]
    )

    reduced = reduce_session(session)

    assert [(note.message_id, str(note.reason), note.result_from) for note in reduced.notes] == [
        ("m01", "command", ()),
        ("m02", "command", ()),
    ]
    assert [message.id for message in reduced.session.messages] == ["m00", "m03"]
    contents = [message.content for message in reduced.session.messages]
    assert "/test tests/api" not in contents, "the echo must not survive as a result"
    assert "/model claude-opus-5" not in contents
    # The invariants of the stage hold on this shape too: the chunk count, the
    # partition over the reduced transcript, and idempotence.
    assert len(reduced.chunks) == len(chunk_session(session)) == 1
    assert [
        index
        for reduced_chunk in reduced.chunks
        for index in range(reduced_chunk.chunk.start_index, reduced_chunk.chunk.end_index)
    ] == list(range(len(reduced.session.messages)))
    again = reduce_session(reduced.session)
    assert again.notes == ()
    assert again.session == reduced.session


def test_a_command_answered_by_an_empty_message_is_dropped():
    session = synthetic_session(
        [
            {"role": "user", "content": "Set the model up."},
            {"role": "user", "content": "/model claude-opus-5", "is_command": True},
            {"role": "system", "content": "   ", "is_system_injected": True},
        ]
    )

    reduced = reduce_session(session)

    assert [note.message_id for note in reduced.notes] == ["m01", "m02"]
    assert all(note.was_replaced is False for note in reduced.notes)
    assert [message.id for message in reduced.session.messages] == ["m00"]


def test_a_blank_answer_is_consumed_and_listed_but_contributes_no_text():
    # What `ReductionNote.result_from` actually records: every answer that was
    # consumed, not only the ones whose text survived. The blank answer leaves
    # the transcript like the other one, so naming it is what keeps its removal
    # auditable — but the result text is the non-blank answer alone.
    session = synthetic_session(
        [
            {"role": "user", "content": "Set the model up."},
            {"role": "user", "content": "/model claude-opus-5", "is_command": True},
            {"role": "system", "content": "   ", "is_system_injected": True},
            {"role": "system", "content": "Model set.", "is_system_injected": True},
        ]
    )

    reduced = reduce_session(session)

    assert [note.result_from for note in reduced.notes] == [("m02", "m03")]
    assert all(note.was_replaced for note in reduced.notes)
    assert [message.id for message in reduced.session.messages] == ["m00", "m01"]
    assert reduced.session.messages[1].content == "Model set."


def test_the_result_run_stops_at_the_next_user_turn():
    session = synthetic_session(
        [
            {"role": "user", "content": "First intent."},
            {"role": "user", "content": "/model claude-opus-5", "is_command": True},
            {"role": "user", "content": "Second intent."},
        ]
    )

    reduced = reduce_session(session)

    assert [note.result_from for note in reduced.notes] == [()]
    assert [message.id for message in reduced.session.messages] == ["m00", "m02"]
    assert len(reduced.chunks) == 2


# --- the two entry points --------------------------------------------------


@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_reduce_session_is_reduce_chunks_on_the_sessions_own_chunks(name):
    session = CORPUS.session(name)

    assert reduce_session(session) == reduce_chunks(session, chunk_session(session))


def test_reduce_chunks_rejects_chunks_that_do_not_partition_the_session():
    session = CORPUS.session("slash_command")
    chunks = chunk_session(session)
    first = chunks[0]

    truncated = Chunk(
        id=first.id,
        start_index=first.start_index,
        end_index=first.end_index - 1,
        is_latest=first.is_latest,
        origin=first.origin,
    )

    with pytest.raises(ValueError, match="partition"):
        reduce_chunks(session, [truncated])


def test_the_reduction_types_are_what_the_stage_returns():
    reduced = reduce_session(CORPUS.session("slash_command"))

    assert isinstance(reduced, ReducedSession)
    assert all(isinstance(chunk, ReducedChunk) for chunk in reduced.chunks)
    assert all(isinstance(note, ReductionNote) for note in reduced.notes)
    assert reduced.notes == tuple(
        note for reduced_chunk in reduced.chunks for note in reduced_chunk.notes
    )


def test_a_reduction_reason_is_its_wire_string():
    assert ReductionReason.COMMAND == "command"
    assert ReductionReason.SYSTEM_INJECTED == "system-injected"


def test_the_corpus_covers_both_reduction_reasons():
    # Guards the expectation table: a table that lost its command or its
    # injected case would still be internally consistent.
    reasons = {reason for notes in EXPECTED_NOTES.values() for _c, _m, reason, _r in notes}

    assert reasons == {str(reason) for reason in ReductionReason}
