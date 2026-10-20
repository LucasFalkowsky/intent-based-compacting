"""User-boundary chunking: one chunk per user turn, lossless over the transcript.

Two kinds of test live here and they check different things:

- **The expectation table.** `EXPECTED_RANGES` is read off the fixtures by hand,
  not computed, so the per-fixture tests have an oracle the implementation
  cannot agree with by accident.
- **The corpus-wide invariants.** Everything parametrized over `FIXTURE_NAMES`
  is discovered from `tests/fixtures/` at collection time, so a fixture added
  by a later ticket is covered without touching this module. The central one is
  `test_chunk_ranges_partition_the_messages`: a `Chunk` is a half-open range
  over `Session.messages` and copies nothing (T1.2, #3), so chunking is
  lossless and duplicate-free exactly when the ranges *partition* the index
  range of the transcript — contiguous, no gap, no overlap, union equal to the
  whole.

What counts as a user turn is restated locally on purpose: it is the chunker's
own decision, and a test that imported the implementation's predicate would
only check that the code agrees with itself. The corpus describes its structure
the same way in `tests/test_fixture_cases.py`.
"""

import json
from itertools import pairwise

import pytest

from fixture_corpus import CORPUS
from idc.chunk import chunk_session
from idc.schema import Chunk, ChunkOrigin, Message, Role, Session, dumps, loads

FIXTURE_NAMES = CORPUS.names()
"""Every fixture of the shared corpus, discovered at collection time."""

EXPECTED_RANGES: dict[str, list[tuple[int, int]]] = {
    # One user turn at index 0, carried to a green test run.
    "single_intent": [(0, 9)],
    # Two user turns: indices 0 and 10.
    "two_sequential_intents": [(0, 10), (10, 16)],
    # Three user turns: indices 0, 11 and 21 (the A, B, A case).
    "interleaved_intents": [(0, 11), (11, 21), (21, 29)],
    # Three user-role messages, but the two slash commands are not turns; the
    # real turn is index 2, and indices 0-1 precede it.
    "slash_command": [(0, 15)],
    # One turn at index 1; the injected user-role message at index 4 is not a
    # turn, and the injected system message at index 0 precedes the turn.
    "system_injected_message": [(0, 11)],
    # One turn at index 0; the subagent span stays inside it until T3.3.
    "subagent_result": [(0, 17)],
    "failed_then_passing_command": [(0, 14)],
    "write_then_two_edits": [(0, 18)],
    # The rendered summary block at index 0 precedes the single turn at index 1.
    "already_compacted_block": [(0, 12)],
}
"""The chunk ranges each fixture must produce, read off the corpus by hand.

Half-open, `start_index` included and `end_index` excluded, in transcript order.
The table pins the fixtures that exist today; the invariants below are
discovery-driven, so it does not have to list a fixture added later.
"""

CREATED_AT = "2026-10-20T09:00:00+00:00"
"""Fixed instant for the synthetic sessions; no test reads the clock."""


def user_turn_indices(session: Session) -> list[int]:
    """Return the indices of the real user turns of `session`.

    A real user turn is a `user` message the harness did not inject and that is
    not a slash command — restated here instead of imported, see the module
    docstring.
    """
    return [
        index
        for index, message in enumerate(session.messages)
        if message.role is Role.USER and not message.is_system_injected and not message.is_command
    ]


def synthetic_session(messages: list[dict[str, object]]) -> Session:
    """Return a session built from message stubs, validated through `loads`.

    Each stub needs a `role` only; the id and the timestamp are filled in. The
    document goes through `loads` rather than `Session.model_validate`, which
    is the load path every stage uses (#3).
    """
    document = {
        "id": "sess-synthetic",
        "agent": "claude-code",
        "created_at": CREATED_AT,
        "messages": [
            {"id": f"m{index:02d}", "timestamp": CREATED_AT, **message}
            for index, message in enumerate(messages)
        ],
    }
    return loads(Session, json.dumps(document))


def spanned_messages(session: Session, chunk: Chunk) -> list[Message]:
    """Return the messages a chunk's range selects from the session."""
    return session.messages[chunk.start_index : chunk.end_index]


def test_the_corpus_was_discovered():
    # Guards every parametrized test below: an empty corpus would silently turn
    # them into zero test cases instead of into failures.
    assert FIXTURE_NAMES


def test_the_expectation_table_names_existing_fixtures():
    assert set(EXPECTED_RANGES) <= set(FIXTURE_NAMES)


# --- the expectation table -------------------------------------------------


@pytest.mark.parametrize("name", list(EXPECTED_RANGES))
def test_a_fixture_with_n_user_turns_yields_n_chunks(name):
    session = CORPUS.session(name)

    assert len(chunk_session(session)) == len(EXPECTED_RANGES[name])


@pytest.mark.parametrize("name", list(EXPECTED_RANGES))
def test_the_chunk_ranges_are_the_expected_boundaries(name):
    session = CORPUS.session(name)

    ranges = [(chunk.start_index, chunk.end_index) for chunk in chunk_session(session)]

    assert ranges == EXPECTED_RANGES[name]


# --- corpus-wide invariants ------------------------------------------------


@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_chunk_ranges_partition_the_messages(name):
    session = CORPUS.session(name)
    chunks = chunk_session(session)

    covered = [index for chunk in chunks for index in range(chunk.start_index, chunk.end_index)]

    # One assertion for the whole property: the covered indices in the order the
    # chunks list them are exactly 0 … n-1, which holds only if the ranges start
    # at the first message, end at the last, follow each other without a gap,
    # and never cover an index twice.
    assert covered == list(range(len(session.messages)))


@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_chunk_ranges_are_contiguous(name):
    session = CORPUS.session(name)
    chunks = chunk_session(session)

    assert all(earlier.end_index == later.start_index for earlier, later in pairwise(chunks))


@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_the_first_chunk_starts_at_the_first_message(name):
    session = CORPUS.session(name)

    assert chunk_session(session)[0].start_index == 0


@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_the_last_chunk_ends_at_the_last_message(name):
    session = CORPUS.session(name)

    assert chunk_session(session)[-1].end_index == len(session.messages)


@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_the_chunks_reproduce_the_transcript_without_copying_it(name):
    session = CORPUS.session(name)
    chunks = chunk_session(session)

    rebuilt = [message for chunk in chunks for message in spanned_messages(session, chunk)]

    assert len(rebuilt) == len(session.messages)
    # Identity, not equality: the chunks reference the session's own messages,
    # so nothing downstream can read a stale copy of a message.
    assert all(a is b for a, b in zip(rebuilt, session.messages, strict=True))


@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_is_latest_is_set_exactly_once(name):
    session = CORPUS.session(name)

    chunks = chunk_session(session)

    assert [chunk.is_latest for chunk in chunks].count(True) == 1


@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_is_latest_marks_the_last_chunk(name):
    session = CORPUS.session(name)

    chunks = chunk_session(session)

    assert chunks[-1].is_latest is True
    assert all(chunk.is_latest is False for chunk in chunks[:-1])


@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_every_chunk_holds_exactly_one_user_turn(name):
    session = CORPUS.session(name)
    turns = set(user_turn_indices(session))

    for chunk in chunk_session(session):
        covered = turns & set(range(chunk.start_index, chunk.end_index))
        assert len(covered) == 1, f"chunk {chunk.id} covers the user turns {sorted(covered)}"


@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_every_chunk_after_the_first_starts_at_a_user_turn(name):
    session = CORPUS.session(name)
    turns = user_turn_indices(session)

    assert all(chunk.start_index in turns for chunk in chunk_session(session)[1:])


@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_chunk_ids_are_unique_within_a_session(name):
    session = CORPUS.session(name)

    ids = [chunk.id for chunk in chunk_session(session)]

    assert len(set(ids)) == len(ids)


@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_chunks_are_numbered_from_one_in_transcript_order(name):
    session = CORPUS.session(name)

    ids = [chunk.id for chunk in chunk_session(session)]

    assert ids == [f"K{position}" for position in range(1, len(ids) + 1)]


@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_every_chunk_originates_from_a_user_message(name):
    session = CORPUS.session(name)

    assert all(chunk.origin is ChunkOrigin.USER for chunk in chunk_session(session))


@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_chunking_the_same_session_twice_yields_identical_chunks(name):
    session = CORPUS.session(name)

    assert chunk_session(session) == chunk_session(CORPUS.session(name))


@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_every_chunk_is_a_valid_canonical_document(name):
    session = CORPUS.session(name)

    for chunk in chunk_session(session):
        assert loads(Chunk, dumps(chunk)) == chunk


# --- the cases that make the boundary non-trivial --------------------------


def test_slash_commands_do_not_open_a_chunk():
    session = CORPUS.session("slash_command")
    user_role = [index for index, m in enumerate(session.messages) if m.role is Role.USER]

    chunks = chunk_session(session)

    assert len(user_role) == 3, "fixture precondition: three user-role messages"
    assert len(chunks) == 1, "only the one real turn is a boundary"
    assert (chunks[0].start_index, chunks[0].end_index) == (0, len(session.messages))


def test_a_command_inside_a_turn_stays_in_that_turn():
    session = CORPUS.session("slash_command")
    command_inside = next(
        index
        for index, message in enumerate(session.messages)
        if message.is_command and index > user_turn_indices(session)[0]
    )

    chunk = chunk_session(session)[0]

    assert chunk.start_index <= command_inside < chunk.end_index


def test_an_injected_user_message_does_not_open_a_chunk():
    session = CORPUS.session("system_injected_message")
    injected_user = [
        index
        for index, message in enumerate(session.messages)
        if message.role is Role.USER and message.is_system_injected
    ]

    chunks = chunk_session(session)

    assert injected_user, "fixture precondition: an injected message in the user role"
    assert len(chunks) == 1
    assert all(chunks[0].start_index <= index < chunks[0].end_index for index in injected_user)


def test_a_preamble_before_the_first_turn_belongs_to_the_first_chunk():
    # The decision of this ticket: a transcript that does not open with a user
    # message has no chunk of its own for what precedes the first turn — the
    # first chunk starts at index 0 and carries it. `already_compacted_block`
    # opens with a rendered summary block that the parser (T7) has to find
    # again, so dropping it or isolating it is not an option, and `ChunkOrigin`
    # has no member for a chunk that starts anywhere but at a user message.
    session = CORPUS.session("already_compacted_block")
    first_turn = user_turn_indices(session)[0]

    chunks = chunk_session(session)

    assert first_turn == 1, "fixture precondition: a message precedes the first user turn"
    assert session.messages[0].role is Role.SYSTEM
    assert session.messages[0].is_system_injected
    assert len(chunks) == 1
    assert chunks[0].start_index == 0 < first_turn
    assert "[C1]" in session.messages[chunks[0].start_index].content, (
        "the rendered block must stay inside the first chunk, not fall out of the chunking"
    )


@pytest.mark.parametrize("name", list(EXPECTED_RANGES))
def test_a_preamble_is_never_a_chunk_of_its_own(name):
    session = CORPUS.session(name)
    turns = user_turn_indices(session)

    chunks = chunk_session(session)

    assert len(chunks) == len(turns), f"user turns {turns} must yield as many chunks"


# --- degenerate transcripts ------------------------------------------------


def test_a_transcript_without_messages_yields_no_chunks():
    session = synthetic_session([])

    assert chunk_session(session) == []


def test_a_transcript_without_a_user_turn_yields_one_covering_chunk():
    # The same rule as the preamble, taken to its end: everything before the
    # first user turn joins the first chunk, and when there is no turn at all
    # the whole transcript is that chunk. Dropping it would lose messages.
    session = synthetic_session(
        [
            {
                "role": "system",
                "content": "<system-reminder>…</system-reminder>",
                "is_system_injected": True,
            },
            {"role": "user", "content": "/model claude-opus-5", "is_command": True},
            {"role": "assistant", "content": "Model set."},
        ]
    )

    chunks = chunk_session(session)

    assert len(chunks) == 1
    assert (chunks[0].start_index, chunks[0].end_index) == (0, 3)
    assert chunks[0].is_latest is True


def test_a_transcript_that_opens_with_a_user_turn_has_no_preamble():
    session = synthetic_session(
        [
            {"role": "user", "content": "First intent."},
            {"role": "assistant", "content": "Done."},
            {"role": "user", "content": "Second intent."},
        ]
    )

    chunks = chunk_session(session)

    assert [(chunk.start_index, chunk.end_index) for chunk in chunks] == [(0, 2), (2, 3)]


def test_consecutive_user_turns_each_get_their_own_chunk():
    session = synthetic_session(
        [
            {"role": "user", "content": "First intent."},
            {"role": "user", "content": "Actually, this one."},
            {"role": "assistant", "content": "Done."},
        ]
    )

    chunks = chunk_session(session)

    assert [(chunk.start_index, chunk.end_index) for chunk in chunks] == [(0, 1), (1, 3)]
