"""Subagent extraction: the delegated work becomes a chunk of its own.

A subagent invocation is a tool call the registry marks as `subagent` (T2.1,
#6 — `Task` for Claude Code); the work it delegated is the span of messages
between that call and its result, the sidechain the importer flattened into
transcript order (#4). This stage lifts that span into a chunk with
`origin = subagent`, placed immediately after the chunk that spawned it.

Three kinds of test live here, following `tests/test_chunk_user_boundaries.py`
and `tests/test_chunk_reduction.py`:

- **The expectation table.** `EXPECTED_CHUNKS` is read off the fixtures by
  hand, not computed, so the per-fixture tests have an oracle the
  implementation cannot agree with by accident. It is pinned to the discovered
  corpus in both directions, so a fixture added later fails loudly until its
  chunks have been read off by hand.
- **The corpus-wide invariants.** Everything parametrized over `FIXTURE_NAMES`
  is discovered from `tests/fixtures/` at collection time, so a tenth fixture
  is covered by them without touching their code. They state what inserting a
  chunk must not break — the partition, the no-copy property, `is_latest` set
  exactly once, idempotence — and that the chunks this stage produces still
  compose with the reduction of T3.2 (#13).
- **The synthetic cases**, for the shapes the corpus does not hold: a call
  whose result never arrives, nested and consecutive calls, a call in the
  latest chunk, a span that would swallow a user turn.

Two things are restated locally on purpose. *Which* tool spawns a subagent is
registry data, so the tests pass their own predicate with the same signature as
`ToolRegistry.is_subagent_tool` instead of importing the registry — a test that
imported it would check that the code agrees with its own data source. And what
makes a message reducible is restated for the same reason
`tests/test_chunk_reduction.py` restates it.
"""

import json
from itertools import pairwise

import pytest

from fixture_corpus import CORPUS
from idc.chunk import (
    SubagentSpan,
    chunk_session,
    chunk_with_subagents,
    extract_subagents,
    reduce_chunks,
    reduce_session,
    subagent_spans,
)
from idc.schema import Agent, Chunk, ChunkOrigin, Message, Role, Session, dumps, loads

FIXTURE_NAMES = CORPUS.names()
"""Every fixture of the shared corpus, discovered at collection time."""

SUBAGENT_TOOLS: dict[Agent, frozenset[str]] = {Agent.CLAUDE_CODE: frozenset({"Task"})}
"""Which tool spawns a subagent, per agent — the registry's data, restated.

`subagent = true` sits on Claude Code's `Task` in the tool registry (#6) and on
nothing else today. The tests carry their own copy for the reason the module
docstring gives, and `is_subagent_tool` below has the signature of the registry
method it stands in for.
"""

EXPECTED_CHUNKS: dict[str, list[tuple[int, int, str]]] = {
    # No subagent call at all: the user-boundary chunks of T3.1, unchanged.
    "single_intent": [(0, 9, "user")],
    "two_sequential_intents": [(0, 10, "user"), (10, 16, "user")],
    "interleaved_intents": [(0, 11, "user"), (11, 21, "user"), (21, 29, "user")],
    "slash_command": [(0, 15, "user")],
    "system_injected_message": [(0, 11, "user")],
    "failed_then_passing_command": [(0, 14, "user")],
    "write_then_two_edits": [(0, 18, "user")],
    "already_compacted_block": [(0, 12, "user")],
    # The one fixture with a subagent call: `Task` (c01) is issued by m02 at
    # index 1 and answered by m08 at index 7, so the subagent's span is the
    # messages m03-m08 (indices 2-7) — its work plus the report it returned.
    # The user turn and the delegating message stay in front of it, the main
    # agent's work on the report follows it.
    "subagent_result": [(0, 2, "user"), (2, 8, "subagent"), (8, 17, "user")],
}
"""The chunks each fixture must produce, read off the corpus by hand.

One `(start_index, end_index, origin)` tuple per chunk, half-open and in
transcript order. The table has to name *every* fixture of the corpus: a
fixture added later without its hand-read chunks would quietly leave the
table-driven tests below with one case less, and
`test_the_expectation_table_covers_the_discovered_fixtures` turns that into a
failure instead.
"""

SUBAGENT_FIXTURE = "subagent_result"
"""The fixture the two acceptance criteria of this ticket are stated over."""

CREATED_AT = "2026-10-20T09:00:00+00:00"
"""Fixed instant for the synthetic sessions; no test reads the clock."""


def is_subagent_tool(agent: Agent, name: str) -> bool:
    """Return whether that tool spawns a subagent, as the registry would.

    The signature is `ToolRegistry.is_subagent_tool`'s, so the production call
    is the registry's bound method where this is the tests' stand-in.
    """
    return name in SUBAGENT_TOOLS.get(agent, frozenset())


def nothing_is_a_subagent_tool(agent: Agent, name: str) -> bool:
    """Return `False` for every tool — a transcript with no subagent work."""
    return False


class FakeRegistry:
    """A stand-in with the shape of the tool registry's subagent lookup."""

    def __init__(self, tools: dict[Agent, frozenset[str]]) -> None:
        """Bind the registry to its subagent tools, per agent."""
        self.tools = tools

    def is_subagent_tool(self, agent: Agent | str, name: str) -> bool:
        """Return whether that tool delegates its work to a subagent."""
        return name in self.tools.get(Agent(agent), frozenset())


def synthetic_session(messages: list[dict[str, object]], **session: object) -> Session:
    """Return a session built from message stubs, validated through `loads`.

    Each stub needs a `role` only; the id and the timestamp are filled in.
    Keyword arguments override the session's own fields (`agent`, `source`, …).
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


def delegating(*call_ids: str, content: str = "Delegating.") -> dict[str, object]:
    """Return an assistant stub that issues one `Task` call per id."""
    return {
        "role": "assistant",
        "content": content,
        "tool_calls": [
            {"id": call_id, "name": "Task", "category": "other", "args": {}} for call_id in call_ids
        ],
    }


def reporting(call_id: str, *, output: str = "Report.", **extra: object) -> dict[str, object]:
    """Return a tool stub that answers `call_id` with a report."""
    return {
        "role": "tool",
        "content": "",
        "tool_result": {"call_id": call_id, "output": output},
        **extra,
    }


def chunk_document(
    position: int,
    start_index: int,
    end_index: int,
    *,
    is_latest: bool = False,
    origin: str = "user",
) -> Chunk:
    """Return a hand-made chunk, validated through `loads` like every stage."""
    return loads(
        Chunk,
        json.dumps(
            {
                "id": f"K{position}",
                "start_index": start_index,
                "end_index": end_index,
                "is_latest": is_latest,
                "origin": origin,
            }
        ),
    )


def ranges(chunks: list[Chunk]) -> list[tuple[int, int, str]]:
    """Return the `(start, end, origin)` triples of `chunks`, in order."""
    return [(chunk.start_index, chunk.end_index, str(chunk.origin)) for chunk in chunks]


def spanned_messages(session: Session, chunk: Chunk) -> list[Message]:
    """Return the messages a chunk's range selects from the session."""
    return session.messages[chunk.start_index : chunk.end_index]


def subagent_chunks(chunks: list[Chunk]) -> list[Chunk]:
    """Return the chunks that hold subagent work, in transcript order."""
    return [chunk for chunk in chunks if chunk.origin is ChunkOrigin.SUBAGENT]


def is_user_turn(message: Message) -> bool:
    """Return whether `message` is a real user turn — restated, see T3.1."""
    return message.role is Role.USER and not message.is_system_injected and not message.is_command


def is_reducible(message: Message) -> bool:
    """Return whether reduction may remove `message` — restated, see T3.2."""
    return (
        (message.is_command or message.is_system_injected)
        and not message.tool_calls
        and message.tool_result is None
    )


def flat_notes(session: Session, chunks: list[Chunk]) -> list[tuple[str, str, tuple[str, ...]]]:
    """Return every reduction note of `chunks`, without the chunk it sits on.

    The chunk a note sits on changes when a chunk is inserted in front of it;
    *what* was reduced must not, which is what the tuples compare.
    """
    return [
        (note.message_id, str(note.reason), note.result_from)
        for reduced_chunk in reduce_chunks(session, chunks).chunks
        for note in reduced_chunk.notes
    ]


def test_the_corpus_was_discovered():
    # Guards every parametrized test below: an empty corpus would silently turn
    # them into zero test cases instead of into failures.
    assert FIXTURE_NAMES


def test_the_expectation_table_covers_the_discovered_fixtures():
    # Guards the table-driven tests below — the only oracle in this suite that
    # running the code cannot reproduce. An empty or shrunken table would
    # silently turn them into fewer test cases instead of into failures, so it
    # is pinned to the corpus in both directions.
    assert EXPECTED_CHUNKS, "the hand-read expectation table must not be empty"
    missing = sorted(set(FIXTURE_NAMES) - set(EXPECTED_CHUNKS))
    unknown = sorted(set(EXPECTED_CHUNKS) - set(FIXTURE_NAMES))

    assert missing == [], f"fixtures without hand-read expected chunks: {missing}"
    assert unknown == [], f"expected chunks for fixtures that do not exist: {unknown}"


def test_the_subagent_fixture_is_the_only_one_with_subagent_work():
    # Guards the two tests of the acceptance criteria and the "unchanged"
    # invariants below, which split the corpus along exactly this line.
    with_calls = sorted(
        name for name in FIXTURE_NAMES if subagent_spans(CORPUS.session(name), is_subagent_tool)
    )

    assert with_calls == [SUBAGENT_FIXTURE]


# --- the two acceptance criteria -------------------------------------------


def test_the_subagent_fixture_yields_one_extra_chunk_after_its_parent():
    session = CORPUS.session(SUBAGENT_FIXTURE)
    before = chunk_session(session)
    call_index = next(
        index
        for index, message in enumerate(session.messages)
        if any(call.name == "Task" for call in message.tool_calls)
    )

    chunks = chunk_with_subagents(session, is_subagent_tool)
    lifted = subagent_chunks(chunks)

    assert len(before) == 1, "fixture precondition: one user turn, one chunk before this stage"
    assert len(lifted) == 1, "one subagent invocation, one chunk of its own"
    parent = chunks[chunks.index(lifted[0]) - 1]
    # Immediately after the chunk that spawned it: the parent chunk is the one
    # holding the `Task` call, and the subagent chunk starts where it ends.
    assert parent.start_index <= call_index < parent.end_index
    assert parent.end_index == lifted[0].start_index
    assert parent.origin is ChunkOrigin.USER


def test_the_parent_chunk_no_longer_holds_the_subagent_transcript():
    session = CORPUS.session(SUBAGENT_FIXTURE)
    span = subagent_spans(session, is_subagent_tool)[0]
    delegated = range(span.start_index, span.end_index)

    before = chunk_session(session)[0]
    chunks = chunk_with_subagents(session, is_subagent_tool)
    lifted = subagent_chunks(chunks)[0]
    parent = chunks[chunks.index(lifted) - 1]

    # Both directions, so the test states the change and not just the result:
    # the user-boundary chunk held the whole span, the parent chunk holds none
    # of it, and every one of its messages sits in the subagent chunk instead.
    assert all(index in range(before.start_index, before.end_index) for index in delegated)
    assert not any(index in range(parent.start_index, parent.end_index) for index in delegated)
    assert (lifted.start_index, lifted.end_index) == (span.start_index, span.end_index)


def test_the_subagent_chunk_holds_the_report_the_call_was_answered_with():
    session = CORPUS.session(SUBAGENT_FIXTURE)
    span = subagent_spans(session, is_subagent_tool)[0]

    lifted = subagent_chunks(chunk_with_subagents(session, is_subagent_tool))[0]
    last = spanned_messages(session, lifted)[-1]

    assert last.tool_result is not None
    assert last.tool_result.call_id == span.call_id
    assert "Three sites construct a Decimal from a float" in last.tool_result.output


# --- the expectation table -------------------------------------------------


@pytest.mark.parametrize("name", list(EXPECTED_CHUNKS))
def test_the_chunks_are_the_expected_ranges_and_origins(name):
    session = CORPUS.session(name)

    assert ranges(chunk_with_subagents(session, is_subagent_tool)) == EXPECTED_CHUNKS[name]


@pytest.mark.parametrize("name", list(EXPECTED_CHUNKS))
def test_the_expected_number_of_chunks_is_produced(name):
    session = CORPUS.session(name)

    assert len(chunk_with_subagents(session, is_subagent_tool)) == len(EXPECTED_CHUNKS[name])


# --- corpus-wide invariants ------------------------------------------------


@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_the_chunk_ranges_partition_the_messages(name):
    session = CORPUS.session(name)
    chunks = chunk_with_subagents(session, is_subagent_tool)

    covered = [index for chunk in chunks for index in range(chunk.start_index, chunk.end_index)]

    # One assertion for the whole property, as in T3.1: the covered indices in
    # the order the chunks list them are exactly 0 … n-1, which holds only if
    # the ranges start at the first message, end at the last, follow each other
    # without a gap, and never cover an index twice.
    assert covered == list(range(len(session.messages)))


@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_the_chunk_ranges_are_contiguous(name):
    session = CORPUS.session(name)
    chunks = chunk_with_subagents(session, is_subagent_tool)

    assert all(earlier.end_index == later.start_index for earlier, later in pairwise(chunks))


@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_the_chunks_reproduce_the_transcript_without_copying_it(name):
    session = CORPUS.session(name)
    chunks = chunk_with_subagents(session, is_subagent_tool)

    rebuilt = [message for chunk in chunks for message in spanned_messages(session, chunk)]

    assert len(rebuilt) == len(session.messages)
    # Identity, not equality: inserting a chunk must not copy a message.
    assert all(a is b for a, b in zip(rebuilt, session.messages, strict=True))


@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_is_latest_is_set_exactly_once(name):
    session = CORPUS.session(name)

    chunks = chunk_with_subagents(session, is_subagent_tool)

    assert [chunk.is_latest for chunk in chunks].count(True) == 1


@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_is_latest_marks_the_last_chunk_that_is_not_subagent_work(name):
    session = CORPUS.session(name)

    chunks = chunk_with_subagents(session, is_subagent_tool)
    latest = next(chunk for chunk in chunks if chunk.is_latest)

    assert latest.origin is ChunkOrigin.USER
    assert all(chunk.origin is ChunkOrigin.SUBAGENT for chunk in chunks[chunks.index(latest) + 1 :])


@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_chunks_are_numbered_from_one_in_transcript_order(name):
    session = CORPUS.session(name)

    ids = [chunk.id for chunk in chunk_with_subagents(session, is_subagent_tool)]

    assert ids == [f"K{position}" for position in range(1, len(ids) + 1)]
    assert len(set(ids)) == len(ids)


@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_no_subagent_chunk_swallows_a_real_user_turn(name):
    session = CORPUS.session(name)

    for chunk in subagent_chunks(chunk_with_subagents(session, is_subagent_tool)):
        swallowed = [
            index
            for index, message in enumerate(spanned_messages(session, chunk), chunk.start_index)
            if is_user_turn(message)
        ]
        assert swallowed == [], f"chunk {chunk.id} covers the user turns {swallowed}"


@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_every_subagent_chunk_runs_from_its_call_to_its_result(name):
    session = CORPUS.session(name)
    chunks = chunk_with_subagents(session, is_subagent_tool)

    for chunk in subagent_chunks(chunks):
        call = session.messages[chunk.start_index - 1]
        result = session.messages[chunk.end_index - 1]
        spawned = [c for c in call.tool_calls if is_subagent_tool(session.agent, c.name)]
        assert spawned, f"the message before {chunk.id} must issue the subagent call"
        assert result.tool_result is not None
        assert result.tool_result.call_id in {c.id for c in spawned}


@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_a_transcript_without_subagent_work_is_chunked_as_before(name):
    session = CORPUS.session(name)

    assert chunk_with_subagents(session, nothing_is_a_subagent_tool) == chunk_session(session)


@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_a_subagent_tool_of_another_agent_is_not_detected(name):
    session = CORPUS.session(name)
    foreign = FakeRegistry({Agent.GEMINI_CLI: frozenset({"Task"})})

    assert chunk_with_subagents(session, foreign.is_subagent_tool) == chunk_session(session)


@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_extracting_twice_changes_nothing(name):
    session = CORPUS.session(name)
    chunks = chunk_with_subagents(session, is_subagent_tool)

    assert extract_subagents(session, chunks, is_subagent_tool) == chunks


@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_extracting_the_same_session_twice_yields_identical_chunks(name):
    session = CORPUS.session(name)

    assert chunk_with_subagents(session, is_subagent_tool) == chunk_with_subagents(
        CORPUS.session(name), is_subagent_tool
    )


@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_every_chunk_is_a_valid_canonical_document(name):
    session = CORPUS.session(name)

    for chunk in chunk_with_subagents(session, is_subagent_tool):
        assert loads(Chunk, dumps(chunk)) == chunk


@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_chunk_with_subagents_is_extract_subagents_on_the_sessions_own_chunks(name):
    session = CORPUS.session(name)

    assert chunk_with_subagents(session, is_subagent_tool) == extract_subagents(
        session, chunk_session(session), is_subagent_tool
    )


# --- composing with the reduction of T3.2 ----------------------------------


@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_an_inserted_boundary_never_follows_a_reducible_message(name):
    # Why the reduction cannot lose a command's result text through a boundary
    # this stage inserted: a command's answer is the *run* of reducible
    # injected messages directly behind it, and a run cannot cross a boundary
    # unless the message in front of the boundary belongs to it. Every boundary
    # inserted here sits behind the message that issued the subagent call or
    # behind the one that reported its result, and a message carrying a tool
    # call or a tool result is never reducible.
    session = CORPUS.session(name)
    chunks = chunk_with_subagents(session, is_subagent_tool)
    inserted = {chunk.start_index for chunk in chunks} - {
        chunk.start_index for chunk in chunk_session(session)
    }

    for boundary in inserted:
        preceding = session.messages[boundary - 1]
        assert not is_reducible(preceding), f"boundary {boundary} splits a reducible message off"
        assert preceding.tool_calls or preceding.tool_result is not None


@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_the_reduction_of_the_subagent_chunks_is_the_reduction_of_the_session(name):
    session = CORPUS.session(name)
    chunks = chunk_with_subagents(session, is_subagent_tool)

    reduced = reduce_chunks(session, chunks)

    # Reduction accepts the inserted chunks (it raises unless the chunks
    # partition the session) and reduces exactly what it reduced before: the
    # same transcript, and the same removals — only the chunk a note sits on
    # may differ, which is what `flat_notes` leaves out.
    assert reduced.session == reduce_session(session).session
    assert flat_notes(session, chunks) == flat_notes(session, chunk_session(session))


@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_the_reduced_subagent_chunks_still_partition_the_reduced_transcript(name):
    session = CORPUS.session(name)

    reduced = reduce_chunks(session, chunk_with_subagents(session, is_subagent_tool))

    covered = [
        index
        for reduced_chunk in reduced.chunks
        for index in range(reduced_chunk.chunk.start_index, reduced_chunk.chunk.end_index)
    ]

    assert covered == list(range(len(reduced.session.messages)))


def test_a_command_before_the_report_keeps_its_reduction_across_the_boundary():
    # The closest a transcript gets to the hazard above: a slash command sits
    # inside the subagent's span and the harness answers it only after the
    # report, so the inserted boundary falls between the two. The answer run
    # stops at the report either way — it is a tool result and therefore not
    # reducible — so the command is dropped with the same note as before.
    session = synthetic_session(
        [
            {"role": "user", "content": "Delegate the sweep."},
            delegating("c01"),
            {"role": "assistant", "content": "Searching."},
            {"role": "user", "content": "/cost", "is_command": True},
            reporting("c01"),
            {"role": "user", "content": "Cost: $1.20", "is_system_injected": True},
            {"role": "assistant", "content": "Acting on the report."},
        ]
    )
    chunks = chunk_with_subagents(session, is_subagent_tool)

    reduced = reduce_chunks(session, chunks)

    assert ranges(chunks) == [(0, 2, "user"), (2, 5, "subagent"), (5, 7, "user")]
    assert reduced.session == reduce_session(session).session
    assert flat_notes(session, chunks) == flat_notes(session, chunk_session(session))


def test_a_continuation_of_nothing_but_noise_keeps_the_noise_reduce_session_drops():
    # The limit of the composition, pinned instead of described: the two
    # stages do **not** commute when the continuation piece this stage splits
    # off consists of nothing but reducible harness noise. `reduce_chunks` must
    # not empty a chunk — a `Chunk` spans at least one message — so on that
    # one-message chunk the guard fires, the reminder survives and no note is
    # written, where `reduce_session` sees the same message inside the one
    # chunk the user turn spans and drops it.
    #
    # Nothing is lost and nothing raises: what differs is an injected reminder
    # and the note about it, never a command's result text — that is what
    # `test_an_inserted_boundary_never_follows_a_reducible_message` pins, and
    # it holds structurally. Relaxing the guard to "the only chunk" would make
    # the stages commute, but the guard belongs to #13 and changing it is a
    # ticket of its own.
    session = synthetic_session(
        [
            {"role": "user", "content": "Delegate the sweep."},
            delegating("c01"),
            {"role": "assistant", "content": "Searching."},
            reporting("c01"),
            {"role": "user", "content": "Reminder.", "is_system_injected": True},
        ]
    )
    chunks = chunk_with_subagents(session, is_subagent_tool)

    lifted = reduce_chunks(session, chunks)
    alone = reduce_session(session)

    assert ranges(chunks) == [(0, 2, "user"), (2, 4, "subagent"), (4, 5, "user")]
    # Both sides, so a change to either stage has to confront the difference
    # rather than silently move it.
    assert [message.id for message in lifted.session.messages] == [
        "m00",
        "m01",
        "m02",
        "m03",
        "m04",
    ]
    assert flat_notes(session, chunks) == []
    assert [message.id for message in alone.session.messages] == ["m00", "m01", "m02", "m03"]
    assert flat_notes(session, chunk_session(session)) == [("m04", "system-injected", ())]


# --- the shapes the corpus does not hold -----------------------------------


def test_a_call_whose_result_never_arrives_yields_no_subagent_chunk():
    # Without the result the span has no end: everything behind the call could
    # be the subagent's work or the main agent's, and guessing would move
    # messages into a chunk they may not belong to. The transcript is therefore
    # chunked as it was before this stage.
    session = synthetic_session(
        [
            {"role": "user", "content": "Delegate the sweep."},
            delegating("c01"),
            {"role": "assistant", "content": "Searching."},
            {"role": "assistant", "content": "Still searching."},
        ]
    )

    assert subagent_spans(session, is_subagent_tool) == []
    assert chunk_with_subagents(session, is_subagent_tool) == chunk_session(session)


def test_a_result_before_its_call_is_not_a_span():
    # A transcript can be truncated at the front, so a result whose call comes
    # later in the file answers an earlier call that was cut off (#3). It must
    # not be read as the end of the span that follows it.
    session = synthetic_session(
        [
            {"role": "user", "content": "Delegate the sweep."},
            reporting("c01"),
            delegating("c01", content="Delegating again."),
            {"role": "assistant", "content": "Searching."},
        ]
    )

    assert subagent_spans(session, is_subagent_tool) == []
    assert chunk_with_subagents(session, is_subagent_tool) == chunk_session(session)


def test_a_message_that_answers_its_own_call_is_not_a_span():
    # The schema does not forbid one message both issuing `Task c01` and
    # reporting c01's result (#3), and the result lookup is strict about it:
    # the span would start behind the call and end at the call, which is no
    # span at all. A lookup that accepted the call's own index instead would
    # build a reversed span and make `subagent_spans` raise.
    session = synthetic_session(
        [
            {"role": "user", "content": "Delegate the sweep."},
            reporting(
                "c01",
                output="Report.",
                tool_calls=[{"id": "c01", "name": "Task", "category": "other", "args": {}}],
            ),
            {"role": "assistant", "content": "Acting on the report."},
        ]
    )

    assert subagent_spans(session, is_subagent_tool) == []
    assert chunk_with_subagents(session, is_subagent_tool) == chunk_session(session)


def test_a_nested_call_stays_inside_the_outer_subagent_chunk():
    # A subagent that delegates again: the outer span already holds the inner
    # one, and a second chunk over the same messages would break the partition.
    session = synthetic_session(
        [
            {"role": "user", "content": "Delegate the sweep."},
            delegating("c01"),
            delegating("c02", content="Delegating further."),
            reporting("c02", output="Inner report."),
            reporting("c01", output="Outer report."),
            {"role": "assistant", "content": "Acting on the report."},
        ]
    )

    chunks = chunk_with_subagents(session, is_subagent_tool)

    assert [span.call_id for span in subagent_spans(session, is_subagent_tool)] == ["c01"]
    assert ranges(chunks) == [(0, 2, "user"), (2, 5, "subagent"), (5, 6, "user")]


def test_consecutive_calls_each_get_their_own_chunk():
    session = synthetic_session(
        [
            {"role": "user", "content": "Delegate both sweeps."},
            delegating("c01"),
            reporting("c01", output="First report."),
            delegating("c02", content="Delegating the second."),
            reporting("c02", output="Second report."),
            {"role": "assistant", "content": "Acting on both reports."},
        ]
    )

    chunks = chunk_with_subagents(session, is_subagent_tool)

    assert ranges(chunks) == [
        (0, 2, "user"),
        (2, 3, "subagent"),
        (3, 4, "user"),
        (4, 5, "subagent"),
        (5, 6, "user"),
    ]
    assert [chunk.is_latest for chunk in chunks] == [False, False, False, False, True]


def test_two_calls_in_one_message_yield_the_first_span_only():
    # Two subagents started in parallel: the importer flattens both sidechains
    # into one transcript order, so where the first one's work ends and the
    # second one's begins is not in the transcript. Only the span that is
    # delimited by its own result becomes a chunk.
    session = synthetic_session(
        [
            {"role": "user", "content": "Delegate both sweeps."},
            delegating("c01", "c02"),
            reporting("c01", output="First report."),
            reporting("c02", output="Second report."),
            {"role": "assistant", "content": "Acting on both reports."},
        ]
    )

    chunks = chunk_with_subagents(session, is_subagent_tool)

    assert [span.call_id for span in subagent_spans(session, is_subagent_tool)] == ["c01"]
    assert ranges(chunks) == [(0, 2, "user"), (2, 3, "subagent"), (3, 5, "user")]


def test_a_report_that_carries_the_next_call_leaves_no_empty_chunk_behind():
    # The message reporting the first subagent issues the second call, so the
    # second span starts exactly where the first one ends and there is no main
    # agent message between them to put into a chunk.
    session = synthetic_session(
        [
            {"role": "user", "content": "Delegate the sweep."},
            delegating("c01"),
            reporting(
                "c01",
                output="First report.",
                tool_calls=[{"id": "c02", "name": "Task", "category": "other", "args": {}}],
            ),
            reporting("c02", output="Second report."),
            {"role": "assistant", "content": "Acting on both reports."},
        ]
    )

    chunks = chunk_with_subagents(session, is_subagent_tool)

    assert ranges(chunks) == [
        (0, 2, "user"),
        (2, 3, "subagent"),
        (3, 4, "subagent"),
        (4, 5, "user"),
    ]


def test_a_span_that_would_swallow_a_user_turn_is_not_extracted():
    # The user wrote while the subagent was running, so the span reaches across
    # a chunk boundary. Lifting it would put a user turn into a subagent chunk
    # and merge two user turns into one range; the span is left alone instead.
    session = synthetic_session(
        [
            {"role": "user", "content": "Delegate the sweep."},
            delegating("c01"),
            {"role": "user", "content": "Also check the report module."},
            reporting("c01"),
        ]
    )

    assert subagent_spans(session, is_subagent_tool) == []
    assert chunk_with_subagents(session, is_subagent_tool) == chunk_session(session)


def test_a_call_in_the_latest_chunk_is_lifted_out_of_it():
    session = synthetic_session(
        [
            {"role": "user", "content": "First intent."},
            {"role": "user", "content": "Delegate the sweep."},
            delegating("c01"),
            {"role": "assistant", "content": "Searching."},
            reporting("c01"),
            {"role": "assistant", "content": "Acting on the report."},
        ]
    )

    chunks = chunk_with_subagents(session, is_subagent_tool)

    assert ranges(chunks) == [(0, 1, "user"), (1, 3, "user"), (3, 5, "subagent"), (5, 6, "user")]
    assert [chunk.is_latest for chunk in chunks] == [False, False, False, True]


def test_a_span_at_the_end_of_the_transcript_does_not_become_the_latest_chunk():
    # The report is the last thing in the transcript, so the subagent chunk is
    # the last one — but `is_latest` marks what the renderer appends verbatim,
    # and lifting the subagent's work out is what makes it compactable. The
    # latest chunk is therefore the last one that is not subagent work.
    session = synthetic_session(
        [
            {"role": "user", "content": "Delegate the sweep."},
            delegating("c01"),
            {"role": "assistant", "content": "Searching."},
            reporting("c01"),
        ]
    )

    chunks = chunk_with_subagents(session, is_subagent_tool)

    assert ranges(chunks) == [(0, 2, "user"), (2, 4, "subagent")]
    assert [chunk.is_latest for chunk in chunks] == [True, False]


def test_a_transcript_without_messages_yields_no_chunks():
    session = synthetic_session([])

    assert chunk_with_subagents(session, is_subagent_tool) == []
    assert subagent_spans(session, is_subagent_tool) == []


def test_a_transcript_without_a_user_turn_keeps_its_one_covering_chunk():
    session = synthetic_session(
        [
            {"role": "user", "content": "/compact", "is_command": True},
            delegating("c01"),
            reporting("c01"),
        ]
    )

    chunks = chunk_with_subagents(session, is_subagent_tool)

    assert ranges(chunks) == [(0, 2, "user"), (2, 3, "subagent")]
    assert [chunk.is_latest for chunk in chunks] == [True, False]


# --- the chunk list this stage is given ------------------------------------


def test_a_span_split_by_the_given_chunks_is_not_extracted():
    # `extract_subagents` takes the chunks it is given, so the span may cross
    # one of their boundaries even though no user turn sits inside it. Lifting
    # it would merge two chunks into one range; it is left alone instead.
    session = CORPUS.session(SUBAGENT_FIXTURE)
    given = [
        chunk_document(1, 0, 5),
        chunk_document(2, 5, len(session.messages), is_latest=True),
    ]

    chunks = extract_subagents(session, given, is_subagent_tool)

    assert ranges(chunks) == ranges(given)


def test_a_subagent_chunk_in_the_given_chunks_is_passed_through():
    session = CORPUS.session(SUBAGENT_FIXTURE)
    given = [chunk_document(1, 0, len(session.messages), is_latest=True, origin="subagent")]

    chunks = extract_subagents(session, given, is_subagent_tool)

    # Nothing to lift out of a chunk that already holds subagent work — and
    # with no other chunk to carry it, `is_latest` stays where it was.
    assert ranges(chunks) == ranges(given)
    assert [chunk.is_latest for chunk in chunks] == [True]


@pytest.mark.parametrize(
    "given",
    [
        pytest.param([(0, 5)], id="gap-at-the-end"),
        pytest.param([(1, 17)], id="gap-at-the-front"),
        pytest.param([(0, 5), (6, 17)], id="gap-in-the-middle"),
        pytest.param([(0, 6), (5, 17)], id="overlap"),
        pytest.param([(0, 17), (0, 17)], id="duplicate"),
        pytest.param([], id="no-chunk-at-all"),
    ],
)
def test_chunks_that_do_not_partition_the_session_are_rejected(given):
    session = CORPUS.session(SUBAGENT_FIXTURE)
    chunks = [
        chunk_document(position, start, end) for position, (start, end) in enumerate(given, 1)
    ]

    with pytest.raises(ValueError, match="partition"):
        extract_subagents(session, chunks, is_subagent_tool)


# --- the detection -------------------------------------------------------


def test_the_registry_is_asked_per_agent_and_tool_name():
    session = CORPUS.session(SUBAGENT_FIXTURE)
    asked: list[tuple[Agent, str]] = []

    def recording(agent: Agent, name: str) -> bool:
        asked.append((agent, name))
        return is_subagent_tool(agent, name)

    chunk_with_subagents(session, recording)

    assert asked, "the detection must consult the registry"
    assert {agent for agent, _ in asked} == {Agent.CLAUDE_CODE}
    assert [name for _, name in asked] == [
        call.name for message in session.messages for call in message.tool_calls
    ]


def test_a_registry_object_can_be_passed_as_the_predicate():
    # The production caller passes `tool_registry().is_subagent_tool` (#6);
    # this pins that a bound method of that shape is all this stage needs.
    session = CORPUS.session(SUBAGENT_FIXTURE)
    registry = FakeRegistry({Agent.CLAUDE_CODE: frozenset({"Task"})})

    assert chunk_with_subagents(session, registry.is_subagent_tool) == chunk_with_subagents(
        session, is_subagent_tool
    )


def test_a_span_names_its_call_and_the_messages_it_delegates():
    session = CORPUS.session(SUBAGENT_FIXTURE)

    span = subagent_spans(session, is_subagent_tool)[0]

    assert span == SubagentSpan(call_id="c01", tool_name="Task", call_index=1, result_index=7)
    assert (span.start_index, span.end_index) == (2, 8)


def test_a_span_cannot_end_before_it_starts():
    with pytest.raises(ValueError, match="result_index"):
        SubagentSpan(call_id="c01", tool_name="Task", call_index=4, result_index=4)
