"""Feeding parsed chains back into the classifier as candidates (T8.2, #35).

T8.1 (#34) reads a rendered block sequence back into `Chain` objects; T4.1 (#16)
already takes carried-over chains through `classify(candidates=…)`. What is
pinned here is the seam between them, and the one thing that was missing on it:
a chain parsed out of a block has **no user turn in this transcript** — its
chunk ids are placeholders, not ranges — so its candidate used to be offered
with an empty last user message, and an adapter saw a carried intent with
nothing but a label. It now carries its Intent label as that message, because
the label is the only thing such a chain has to say what it is about.

The oracle is hand-written, not computed: `CSV_LABEL` and `LIMITER_LABEL` are
read off the two `## [C…] Intent:` headings of the block that
`tests/fixtures/already_compacted_block.json` carries in `messages[0].content`,
and `EXTENDED` is the chain that fixture's gold note names as the one its single
new turn continues. The assertions are therefore on what the `FakeAdapter`
**recorded** wherever the point is what the adapter was shown: a test that only
looked at the resulting chains would pass with an empty candidate message, which
is the whole of this ticket.

Two cases need a transcript with more than one new turn, and
`already_compacted_block` holds exactly one — so they seed that fixture's block
into another fixture's transcript. That is not a contrivance: a carried-over
chain is by definition not a range over the transcript being classified, which
is why the store resolves nothing from its chunk ids, and a second turn is the
only way to observe a candidate *after* the chain was extended.

No test here calls a model: the decision backend is `FakeAdapter` throughout.
"""

from datetime import UTC, datetime

import pytest

from fixture_corpus import CORPUS
from idc.chunk import chunk_session
from idc.classify import NEW, ChainStore, FakeAdapter, FakeCall, classify, classify_chunks
from idc.parse import parse_chains, placeholder_chunk_id, split_blocks
from idc.schema import Agent, Chain, Chunk, Message, Role, Session

BLOCK_FIXTURE = "already_compacted_block"
"""The fixture whose first message holds an earlier run's two-chain block."""

UNTOUCHED = "C1"
"""The block's first chain — the CSV export, which no new turn is about."""

EXTENDED = "C2"
"""The chain the fixture's one new turn continues, per its gold note."""

CSV_LABEL = "Make the CSV export emit UTC timestamps"
"""`C1`'s Intent heading in that block, by hand."""

LIMITER_LABEL = "Add a per-token rate limiter to the API"
"""`C2`'s Intent heading in that block, by hand."""

GROWN = "two_sequential_intents"
"""A transcript with two new turns, for the cases that need a second decision."""

OPENED = "interleaved_intents"
"""A transcript whose turns are long enough that a label is a *cut* of one.

Its first turn runs past `PROVISIONAL_LABEL_MAX_LENGTH`, so the label of a chain
this run opens differs from the turn it was derived from — which is what makes
"the candidate shows the turn, not the label" an assertion with two outcomes.
"""


def block_text() -> str:
    """Return the compacted block the fixture's transcript carries."""
    content = CORPUS.session(BLOCK_FIXTURE).messages[0].content
    assert content.startswith("## [C"), f"{BLOCK_FIXTURE} no longer opens with a block"
    return content


def parsed() -> list[Chain]:
    """Return the block's chains as T8.1 reads them back."""
    chains = parse_chains(block_text())
    assert [chain.id for chain in chains] == [UNTOUCHED, EXTENDED]
    assert [chain.label for chain in chains] == [CSV_LABEL, LIMITER_LABEL]
    return chains


def session_and_chunks(name: str) -> tuple[Session, list[Chunk]]:
    """Return a fixture's session and its user-boundary chunks."""
    session = CORPUS.session(name)
    return session, chunk_session(session)


def shown(call: FakeCall) -> dict[str, str]:
    """Return the last user message of every candidate of one recorded call."""
    return {candidate.id: candidate.last_user_message for candidate in call.candidates}


# --- the ticket's own test ------------------------------------------------------


def test_the_new_chunk_is_assigned_to_a_parsed_chain_and_marked_extended():
    session, chunks = session_and_chunks(BLOCK_FIXTURE)
    adapter = FakeAdapter([EXTENDED])

    result = classify_chunks(session, chunks, adapter, candidates=parsed())

    assert [chunk.id for chunk in chunks] == ["K1"]
    assert [(chain.id, chain.chunk_ids) for chain in result.chains] == [
        (UNTOUCHED, [placeholder_chunk_id(UNTOUCHED)]),
        (EXTENDED, [placeholder_chunk_id(EXTENDED), "K1"]),
    ]
    assert result.chain_of("K1") == EXTENDED
    assert result.assignments[0].opened is False


# --- what the adapter is actually shown ----------------------------------------


def test_a_parsed_chains_candidate_carries_its_intent_label_as_the_last_message():
    # The assertion is on the record, not on the chains: a parsed chain's label
    # travels anyway, and only the recorded candidates say whether the adapter
    # was given anything to decide the carried intents from.
    session, chunks = session_and_chunks(BLOCK_FIXTURE)
    adapter = FakeAdapter([EXTENDED])

    classify(session, chunks, adapter, candidates=parsed())

    assert [
        (candidate.id, candidate.label, candidate.last_user_message)
        for candidate in adapter.calls[0].candidates
    ] == [
        (UNTOUCHED, CSV_LABEL, CSV_LABEL),
        (EXTENDED, LIMITER_LABEL, LIMITER_LABEL),
    ]


def test_once_a_parsed_chain_holds_a_chunk_of_this_run_its_candidate_shows_that_turn():
    # The case that decides where the rule lives: after the extension the chain
    # *does* have a user turn, and the label it was seeded with must not stand
    # in front of it any more.
    session, chunks = session_and_chunks(GROWN)
    adapter = FakeAdapter([EXTENDED, EXTENDED])

    classify(session, chunks, adapter, candidates=parsed())

    first_turn = session.messages[0].content
    assert shown(adapter.calls[0]) == {UNTOUCHED: CSV_LABEL, EXTENDED: LIMITER_LABEL}
    assert shown(adapter.calls[1]) == {UNTOUCHED: CSV_LABEL, EXTENDED: first_turn}
    assert first_turn != LIMITER_LABEL


def test_a_parsed_chain_extended_by_a_turn_without_text_shows_that_empty_turn():
    # The discriminating case for *where* the rule lives. It is applied when the
    # chain is seeded, so the chain carries the label like any other last user
    # message and `append` overwrites it with whatever the new chunk's turn
    # holds — the empty string included. A candidate built at candidate time as
    # "the last user message, or the label when it is empty" would show the
    # label here, although the chain does hold a turn of this run.
    #
    # The session is built in memory: the corpus states what the three agents
    # send, and a turn the harness recorded without text is not one of them.
    session = Session(
        id="sess-user-turn-without-text",
        agent=Agent.CLAUDE_CODE,
        created_at=datetime(2026, 9, 10, 9, 0, tzinfo=UTC),
        messages=[
            Message(
                id="m01",
                role=Role.USER,
                content="",
                timestamp=datetime(2026, 9, 10, 9, 0, tzinfo=UTC),
            )
        ],
    )
    store = ChainStore(session, parsed())

    store.append(EXTENDED, Chunk(id="K1", start_index=0, end_index=1))

    assert {candidate.id: candidate.last_user_message for candidate in store.candidates()} == {
        UNTOUCHED: CSV_LABEL,
        EXTENDED: "",
    }


# --- chains this run opened ----------------------------------------------------


def test_a_chain_this_run_opened_shows_its_own_user_turn_and_not_its_label():
    session, chunks = session_and_chunks(OPENED)
    adapter = FakeAdapter([NEW, NEW, "C3"])

    classify(session, chunks, adapter, candidates=parsed())

    first_turn = session.messages[0].content
    opened = [
        candidate
        for candidate in adapter.calls[1].candidates
        if candidate.id not in {UNTOUCHED, EXTENDED}
    ]
    assert [(candidate.id, candidate.last_user_message) for candidate in opened] == [
        ("C3", first_turn)
    ]
    assert opened[0].label != first_turn, "the fixture's first turn must be long enough to be cut"


def test_seeding_changes_nothing_about_the_candidates_of_this_runs_own_chains():
    session, chunks = session_and_chunks(OPENED)
    plain = FakeAdapter([NEW, NEW, "C1"])
    seeded = FakeAdapter([NEW, NEW, "C3"])

    classify(session, chunks, plain)
    classify(session, chunks, seeded, candidates=parsed())

    for plain_call, seeded_call in zip(plain.calls, seeded.calls, strict=True):
        own = [
            candidate
            for candidate in seeded_call.candidates
            if candidate.id not in {UNTOUCHED, EXTENDED}
        ]
        assert [(c.label, c.last_user_message) for c in own] == [
            (c.label, c.last_user_message) for c in plain_call.candidates
        ]


def test_a_chain_this_run_opens_is_numbered_past_the_highest_parsed_id():
    session, chunks = session_and_chunks(BLOCK_FIXTURE)

    chains = classify(session, chunks, FakeAdapter([NEW]), candidates=parsed())

    assert [chain.id for chain in chains] == [UNTOUCHED, EXTENDED, "C3"]
    assert (chains[-1].compacted, chains[-1].compacted_text) == (False, None)


# --- what a parsed chain keeps -------------------------------------------------


@pytest.mark.parametrize("script", [[EXTENDED], [NEW]], ids=["extended", "untouched"])
def test_compacted_and_its_text_survive_classification_block_by_block(script):
    # The compactor needs the old block to merge against (#28), so the text has
    # to come out byte for byte, the extended chain included.
    session, chunks = session_and_chunks(BLOCK_FIXTURE)
    blocks = split_blocks(block_text())

    chains = classify(session, chunks, FakeAdapter(script), candidates=parsed())

    assert len(blocks) == 2
    assert [(chain.compacted, chain.compacted_text) for chain in chains[:2]] == [
        (True, blocks[0]),
        (True, blocks[1]),
    ]


def test_the_same_block_and_the_same_script_classify_the_same_way_twice():
    session, chunks = session_and_chunks(BLOCK_FIXTURE)
    adapters = [FakeAdapter([EXTENDED]), FakeAdapter([EXTENDED])]

    runs = [classify(session, chunks, adapter, candidates=parsed()) for adapter in adapters]

    assert runs[0] == runs[1]
    assert adapters[0].calls == adapters[1].calls
