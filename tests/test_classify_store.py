"""The chain store: ordered open chains, one chunk per chain, ids by position.

The store is the part of T4.1 (#16) that has to be trustworthy without a model
in the loop: it assigns the chain ids, it offers the candidates an adapter sees,
and it is the only place that can let a chunk end up in two chains or in none.
Everything it refuses is pinned here, because a silent repair would be a wrong
classification nobody notices:

- a decision naming a chain the store does not hold (`UnknownChainError`),
- the same chunk assigned twice (`ChunkAlreadyAssignedError`),
- a chunk that is not a range over this session (`ForeignChunkError`) — without
  that guard the last user message of a foreign chunk would silently come out
  empty,
- a chunk appended behind the chain's own chunks in transcript order
  (`ChunkOrderError`), which `Chain.chunk_ids` requires to be ordered.

Chain ids are `C<n>` by **position**, deliberately not content-hashed, so a
re-run over a transcript that only grew keeps the earlier ids — T6.3 (#28) and
T8 (#33) depend on it, and `test_chain_ids_stay_stable_when_the_transcript_grows`
is what pins it.

The expected labels and last user messages are read off the fixtures by hand
(`tests/fixtures/README.md` names the message each one comes from), so the
assertions have an oracle the implementation cannot agree with by accident.
"""

import pytest

from fixture_corpus import CORPUS
from idc.chunk import chunk_session, chunk_with_subagents
from idc.classify import (
    CHAIN_ID_PREFIX,
    PROVISIONAL_LABEL_MAX_LENGTH,
    ChainCandidate,
    ChainStore,
    ChunkAlreadyAssignedError,
    ChunkOrderError,
    ForeignChunkError,
    UnknownChainError,
    chain_id,
    chunk_user_message,
    is_chain_id,
    provisional_label,
)
from idc.schema import Agent, Chain, Chunk, CompactedBlock, Session, dumps, loads

SAMPLER_LABEL = "`tests/rates/test_sampler.py::test_weighted_pick` fails…"
"""The provisional label of `interleaved_intents`' first turn (m01), by hand."""

RENAME_LABEL = "Unrelated: rename the config key `retry_limit` to…"
"""The provisional label of `interleaved_intents`' second turn (m12), by hand."""


def session_and_chunks(name: str) -> tuple[Session, list[Chunk]]:
    session = CORPUS.session(name)
    return session, chunk_session(session)


def is_task_tool(agent: Agent, tool_name: str) -> bool:
    """Stand in for `ToolRegistry.is_subagent_tool` (#6), as T3.3's tests do."""
    return agent is Agent.CLAUDE_CODE and tool_name == "Task"


def test_chain_ids_are_c_n_counted_from_one():
    assert CHAIN_ID_PREFIX == "C"
    assert [chain_id(position) for position in (1, 2, 10)] == ["C1", "C2", "C10"]
    assert is_chain_id("C1")
    assert not is_chain_id("K1")
    assert not is_chain_id("C0")


def test_a_chain_id_needs_a_position_counted_from_one():
    with pytest.raises(ValueError, match="position"):
        chain_id(0)


def test_the_generated_chain_ids_are_what_the_canonical_schema_accepts():
    chain = Chain(id=chain_id(7), label="an intent", chunk_ids=["K1"])

    assert loads(Chain, dumps(chain)) == chain


def test_a_new_store_holds_no_chain_and_offers_no_candidate():
    session, _ = session_and_chunks("single_intent")

    store = ChainStore(session)

    assert len(store) == 0
    assert store.chains() == []
    assert store.candidates() == []
    assert "C1" not in store


def test_opening_a_chain_numbers_it_by_position_and_labels_it_from_the_user_message():
    session, chunks = session_and_chunks("interleaved_intents")
    store = ChainStore(session)

    first = store.open_chain(chunks[0])
    second = store.open_chain(chunks[1])

    assert (first.id, first.label, first.chunk_ids) == ("C1", SAMPLER_LABEL, ["K1"])
    assert (second.id, second.label, second.chunk_ids) == ("C2", RENAME_LABEL, ["K2"])
    assert len(store) == 2
    assert "C2" in store


def test_an_opened_chain_is_not_compacted_yet():
    session, chunks = session_and_chunks("single_intent")

    chain = ChainStore(session).open_chain(chunks[0])

    assert chain.compacted is False
    assert chain.compacted_text is None


def test_an_explicit_label_replaces_the_provisional_one():
    session, chunks = session_and_chunks("single_intent")

    chain = ChainStore(session).open_chain(chunks[0], label="timezone in the CSV export")

    assert chain.label == "timezone in the CSV export"


def test_an_explicit_label_may_not_be_empty():
    session, chunks = session_and_chunks("single_intent")

    with pytest.raises(ValueError, match="label"):
        ChainStore(session).open_chain(chunks[0], label="")


def test_appending_extends_the_chain_in_transcript_order():
    session, chunks = session_and_chunks("interleaved_intents")
    store = ChainStore(session)
    store.open_chain(chunks[0])
    store.open_chain(chunks[1])

    extended = store.append("C1", chunks[2])

    assert extended.chunk_ids == ["K1", "K3"]
    assert [chain.chunk_ids for chain in store.chains()] == [["K1", "K3"], ["K2"]]
    assert len(store) == 2


def test_the_candidates_are_the_open_chains_in_the_order_they_were_opened():
    session, chunks = session_and_chunks("interleaved_intents")
    store = ChainStore(session)
    store.open_chain(chunks[0])
    store.open_chain(chunks[1])

    assert store.candidates() == [
        ChainCandidate(id="C1", label=SAMPLER_LABEL, last_user_message=session.messages[0].content),
        ChainCandidate(id="C2", label=RENAME_LABEL, last_user_message=session.messages[11].content),
    ]


def test_a_candidates_last_user_message_follows_the_chains_last_chunk():
    session, chunks = session_and_chunks("interleaved_intents")
    store = ChainStore(session)
    store.open_chain(chunks[0])

    store.append("C1", chunks[2])

    assert store.candidates()[0].last_user_message == session.messages[21].content


def test_the_candidate_list_is_the_stores_own_copy():
    session, chunks = session_and_chunks("single_intent")
    store = ChainStore(session)
    store.open_chain(chunks[0])

    store.candidates().clear()

    assert len(store.candidates()) == 1


def test_appending_to_a_chain_the_store_does_not_hold_fails_loudly():
    session, chunks = session_and_chunks("two_sequential_intents")
    store = ChainStore(session)
    store.open_chain(chunks[0])

    with pytest.raises(UnknownChainError, match=r"'C7'.*C1") as raised:
        store.append("C7", chunks[1])

    assert raised.value.chain_id == "C7"
    assert raised.value.known == ("C1",)


def test_the_unknown_chain_error_says_so_when_nothing_was_on_offer():
    session, chunks = session_and_chunks("single_intent")

    with pytest.raises(UnknownChainError, match="none"):
        ChainStore(session).append("C1", chunks[0])


def test_a_chunk_belongs_to_exactly_one_chain():
    session, chunks = session_and_chunks("two_sequential_intents")
    store = ChainStore(session)
    store.open_chain(chunks[0])
    store.open_chain(chunks[1])

    with pytest.raises(ChunkAlreadyAssignedError, match=r"'K1'.*'C1'"):
        store.append("C2", chunks[0])


def test_a_chunk_cannot_be_appended_to_its_own_chain_twice():
    session, chunks = session_and_chunks("single_intent")
    store = ChainStore(session)
    store.open_chain(chunks[0])

    with pytest.raises(ChunkAlreadyAssignedError, match="'K1'"):
        store.append("C1", chunks[0])


def test_a_chunk_that_is_not_a_range_over_this_session_fails_loudly():
    session, _ = session_and_chunks("single_intent")
    foreign = Chunk(id="K9", start_index=90, end_index=99)

    with pytest.raises(ForeignChunkError, match="K9"):
        ChainStore(session).open_chain(foreign)


def test_a_chunk_out_of_transcript_order_fails_loudly():
    session, chunks = session_and_chunks("interleaved_intents")
    store = ChainStore(session)
    store.open_chain(chunks[2])

    with pytest.raises(ChunkOrderError, match="K1"):
        store.append("C1", chunks[0])


def test_the_store_knows_which_chain_a_chunk_went_to():
    session, chunks = session_and_chunks("two_sequential_intents")
    store = ChainStore(session)
    store.open_chain(chunks[0])
    store.open_chain(chunks[1])

    assert store.chain_of("K1") == "C1"
    assert store.chain_of("K2") == "C2"
    assert store.chain_of("K3") is None


def test_chain_ids_stay_stable_when_the_transcript_grows():
    session, chunks = session_and_chunks("interleaved_intents")

    before = ChainStore(session)
    for chunk in chunks[:2]:
        before.open_chain(chunk)
    after = ChainStore(session)
    for chunk in chunks:
        after.open_chain(chunk)

    # The third turn only adds `C3`; the ids of the first two chains are the
    # same in both runs, which is what #28 and #33 rely on.
    assert [chain.id for chain in before.chains()] == ["C1", "C2"]
    assert [chain.id for chain in after.chains()] == ["C1", "C2", "C3"]
    assert [chain.label for chain in after.chains()][:2] == [
        chain.label for chain in before.chains()
    ]


def test_the_chains_of_a_store_are_a_valid_compacted_block():
    session, chunks = session_and_chunks("interleaved_intents")
    store = ChainStore(session)
    store.open_chain(chunks[0])
    store.open_chain(chunks[1])
    store.append("C1", chunks[2])

    block = CompactedBlock(
        chains=store.chains(), latest_chunk_id=chunks[-1].id, rendered_for=session.agent
    )

    assert loads(CompactedBlock, dumps(block)) == block


def test_a_store_is_seeded_with_the_chains_of_an_earlier_run():
    session, _ = session_and_chunks("already_compacted_block")
    carried = Chain(
        id="C1",
        label="rate limiter",
        chunk_ids=["K8"],
        compacted=True,
        compacted_text="## C1 — rate limiter\n",
    )

    store = ChainStore(session, [carried])

    assert store.candidates() == [
        ChainCandidate(id="C1", label="rate limiter", last_user_message="")
    ]
    assert store.chains() == [carried]
    assert store.chain_of("K8") == "C1"


def test_a_seeded_chain_keeps_its_summary_when_it_is_extended():
    session, chunks = session_and_chunks("already_compacted_block")
    carried = Chain(
        id="C1",
        label="rate limiter",
        chunk_ids=["K8"],
        compacted=True,
        compacted_text="## C1 — rate limiter\n",
    )
    store = ChainStore(session, [carried])

    extended = store.append("C1", chunks[0])

    assert extended.chunk_ids == ["K8", "K1"]
    assert (extended.compacted, extended.compacted_text) == (True, "## C1 — rate limiter\n")
    assert store.candidates()[0].last_user_message == session.messages[1].content


def test_a_new_chain_is_numbered_past_the_highest_seeded_one():
    session, chunks = session_and_chunks("already_compacted_block")
    seeded = [
        Chain(id="C1", label="an earlier intent", chunk_ids=["K8"]),
        Chain(id="C5", label="another earlier intent", chunk_ids=["K9"]),
    ]

    store = ChainStore(session, seeded)

    assert store.open_chain(chunks[0]).id == "C6"


def test_two_seeded_chains_may_not_share_an_id():
    session, _ = session_and_chunks("single_intent")
    seeded = [
        Chain(id="C1", label="an intent", chunk_ids=["K8"]),
        Chain(id="C1", label="another intent", chunk_ids=["K9"]),
    ]

    with pytest.raises(ValueError, match="C1"):
        ChainStore(session, seeded)


def test_a_chunk_id_a_seeded_chain_already_claims_cannot_be_assigned_again():
    # The chunk ids of a carried-over chain are positions in the transcript of
    # the *earlier* run, so they can collide with this run's ids. That has to be
    # re-mapped before seeding (#33); the store refuses loudly rather than
    # putting one id into two chains.
    session, chunks = session_and_chunks("already_compacted_block")
    carried = Chain(id="C1", label="an earlier intent", chunk_ids=[chunks[0].id])
    store = ChainStore(session, [carried])

    with pytest.raises(ChunkAlreadyAssignedError, match=chunks[0].id):
        store.open_chain(chunks[0])

    assert store.chains() == [carried]


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("single_intent", 0),
        ("two_sequential_intents", 0),
        ("slash_command", 2),
        ("system_injected_message", 1),
    ],
)
def test_the_user_message_of_a_chunk_is_its_real_user_turn(name, expected):
    session, chunks = session_and_chunks(name)

    assert chunk_user_message(session, chunks[0]) == session.messages[expected].content


def test_a_chunk_without_a_user_turn_has_no_user_message():
    session = CORPUS.session("subagent_result")
    chunks = chunk_with_subagents(session, is_task_tool)
    subagent_chunks = [chunk for chunk in chunks if chunk.origin.value == "subagent"]

    assert subagent_chunks, "the fixture must hold a subagent chunk"
    assert all(chunk_user_message(session, chunk) is None for chunk in subagent_chunks)


def test_the_user_message_of_a_foreign_chunk_is_not_silently_empty():
    session, _ = session_and_chunks("single_intent")

    with pytest.raises(ForeignChunkError, match="K9"):
        chunk_user_message(session, Chunk(id="K9", start_index=90, end_index=99))


def test_a_chain_opened_on_a_chunk_without_a_user_turn_falls_back_to_its_position():
    session = CORPUS.session("subagent_result")
    chunks = chunk_with_subagents(session, is_task_tool)
    store = ChainStore(session)
    store.open_chain(chunks[0])

    chain = store.open_chain(next(chunk for chunk in chunks if chunk.origin.value == "subagent"))

    assert chain.label == "intent 2"


def test_a_provisional_label_is_the_user_message_when_it_is_short_enough():
    assert provisional_label("Now document the new option in README.md.", 1) == (
        "Now document the new option in README.md."
    )


def test_a_provisional_label_collapses_whitespace():
    assert provisional_label("  rename\tthe   key\neverywhere  ", 1) == "rename the key everywhere"


def test_a_long_provisional_label_is_cut_at_a_word_boundary():
    label = provisional_label(
        "`tests/rates/test_sampler.py::test_weighted_pick` fails about one run in ten.", 1
    )

    assert label == SAMPLER_LABEL
    assert len(label) <= PROVISIONAL_LABEL_MAX_LENGTH


@pytest.mark.parametrize("text", ["", "   ", "\n\t "])
def test_a_provisional_label_falls_back_to_the_position_without_a_user_message(text):
    assert provisional_label(text, 3) == "intent 3"


def test_a_provisional_label_is_a_label_the_canonical_schema_accepts():
    label = provisional_label("x" * 500, 1)

    assert len(label) <= PROVISIONAL_LABEL_MAX_LENGTH
    assert Chain(id="C1", label=label, chunk_ids=["K1"]).label == label
