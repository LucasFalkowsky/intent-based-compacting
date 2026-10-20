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
    ChunkUnderDecision,
    ClassifyError,
    ForeignChunkError,
    UnknownChainError,
    chain_id,
    chunk_user_message,
    is_chain_id,
    provisional_label,
)
from idc.classify import store as store_module
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


@pytest.mark.parametrize("label", ["", "   ", "\n\t "])
def test_an_explicit_label_must_hold_a_word(label):
    # The *store's* guard, not the schema's: `Chain(label="")` raises a
    # `SchemaValidationError` — itself a `ValueError` whose message contains
    # "label" — so the match has to name the sentence only this guard writes.
    # A blank label is refused for the same reason an empty one is: it is the
    # absence `provisional_label` falls back on, and `"   "` would otherwise
    # become a chain's name because the schema accepts any non-empty string.
    session, chunks = session_and_chunks("single_intent")

    with pytest.raises(ValueError, match="omit it for a provisional one"):
        ChainStore(session).open_chain(chunks[0], label=label)


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


def test_the_subject_of_a_chunk_is_what_an_adapter_decides_about():
    # The chunk itself carries indices and no text, so the store — the only side
    # of the stage that may read the session — resolves the chunk's own user
    # message and hands it over. Without it an adapter would see less of the
    # chunk it classifies than of the chains it compares it against.
    session, chunks = session_and_chunks("interleaved_intents")

    assert ChainStore(session).subject(chunks[0]) == ChunkUnderDecision(
        id="K1", user_message=session.messages[0].content
    )


def test_the_subject_of_a_chunk_without_a_user_turn_carries_no_text():
    session = CORPUS.session("subagent_result")
    chunks = chunk_with_subagents(session, is_task_tool)
    subagent = next(chunk for chunk in chunks if chunk.origin.value == "subagent")

    assert ChainStore(session).subject(subagent) == ChunkUnderDecision(id=subagent.id)


def test_the_subject_of_a_foreign_chunk_is_not_silently_empty():
    session, _ = session_and_chunks("single_intent")

    with pytest.raises(ForeignChunkError, match="K9"):
        ChainStore(session).subject(Chunk(id="K9", start_index=90, end_index=99))


def test_a_chunks_user_message_is_read_once_per_chunk(monkeypatch):
    # The store resolves the message to build the subject and needs the same
    # string to label the chain, so it reads the transcript once and not twice:
    # the subject and the provisional label are the same resolution.
    session, chunks = session_and_chunks("interleaved_intents")
    read: list[str] = []
    resolve = store_module.chunk_user_message

    def counted(session_argument: Session, chunk_argument: Chunk) -> str | None:
        read.append(chunk_argument.id)
        return resolve(session_argument, chunk_argument)

    monkeypatch.setattr(store_module, "chunk_user_message", counted)
    store = ChainStore(session)

    store.subject(chunks[0])
    store.open_chain(chunks[0])
    store.subject(chunks[2])
    store.append("C1", chunks[2])

    assert read == ["K1", "K3"]
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


def test_seeded_chains_must_arrive_in_the_order_they_were_opened():
    # "Ordered by opening" is what `candidates()` and `chains()` promise, and a
    # chain id *is* its opening position — so a caller handing them over in
    # another order states two contradicting orders. The store refuses instead
    # of silently offering `C3` before `C1` — and it refuses as a *failure of
    # the stage*, because these chains were parsed out of a compacted block in a
    # transcript file (#33) and not constructed by a programmer.
    session, _ = session_and_chunks("single_intent")
    seeded = [
        Chain(id="C3", label="the later intent", chunk_ids=["K8"]),
        Chain(id="C1", label="the earlier intent", chunk_ids=["K9"]),
    ]

    with pytest.raises(ClassifyError, match="order") as raised:
        ChainStore(session, seeded)

    assert (raised.value.chain_id, raised.value.previous_chain_id) == ("C1", "C3")


def test_two_seeded_chains_may_not_share_an_id():
    # Untrusted input again, so the same family: a hand-edited summary block may
    # hold the same chain twice, and T10's CLI catches it with the one `except`
    # the stage promises instead of printing a traceback.
    session, _ = session_and_chunks("single_intent")
    seeded = [
        Chain(id="C1", label="an intent", chunk_ids=["K8"]),
        Chain(id="C1", label="another intent", chunk_ids=["K9"]),
    ]

    with pytest.raises(ClassifyError, match="C1") as raised:
        ChainStore(session, seeded)

    assert raised.value.chain_id == "C1"


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


def test_the_already_assigned_message_names_a_carried_over_chain_as_the_cause():
    # The normal case for #33, because both runs number their chunks
    # positionally: the clash is a stale id, not a chunk in two chains, and the
    # message has to say which of the two it is looking at.
    session, chunks = session_and_chunks("already_compacted_block")
    carried = Chain(id="C1", label="an earlier intent", chunk_ids=[chunks[0].id])
    store = ChainStore(session, [carried])

    with pytest.raises(ChunkAlreadyAssignedError, match="carried over from an earlier run") as e:
        store.open_chain(chunks[0])

    assert e.value.carried_over is True


def test_the_already_assigned_message_states_the_partition_for_this_runs_chains():
    session, chunks = session_and_chunks("two_sequential_intents")
    store = ChainStore(session)
    store.open_chain(chunks[0])
    store.open_chain(chunks[1])

    with pytest.raises(ChunkAlreadyAssignedError, match="exactly one chain") as e:
        store.append("C2", chunks[0])

    assert e.value.carried_over is False


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


def test_the_user_message_of_a_chunk_holding_two_user_turns_is_the_last_one():
    # The chunker never builds such a chunk — it splits *at* every real user
    # turn, so a chunk holds at most one (T3.1). `chunk_user_message` is public
    # and takes any range, though, and it is documented to return the chunk's
    # *last* user turn: a forward scan would answer with the first.
    session, _ = session_and_chunks("two_sequential_intents")
    both = Chunk(id="K1", start_index=0, end_index=len(session.messages))

    assert chunk_user_message(session, both) == session.messages[10].content


def test_a_chunk_that_fits_resolves_against_this_session_whatever_it_came_from():
    # The documented limit of the guard, not an oversight: a `Chunk` carries no
    # session id (#3), so "is this a chunk of this transcript" is only answerable
    # as "does its range fit". A chunk of another run that happens to fit
    # resolves against *this* session's messages, and only the caller can
    # prevent that.
    session, _ = session_and_chunks("interleaved_intents")
    fits = Chunk(id="K7", start_index=0, end_index=1)

    assert chunk_user_message(session, fits) == session.messages[0].content


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


def test_a_cut_provisional_label_drops_the_punctuation_before_the_ellipsis():
    # `write_then_two_edits`' own turn, whose word boundary falls right after a
    # colon: without the trim the label would read "with tests:…".
    label = provisional_label(
        "Add a `slugify` helper in `src/text/slug.py` with tests: "
        "collapse whitespace to single hyphens.",
        1,
    )

    assert label == "Add a `slugify` helper in `src/text/slug.py` with tests…"


@pytest.mark.parametrize("text", ["", "   ", "\n\t "])
def test_a_provisional_label_falls_back_to_the_position_without_a_user_message(text):
    assert provisional_label(text, 3) == "intent 3"


def test_a_provisional_label_is_a_label_the_canonical_schema_accepts():
    label = provisional_label("x" * 500, 1)

    assert len(label) <= PROVISIONAL_LABEL_MAX_LENGTH
    assert Chain(id="C1", label=label, chunk_ids=["K1"]).label == label
