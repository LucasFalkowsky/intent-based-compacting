"""Sequential assignment: scripted decisions reproduce the expected chains.

This is the test the ticket asks for (#16). Three kinds of test live here,
following `tests/test_chunk_subagents.py`:

- **The expectation table.** `EXPECTED_CHAINS` holds, per fixture, the scripted
  decision list and the chains it must produce — chain id, chunk ids and
  provisional label, all read off the fixtures by hand, so the per-fixture tests
  have an oracle the implementation cannot agree with by accident. Unlike the
  tables of T3 it covers three fixtures rather than the whole corpus: a hand-made
  script only means something for a transcript whose intents are known, and the
  rest of the corpus is covered by the generated scripts below.
- **The corpus-wide invariants.** Everything parametrized over `FIXTURE_NAMES`
  and `SCRIPT_BUILDERS` is discovered from `tests/fixtures/` at collection time,
  so a tenth fixture is covered without touching their code. They state what no
  adapter may be able to break: every chunk lands in exactly one chain, the
  chains are numbered `C1…Ck` in the order they were opened, their chunk ids
  stay in transcript order, the same script yields the same result twice, and
  the chains are a valid canonical `CompactedBlock`.
- **The loud failures**, driven through the loop rather than against the store:
  a script that ran out, a decision naming a chain that was never a candidate,
  and an adapter that returns something that is not a decision.

`interleaved_intents` is the discriminating case and the reason chains exist
rather than a running counter: its turns are A, B, A, so the third turn has to
rejoin **C1** instead of opening a third chain — and the test pins both, the
rejoining and the fact that a different script over the same transcript would
have opened three chains. Nothing here calls a model: the fake adapter is the
whole decision backend, which is what keeps the end-to-end test (#42)
network-free.
"""

import pytest

from fixture_corpus import CORPUS
from idc.chunk import chunk_session, chunk_with_subagents
from idc.classify import (
    NEW,
    PROVISIONAL_LABEL_MAX_LENGTH,
    AdapterContractError,
    Assignment,
    ChainCandidate,
    Decision,
    FakeAdapter,
    ScriptExhaustedError,
    UnknownChainError,
    classify,
    classify_chunks,
)
from idc.schema import Agent, Chain, Chunk, CompactedBlock, Session, dumps, loads

FIXTURE_NAMES = CORPUS.names()
"""Every fixture of the shared corpus, discovered at collection time."""

EXPECTED_CHAINS: dict[str, dict[str, object]] = {
    # One intent, one chunk, one chain: the minimal case.
    "single_intent": {
        "script": [NEW],
        "chains": [("C1", ["K1"], "The CSV export of the billing report writes naive…")],
    },
    # Two unrelated intents over disjoint file sets: the second turn opens a
    # chain of its own and the first one is not extended.
    "two_sequential_intents": {
        "script": [NEW, NEW],
        "chains": [
            ("C1", ["K1"], "Add a `--format json` option to the `report` command. The…"),
            ("C2", ["K2"], "Now document the new option in README.md, nothing else."),
        ],
    },
    # A, B, A — the third turn returns to the first intent, so it is appended to
    # `C1` and the transcript has two chains, not three.
    "interleaved_intents": {
        "script": [NEW, NEW, "C1"],
        "chains": [
            ("C1", ["K1", "K3"], "`tests/rates/test_sampler.py::test_weighted_pick` fails…"),
            ("C2", ["K2"], "Unrelated: rename the config key `retry_limit` to…"),
        ],
    },
}
"""Per fixture: the scripted decisions and the chains they must produce."""


def all_new(count: int) -> list[object]:
    """Return a script that opens a chain for every chunk."""
    return [NEW] * count


def all_one(count: int) -> list[object]:
    """Return a script that puts every chunk into the first chain."""
    return [NEW, *["C1"] * (count - 1)]


def alternating(count: int) -> list[object]:
    """Return a script that opens two chains and then alternates between them."""
    return [NEW if position < 2 else f"C{position % 2 + 1}" for position in range(count)]


SCRIPT_BUILDERS = (all_new, all_one, alternating)
"""Generated scripts, so the corpus-wide invariants need no hand-written one."""


def is_task_tool(agent: Agent, tool_name: str) -> bool:
    """Stand in for `ToolRegistry.is_subagent_tool` (#6), as T3.3's tests do."""
    return agent is Agent.CLAUDE_CODE and tool_name == "Task"


def session_and_chunks(name: str) -> tuple[Session, list[Chunk]]:
    session = CORPUS.session(name)
    return session, chunk_session(session)


class RawAdapter:
    """An adapter that answers with a bare chain id instead of a `Decision`.

    The interface tolerates that form (`as_decision` reads it), and this adapter
    is what pins the tolerance through the loop.
    """

    def decide(self, chunk: Chunk, candidates: list[ChainCandidate]) -> str:
        """Return the id of the first candidate, or `NEW` when there is none."""
        return candidates[0].id if candidates else NEW


class SilentAdapter:
    """An adapter that decides nothing — the contract violation to catch."""

    def decide(self, chunk: Chunk, candidates: list[ChainCandidate]) -> None:
        """Return `None`, which is not a decision."""
        return None


def test_the_expectation_table_names_only_fixtures_the_corpus_holds():
    # Pins the table to the discovered corpus the way T3's tables do: a renamed
    # or removed fixture fails here instead of silently losing a case.
    assert set(EXPECTED_CHAINS) <= set(FIXTURE_NAMES)
    assert set(EXPECTED_CHAINS) == {
        "single_intent",
        "two_sequential_intents",
        "interleaved_intents",
    }


@pytest.mark.parametrize("name", list(EXPECTED_CHAINS))
def test_scripted_decisions_reproduce_the_expected_chains(name):
    session, chunks = session_and_chunks(name)
    expected = EXPECTED_CHAINS[name]

    chains = classify(session, chunks, FakeAdapter(expected["script"]))

    assert [(chain.id, chain.chunk_ids, chain.label) for chain in chains] == expected["chains"]
    assert all(isinstance(chain, Chain) for chain in chains)
    assert all(chain.compacted is False for chain in chains)


def test_the_interleaved_third_turn_rejoins_the_first_chain():
    session, chunks = session_and_chunks("interleaved_intents")

    chains = classify(session, chunks, FakeAdapter([NEW, NEW, "C1"]))

    assert len(chunks) == 3
    assert [chain.id for chain in chains] == ["C1", "C2"]
    assert chains[0].chunk_ids == ["K1", "K3"]


def test_a_third_chain_is_opened_only_when_the_adapter_says_so():
    # The counterpart of the test above: the transcript does not decide this,
    # the adapter does — which is why the chain store may not infer anything.
    session, chunks = session_and_chunks("interleaved_intents")

    chains = classify(session, chunks, FakeAdapter([NEW, NEW, NEW]))

    assert [(chain.id, chain.chunk_ids) for chain in chains] == [
        ("C1", ["K1"]),
        ("C2", ["K2"]),
        ("C3", ["K3"]),
    ]


def test_the_candidates_offered_are_the_chains_open_at_that_moment():
    session, chunks = session_and_chunks("interleaved_intents")
    adapter = FakeAdapter([NEW, NEW, "C1"])

    classify(session, chunks, adapter)

    assert [call.chunk_id for call in adapter.calls] == ["K1", "K2", "K3"]
    assert [call.candidate_ids for call in adapter.calls] == [(), ("C1",), ("C1", "C2")]
    assert adapter.calls[2].candidate_ids == ("C1", "C2")


def test_a_candidate_carries_the_last_user_message_of_its_chain():
    session, chunks = session_and_chunks("interleaved_intents")
    adapter = FakeAdapter([NEW, NEW, "C1"])

    classify(session, chunks, adapter)

    assert adapter.calls[2].candidate_ids == ("C1", "C2")
    offered = {candidate.id: candidate for candidate in adapter.calls[2].candidates}
    assert offered["C1"].last_user_message == session.messages[0].content
    assert offered["C2"].last_user_message == session.messages[11].content


def test_the_assignments_record_what_was_decided_per_chunk():
    session, chunks = session_and_chunks("interleaved_intents")
    script = [Decision.new(confidence=0.9), Decision.new(), Decision.extend("C1", confidence=0.42)]

    result = classify_chunks(session, chunks, FakeAdapter(script))

    assert result.assignments == (
        Assignment(chunk_id="K1", chain_id="C1", opened=True, confidence=0.9),
        Assignment(chunk_id="K2", chain_id="C2", opened=True, confidence=None),
        Assignment(chunk_id="K3", chain_id="C1", opened=False, confidence=0.42),
    )
    assert result.chain_of("K3") == "C1"
    assert result.chain_of("K9") is None
    assert [chain.id for chain in result.chains] == ["C1", "C2"]


def test_classification_is_deterministic_for_the_same_script():
    session, chunks = session_and_chunks("interleaved_intents")

    first = classify_chunks(session, chunks, FakeAdapter([NEW, NEW, "C1"]))
    second = classify_chunks(session, chunks, FakeAdapter([NEW, NEW, "C1"]))

    assert first == second
    assert [dumps(chain) for chain in first.chains] == [dumps(chain) for chain in second.chains]


def test_a_bare_chain_id_from_an_adapter_is_read_as_a_decision():
    session, chunks = session_and_chunks("two_sequential_intents")

    chains = classify(session, chunks, RawAdapter())

    assert [(chain.id, chain.chunk_ids) for chain in chains] == [("C1", ["K1", "K2"])]


def test_an_adapter_that_returns_something_else_fails_loudly():
    session, chunks = session_and_chunks("single_intent")

    with pytest.raises(AdapterContractError, match="Decision"):
        classify(session, chunks, SilentAdapter())


def test_a_script_that_runs_out_fails_loudly():
    session, chunks = session_and_chunks("interleaved_intents")

    with pytest.raises(ScriptExhaustedError, match="K3"):
        classify(session, chunks, FakeAdapter([NEW, NEW]))


def test_a_decision_naming_a_chain_that_was_never_a_candidate_fails_loudly():
    session, chunks = session_and_chunks("two_sequential_intents")

    with pytest.raises(UnknownChainError, match="'C4'"):
        classify(session, chunks, FakeAdapter([NEW, "C4"]))


def test_a_transcript_without_a_chunk_yields_no_chain():
    session = CORPUS.session("single_intent")
    adapter = FakeAdapter([])

    assert classify(session, [], adapter) == []
    assert adapter.calls == ()


def test_chains_carried_over_from_an_earlier_run_are_candidates_again():
    session, chunks = session_and_chunks("already_compacted_block")
    carried = Chain(
        id="C1",
        label="Make the CSV export emit UTC timestamps",
        chunk_ids=["K8"],
        compacted=True,
        compacted_text="## [C1] Intent: Make the CSV export emit UTC timestamps\n",
    )
    adapter = FakeAdapter(["C1"])

    chains = classify(session, chunks, adapter, candidates=[carried])

    assert adapter.calls[0].candidate_ids == ("C1",)
    assert adapter.calls[0].candidates[0].label == "Make the CSV export emit UTC timestamps"
    assert [(chain.id, chain.chunk_ids, chain.compacted) for chain in chains] == [
        ("C1", ["K8", "K1"], True)
    ]


def test_the_latest_chunk_is_classified_like_every_other_chunk():
    # It is rendered verbatim instead of compacted (T7), but it still belongs to
    # an intent, so the classifier must not skip it.
    session, chunks = session_and_chunks("two_sequential_intents")

    result = classify_chunks(session, chunks, FakeAdapter([NEW, "C1"]))

    assert chunks[-1].is_latest is True
    assert result.chain_of(chunks[-1].id) == "C1"


def test_a_subagent_chunk_is_classified_like_every_other_chunk():
    session = CORPUS.session("subagent_result")
    chunks = chunk_with_subagents(session, is_task_tool)

    result = classify_chunks(session, chunks, FakeAdapter(all_one(len(chunks))))

    assert len(chunks) > 1
    assert [chain.chunk_ids for chain in result.chains] == [[chunk.id for chunk in chunks]]


@pytest.mark.parametrize("name", FIXTURE_NAMES)
@pytest.mark.parametrize("builder", SCRIPT_BUILDERS, ids=lambda build: build.__name__)
def test_every_chunk_lands_in_exactly_one_chain(name, builder):
    session, chunks = session_and_chunks(name)

    result = classify_chunks(session, chunks, FakeAdapter(builder(len(chunks))))

    assigned = [chunk_id for chain in result.chains for chunk_id in chain.chunk_ids]
    assert sorted(assigned) == sorted(chunk.id for chunk in chunks)
    assert len(assigned) == len(set(assigned))
    assert [assignment.chunk_id for assignment in result.assignments] == [
        chunk.id for chunk in chunks
    ]


@pytest.mark.parametrize("name", FIXTURE_NAMES)
@pytest.mark.parametrize("builder", SCRIPT_BUILDERS, ids=lambda build: build.__name__)
def test_the_chains_are_numbered_by_position_and_ordered_by_their_first_chunk(name, builder):
    session, chunks = session_and_chunks(name)
    positions = {chunk.id: index for index, chunk in enumerate(chunks)}

    result = classify_chunks(session, chunks, FakeAdapter(builder(len(chunks))))

    assert [chain.id for chain in result.chains] == [
        f"C{position}" for position in range(1, len(result.chains) + 1)
    ]
    assert [positions[chain.chunk_ids[0]] for chain in result.chains] == sorted(
        positions[chain.chunk_ids[0]] for chain in result.chains
    )
    for chain in result.chains:
        indices = [positions[chunk_id] for chunk_id in chain.chunk_ids]
        assert indices == sorted(indices), chain.id


@pytest.mark.parametrize("name", FIXTURE_NAMES)
@pytest.mark.parametrize("builder", SCRIPT_BUILDERS, ids=lambda build: build.__name__)
def test_every_chain_carries_a_usable_provisional_label(name, builder):
    session, chunks = session_and_chunks(name)

    result = classify_chunks(session, chunks, FakeAdapter(builder(len(chunks))))

    for chain in result.chains:
        assert chain.label
        assert "\n" not in chain.label
        assert len(chain.label) <= PROVISIONAL_LABEL_MAX_LENGTH


@pytest.mark.parametrize("name", FIXTURE_NAMES)
@pytest.mark.parametrize("builder", SCRIPT_BUILDERS, ids=lambda build: build.__name__)
def test_the_chains_round_trip_as_a_canonical_compacted_block(name, builder):
    session, chunks = session_and_chunks(name)

    chains = classify(session, chunks, FakeAdapter(builder(len(chunks))))

    latest = next(chunk for chunk in chunks if chunk.is_latest)
    block = CompactedBlock(chains=chains, latest_chunk_id=latest.id, rendered_for=session.agent)
    assert loads(CompactedBlock, dumps(block)) == block
    assert all(loads(Chain, dumps(chain)) == chain for chain in chains)


@pytest.mark.parametrize("name", FIXTURE_NAMES)
@pytest.mark.parametrize("builder", SCRIPT_BUILDERS, ids=lambda build: build.__name__)
def test_the_same_script_classifies_the_same_way_twice(name, builder):
    session, chunks = session_and_chunks(name)

    first = classify_chunks(session, chunks, FakeAdapter(builder(len(chunks))))
    second = classify_chunks(session, chunks, FakeAdapter(builder(len(chunks))))

    assert first == second
