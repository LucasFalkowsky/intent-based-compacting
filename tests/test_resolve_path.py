"""The path of a chain and what the chain resolves to (T5.3, #23).

"Everything not in the solution set is path" (#23) is a *partition*, so the
test that matters is the invariant and not a case list: for every chain of
every fixture, each tool call of the chain is in the solution or in the path,
exactly once, and the two together are the chain's whole call set. The cases
the ticket names — reads, searches, failed execs, superseded edits,
non-verification execs — are then one example each, read off the corpus rather
than written here.

The oracle is the corpus: a chain is built the way the pipeline builds it
(`chunk_session`, then `classify` with the fixture's own gold script), and the
call set and the transcript order a chain is checked against are read from the
fixture's messages through `tests/fixture_corpus.py` — never from
`chain_calls`, the function the partition is built on.

Two cases the corpus cannot hold are made by rewriting one fixture's own
document and loading it through `loads`, the way `tests/test_resolve_writes.py`
and `tests/test_resolve_verification.py` do: an `exec` call that runs no
verification (every `exec` call of every fixture runs pytest), and a chain with
no tool calls at all (every chunk of every fixture carries at least one).
"""

import importlib
import json

import pytest

from fixture_corpus import COMMAND_KEYS as CORPUS_COMMAND_KEYS
from fixture_corpus import CORPUS, tool_calls
from idc.chunk import chunk_session
from idc.classify import FakeAdapter, classify, gold_script
from idc.resolve import (
    ForeignChunkError,
    Resolution,
    ResolveError,
    UnknownChunkError,
    path_call_ids,
    resolve,
    solution_call_ids,
    solution_files,
    solution_verification,
)
from idc.schema import Chain, Chunk, Session, loads

FIXTURE_NAMES = CORPUS.names()
"""Every fixture of the shared corpus, discovered at collection time."""

SINGLE = "single_intent"
"""Two reads, one edit and one passing command — the smallest whole chain."""

REWRITTEN = "rewritten_after_failed_edit"
"""A superseded write and a write the agent saw fail, in one chain."""

THREE_RUNS = "write_then_two_edits"
"""Two failing verification runs and a passing one — two discarded runs."""

FAILED_THEN_PASSING = "failed_then_passing_command"
"""A failing command and the passing rerun of it."""

INTERLEAVED = "interleaved_intents"
"""Two chains in one transcript — one of them opens with a search."""

SUBAGENT = "subagent_result"
"""A chain that opens with a `Task` call, which is category `other`."""


def chains_of(session: Session, name: str) -> tuple[list[Chunk], tuple[Chain, ...]]:
    """Return a session's chunks and chains, built the way the pipeline builds them."""
    chunks = chunk_session(session)
    chains = tuple(classify(session, chunks, FakeAdapter(gold_script(CORPUS.gold(name)))))
    return chunks, chains


def classified(name: str) -> tuple[Session, list[Chunk], tuple[Chain, ...]]:
    """Return a fixture's session, chunks and chains.

    The chains come from `classify` playing the fixture's gold script, so the
    chunk-to-chain mapping is the corpus' own hand-written labelling and not an
    expectation this module invents.
    """
    session = CORPUS.session(name)
    chunks, chains = chains_of(session, name)
    return session, chunks, chains


def only_chain(name: str) -> tuple[Session, list[Chunk], Chain]:
    """Return a single-chain fixture's session, chunks and that one chain."""
    session, chunks, chains = classified(name)
    assert len(chains) == 1, f"{name} is used as a single-chain fixture"
    return session, chunks, chains[0]


def chain_call_ids(session: Session, chunks: list[Chunk], chain: Chain) -> list[str]:
    """Return every tool-call id the chain's chunks carry, read off the corpus.

    The oracle of the completeness test: the chain's chunk ids are resolved
    against the chunk list here, and the calls are read from the fixture's own
    messages through `fixture_corpus.tool_calls` — so the partition is not
    checked against the chain walk it is built on.
    """
    by_id = {chunk.id: chunk for chunk in chunks}
    ids: list[str] = []
    for chunk_id in chain.chunk_ids:
        chunk = by_id[chunk_id]
        ids += [
            call.id for call in tool_calls(session.messages[chunk.start_index : chunk.end_index])
        ]
    return ids


def transcript_positions(session: Session) -> dict[str, int]:
    """Return the position of every tool call of the session, in transcript order."""
    return {call.id: position for position, call in enumerate(tool_calls(session.messages))}


def session_with_command(name: str, call_id: str, command: str) -> Session:
    """Return a fixture with one `exec` call's command rewritten, via the load path.

    Every `exec` call of the corpus runs pytest, so an `exec` call that runs no
    verification is made by rewriting the fixture's own document and loading it
    through `loads` — not by writing a transcript into this file.
    """
    document = json.loads(CORPUS.text(name))
    rewritten = 0
    for message in document["messages"]:
        for call in message.get("tool_calls", []):
            if call["id"] == call_id:
                for key in CORPUS_COMMAND_KEYS:
                    if key in call["args"]:
                        call["args"][key] = command
                        rewritten += 1
    assert rewritten == 1, f"{call_id} must name its command exactly once in {name}"
    return loads(Session, json.dumps(document))


# --- the acceptance criterion: the partition ------------------------------


def test_the_corpus_holds_fixtures_to_partition():
    # The parametrized tests below are a table over the corpus, and an empty
    # table is a skip and not a pass (T5.1's lesson, #21).
    assert FIXTURE_NAMES


@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_the_partition_is_complete_and_disjoint_for_every_fixture(name):
    session, chunks, chains = classified(name)

    for chain in chains:
        resolution = resolve(session, chunks, chain)
        solution = list(resolution.solution_call_ids)
        path = list(resolution.path_call_ids)

        assert sorted(solution + path) == sorted(set(chain_call_ids(session, chunks, chain)))
        assert not set(solution) & set(path), "a call is in the solution or in the path"
        assert len(set(path)) == len(path), "a call is listed once"


@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_the_path_names_calls_of_the_chain_in_transcript_order(name):
    session, chunks, chains = classified(name)
    positions = transcript_positions(session)

    for chain in chains:
        path = path_call_ids(session, chunks, chain)

        assert all(isinstance(call_id, str) for call_id in path)
        assert [positions[call_id] for call_id in path] == sorted(
            positions[call_id] for call_id in path
        )


@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_the_solution_ids_are_the_kept_writes_and_the_kept_run(name):
    session, chunks, chains = classified(name)

    for chain in chains:
        expected = [entry.call_ids for entry in solution_files(session, chunks, chain)]
        flat = [call_id for ids in expected for call_id in ids]
        verification = solution_verification(session, chunks, chain)
        if verification is not None:
            flat.append(verification.run.id)

        assert sorted(solution_call_ids(session, chunks, chain)) == sorted(set(flat))


@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_resolving_the_same_chain_twice_gives_the_same_answer(name):
    session, chunks, chains = classified(name)

    for chain in chains:
        assert resolve(session, chunks, chain) == resolve(session, chunks, chain)


# --- one example per kind of path call ------------------------------------


def test_reads_are_path():
    session, chunks, chain = only_chain(SINGLE)

    resolution = resolve(session, chunks, chain)

    assert resolution.path_call_ids == ("c01", "c02")
    assert resolution.solution_call_ids == ("c03", "c04")


def test_a_search_is_path():
    session, chunks, chains = classified(INTERLEAVED)

    second = resolve(session, chunks, chains[1])

    assert "c06" in second.path_call_ids


def test_a_superseded_write_and_a_failed_write_are_path():
    session, chunks, chain = only_chain(REWRITTEN)

    resolution = resolve(session, chunks, chain)

    assert resolution.path_call_ids == ("c01", "c02", "c03")
    assert resolution.solution_call_ids == ("c04",)


def test_a_discarded_verification_run_is_path():
    session, chunks, chain = only_chain(THREE_RUNS)

    resolution = resolve(session, chunks, chain)

    assert {"c04", "c06"} <= set(resolution.path_call_ids)
    assert resolution.verification is not None
    assert resolution.verification.run.id == "c08"
    assert "c08" not in resolution.path_call_ids


def test_a_command_the_agent_saw_fail_is_path():
    session, chunks, chain = only_chain(FAILED_THEN_PASSING)

    resolution = resolve(session, chunks, chain)

    assert "c03" in resolution.path_call_ids


def test_a_call_of_neither_write_nor_exec_category_is_path():
    session, chunks, chain = only_chain(SUBAGENT)

    resolution = resolve(session, chunks, chain)

    assert "c01" in resolution.path_call_ids


def test_an_exec_call_that_runs_no_verification_is_path():
    session = session_with_command(SINGLE, "c04", "git status")
    chunks, chains = chains_of(session, SINGLE)

    resolution = resolve(session, chunks, chains[0])

    assert resolution.verification is None
    assert "c04" in resolution.path_call_ids


def test_a_file_edited_by_a_shell_command_is_in_no_solution_and_the_call_is_path():
    # The limitation the module docstring records: the shell call names no
    # file, so no `SolutionFile` is about the file it changed, and nothing
    # matches it as a verification either — so it is path, and a consumer that
    # drops the path drops the change with it.
    session = session_with_command(SINGLE, "c04", "sed -i 's/old/new/' src/billing/rounding.py")
    chunks, chains = chains_of(session, SINGLE)

    resolution = resolve(session, chunks, chains[0])

    assert "c04" in resolution.path_call_ids
    assert "src/billing/rounding.py" not in [entry.path for entry in resolution.solution_files]


# --- the stage's entry point ----------------------------------------------


@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_resolve_holds_the_three_fields_the_stage_resolves(name):
    session, chunks, chains = classified(name)

    for chain in chains:
        resolution = resolve(session, chunks, chain)

        assert resolution.solution_files == solution_files(session, chunks, chain)
        assert resolution.verification == solution_verification(session, chunks, chain)
        assert resolution.path_call_ids == path_call_ids(session, chunks, chain)


def test_a_chain_that_made_no_tool_call_resolves_to_nothing():
    # No chunk of the corpus is call-free, so the chunk is built here over a
    # message range of a fixture that carries no tool call.
    session = CORPUS.session(SINGLE)
    quiet = Chunk(id="K1", start_index=0, end_index=1)
    assert not tool_calls(session.messages[0:1]), "the range must carry no tool call"
    chain = Chain(id="C1", label="the one intent", chunk_ids=["K1"])

    resolution = resolve(session, [quiet], chain)

    assert resolution == Resolution(solution_files=(), verification=None, path_call_ids=())
    assert resolution.solution_call_ids == ()


def test_the_partition_lists_a_call_two_chunks_both_cover_once():
    # Two chunks whose ranges overlap locate the same call twice. The chunker
    # produces no such list, but the stage is handed its chunks and does not
    # require it — and the sides of the partition are ids, so one call is one
    # entry. The range stops before the fixture's command so that the chain
    # resolves to writes and reads only.
    session = CORPUS.session(SINGLE)
    covered = session.messages[0:6]
    assert [call.id for call in tool_calls(covered)] == ["c01", "c02", "c03"]
    overlapping = [
        Chunk(id="K1", start_index=0, end_index=6),
        Chunk(id="K2", start_index=0, end_index=6),
    ]
    chain = Chain(id="C1", label="the one intent", chunk_ids=["K1", "K2"])

    resolution = resolve(session, overlapping, chain)
    path = resolution.path_call_ids

    assert len(set(path)) == len(path)
    assert not set(path) & set(resolution.solution_call_ids)
    assert sorted(set(path) | set(resolution.solution_call_ids)) == ["c01", "c02", "c03"]


def test_a_resolution_that_counts_a_call_twice_is_a_programming_error():
    session, chunks, chain = only_chain(SINGLE)
    resolution = resolve(session, chunks, chain)

    with pytest.raises(ValueError, match="c03"):
        Resolution(
            solution_files=resolution.solution_files,
            verification=resolution.verification,
            path_call_ids=(*resolution.path_call_ids, "c03"),
        )
    with pytest.raises(ValueError, match="c01"):
        Resolution(
            solution_files=resolution.solution_files,
            verification=resolution.verification,
            path_call_ids=(*resolution.path_call_ids, "c01"),
        )


def test_the_refusals_of_the_entry_point_are_the_stages_own():
    session, chunks, _chain = only_chain(SINGLE)
    unknown = Chain(id="C9", label="a carried-over chain", chunk_ids=["K404"])
    too_long = Chunk(id="K1", start_index=0, end_index=len(session.messages) + 1)

    with pytest.raises(UnknownChunkError) as missing:
        resolve(session, chunks, unknown)
    with pytest.raises(ForeignChunkError) as foreign:
        resolve(session, [too_long], Chain(id="C1", label="one intent", chunk_ids=["K1"]))

    assert missing.value.chunk_id == "K404"
    assert foreign.value.chunk_id == "K1"
    assert isinstance(missing.value, ResolveError)
    assert isinstance(foreign.value, ResolveError)


def test_the_path_holds_no_content_of_the_calls_it_names():
    # The parent's second criterion (#20): a tool result is never summarized,
    # so the path side names calls and carries nothing of what they said.
    session, chunks, chain = only_chain(FAILED_THEN_PASSING)
    outputs = [
        message.tool_result.output
        for message in session.messages
        if message.tool_result is not None
    ]
    assert outputs, "the fixture must hold tool results to look for"

    path = resolve(session, chunks, chain).path_call_ids

    assert set(path) <= {call.id for call in tool_calls(session.messages)}
    assert not [entry for entry in path if any(entry in output for output in outputs)]


def test_the_module_documents_the_limitation_of_the_partition():
    documentation = importlib.import_module("idc.resolve.resolution").__doc__ or ""

    assert "sed" in documentation
    assert "not detected" in documentation
