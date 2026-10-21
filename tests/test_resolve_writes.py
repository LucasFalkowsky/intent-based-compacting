"""The last write per file of a chain (T5.1, #21).

The oracle is the corpus and nothing else: a chain is built the way the
pipeline builds it — `chunk_session`, then `classify` with the fixture's own
gold script — so no test here states a chain by hand, and no test calls a
model.

Two fixtures carry the ticket's two cases. `write_then_two_edits` has one file
created and then corrected twice, which is the "last full write plus the edits
behind it" case, and a second file written once, which is what makes the
grouping per *file*. `rewritten_after_failed_edit` has the same file written
whole twice with a refused edit between them, which is the "only the second
write" case and the one where a write the agent saw fail is neither the
accepted one nor a superseded one.

The corpus-wide tests at the end are the invariants: every write of a chain
that names a file is kept, superseded or failed exactly once, the entries are
in transcript order, and resolving twice gives the same answer.
"""

import json

import pytest

from fixture_corpus import CORPUS
from idc.chunk import chunk_session
from idc.classify import FakeAdapter, classify, gold_script
from idc.resolve import (
    CONTENT_KEYS,
    FILE_PATH_KEYS,
    ForeignChunkError,
    LocatedCall,
    ResolveError,
    SolutionFile,
    UnknownChunkError,
    WriteKind,
    chain_calls,
    chain_chunks,
    content_of,
    file_of,
    solution_files,
    write_calls,
    write_kind,
)
from idc.schema import Chain, Chunk, Session, ToolCall, ToolCategory, ToolResult, loads

FIXTURE_NAMES = CORPUS.names()
"""Every fixture of the shared corpus, discovered at collection time."""

WRITE_THEN_TWO_EDITS = "write_then_two_edits"
"""The fixture of the first acceptance criterion: one write, then two edits."""

REWRITTEN = "rewritten_after_failed_edit"
"""The fixture of the second one: the same file written whole twice."""


def classified(name: str) -> tuple[Session, list[Chunk], tuple[Chain, ...]]:
    """Return a fixture's session, chunks and chains, built by the pipeline.

    The chains come from `classify` playing the fixture's gold script, so the
    chunk-to-chain mapping is the corpus' own hand-written labelling and not an
    expectation this module invents.
    """
    session = CORPUS.session(name)
    chunks = chunk_session(session)
    chains = tuple(classify(session, chunks, FakeAdapter(gold_script(CORPUS.gold(name)))))
    return session, chunks, chains


def only_chain(name: str) -> tuple[Session, list[Chunk], Chain]:
    """Return a single-chain fixture's session, chunks and that one chain."""
    session, chunks, chains = classified(name)
    assert len(chains) == 1, f"{name} is used as a single-chain fixture"
    return session, chunks, chains[0]


def files_of(name: str) -> dict[str, SolutionFile]:
    """Return the solution files of a single-chain fixture, keyed by path."""
    session, chunks, chain = only_chain(name)
    return {entry.path: entry for entry in solution_files(session, chunks, chain)}


def ids(calls: tuple[LocatedCall, ...]) -> list[str]:
    """Return the tool-call ids of `calls`, in the order they are held."""
    return [located.id for located in calls]


def all_write_calls(session: Session) -> list[ToolCall]:
    """Return every write-category call of a session, in transcript order."""
    return [
        call
        for message in session.messages
        for call in message.tool_calls
        if call.category is ToolCategory.WRITE
    ]


def session_with_renamed_argument(name: str, call_id: str, old: str, new: str) -> Session:
    """Return a fixture with one call's argument key renamed, via the load path.

    The corpus records only the three argument spellings the agents were
    *observed* to send (`tests/fixtures/README.md`), so the one case it cannot
    hold is a call whose file argument is spelled in a fourth way. It is built
    here by renaming a key in the fixture's own document and loading that
    through `loads`, the same path every stage uses — not by writing a
    transcript into this file.
    """
    document = json.loads(CORPUS.text(name))
    renamed = 0
    for message in document["messages"]:
        for call in message["tool_calls"]:
            if call["id"] == call_id:
                call["args"][new] = call["args"].pop(old)
                renamed += 1
    assert renamed == 1, f"{call_id} must occur exactly once in {name}"
    return loads(Session, json.dumps(document))


# --- the two acceptance criteria -------------------------------------------


def test_a_file_written_then_edited_twice_keeps_the_write_and_both_edits():
    entry = files_of(WRITE_THEN_TWO_EDITS)["/repo/src/text/slug.py"]

    assert entry.write is not None
    assert entry.write.id == "c03", "the full write is the accepted one"
    assert ids(entry.edits) == ["c05", "c07"], "both edits follow it, in transcript order"
    assert entry.superseded == (), "nothing precedes the only full write"
    assert entry.failed == ()
    assert ids(entry.kept) == ["c03", "c05", "c07"]


def test_a_file_written_twice_keeps_only_the_second_write():
    entry = files_of(REWRITTEN)["docs/install.md"]

    assert entry.write is not None
    assert entry.write.id == "c04", "the second full write is the accepted one"
    assert ids(entry.superseded) == ["c02"], "the first one is overwritten, so it is path"
    assert entry.edits == (), "the edit between the two writes did not survive it"
    assert ids(entry.kept) == ["c04"]


# --- the grouping ----------------------------------------------------------


def test_each_file_of_a_chain_becomes_an_entry_of_its_own():
    entries = files_of(WRITE_THEN_TWO_EDITS)

    assert list(entries) == ["/repo/tests/text/test_slug.py", "/repo/src/text/slug.py"], (
        "one entry per file, ordered by the first write that touched it"
    )
    assert ids(entries["/repo/tests/text/test_slug.py"].kept) == ["c02"]


def test_a_file_that_was_only_ever_edited_keeps_the_whole_edit_sequence():
    entry = files_of("slash_command")["src/api/orders.py"]

    assert entry.write is None, "the chain never wrote the file whole"
    assert ids(entry.edits) == ["c02", "c04"], "so every edit is kept, in transcript order"
    assert entry.superseded == ()


def test_the_edits_of_one_file_are_collected_across_the_chunks_of_the_chain():
    session, chunks, chains = classified("interleaved_intents")
    resumed = next(chain for chain in chains if chain.id == "C1")
    entries = {entry.path: entry for entry in solution_files(session, chunks, resumed)}

    assert resumed.chunk_ids == ["K1", "K3"], "the A, B, A chain the corpus labels"
    assert ids(entries["src/rates/sampler.py"].edits) == ["c03", "c11"], (
        "the edit of the first turn and the one of the resumed turn, in transcript order"
    )


def test_a_chain_resolves_only_the_writes_of_its_own_chunks():
    session, chunks, chains = classified("interleaved_intents")
    resumed, foreign = chains
    resolved = {
        chain.id: {entry.path for entry in solution_files(session, chunks, chain)}
        for chain in (resumed, foreign)
    }

    assert resolved["C1"] == {"src/rates/sampler.py", "tests/rates/test_sampler.py"}
    assert resolved["C2"] == {"config/settings.toml", "src/config/loader.py"}
    assert resolved["C1"] & resolved["C2"] == set()
    assert resolved["C1"] | resolved["C2"] == {
        path for call in all_write_calls(session) if (path := file_of(call)) is not None
    }


def test_the_order_of_a_chains_chunk_ids_does_not_change_what_is_resolved():
    # `Chain.chunk_ids` is documented to be in transcript order, and the chains
    # the classifier builds are. The resolver orders the chunks by their
    # position anyway, because "the last write" is a statement about the
    # transcript and must not depend on how a chain was assembled.
    session, chunks, chains = classified("interleaved_intents")
    resumed = next(chain for chain in chains if chain.chunk_ids == ["K1", "K3"])
    reversed_chain = Chain(
        id=resumed.id, label=resumed.label, chunk_ids=list(reversed(resumed.chunk_ids))
    )

    assert solution_files(session, chunks, reversed_chain) == solution_files(
        session, chunks, resumed
    )


# --- a write the agent saw fail --------------------------------------------


def test_a_write_the_agent_saw_fail_is_neither_accepted_nor_superseded():
    entry = files_of(REWRITTEN)["docs/install.md"]

    assert ids(entry.failed) == ["c03"], "the refused edit is reported as the failure it was"
    assert "c03" not in ids(entry.kept), "a write that did not happen is no solution"
    assert "c03" not in ids(entry.superseded), "and it overwrote nothing either"


def test_a_failed_write_does_not_hide_the_write_before_it():
    # The refused edit is the last write-category call for the file except for
    # the rewrite behind it; reading it as the accepted one would present an
    # empty change as the solution, and reading it as a superseded one would
    # claim it ever took effect.
    entry = files_of(REWRITTEN)["docs/install.md"]
    whole = [located for located in (entry.write, *entry.superseded) if located is not None]

    assert [located.kind for located in whole] == [WriteKind.FULL, WriteKind.FULL]
    assert [located.failed for located in whole] == [False, False]


# --- the tool results ------------------------------------------------------


def test_a_kept_write_carries_the_result_the_agent_received_verbatim():
    session, chunks, chain = only_chain(WRITE_THEN_TWO_EDITS)
    entry = {item.path: item for item in solution_files(session, chunks, chain)}[
        "/repo/src/text/slug.py"
    ]
    recorded = {
        message.tool_result.call_id: message.tool_result
        for message in session.messages
        if message.tool_result is not None
    }

    assert entry.write is not None
    assert entry.write.result is recorded["c03"], "the result object itself, not a copy of it"
    assert entry.write.result.output == "Wrote /repo/src/text/slug.py (16 lines)"
    assert [located.result.output for located in entry.edits] == [
        "Modified /repo/src/text/slug.py (1 replacement)",
        "Modified /repo/src/text/slug.py (1 replacement)",
    ]


# --- reading one call ------------------------------------------------------


@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_a_write_is_full_exactly_when_it_carries_the_whole_file_content(name):
    session = CORPUS.session(name)

    for call in all_write_calls(session):
        whole = any(isinstance(call.args.get(key), str) for key in CONTENT_KEYS)
        assert (write_kind(call) is WriteKind.FULL) is whole, call.id


def test_the_content_of_a_full_write_is_the_text_it_wrote():
    session = CORPUS.session(WRITE_THEN_TWO_EDITS)
    by_id = {call.id: call for call in all_write_calls(session)}

    assert content_of(by_id["c03"]) == by_id["c03"].args["content"]
    assert content_of(by_id["c03"]).startswith('"""Turn arbitrary text into a URL slug."""')
    assert content_of(by_id["c05"]) is None, "an edit holds a replacement, not a file"


@pytest.mark.parametrize("key", FILE_PATH_KEYS)
def test_every_argument_spelling_the_agents_use_names_the_file(key):
    call = ToolCall(id="c01", name="Write", category=ToolCategory.WRITE, args={key: "src/a.py"})

    assert file_of(call) == "src/a.py"


def test_a_write_that_carries_a_replacement_and_a_content_argument_is_read_as_an_edit():
    # The one shape where the two signals disagree, and the reason
    # `REPLACEMENT_KEYS` is consulted first. No agent of the study sends
    # it, so it is built from the corpus: `c05` is an edit whose `new_string`
    # is renamed to `content`, which leaves the `old_string` that proves it
    # patches. Read as a full write it would supersede the write it patched and
    # the file's real content would be gone from the solution; read as an edit
    # it only keeps one write more than needed. The lossless reading wins.
    session = session_with_renamed_argument(WRITE_THEN_TWO_EDITS, "c05", "new_string", "content")
    chunks = chunk_session(session)
    chain = Chain(id="C1", label="the one intent", chunk_ids=[chunks[0].id])
    patched = next(call for call in all_write_calls(session) if call.id == "c05")
    entry = {item.path: item for item in solution_files(session, chunks, chain)}[
        "/repo/src/text/slug.py"
    ]

    assert "old_string" in patched.args
    assert "content" in patched.args
    assert write_kind(patched) is WriteKind.EDIT
    assert content_of(patched) is None
    assert entry.write is not None
    assert entry.write.id == "c03", "the full write it patched is still the accepted one"
    assert ids(entry.edits) == ["c05", "c07"]


def test_a_call_whose_arguments_name_no_file_is_attributed_to_none():
    call = ToolCall(id="c01", name="Write", category=ToolCategory.WRITE, args={"content": "x"})

    assert file_of(call) is None


def test_a_write_whose_file_argument_is_spelled_unknown_is_attributed_to_no_file():
    session = session_with_renamed_argument(WRITE_THEN_TWO_EDITS, "c03", "file_path", "target")
    chunks = chunk_session(session)
    chain = Chain(id="C1", label="the one intent", chunk_ids=[chunks[0].id])
    entries = {entry.path: entry for entry in solution_files(session, chunks, chain)}

    assert [located.path for located in write_calls(session, chunks, chain)] == [
        "/repo/tests/text/test_slug.py",
        None,
        "/repo/src/text/slug.py",
        "/repo/src/text/slug.py",
    ], "the call is still reported, it is only attributed to no file"
    assert entries["/repo/src/text/slug.py"].write is None, (
        "the write that cannot be attributed does not become the file's accepted write"
    )
    assert ids(entries["/repo/src/text/slug.py"].edits) == ["c05", "c07"]


# --- the chain's calls -----------------------------------------------------


@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_the_chunks_of_a_chain_are_resolved_in_transcript_order(name):
    _session, chunks, chains = classified(name)
    by_id = {chunk.id: chunk for chunk in chunks}

    for chain in chains:
        resolved = chain_chunks(chunks, chain)
        assert [chunk.id for chunk in resolved] == sorted(
            chain.chunk_ids, key=lambda identifier: by_id[identifier].start_index
        )
        assert all(chunk is by_id[chunk.id] for chunk in resolved)


@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_the_calls_of_a_chain_are_its_chunks_calls_with_their_results(name):
    session, chunks, chains = classified(name)
    recorded = {
        message.tool_result.call_id: message.tool_result
        for message in session.messages
        if message.tool_result is not None
    }

    for chain in chains:
        expected = [
            (call.id, index, message.id, chunk.id)
            for chunk in chain_chunks(chunks, chain)
            for index in range(chunk.start_index, chunk.end_index)
            if (message := session.messages[index]).tool_calls
            for call in message.tool_calls
        ]
        located = chain_calls(session, chunks, chain)

        assert [
            (item.id, item.message_index, item.message_id, item.chunk_id) for item in located
        ] == expected
        assert [item.result for item in located] == [
            recorded[call_id] for call_id, _, _, _ in expected
        ]


# --- corpus-wide invariants ------------------------------------------------


@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_every_write_that_names_a_file_is_kept_superseded_or_failed_once(name):
    session, chunks, chains = classified(name)

    for chain in chains:
        attributable = [
            located for located in write_calls(session, chunks, chain) if located.path is not None
        ]
        accounted: list[str] = []
        for entry in solution_files(session, chunks, chain):
            accounted += ids(entry.kept) + ids(entry.superseded) + ids(entry.failed)

        assert sorted(accounted) == sorted(located.id for located in attributable)
        assert len(set(accounted)) == len(accounted), "a write is in exactly one of the three"


@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_the_writes_of_a_chain_are_the_write_calls_of_its_chunks_in_order(name):
    session, chunks, chains = classified(name)

    for chain in chains:
        writes = write_calls(session, chunks, chain)

        assert ids(writes) == [
            located.id
            for located in chain_calls(session, chunks, chain)
            if located.category is ToolCategory.WRITE
        ]
        assert [located.message_index for located in writes] == sorted(
            located.message_index for located in writes
        )


@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_an_entry_holds_its_writes_in_transcript_order(name):
    session, chunks, chains = classified(name)

    for chain in chains:
        for entry in solution_files(session, chunks, chain):
            held = (*entry.superseded, *entry.kept)
            assert [located.path for located in held] == [entry.path] * len(held)
            for located in entry.superseded:
                assert entry.write is None or located.message_index < entry.write.message_index
            assert [located.message_index for located in entry.edits] == sorted(
                located.message_index for located in entry.edits
            )


@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_resolving_the_same_chain_twice_gives_the_same_answer(name):
    session, chunks, chains = classified(name)

    for chain in chains:
        assert solution_files(session, chunks, chain) == solution_files(session, chunks, chain)


# --- what the stage refuses ------------------------------------------------


def test_a_chain_naming_a_chunk_that_was_not_offered_is_refused():
    session, chunks, chain = only_chain(WRITE_THEN_TWO_EDITS)
    unknown = Chain(id=chain.id, label=chain.label, chunk_ids=["K9"])

    with pytest.raises(UnknownChunkError) as error:
        solution_files(session, chunks, unknown)

    assert error.value.chunk_id == "K9"
    assert error.value.chain_id == chain.id
    assert "K1" in error.value.known
    assert isinstance(error.value, ResolveError)


def test_a_chunk_that_reaches_past_the_transcript_is_refused():
    session = CORPUS.session(WRITE_THEN_TWO_EDITS)
    too_long = Chunk(id="K1", start_index=0, end_index=len(session.messages) + 1)
    chain = Chain(id="C1", label="the one intent", chunk_ids=["K1"])

    with pytest.raises(ForeignChunkError) as error:
        solution_files(session, [too_long], chain)

    assert error.value.chunk_id == "K1"
    assert error.value.session_id == session.id
    assert isinstance(error.value, ResolveError)


def test_the_refusals_are_one_family():
    assert issubclass(UnknownChunkError, ResolveError)
    assert issubclass(ForeignChunkError, ResolveError)


def test_a_solution_file_names_the_ids_its_solution_is_expressed_in():
    entry = files_of(WRITE_THEN_TWO_EDITS)["/repo/src/text/slug.py"]

    assert entry.call_ids == ("c03", "c05", "c07")
    assert entry.call_ids == tuple(ids(entry.kept))


def test_a_solution_file_that_states_nothing_is_a_programming_error():
    # A plain `ValueError`, not a `ResolveError`: the mistake is in a caller's
    # own construction and not in input the stage was handed.
    located = files_of(REWRITTEN)["docs/install.md"].write
    with pytest.raises(ValueError, match="name the file"):
        SolutionFile(path="   ", write=located, edits=(), superseded=(), failed=())
    with pytest.raises(ValueError, match="no write at all"):
        SolutionFile(path="docs/install.md", write=None, edits=(), superseded=(), failed=())


def test_a_file_whose_every_write_failed_is_an_entry_with_an_empty_solution():
    # Reachable, and the one entry whose `kept` is empty: the writes of the file
    # are reported as failed, so the partition stays complete and nothing claims
    # the file was changed. No corpus transcript gives up on a file like that,
    # so the shape is pinned on the value object.
    failed = files_of(REWRITTEN)["docs/install.md"].failed
    entry = SolutionFile(path="docs/install.md", write=None, edits=(), superseded=(), failed=failed)

    assert entry.kept == ()
    assert entry.call_ids == ()
    assert ids(entry.failed) == ["c03"]


def test_a_located_call_reports_what_the_agent_saw() -> None:
    result = ToolResult(call_id="c01", output="boom", is_error=True)
    call = ToolCall(id="c01", name="Edit", category=ToolCategory.WRITE, args={"file_path": "a.py"})
    located = LocatedCall(
        call=call, chunk_id="K1", message_id="m01", message_index=0, result=result
    )

    assert located.id == "c01"
    assert located.name == "Edit"
    assert located.category is ToolCategory.WRITE
    assert located.path == "a.py"
    assert located.kind is WriteKind.EDIT
    assert located.failed is True
