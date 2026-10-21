"""Assembly of the rendered output: the summary block, then the latest chunk (T7.1, #31).

This stage does two things and nothing else: it puts the block's chains into one
Markdown text in the order the block holds them, and it hands on the messages of
the chunk the block names as the one to keep verbatim. The agent's native tags,
roles and metadata are T7.2's and no test here expects them.

Three properties carry it, and the tests are built so that a plausible
implementation mistake breaks one of them:

- **The order is the input's.** The order tests hand in chains whose ids are not
  ascending, so an implementation that sorted them — by id, or by rebuilding the
  order from anywhere else — would produce a different text and fail.
- **The latest chunk is byte-identical.** Asserted as object *identity* against
  the session that went in, and in addition by value against a freshly loaded
  copy of the same fixture, so a stage that handed back equal-but-rebuilt
  messages passes the first assertion and a stage that mutated the session's own
  objects fails the second.
- **What is written is what `idc.parse` reads.** The summary is round-tripped
  through `parse_chains`, which is the reader of the next run.

Nothing here calls a model, reads a transcript of this machine or touches a
clock: the only input is the shared corpus and its golden blocks.
"""

import pytest

from fixture_corpus import CORPUS
from idc.chunk import chunk_session
from idc.parse import BLOCK_SEPARATOR, join_blocks, parse_chains, split_blocks
from idc.render import (
    AssembledTranscript,
    RenderError,
    UncompactedChainError,
    UnknownLatestChunkError,
    assemble,
)
from idc.schema import Agent, Chain, Chunk, CompactedBlock, Message, Session

COMPACT_NAMES = sorted(CORPUS.compact_names())
"""Every corpus fixture that has a golden block sequence, which is all of them."""

TWO_CHAINS = "interleaved_intents"
"""A fixture whose golden sequence holds two blocks, so order is observable."""

ANOTHER_TWO_CHAINS = "two_sequential_intents"
"""A second two-block fixture, so a third chain id can be borrowed from it."""


def test_the_corpus_offers_fixtures_with_golden_blocks() -> None:
    """Guard the parametrization: an empty corpus would skip, not pass."""
    assert COMPACT_NAMES


def golden_chains(name: str) -> list[Chain]:
    """Return the chains of a fixture's golden block sequence, in document order."""
    return parse_chains(CORPUS.compact_markdown(name))


def block_of(name: str, *, agent: Agent = Agent.CLAUDE_CODE) -> CompactedBlock:
    """Build the `CompactedBlock` of a fixture: its golden chains, its last chunk."""
    chunks = chunk_session(CORPUS.session(name))
    return CompactedBlock(
        chains=golden_chains(name),
        latest_chunk_id=chunks[-1].id,
        rendered_for=agent,
    )


def messages_of(session: Session, chunk: Chunk) -> list[Message]:
    """Return the messages a chunk spans, as the session's own objects."""
    return list(session.messages[chunk.start_index : chunk.end_index])


def renamed(text: str, chain_id: str) -> str:
    """Return a golden block with its heading's chain id replaced.

    Lets a third chain be borrowed from another fixture's block without two
    chains of one `CompactedBlock` sharing an id, which the schema refuses.
    """
    heading, newline, rest = text.partition("\n")
    _prefix, bracket, label = heading.partition("]")
    assert bracket == "]", heading
    return f"## [{chain_id}]{label}{newline}{rest}"


@pytest.mark.parametrize("name", COMPACT_NAMES)
def test_assemble_returns_the_golden_block_and_the_latest_chunk(name: str) -> None:
    """The summary is the chains' blocks joined; the messages are the chunk's."""
    session = CORPUS.session(name)
    chunks = chunk_session(session)
    block = block_of(name)

    assembled = assemble(block, session=session, chunks=chunks)

    assert isinstance(assembled, AssembledTranscript)
    assert assembled.summary == CORPUS.compact_markdown(name)
    assert list(assembled.messages) == messages_of(session, chunks[-1])


@pytest.mark.parametrize("name", COMPACT_NAMES)
def test_the_summary_is_the_blocks_joined_by_the_parsers_separator(name: str) -> None:
    """What this stage writes between two blocks is what `idc.parse` splits on."""
    session = CORPUS.session(name)
    chunks = chunk_session(session)
    block = block_of(name)
    texts = [chain.compacted_text for chain in block.chains]

    assembled = assemble(block, session=session, chunks=chunks)

    assert assembled.summary == BLOCK_SEPARATOR.join(texts)
    assert assembled.summary == join_blocks(texts)
    assert len(split_blocks(assembled.summary)) == len(block.chains)


@pytest.mark.parametrize("name", COMPACT_NAMES)
def test_the_summary_round_trips_through_parse_chains(name: str) -> None:
    """Reading the summary back yields the chains that went in, byte for byte."""
    session = CORPUS.session(name)
    chunks = chunk_session(session)
    block = block_of(name)

    assembled = assemble(block, session=session, chunks=chunks)

    read_back = parse_chains(assembled.summary)
    assert [chain.id for chain in read_back] == [chain.id for chain in block.chains]
    assert [chain.label for chain in read_back] == [chain.label for chain in block.chains]
    assert [chain.compacted_text for chain in read_back] == [
        chain.compacted_text for chain in block.chains
    ]


@pytest.mark.parametrize("name", COMPACT_NAMES)
def test_assembling_twice_gives_the_identical_result(name: str) -> None:
    """Nothing in this stage depends on a clock, a hash seed or a call count."""
    session = CORPUS.session(name)
    chunks = chunk_session(session)
    block = block_of(name)

    first = assemble(block, session=session, chunks=chunks)
    second = assemble(block, session=session, chunks=chunks)

    assert first == second
    assert first.summary == second.summary


def test_the_chains_keep_the_order_the_block_holds_them_in() -> None:
    """Three chains in descending-then-ascending id order come out in that order."""
    session = CORPUS.session(TWO_CHAINS)
    chunks = chunk_session(session)
    first, second = golden_chains(TWO_CHAINS)
    borrowed = golden_chains(ANOTHER_TWO_CHAINS)[0]
    third = Chain(
        id="C3",
        label=borrowed.label,
        chunk_ids=borrowed.chunk_ids,
        compacted=True,
        compacted_text=renamed(borrowed.compacted_text, "C3"),
    )
    block = CompactedBlock(
        chains=[third, first, second],
        latest_chunk_id=chunks[-1].id,
        rendered_for=Agent.CLAUDE_CODE,
    )

    assembled = assemble(block, session=session, chunks=chunks)

    assert [chain.id for chain in parse_chains(assembled.summary)] == ["C3", "C1", "C2"]
    assert assembled.summary == join_blocks(
        [third.compacted_text, first.compacted_text, second.compacted_text]
    )


def test_the_reversed_order_is_passed_through_unsorted() -> None:
    """A block whose chains are not chronological is written as it stands."""
    session = CORPUS.session(TWO_CHAINS)
    chunks = chunk_session(session)
    first, second = golden_chains(TWO_CHAINS)
    block = CompactedBlock(
        chains=[second, first],
        latest_chunk_id=chunks[-1].id,
        rendered_for=Agent.CLAUDE_CODE,
    )

    assembled = assemble(block, session=session, chunks=chunks)

    assert [chain.id for chain in parse_chains(assembled.summary)] == ["C2", "C1"]


@pytest.mark.parametrize("name", COMPACT_NAMES)
def test_the_latest_chunk_messages_are_the_sessions_own_objects(name: str) -> None:
    """Byte-identity is identity: the messages are not copied, let alone rebuilt."""
    session = CORPUS.session(name)
    chunks = chunk_session(session)
    block = block_of(name)

    assembled = assemble(block, session=session, chunks=chunks)

    spanned = messages_of(session, chunks[-1])
    assert len(assembled.messages) == len(spanned)
    for rendered, original in zip(assembled.messages, spanned, strict=True):
        assert rendered is original


@pytest.mark.parametrize("name", COMPACT_NAMES)
def test_the_latest_chunk_is_unchanged_against_a_freshly_loaded_fixture(name: str) -> None:
    """Nothing is normalized, re-timed, re-id'd, re-ordered or truncated."""
    session = CORPUS.session(name)
    chunks = chunk_session(session)
    block = block_of(name)

    assembled = assemble(block, session=session, chunks=chunks)

    fresh = CORPUS.session(name)
    expected = messages_of(fresh, chunks[-1])
    assert [message.model_dump() for message in assembled.messages] == [
        message.model_dump() for message in expected
    ]


def test_a_tool_result_in_the_latest_chunk_survives_in_full() -> None:
    """The dump comparison is not vacuous: a result's whole output comes through.

    `failed_then_passing_command` is one single chunk — reads, a failed test run
    and a passing one — so its latest chunk carries six tool results with
    output. If that stopped being true the assertion below fails rather than
    passing emptily.
    """
    name = "failed_then_passing_command"
    session = CORPUS.session(name)
    chunks = chunk_session(session)
    block = block_of(name)

    assembled = assemble(block, session=session, chunks=chunks)

    results = [
        message.tool_result for message in assembled.messages if message.tool_result is not None
    ]
    assert results, "fixture no longer ends on a chunk with a tool result"
    original = [
        message.tool_result
        for message in messages_of(session, chunks[-1])
        if message.tool_result is not None
    ]
    assert [result.output for result in results] == [result.output for result in original]
    assert any(result.output for result in results)


def test_a_block_naming_an_earlier_chunk_renders_that_chunk() -> None:
    """`latest_chunk_id` is the authority on what stays verbatim, not `is_latest`."""
    session = CORPUS.session(TWO_CHAINS)
    chunks = chunk_session(session)
    assert len(chunks) > 1
    block = CompactedBlock(
        chains=golden_chains(TWO_CHAINS),
        latest_chunk_id=chunks[0].id,
        rendered_for=Agent.CLAUDE_CODE,
    )

    assembled = assemble(block, session=session, chunks=chunks)

    assert list(assembled.messages) == messages_of(session, chunks[0])


def test_a_block_without_chains_assembles_to_an_empty_summary() -> None:
    """A first run that compacted nothing still renders its latest chunk."""
    session = CORPUS.session(TWO_CHAINS)
    chunks = chunk_session(session)
    block = CompactedBlock(chains=[], latest_chunk_id=chunks[-1].id, rendered_for=Agent.CLAUDE_CODE)

    assembled = assemble(block, session=session, chunks=chunks)

    assert assembled.summary == ""
    assert list(assembled.messages) == messages_of(session, chunks[-1])


def test_an_unknown_latest_chunk_id_is_refused() -> None:
    """A block naming no chunk of the session is bad input, not a crash."""
    session = CORPUS.session(TWO_CHAINS)
    chunks = chunk_session(session)
    block = CompactedBlock(
        chains=golden_chains(TWO_CHAINS),
        latest_chunk_id="K999",
        rendered_for=Agent.CLAUDE_CODE,
    )

    with pytest.raises(UnknownLatestChunkError) as excinfo:
        assemble(block, session=session, chunks=chunks)

    assert isinstance(excinfo.value, RenderError)
    assert not isinstance(excinfo.value, ValueError)
    assert excinfo.value.chunk_id == "K999"
    assert excinfo.value.known == tuple(chunk.id for chunk in chunks)
    assert "K999" in str(excinfo.value)


def test_no_chunks_at_all_is_refused_as_an_unknown_chunk() -> None:
    """An empty chunk sequence names nothing, so the same refusal states it."""
    session = CORPUS.session(TWO_CHAINS)
    block = CompactedBlock(
        chains=golden_chains(TWO_CHAINS),
        latest_chunk_id="K1",
        rendered_for=Agent.CLAUDE_CODE,
    )

    with pytest.raises(UnknownLatestChunkError) as excinfo:
        assemble(block, session=session, chunks=[])

    assert excinfo.value.known == ()


def test_an_uncompacted_chain_is_refused_instead_of_writing_none() -> None:
    """`compacted_text is None` must never reach the output as the text `None`."""
    session = CORPUS.session(TWO_CHAINS)
    chunks = chunk_session(session)
    first, second = golden_chains(TWO_CHAINS)
    open_chain = Chain(id="C3", label="still open", chunk_ids=["K1"])
    block = CompactedBlock(
        chains=[first, open_chain, second],
        latest_chunk_id=chunks[-1].id,
        rendered_for=Agent.CLAUDE_CODE,
    )

    with pytest.raises(UncompactedChainError) as excinfo:
        assemble(block, session=session, chunks=chunks)

    assert isinstance(excinfo.value, RenderError)
    assert not isinstance(excinfo.value, ValueError)
    assert excinfo.value.chain_id == "C3"
    assert excinfo.value.position == 2
    assert "C3" in str(excinfo.value)


def test_a_chunk_range_outside_the_session_is_a_value_error() -> None:
    """Pairing chunks with the wrong session is the caller's own construction."""
    session = CORPUS.session(TWO_CHAINS)
    beyond = Chunk(
        id="K99",
        start_index=len(session.messages),
        end_index=len(session.messages) + 1,
        is_latest=True,
    )
    block = CompactedBlock(
        chains=golden_chains(TWO_CHAINS),
        latest_chunk_id=beyond.id,
        rendered_for=Agent.CLAUDE_CODE,
    )

    with pytest.raises(ValueError, match="K99") as excinfo:
        assemble(block, session=session, chunks=[beyond])

    assert not isinstance(excinfo.value, RenderError)
    assert "K99" in str(excinfo.value)


def test_the_assembled_result_is_immutable() -> None:
    """T7.2 wraps the two parts; it does not rewrite them in place."""
    session = CORPUS.session(TWO_CHAINS)
    chunks = chunk_session(session)

    assembled = assemble(block_of(TWO_CHAINS), session=session, chunks=chunks)

    assert isinstance(assembled.messages, tuple)
    with pytest.raises(AttributeError):
        assembled.summary = "rewritten"  # type: ignore[misc]
