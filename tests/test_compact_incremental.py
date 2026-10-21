"""Incremental compaction: only chains with new chunks are compacted again (T6.3, #28).

Two properties carry this stage, and a test that proves one without the other
proves nothing:

- **A chain without new chunks is copied, not re-rendered.** The tests therefore
  never compare a copied block only against what the writer would produce for it
  — a re-render that happens to match would pass that. They compare against the
  bytes that went in, and one of them carries a section the writer *cannot*
  write, so only a copy can reproduce it.
- **A copy costs no model call.** The scripts handed to `FakeNarrativeClient`
  hold exactly as many responses as chains are expected to be compacted, so a
  stage that asked the client for every chain and threw the answer away would
  run the script out and raise instead of passing quietly. The call record is
  asserted on as well, chain id by chain id, because the number of calls is what
  the study pays.

The realistic input is `tests/fixtures/already_compacted_block.json`: its
transcript opens with an earlier run's two-chain block and holds one new user
turn, which — per that fixture's gold note — continues the block's `C2`. That
single run produces all three cases this stage has to tell apart: `C1`
untouched, `C2` extended, and (with a chain of its own added) one the block does
not hold at all.

No test here calls a model, and none reads a transcript of this machine.
"""

from collections.abc import Callable

import pytest

from fixture_corpus import CORPUS
from idc.chunk import chunk_session
from idc.classify import FakeAdapter, classify
from idc.compact import (
    DECISIONS_HEADER,
    REJECTED_HEADER,
    CompactError,
    FakeNarrativeClient,
    chain_heading,
    compact_incrementally,
    incremental_blocks,
    is_carried_over,
    transcript_chain,
    transcript_chunk_ids,
)
from idc.parse import (
    PLACEHOLDER_CHUNK_PREFIX,
    ParseError,
    join_blocks,
    parse_chains,
    placeholder_chunk_id,
    split_blocks,
)
from idc.resolve import Resolution, UnknownChunkError, resolve
from idc.schema import Chain

SEEDED = "already_compacted_block"
"""The fixture whose transcript opens with an earlier run's block."""

EXTENDED = "C2"
"""The chain the fixture's one new chunk continues, per its gold note."""

UNTOUCHED = "C1"
"""The chain of the same block that the new turn has nothing to do with."""

COMPACT_NAMES = CORPUS.compact_names()
"""Every golden block sequence in the corpus, for the round-trip case."""

EMPTY_RESOLUTION = Resolution(solution_files=(), verification=None, path_call_ids=())
"""What a chain resolves to when it wrote no file and ran no command."""


def narrative(label: str) -> str:
    """Return a well-formed narrative response for a chain with that intent."""
    return (
        f"## Intent: {label}\n"
        f"{DECISIONS_HEADER}\n"
        f"- The limiter keeps its in-process bucket.\n"
        f"{REJECTED_HEADER}\n"
        f"- A shared Redis bucket, deferred again.\n"
    )


def block_text() -> str:
    """Return the compacted block the fixture's transcript carries."""
    session = CORPUS.session(SEEDED)
    blocks = [m.content for m in session.messages if m.content.startswith("## [C")]
    assert len(blocks) == 1, f"{SEEDED} is used as the one-block fixture, found {len(blocks)}"
    return blocks[0]


def seeded_run() -> tuple[list[Chain], Callable[[Chain], Resolution]]:
    """Return the chains of a run seeded with the fixture's block, and its resolver.

    The adapter is scripted with the one decision the fixture's gold note
    states: its single new chunk continues the block's `C2`.
    """
    session = CORPUS.session(SEEDED)
    chunks = chunk_session(session)
    chains = classify(
        session,
        chunks,
        FakeAdapter([EXTENDED]),
        candidates=parse_chains(block_text()),
    )

    def resolution_of(chain: Chain) -> Resolution:
        return resolve(session, chunks, chain)

    return chains, resolution_of


def by_id(blocks: list[str]) -> dict[str, str]:
    """Return the blocks of a rendered sequence, keyed by the chain they summarize."""
    return {chain.id: chain.compacted_text for chain in parse_chains(join_blocks(blocks))}


def _unexpected(chain: Chain) -> Resolution:
    """A resolver that fails: handed to runs in which no chain may be resolved."""
    raise AssertionError(f"chain {chain.id!r} was resolved, and it has nothing to resolve")


def test_the_fixture_run_produces_one_untouched_and_one_extended_chain() -> None:
    """The seeded run is the input this ticket is about, so pin what it is."""
    chains, _ = seeded_run()

    assert [chain.id for chain in chains] == [UNTOUCHED, EXTENDED]
    untouched, extended = chains
    assert untouched.chunk_ids == [placeholder_chunk_id(UNTOUCHED)]
    assert placeholder_chunk_id(EXTENDED) in extended.chunk_ids
    assert [
        chunk_id
        for chunk_id in extended.chunk_ids
        if not chunk_id.startswith(PLACEHOLDER_CHUNK_PREFIX)
    ]


def test_only_the_extended_chain_changes() -> None:
    """The ticket's own test: one new chunk on C2 leaves every other block alone."""
    chains, resolution_of = seeded_run()
    before = by_id(split_blocks(block_text()))
    client = FakeNarrativeClient([narrative("Give the rate limiter a Retry-After header")])

    written = compact_incrementally(chains, resolution_of=resolution_of, client=client)

    after = by_id(split_blocks(written))

    assert after[UNTOUCHED] == before[UNTOUCHED]
    assert after[EXTENDED] != before[EXTENDED]
    assert [call.chain_id for call in client.calls] == [EXTENDED]


def test_an_untouched_chain_costs_no_model_call() -> None:
    """A script of one response is enough for a two-chain block with one extension.

    The negative half is the fake's: a stage that asked the client for every
    chain would find the script empty on the second one and raise
    `NarrativeScriptExhaustedError` rather than pass.
    """
    chains, resolution_of = seeded_run()
    client = FakeNarrativeClient([narrative("Give the rate limiter a Retry-After header")])

    compact_incrementally(chains, resolution_of=resolution_of, client=client)

    assert len(client.calls) == 1
    assert client.remaining == 0


def test_a_carried_over_chain_is_not_resolved() -> None:
    """The copied chain's resolution is never asked for — it has nothing to resolve."""
    chains, resolution_of = seeded_run()
    asked: list[str] = []

    def recording(chain: Chain) -> Resolution:
        asked.append(chain.id)
        return resolution_of(chain)

    client = FakeNarrativeClient([narrative("Give the rate limiter a Retry-After header")])
    compact_incrementally(chains, resolution_of=recording, client=client)

    assert asked == [EXTENDED]


def test_a_copied_block_keeps_a_section_the_writer_never_wrote() -> None:
    """Byte-identity is a copy, not a re-render that happens to agree.

    The carried block gets a fifth section, which `idc.compact` has no way to
    produce: a stage that re-rendered the chain could not put it back.
    """
    first, second = split_blocks(block_text())
    hand_edited = f"{first}### Notes from the reviewer\n- Added by hand, after the run.\n"
    chains = parse_chains(f"{hand_edited}\n{second}")
    client = FakeNarrativeClient([])

    written = compact_incrementally(chains, resolution_of=_unexpected, client=client)

    assert split_blocks(written)[0] == hand_edited
    assert client.calls == ()


def test_an_untouched_block_comes_back_byte_identical() -> None:
    """The T6 → T8 → T6 round trip, separator included, through this stage."""
    text = block_text()

    written = compact_incrementally(
        parse_chains(text), resolution_of=_unexpected, client=FakeNarrativeClient([])
    )

    assert written == text


def test_the_corpus_has_golden_block_sequences() -> None:
    """The round-trip parametrization below is empty if the corpus holds none."""
    assert COMPACT_NAMES


@pytest.mark.parametrize("name", COMPACT_NAMES)
def test_every_golden_sequence_survives_the_round_trip(name: str) -> None:
    """Every block the writer produced parses and copies back to the same bytes."""
    text = CORPUS.compact_markdown(name)

    written = compact_incrementally(
        parse_chains(text), resolution_of=_unexpected, client=FakeNarrativeClient([])
    )

    assert written == text


def test_a_new_chain_is_appended_after_the_carried_ones() -> None:
    """Document order: the block's order, and a chain it does not hold comes last."""
    chains, resolution_of = seeded_run()
    opened = Chain(id="C3", label="Add a retry budget to the HTTP client", chunk_ids=["K9"])
    client = FakeNarrativeClient(
        [narrative("Give the rate limiter a Retry-After header"), narrative("Add a retry budget")]
    )

    def lookup(chain: Chain) -> Resolution:
        return EMPTY_RESOLUTION if chain.id == "C3" else resolution_of(chain)

    written = compact_incrementally([*chains, opened], resolution_of=lookup, client=client)

    assert [chain.id for chain in parse_chains(written)] == [UNTOUCHED, EXTENDED, "C3"]
    assert [call.chain_id for call in client.calls] == [EXTENDED, "C3"]


def test_the_order_handed_over_is_the_order_written() -> None:
    """The stage writes the sequence it was given and sorts nothing."""
    chains, resolution_of = seeded_run()
    opened = Chain(id="C3", label="Add a retry budget to the HTTP client", chunk_ids=["K9"])
    client = FakeNarrativeClient(
        [narrative("Add a retry budget"), narrative("Give the rate limiter a Retry-After header")]
    )

    def lookup(chain: Chain) -> Resolution:
        return EMPTY_RESOLUTION if chain.id == "C3" else resolution_of(chain)

    written = compact_incrementally([opened, *chains], resolution_of=lookup, client=client)

    assert [chain.id for chain in parse_chains(written)] == ["C3", UNTOUCHED, EXTENDED]


def test_a_new_chain_is_compacted_even_without_a_carried_block() -> None:
    """The third case on its own: a run with nothing carried over compacts everything."""
    opened = Chain(id="C1", label="Add a retry budget to the HTTP client", chunk_ids=["K1"])
    client = FakeNarrativeClient([narrative("Add a retry budget")])

    written = compact_incrementally(
        [opened], resolution_of=lambda _chain: EMPTY_RESOLUTION, client=client
    )

    assert written.startswith(chain_heading(opened, "Add a retry budget"))
    assert [call.chain_id for call in client.calls] == ["C1"]


def test_no_chains_writes_nothing() -> None:
    """An empty sequence is the empty text, and costs no call."""
    client = FakeNarrativeClient([])

    assert compact_incrementally([], resolution_of=_unexpected, client=client) == ""
    assert client.calls == ()


def test_the_blocks_are_the_blocks_of_the_sequence() -> None:
    """`incremental_blocks` and `compact_incrementally` are one run, split two ways."""
    chains, resolution_of = seeded_run()
    script = [narrative("Give the rate limiter a Retry-After header")]

    blocks = incremental_blocks(
        chains, resolution_of=resolution_of, client=FakeNarrativeClient(list(script))
    )
    written = compact_incrementally(
        chains, resolution_of=resolution_of, client=FakeNarrativeClient(list(script))
    )

    assert blocks == split_blocks(written)


def test_the_user_messages_of_an_extended_chain_reach_the_client() -> None:
    """The hook is asked for the chains that are compacted, and its answer is offered."""
    chains, resolution_of = seeded_run()
    client = FakeNarrativeClient([narrative("Give the rate limiter a Retry-After header")])
    asked: list[str] = []

    def messages_of(chain: Chain) -> tuple[str, ...]:
        asked.append(chain.id)
        return ("The rate limiter from before returns 429 without a Retry-After header.",)

    compact_incrementally(
        chains, resolution_of=resolution_of, client=client, user_messages_of=messages_of
    )

    assert asked == [EXTENDED]
    assert client.calls[0].request.user_messages == (
        "The rate limiter from before returns 429 without a Retry-After header.",
    )


def test_without_the_hook_no_user_message_is_claimed() -> None:
    """The default offers none rather than inventing one."""
    chains, resolution_of = seeded_run()
    client = FakeNarrativeClient([narrative("Give the rate limiter a Retry-After header")])

    compact_incrementally(chains, resolution_of=resolution_of, client=client)

    assert client.calls[0].request.user_messages == ()


def test_the_same_inputs_write_the_same_bytes() -> None:
    """Determinism: two runs, one script, one result."""
    script = [narrative("Give the rate limiter a Retry-After header")]
    first_chains, first_resolver = seeded_run()
    second_chains, second_resolver = seeded_run()

    first = compact_incrementally(
        first_chains, resolution_of=first_resolver, client=FakeNarrativeClient(list(script))
    )
    second = compact_incrementally(
        second_chains, resolution_of=second_resolver, client=FakeNarrativeClient(list(script))
    )

    assert first == second


def test_a_malformed_block_is_refused_by_the_parser() -> None:
    """The refusal comes from `idc.parse` and this stage puts nothing in front of it.

    `parse_chains` raises inside the argument expression, so this says nothing
    about the stage's own body on its own — `test_the_stage_wraps_no_refusal_of
    _the_parsers` is the half that does.
    """
    malformed = block_text().replace("## [C1]", "## [c1]", 1)

    with pytest.raises(ParseError) as caught:
        compact_incrementally(
            parse_chains(malformed), resolution_of=_unexpected, client=FakeNarrativeClient([])
        )

    assert not isinstance(caught.value, CompactError)


def test_a_parsed_chain_is_carried_over() -> None:
    """The predicate: a chain straight out of a block holds no chunk of this run."""
    for chain in parse_chains(block_text()):
        assert is_carried_over(chain)
        assert transcript_chunk_ids(chain) == ()


def test_an_extended_chain_is_not_carried_over() -> None:
    """A carried chain that took a chunk of this run is one to compact again."""
    chains, _ = seeded_run()
    untouched, extended = chains

    assert is_carried_over(untouched)
    assert not is_carried_over(extended)
    assert transcript_chunk_ids(extended) == tuple(
        chunk_id
        for chunk_id in extended.chunk_ids
        if not chunk_id.startswith(PLACEHOLDER_CHUNK_PREFIX)
    )


def test_a_new_chain_is_not_carried_over() -> None:
    """A chain of this run alone was never in a block."""
    opened = Chain(id="C7", label="Add a retry budget to the HTTP client", chunk_ids=["K9"])

    assert not is_carried_over(opened)
    assert transcript_chunk_ids(opened) == ("K9",)


def test_the_transcript_chain_drops_the_placeholder() -> None:
    """What the resolver is handed: the chain minus ids that name no chunk here."""
    chains, _ = seeded_run()
    extended = chains[1]

    current = transcript_chain(extended)

    assert current.id == extended.id
    assert current.label == extended.label
    assert current.compacted_text == extended.compacted_text
    assert current.chunk_ids == list(transcript_chunk_ids(extended))
    assert placeholder_chunk_id(EXTENDED) not in current.chunk_ids


def test_the_resolver_is_handed_a_chain_it_can_resolve() -> None:
    """Why `transcript_chain` exists: the carried placeholder names no chunk here.

    The extended chain as the classifier returns it cannot be resolved — its
    first id came out of a block and no chunk of this transcript answers to it —
    so the stage hands the resolver the chain reduced to this transcript's
    chunks, and the run goes through.
    """
    chains, resolution_of = seeded_run()
    extended = chains[1]
    client = FakeNarrativeClient([narrative("Give the rate limiter a Retry-After header")])

    with pytest.raises(UnknownChunkError, match=placeholder_chunk_id(EXTENDED)):
        resolution_of(extended)

    assert compact_incrementally(chains, resolution_of=resolution_of, client=client)


def test_the_transcript_chain_of_a_carried_over_chain_is_refused() -> None:
    """There is no such chain: a carried-over chain holds no chunk of this run."""
    carried = parse_chains(block_text())[0]

    with pytest.raises(ValueError, match=carried.id):
        transcript_chain(carried)


# --- the edge the predicate's two halves meet at --------------------------------


def test_a_chain_that_was_never_compacted_is_not_carried_over() -> None:
    """Both halves of the `iff` are load-bearing, not only the chunk-id half.

    A chain carrying a placeholder id but no summary is not an earlier run's —
    the schema admits it, because `Identifier` is any non-empty string and
    nothing ties the id's spelling to the flag. Judging it carried would copy a
    `compacted_text` that is `None`, which is how the literal text `None` would
    reach the output. Pinned because dropping `chain.compacted` from the
    predicate otherwise leaves the whole suite green.
    """
    never_compacted = Chain(
        id=UNTOUCHED, label="L", chunk_ids=[placeholder_chunk_id(UNTOUCHED)], compacted=False
    )

    assert not is_carried_over(never_compacted)


def test_a_chain_with_no_part_in_this_transcript_is_refused_as_a_compact_error() -> None:
    """The refusal a caller can reach is this stage's own, not a bare `ValueError`.

    `transcript_chain` is asked only about chains that are written, so a chain
    that is neither carried over nor present in this transcript is the one
    input that can reach it. It stays a `ValueError`, so a caller that catches
    that still works, but it is a `CompactError` too — the stage's refusals are
    one family, and a bare `ValueError` would put a raw traceback on the CLI.
    """
    orphan = Chain(
        id=UNTOUCHED, label="L", chunk_ids=[placeholder_chunk_id(UNTOUCHED)], compacted=False
    )

    with pytest.raises(CompactError) as caught:
        compact_incrementally([orphan], resolution_of=_unexpected, client=FakeNarrativeClient([]))
    assert isinstance(caught.value, ValueError)
    assert UNTOUCHED in str(caught.value)


def test_the_stage_wraps_no_refusal_of_the_parsers() -> None:
    """Nothing in this stage catches a `ParseError` and re-raises it as its own.

    The sibling test hands `compact_incrementally` an already-raising argument,
    so it never enters the body. This one puts the refusal inside the body, by
    way of a lookup that raises it, and asserts it comes back out unchanged.
    """

    def raising(chain: Chain) -> Resolution:
        raise ParseError("from inside the stage")

    chains = list(parse_chains(block_text()))
    extended = next(chain for chain in chains if chain.id == EXTENDED)
    chains[chains.index(extended)] = Chain(
        id=extended.id,
        label=extended.label,
        chunk_ids=[*extended.chunk_ids, "K1"],
        compacted=extended.compacted,
        compacted_text=extended.compacted_text,
    )

    with pytest.raises(ParseError, match="from inside the stage"):
        compact_incrementally(chains, resolution_of=raising, client=FakeNarrativeClient(["x"]))


def test_an_id_only_a_looser_reading_calls_a_chunk_fails_towards_rewriting() -> None:
    """The prefix test is looser than the gold reader's, and that is the safe side.

    `Identifier` is any non-empty string, so an id like `Kompakt1` is possible
    even though nothing here produces one. Counting it as a chunk makes the
    chain look extended — it is rewritten, and a wrong resolution is refused
    loudly. The stricter reading would make it look carried, and a chunk the
    chain really gained would vanish from the summary with the suite green.
    """
    odd = Chain(
        id=UNTOUCHED,
        label="L",
        chunk_ids=[placeholder_chunk_id(UNTOUCHED), "Kompakt1"],
        compacted=True,
        compacted_text=parse_chains(block_text())[0].compacted_text,
    )

    assert not is_carried_over(odd)
    assert transcript_chunk_ids(odd) == ("Kompakt1",)
