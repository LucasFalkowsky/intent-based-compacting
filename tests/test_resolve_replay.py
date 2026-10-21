"""Replaying a file's edits into the content the chain left it with (T5.4, #24).

The oracle is the corpus and nothing else: the chains are built the way the
pipeline builds them — `chunk_session`, then `classify` with the fixture's own
gold script — and the expected final content of the one replayable file is the
hand-written `fixture_corpus.FINAL_SLUG_CONTENT`, which no code derives.

`write_then_two_edits` carries the ticket's first case: a file written whole and
then corrected twice, so its content after the chain is determined by the
transcript alone. The second case — the fallback — is almost the whole rest of
the corpus, where a file was only ever edited and the read before the first edit
is proven by its own line-number gutter to have begun past the file's first
line.

The cases the corpus cannot hold (a read that begins at line 1, a read whose
output carries no gutter, an edit whose anchor the base does not hold) are built
by rewriting a fixture's own document and loading it through `loads`, the way
`tests/test_resolve_writes.py` does — never by writing a transcript into this
file.
"""

import json
from collections.abc import Callable

import pytest

from fixture_corpus import CORPUS, FINAL_SLUG_CONTENT, FINAL_SLUG_PATH
from fixture_corpus import REPLACEMENT_PAIRS as CORPUS_REPLACEMENT_PAIRS
from fixture_corpus import results_by_call_id as corpus_results_by_call_id
from idc.chunk import chunk_session
from idc.classify import FakeAdapter, classify, gold_script
from idc.resolve import (
    REPLACEMENT_KEYS,
    FileReplay,
    LocatedCall,
    ReplayStatus,
    SolutionFile,
    rendered_extent,
    replay_file,
    replay_files,
    solution_files,
)
from idc.schema import Chain, Chunk, Session, ToolCall, ToolCategory, loads

FIXTURE_NAMES = CORPUS.names()
"""Every fixture of the shared corpus, discovered at collection time."""

WRITE_THEN_TWO_EDITS = "write_then_two_edits"
"""The fixture of the first acceptance criterion: one write, then two edits."""

ONLY_EDITED = "already_compacted_block"
"""A fixture whose files are only ever edited, after a read of part of them."""

TWO_READS = "interleaved_intents"
"""A fixture that reads one file again *after* having edited it."""


def classified(name: str) -> tuple[Session, list[Chunk], tuple[Chain, ...]]:
    """Return a fixture's session, chunks and chains, built by the pipeline."""
    session = CORPUS.session(name)
    chunks = chunk_session(session)
    chains = tuple(classify(session, chunks, FakeAdapter(gold_script(CORPUS.gold(name)))))
    return session, chunks, chains


def only_chain(name: str) -> tuple[Session, list[Chunk], Chain]:
    """Return a single-chain fixture's session, chunks and that one chain."""
    session, chunks, chains = classified(name)
    assert len(chains) == 1, f"{name} is used as a single-chain fixture"
    return session, chunks, chains[0]


def replays(session: Session, chunks: list[Chunk], chain: Chain) -> dict[str, FileReplay]:
    """Return the chain's replays, keyed by the file they are about."""
    return {replay.path: replay for replay in replay_files(session, chunks, chain)}


def replays_of(name: str) -> dict[str, FileReplay]:
    """Return the replays of a single-chain fixture, keyed by file."""
    return replays(*only_chain(name))


def document_with(name: str, call_id: str, change: Callable[[dict], None]) -> Session:
    """Return a fixture with `change` applied to one call, via the load path.

    The corpus records what the three agents were observed to send, so the
    shapes it cannot hold are built by rewriting the fixture's own document and
    loading it through `loads` — the same path every stage uses.
    """
    document = json.loads(CORPUS.text(name))
    changed = 0
    for message in document["messages"]:
        for call in message["tool_calls"]:
            if call["id"] == call_id:
                change(call)
                changed += 1
    assert changed == 1, f"{call_id} must occur exactly once in {name}"
    return loads(Session, json.dumps(document))


def with_argument(name: str, call_id: str, key: str, value: str) -> Session:
    """Return a fixture with one call's argument set to `value`."""
    return document_with(name, call_id, lambda call: call["args"].__setitem__(key, value))


def with_renamed_arguments(name: str, call_id: str, renames: dict[str, str]) -> Session:
    """Return a fixture with one call's argument keys renamed."""

    def rename(call: dict) -> None:
        for old, new in renames.items():
            call["args"][new] = call["args"].pop(old)

    return document_with(name, call_id, rename)


def with_read_output(name: str, call_id: str, output: str) -> Session:
    """Return a fixture with the result of one call carrying `output`.

    A read's result is the only thing that says what the agent saw of a file, so
    the extents the corpus does not record are made by rewriting it.
    """
    document = json.loads(CORPUS.text(name))
    changed = 0
    for message in document["messages"]:
        result = message["tool_result"]
        if result is not None and result["call_id"] == call_id:
            result["output"] = output
            changed += 1
    assert changed == 1, f"the result of {call_id} must occur exactly once in {name}"
    return loads(Session, json.dumps(document))


def one_chain_over(session: Session) -> tuple[list[Chunk], Chain]:
    """Return a chunking of `session` and one chain over all of its chunks."""
    chunks = chunk_session(session)
    return chunks, Chain(id="C1", label="the one intent", chunk_ids=[chunk.id for chunk in chunks])


def slug_replay(session: Session) -> FileReplay:
    """Return the replay of `FINAL_SLUG_PATH` over a whole rewritten fixture."""
    chunks, chain = one_chain_over(session)
    return replays(session, chunks, chain)[FINAL_SLUG_PATH]


# --- the replay itself -----------------------------------------------------


def test_the_replay_equals_the_known_final_content_of_the_fixture():
    replay = replays_of(WRITE_THEN_TWO_EDITS)[FINAL_SLUG_PATH]

    assert replay.status is ReplayStatus.REPLAYED
    assert replay.content == FINAL_SLUG_CONTENT
    assert replay.base_call_id == "c03", "the last full write is the base"
    assert replay.replayed_call_ids == ("c05", "c07")
    assert replay.replayed is True


def test_the_base_of_a_replay_is_the_content_the_write_call_carries():
    session, chunks, chain = only_chain(WRITE_THEN_TWO_EDITS)
    replay = replays(session, chunks, chain)["/repo/tests/text/test_slug.py"]
    calls = [call for message in session.messages for call in message.tool_calls]
    written = next(call for call in calls if call.id == "c02")

    assert replay.status is ReplayStatus.REPLAYED
    assert replay.replayed_call_ids == (), "a file written once has no edit to replay"
    assert replay.content == written.args["content"]


def test_a_replayed_file_names_its_base_and_then_its_edits():
    session, chunks, chain = only_chain(WRITE_THEN_TWO_EDITS)
    entries = {entry.path: entry for entry in solution_files(session, chunks, chain)}
    replay = replays(session, chunks, chain)[FINAL_SLUG_PATH]

    assert (replay.base_call_id, *replay.replayed_call_ids) == entries[FINAL_SLUG_PATH].call_ids


def test_an_edit_that_replaces_its_anchor_with_nothing_removes_it():
    anchor = (
        '    stripped = _PUNCTUATION.sub("", folded.lower())\n'
        '    return _WHITESPACE.sub("-", stripped).strip("-")'
    )
    session = with_argument(WRITE_THEN_TWO_EDITS, "c07", "new_string", "")

    replay = slug_replay(session)

    assert replay.status is ReplayStatus.REPLAYED, (
        "an empty replacement is a deletion and a value, unlike an empty anchor"
    )
    assert replay.content is not None
    assert anchor not in replay.content
    assert replay.replayed_call_ids == ("c05", "c07")


@pytest.mark.parametrize(("old_key", "new_key"), CORPUS_REPLACEMENT_PAIRS)
def test_every_replacement_spelling_the_agents_use_is_replayed(old_key, new_key):
    renames = {"old_string": old_key, "new_string": new_key}
    document = json.loads(CORPUS.text(WRITE_THEN_TWO_EDITS))
    for message in document["messages"]:
        for call in message["tool_calls"]:
            if call["id"] in {"c05", "c07"}:
                for old, new in renames.items():
                    call["args"][new] = call["args"].pop(old)
    session = loads(Session, json.dumps(document))

    assert slug_replay(session).content == FINAL_SLUG_CONTENT


def test_the_pairs_of_the_corpus_are_the_replacement_keys_of_the_stage():
    # The stage keeps the keys as one flat list, because *one* of them is
    # enough to know a call patches; replaying needs them in pairs. The corpus
    # copy is the pairing, hand-written, and the two are pinned here so that a
    # key added to one of them without the other fails.
    flattened = tuple(key for pair in CORPUS_REPLACEMENT_PAIRS for key in pair)

    assert flattened == REPLACEMENT_KEYS
    assert len(CORPUS_REPLACEMENT_PAIRS) == 2, "the table the spelling test ranges over"


# --- the fallback ----------------------------------------------------------


def test_a_file_whose_read_began_past_its_first_line_falls_back():
    replay = replays_of(ONLY_EDITED)["src/api/limiter.py"]

    assert replay.status is ReplayStatus.PARTIAL_READ
    assert replay.content is None, "a partial read must never be replayed into content"
    assert replay.base_call_id == "c01", "the read that was examined and rejected"
    assert replay.replayed_call_ids == ()
    assert replay.replayed is False


def test_a_read_that_begins_at_the_first_line_is_still_not_proven_whole():
    # The gutter says where a read *began* and that it has no gap. Nothing in a
    # read result states a file's total extent, so a read cut off after line 2
    # cannot be told from one that reached the end — and replaying on it would
    # put a truncated file into the resolved solution.
    rendered = "     1\tclass TokenBucket:\n     2\t    pass\n"
    session = with_read_output(ONLY_EDITED, "c01", rendered)
    chunks, chain = one_chain_over(session)

    replay = replays(session, chunks, chain)["src/api/limiter.py"]

    assert rendered_extent(rendered) == (1, 2)
    assert replay.status is ReplayStatus.UNPROVEN_READ
    assert replay.content is None


def test_a_read_whose_output_carries_no_gutter_states_nothing_about_its_extent():
    session = with_read_output(ONLY_EDITED, "c01", "class TokenBucket:\n    pass\n")
    chunks, chain = one_chain_over(session)

    replay = replays(session, chunks, chain)["src/api/limiter.py"]

    assert replay.status is ReplayStatus.UNPROVEN_READ
    assert replay.content is None


def test_a_file_the_chain_never_read_has_no_base_at_all():
    session, chunks, chains = classified(TWO_READS)
    second = next(chain for chain in chains if chain.id == "C2")

    replay = replays(session, chunks, second)["config/settings.toml"]

    assert replay.status is ReplayStatus.NO_BASE
    assert replay.base_call_id is None
    assert replay.content is None


def test_a_read_that_followed_the_first_edit_is_not_a_base():
    # `interleaved_intents` reads the sampler again after having edited it, so
    # that second read shows the file as the edit left it, not the content the
    # replay would have to start from.
    session, chunks, chains = classified(TWO_READS)
    first = next(chain for chain in chains if chain.id == "C1")

    replay = replays(session, chunks, first)["src/rates/sampler.py"]

    assert replay.base_call_id == "c02", "the last read before the first kept edit"
    assert replay.status is ReplayStatus.PARTIAL_READ


def test_a_read_the_agent_saw_fail_is_no_base():
    document = json.loads(CORPUS.text(ONLY_EDITED))
    marked = 0
    for message in document["messages"]:
        result = message["tool_result"]
        if result is not None and result["call_id"] == "c01":
            result["is_error"] = True
            marked += 1
    assert marked == 1
    session = loads(Session, json.dumps(document))
    chunks, chain = one_chain_over(session)

    replay = replays(session, chunks, chain)["src/api/limiter.py"]

    assert replay.status is ReplayStatus.NO_BASE, "a refused read shows nothing of the file"
    assert replay.base_call_id is None


def test_an_entry_whose_write_is_not_a_full_write_falls_through_to_the_reads():
    # `SolutionFile` does not check that its `write` is a full write, so an
    # entry a caller built from an edit reaches the fall-through in the base
    # lookup. Built by hand here because `solution_files` cannot produce it.
    edit = LocatedCall(
        call=ToolCall(
            id="c01",
            name="Edit",
            category=ToolCategory.WRITE,
            args={"file_path": "src/a.py", "old_string": "a", "new_string": "b"},
        ),
        chunk_id="k01",
        message_id="m01",
        message_index=0,
        result=None,
    )
    entry = SolutionFile(path="src/a.py", write=edit, edits=(), superseded=(), failed=())

    replay = replay_file(entry, ())

    assert replay.status is ReplayStatus.NO_BASE
    assert replay.content is None


# --- an edit the replay cannot place --------------------------------------


def test_an_edit_whose_anchor_the_base_does_not_hold_falls_back():
    session = with_argument(WRITE_THEN_TWO_EDITS, "c05", "old_string", "no such line")

    replay = slug_replay(session)

    assert replay.status is ReplayStatus.ANCHOR_NOT_FOUND
    assert replay.content is None, "a partly replayed file is not the file"
    assert replay.replayed_call_ids == ()


def test_an_edit_that_names_no_replacement_at_all_falls_back():
    session = with_renamed_arguments(
        WRITE_THEN_TWO_EDITS, "c05", {"old_string": "pattern", "new_string": "source"}
    )

    replay = slug_replay(session)

    assert replay.status is ReplayStatus.ANCHOR_NOT_FOUND
    assert replay.content is None


def test_an_anchor_that_holds_no_word_names_no_anchor():
    session = with_argument(WRITE_THEN_TWO_EDITS, "c05", "old_string", "   ")

    assert slug_replay(session).status is ReplayStatus.ANCHOR_NOT_FOUND


def test_an_edit_whose_anchor_occurs_twice_in_the_base_falls_back():
    session = with_argument(WRITE_THEN_TWO_EDITS, "c05", "old_string", "unicodedata")

    replay = slug_replay(session)

    assert replay.status is ReplayStatus.ANCHOR_AMBIGUOUS
    assert replay.content is None


# --- reading a read's extent ----------------------------------------------


def test_the_extent_of_a_read_is_the_first_and_last_line_its_gutter_names():
    results = corpus_results_by_call_id(CORPUS.session(ONLY_EDITED))

    assert rendered_extent(results["c01"].output) == (14, 19)


def test_the_extent_of_a_read_that_covers_a_whole_short_file_starts_at_one():
    results = corpus_results_by_call_id(CORPUS.session("rewritten_after_failed_edit"))

    assert rendered_extent(results["c01"].output) == (1, 11)


@pytest.mark.parametrize(
    ("output", "reason"),
    [
        ("", "nothing was rendered"),
        ("class A:\n    pass\n", "no line carries a gutter"),
        ("     1\ta\nb\n", "one line carries no gutter"),
        ("     1\ta\n     3\tb\n", "the numbering has a gap"),
        ("     3\ta\n     2\tb\n", "the numbering runs backwards"),
    ],
)
def test_an_output_that_is_not_a_contiguous_gutter_has_no_extent(output, reason):
    assert rendered_extent(output) is None, reason


def test_a_read_rendered_with_a_blank_line_outside_the_gutter_has_no_extent():
    # `write_then_two_edits` renders a blank file line as an empty line with no
    # number, so its numbering jumps 1 -> 2 across an unnumbered line. The
    # extent of such an output is not readable, and claiming one would claim a
    # line count the output does not state.
    results = corpus_results_by_call_id(CORPUS.session(WRITE_THEN_TWO_EDITS))

    assert rendered_extent(results["c01"].output) is None


# --- the value object ------------------------------------------------------


def test_a_replay_that_states_a_content_without_having_replayed_is_a_programming_error():
    with pytest.raises(ValueError, match="content"):
        FileReplay(
            path="src/a.py",
            content="x",
            status=ReplayStatus.NO_BASE,
            base_call_id=None,
            replayed_call_ids=(),
        )


def test_a_replay_that_replayed_without_a_content_is_a_programming_error():
    with pytest.raises(ValueError, match="content"):
        FileReplay(
            path="src/a.py",
            content=None,
            status=ReplayStatus.REPLAYED,
            base_call_id="c01",
            replayed_call_ids=(),
        )


def test_a_replay_that_replayed_without_a_base_is_a_programming_error():
    with pytest.raises(ValueError, match="base"):
        FileReplay(
            path="src/a.py",
            content="x",
            status=ReplayStatus.REPLAYED,
            base_call_id=None,
            replayed_call_ids=(),
        )


def test_a_replay_that_names_replayed_edits_without_having_replayed_is_a_programming_error():
    with pytest.raises(ValueError, match="replayed"):
        FileReplay(
            path="src/a.py",
            content=None,
            status=ReplayStatus.NO_BASE,
            base_call_id=None,
            replayed_call_ids=("c01",),
        )


def test_a_replay_that_names_no_file_is_a_programming_error():
    with pytest.raises(ValueError, match="file"):
        FileReplay(
            path="  ",
            content=None,
            status=ReplayStatus.NO_BASE,
            base_call_id=None,
            replayed_call_ids=(),
        )


# --- over the whole corpus -------------------------------------------------


@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_the_replay_is_about_exactly_the_files_of_the_solution(name):
    session, chunks, chains = classified(name)

    for chain in chains:
        assert [replay.path for replay in replay_files(session, chunks, chain)] == [
            entry.path for entry in solution_files(session, chunks, chain)
        ]


@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_a_content_is_stated_exactly_when_the_file_was_replayed(name):
    session, chunks, chains = classified(name)

    for chain in chains:
        for replay in replay_files(session, chunks, chain):
            assert (replay.content is not None) is (replay.status is ReplayStatus.REPLAYED)


@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_a_file_without_an_accepted_full_write_is_never_replayed(name):
    # The honest half of the ticket: the replay starts from content the
    # transcript states verbatim, and only a full write does that. A file the
    # chain only edited falls back, with the reason named.
    session, chunks, chains = classified(name)

    for chain in chains:
        entries = {entry.path: entry for entry in solution_files(session, chunks, chain)}
        for replay in replay_files(session, chunks, chain):
            if entries[replay.path].write is None:
                assert replay.status in {
                    ReplayStatus.NO_BASE,
                    ReplayStatus.PARTIAL_READ,
                    ReplayStatus.UNPROVEN_READ,
                }, replay.path


def test_the_corpus_replays_at_least_one_file():
    # The table the corpus-wide tests range over must not be able to state
    # nothing: without a replayed file every assertion above would hold
    # vacuously.
    replayed = [
        replay.path
        for name in FIXTURE_NAMES
        for session, chunks, chains in (classified(name),)
        for chain in chains
        for replay in replay_files(session, chunks, chain)
        if replay.status is ReplayStatus.REPLAYED
    ]

    assert FINAL_SLUG_PATH in replayed
    assert len(replayed) >= 1


@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_replaying_the_same_chain_twice_gives_the_same_answer(name):
    session, chunks, chains = classified(name)

    for chain in chains:
        assert replay_files(session, chunks, chain) == replay_files(session, chunks, chain)
