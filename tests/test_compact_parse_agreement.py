"""The writer's guard and the reader's structure rules, held against each other (T8.4, #97).

The contract these tests defend is one sentence: **what `idc.compact` renders,
`idc.parse` reads back.** The two sides decide what a structural line is in two
places on purpose — the writer refuses what a model *may write*, the reader
reads what a block *holds* — and they are allowed to differ only in the safe
direction, which is the writer refusing more.

The heading dimension did not differ safely. `idc.compact.narrative` asked for
whitespace after a `#` run, so `##NOTE` was ordinary prose to it, while
`idc.parse` tests the bare `##` prefix and then refuses the line with
`MalformedHeadingError`. A model that wrote one sentence beginning `##NOTE`
produced a block that could never be read again — a dead summary rather than a
re-asked response.

Two tests carry that:

- **The table** walks every `#`-run length one through six, with and without whitespace
  after the run, at indents 0 through 4, and asserts the two verdicts agree:
  either the writer refuses the line, or the block it renders parses.
- **The property** is the criterion that was missing when the defect shipped.
  Over every short string of `#`, space, tab, a backtick, `-` and a letter, no
  response the writer accepts may render a block the reader refuses. It is
  deterministic — `itertools.product`, no randomness and no seed — and it is
  the test that would have caught `##NOTE` without anybody thinking of
  `##NOTE`.

The chain and the resolution come from the shared corpus through the pipeline,
as everywhere else, and no test here calls a model.
"""

import itertools

import pytest

from fixture_corpus import CORPUS
from idc.chunk import chunk_session
from idc.classify import FakeAdapter, classify, gold_script
from idc.compact import (
    DECISIONS_HEADER,
    REJECTED_HEADER,
    Narrative,
    NarrativeContractError,
    as_narrative,
    render_narrative_block,
)
from idc.parse import ParseError, parse_blocks, parse_chains
from idc.resolve import Resolution, resolve
from idc.schema import Chain

ONE_CHAIN = "single_intent"
"""The one-chunk, one-edit, one-run fixture every block here is rendered for."""


@pytest.fixture(scope="module")
def pair() -> tuple[Chain, Resolution]:
    """Return the fixture's single chain and what it resolved to.

    Module-scoped because the property test renders thousands of blocks for it
    and the pipeline's answer for one fixture is the same every time.
    """
    session = CORPUS.session(ONE_CHAIN)
    chunks = chunk_session(session)
    chains = list(classify(session, chunks, FakeAdapter(gold_script(CORPUS.gold(ONE_CHAIN)))))
    assert len(chains) == 1, f"{ONE_CHAIN} is used as a single-chain fixture"
    return chains[0], resolve(session, chunks, chains[0])


def response(*body: str) -> str:
    """Return a well-formed response carrying `body` in **both** narrative sections.

    Args:
        body: The lines under test, in the order a client wrote them.

    Returns:
        The three markers, each once and in order, with `body` as the body of
        `Decisions & constraints` and again as the body of `Rejected
        approaches`. Both positions at once, because they are not the same
        position to the reader: the second body is the last thing above the
        deterministic sections the renderer appends, and a line that enclosed or
        reopened something would do it to `### Solution` from there.
    """
    lines = "".join(f"{line}\n" for line in body)
    return f"## Intent: x\n{DECISIONS_HEADER}\n{lines}{REJECTED_HEADER}\n{lines}"


def written(pair: tuple[Chain, Resolution], *body: str) -> str | None:
    """Return the block the writer renders for `body`, or `None` when it refuses.

    Args:
        pair: The chain and its resolution, as the module fixture holds them.
        body: The lines under test.

    Returns:
        The rendered block, or `None` when `as_narrative` refuses the response.
        A refusal is a verdict and not a failure: the client is asked again, and
        nothing unreadable reaches a file.
    """
    chain, resolution = pair
    try:
        narrative = as_narrative(response(*body))
    except NarrativeContractError:
        return None
    return render_narrative_block(chain, resolution, narrative)


# --- the table over every hash run, spacing and indent --------------------------

HASH_RUNS = range(1, 7)
"""Every ATX run length Markdown knows: one `#` through six."""

INDENTS = range(0, 5)
"""Indents 0 through 4 — past the three the reader still reads as structure."""

HEADING_SHAPES = tuple(itertools.product(HASH_RUNS, (False, True), INDENTS))
"""Every `(run, spaced, indent)` combination the agreement is asserted over."""


def heading_shape(run: int, spaced: bool, indent: int) -> str:
    """Return the body line one table case is about.

    Args:
        run: How many `#` the line opens with, after its indent.
        spaced: Whether a space follows the run — the condition the writer's
            guard used to require before it called the line a heading at all.
        indent: How many leading spaces stand in front of the run.

    Returns:
        The line, always with the same word after the run so that cases differ
        in nothing else.
    """
    return f"{' ' * indent}{'#' * run}{' ' if spaced else ''}NOTE"


def test_the_heading_shape_table_is_not_empty():
    assert len(HEADING_SHAPES) == len(HASH_RUNS) * 2 * len(INDENTS) == 60


@pytest.mark.parametrize(("run", "spaced", "indent"), HEADING_SHAPES)
def test_the_writers_verdict_and_the_readers_agree_on_a_hash_run(pair, run, spaced, indent):
    """Either the writer refuses the line, or the block it renders parses.

    The disagreement this pins out was `##NOTE`: accepted by the writer,
    `MalformedHeadingError` to the reader, at every indent the reader still
    reads as structure.
    """
    line = heading_shape(run, spaced, indent)
    block = written(pair, line)
    if block is None:
        with pytest.raises(NarrativeContractError, match="body line"):
            as_narrative(response(line))
        with pytest.raises(ValueError, match="body line"):
            Narrative(label="x", decisions=(line,))
        with pytest.raises(ValueError, match="body line"):
            Narrative(label="x", rejected=(line,))
        return
    chains = parse_chains(block)
    assert len(chains) == 1
    assert chains[0].id == pair[0].id
    assert line in chains[0].compacted_text


# --- the property the defect slipped through ------------------------------------

ALPHABET = ("#", " ", "\t", "`", "-", "a")
"""The characters a structural line can be built from, and one that cannot.

`#` and the backtick carry the two structures a narrative body may not hold, the
space and the tab are the two indents the reader weighs differently, `-` is a
Setext underline and a bullet, and the letter is content.
"""

SINGLE_LINES = tuple(
    "".join(characters)
    for length in (1, 2, 3, 4, 5)
    for characters in itertools.product(ALPHABET, repeat=length)
)
"""Every string of one to five characters over `ALPHABET`.

Eleven thousand bodies with the pairs below, and under a second of runtime for
them: the generator is wide because it is cheap, not because five is a number
anybody reasoned to.
"""

LINE_PAIRS = tuple(
    itertools.product(
        [line for line in SINGLE_LINES if len(line) <= 2],
        repeat=2,
    )
)
"""Every ordered pair of the one- and two-character strings.

A body of two lines, because two of the shapes a block can be broken by need
one line to open and a second to be swallowed: a fence encloses what follows it,
and a Setext underline turns the line *above* it into a header.
"""

GENERATED_BODIES = tuple([(line,) for line in SINGLE_LINES] + list(LINE_PAIRS))
"""Every generated narrative body, one entry per case the property is checked on."""


def test_the_generated_bodies_are_not_empty():
    assert len(SINGLE_LINES) == 6 + 6**2 + 6**3 + 6**4 + 6**5
    assert len(LINE_PAIRS) == (6 + 6**2) ** 2
    assert len(GENERATED_BODIES) == len(SINGLE_LINES) + len(LINE_PAIRS)


def test_no_response_the_writer_accepts_renders_a_block_the_reader_refuses(pair):
    """The agreement as a property, over bodies nobody chose one by one.

    This is the test the defect would have failed: `##NOTE` is in
    `GENERATED_BODIES` because `#` and a letter are in the alphabet, not because
    anybody thought of it.
    """
    unreadable: list[tuple[tuple[str, ...], str]] = []
    accepted = 0
    for body in GENERATED_BODIES:
        block = written(pair, *body)
        if block is None:
            continue
        accepted += 1
        try:
            parse_chains(block)
        except ParseError as exc:
            unreadable.append((body, f"{type(exc).__name__}: {exc}"))
    assert unreadable == []
    assert accepted > len(GENERATED_BODIES) // 2, (
        "a property over bodies the writer refuses wholesale proves nothing; "
        f"{accepted} of {len(GENERATED_BODIES)} were accepted"
    )


# --- prose both sides accept, and must keep accepting ---------------------------

LEGITIMATE_LINES = {
    "a_one_hash_tag": "#NOTE",
    "a_six_hash_tag": "######NOTE",
    "an_issue_reference": "- fixes #42 in the parser",
    "a_header_inside_inline_code": "- never write `### Solution` yourself",
}
"""Lines both sides accept, which the stricter guard may not take away.

The last one is load-bearing: the instruction tells a model not to write the
deterministic headers itself, and a guard that refused the sentence saying so
would refuse the sentence most worth writing. `#NOTE` is prose to the reader
(one `#` is no marker of its) and `######NOTE` is a section header it tolerates
like any other it did not render, so neither is a disagreement.
"""


@pytest.mark.parametrize("shape", sorted(LEGITIMATE_LINES))
def test_a_legitimate_hash_line_still_round_trips(pair, shape):
    line = LEGITIMATE_LINES[shape]
    block = written(pair, line)
    assert block is not None, f"{line!r} is prose both sides accept"
    chains = parse_chains(block)
    assert len(chains) == 1
    assert line in chains[0].compacted_text
    assert Narrative(label="x", decisions=(line,)).decisions == (line,)


def test_the_legitimate_lines_cover_every_enumerated_shape():
    assert set(LEGITIMATE_LINES) == {
        "a_header_inside_inline_code",
        "a_one_hash_tag",
        "a_six_hash_tag",
        "an_issue_reference",
    }


# --- the bytes on disk the stricter guard may not move --------------------------


def narrative_body_lines(text: str) -> list[str]:
    """Return every narrative body line of a rendered block sequence.

    Args:
        text: One or more blocks, as they stand on disk.

    Returns:
        The non-blank lines of each block's `Decisions & constraints` and
        `Rejected approaches` sections. The reader is what locates them, so the
        statement the caller makes is about the two sections a model writes and
        not about the deterministic tail, whose fenced bodies are the writer's
        own and no model's to produce.
    """
    return [
        line
        for block in parse_blocks(text)
        for header in (DECISIONS_HEADER, REJECTED_HEADER)
        for line in block.body(header).splitlines()
        if line.strip()
    ]


def test_the_golden_blocks_hold_no_narrative_line_the_stricter_guard_refuses():
    """Every narrative body line of the ten golden sequences stays writable.

    The goldens are the writer's own output, so a guard that refused a line in
    one of them would have changed what the writer renders. Read off the files
    rather than re-rendered, which is what makes this a statement about the
    bytes on disk.
    """
    names = CORPUS.compact_names()
    assert len(names) == 10, "the corpus holds ten golden block sequences"
    for name in names:
        lines = narrative_body_lines(CORPUS.compact_markdown(name))
        assert lines, f"{name} holds narrative body lines"
        for line in lines:
            assert Narrative(label="x", decisions=(line,)).decisions == (line,)


def test_the_hand_written_block_still_parses_and_holds_writable_lines():
    """The hand-written block of `already_compacted_block.json` is unaffected.

    Its narrative sections carry real prose — the shape the writer renders once
    a model fills them — so both halves of the contract are asserted on it: the
    reader still reads it, and every line of its narrative sections is still a
    line the writer could have produced.
    """
    text = CORPUS.session("already_compacted_block").messages[0].content
    chains = parse_chains(text)
    assert [chain.id for chain in chains] == ["C1", "C2"]
    lines = narrative_body_lines(text)
    assert len(lines) > 4, "the hand-written block carries prose in both sections"
    for line in lines:
        assert Narrative(label="x", decisions=(line,)).decisions == (line,)
