"""Reading a rendered compacted block back into chains (T8.1, #34).

The oracle is the writer, not a second spelling of the format: the golden block
sequences of `tests/fixtures/gold/compact/<name>.md` are the bytes T6.1 (#26)
renders, and the round-trip tests assert that what the parser hands back
re-joins to those exact bytes. Every block this file *builds* is composed with
`idc.compact`'s own `chain_heading`, `SECTION_HEADERS` and `EMPTY_SECTION`, so a
header string is never written out here; only the deliberately malformed
headings are literals, because being wrong is the point of them.

`tests/fixtures/already_compacted_block.json` carries a second, hand-written
shape in `messages[0].content`: real prose in the narrative sections and a
language tag on a fence. It is pinned here because the writer will render that
shape as soon as T6.2 fills the narrative sections, and a parser that only
accepts the `- (none)` placeholder would break on the day it lands.
"""

import ast
import builtins
import io
import random
import time
from pathlib import Path

import pytest

import idc.parse
from fixture_corpus import CORPUS
from idc.chunk import chunk_session
from idc.compact import EMPTY_SECTION, SECTION_HEADERS, chain_heading
from idc.parse import (
    BLOCK_SEPARATOR,
    HEADING_PATTERN,
    HEADING_SHAPE,
    PLACEHOLDER_CHUNK_PREFIX,
    DuplicateChainError,
    MalformedHeadingError,
    MissingSectionError,
    NoHeadingError,
    ParsedBlock,
    ParsedSection,
    ParseError,
    StrayContentError,
    UnterminatedFenceError,
    join_blocks,
    parse_blocks,
    parse_chains,
    placeholder_chunk_id,
    split_blocks,
)
from idc.schema import Chain

COMPACT_NAMES = CORPUS.compact_names()
"""Every transcript the corpus holds golden compaction Markdown for."""

MULTI_CHAIN_NAMES = ("interleaved_intents", "two_sequential_intents")
"""The golden sequences that hold more than one block."""

VARIABLE_FENCE = "rewritten_after_failed_edit"
"""The golden sequence whose Solution opens a four-backtick fence.

Its fenced content holds a three-backtick run and lines that start with `#`, so
it is the case that tells a fence-aware splitter from a line-prefix one.
"""

PROSE_FIXTURE = "already_compacted_block"
"""The fixture whose first message holds a hand-written block with prose."""

STRUCTURAL_INDENTS = (0, 1, 2, 3)
"""Every indentation at which a line is still structure, written out on purpose.

CommonMark reads a heading, a section header and a code fence that carries up to
three leading spaces as that thing, and a line indented by four as an indented
code block. The numbers are literals here rather than imported from the parser:
a table taken from the thing under test would agree with it by construction.
"""

CONTENT_INDENT = 4
"""The first indentation at which a line is content and no longer structure."""

REFUSALS = (
    DuplicateChainError,
    MalformedHeadingError,
    MissingSectionError,
    NoHeadingError,
    StrayContentError,
    UnterminatedFenceError,
)
"""Every refusal the parser raises for a malformed block."""

PARSE_SOURCES = sorted(Path(__file__).resolve().parents[1].joinpath("src/idc/parse").glob("*.py"))
"""Every source file of the parser — what the import census below reads."""

ALLOWED_IDC_IMPORTS = {
    "idc.compact",
    "idc.parse.blocks",
    "idc.parse.errors",
    "idc.schema",
    "idc.schema.models",
}
"""The only `idc` modules the parser may reach: the writer and the schema."""

ALLOWED_FOREIGN_IMPORTS = {"collections.abc", "dataclasses", "re"}
"""The only modules outside `idc` the parser may reach — all of them stdlib."""


def built(
    chain_id: str = "C1",
    label: str = "Keep the parser and the writer on one format",
    headers: tuple[str, ...] = SECTION_HEADERS,
    bodies: dict[str, str] | None = None,
) -> str:
    """Compose one block out of the writer's own heading and header constants."""
    bodies = bodies or {}
    heading = chain_heading(Chain(id=chain_id, label=label, chunk_ids=["K1"]))
    lines = [heading]
    for header in headers:
        lines.append(header)
        lines.append(bodies.get(header, EMPTY_SECTION))
    return "\n".join(lines) + "\n"


def headings_of(markdown: str) -> list[str]:
    """Return the heading line of every block of a golden sequence."""
    return [block.splitlines()[0] for block in split_blocks(markdown)]


def prose_block() -> str:
    """Return the hand-written block the `already_compacted_block` fixture holds."""
    content = CORPUS.session(PROSE_FIXTURE).messages[0].content
    assert "```python" in content, f"{PROSE_FIXTURE} no longer carries a language-tagged fence"
    assert EMPTY_SECTION not in content, f"{PROSE_FIXTURE} no longer carries real prose"
    return content


# --- the golden round trip ------------------------------------------------------


def test_the_corpus_offers_golden_block_sequences():
    # Every golden test below is parametrized over this list; an empty one would
    # turn all of them into a silent skip.
    assert COMPACT_NAMES


@pytest.mark.parametrize("name", COMPACT_NAMES)
def test_every_golden_sequence_splits_back_into_its_own_bytes(name):
    markdown = CORPUS.compact_markdown(name)

    assert join_blocks(split_blocks(markdown)) == markdown


@pytest.mark.parametrize("name", COMPACT_NAMES)
def test_every_parsed_block_holds_its_own_text_verbatim(name):
    markdown = CORPUS.compact_markdown(name)

    assert join_blocks([block.text for block in parse_blocks(markdown)]) == markdown


@pytest.mark.parametrize("name", COMPACT_NAMES)
def test_every_parsed_chain_carries_its_block_as_compacted_text(name):
    markdown = CORPUS.compact_markdown(name)

    chains = parse_chains(markdown)

    assert [chain.compacted for chain in chains] == [True] * len(chains)
    assert join_blocks([chain.compacted_text for chain in chains]) == markdown


@pytest.mark.parametrize("name", COMPACT_NAMES)
def test_parsing_a_sequence_twice_gives_the_same_blocks(name):
    markdown = CORPUS.compact_markdown(name)

    assert parse_blocks(markdown) == parse_blocks(markdown)


@pytest.mark.parametrize("name", COMPACT_NAMES)
def test_every_parsed_block_carries_the_four_written_sections(name):
    markdown = CORPUS.compact_markdown(name)

    for block in parse_blocks(markdown):
        assert [section.header for section in block.sections] == list(SECTION_HEADERS)


@pytest.mark.parametrize("name", COMPACT_NAMES)
def test_every_parsed_heading_is_the_one_the_writer_would_render(name):
    markdown = CORPUS.compact_markdown(name)

    rendered = [chain_heading(block.as_chain()) for block in parse_blocks(markdown)]

    assert rendered == headings_of(markdown)


# --- more than one chain in one sequence ----------------------------------------


@pytest.mark.parametrize("name", MULTI_CHAIN_NAMES)
def test_a_multi_chain_sequence_yields_one_chain_per_heading(name):
    markdown = CORPUS.compact_markdown(name)

    chains = parse_chains(markdown)

    assert [chain.id for chain in chains] == ["C1", "C2"]
    assert [chain_heading(chain) for chain in chains] == headings_of(markdown)


@pytest.mark.parametrize("name", MULTI_CHAIN_NAMES)
def test_a_multi_chain_sequence_keeps_document_order(name):
    markdown = CORPUS.compact_markdown(name)

    blocks = parse_blocks(markdown)

    assert [block.line for block in blocks] == sorted(block.line for block in blocks)
    assert markdown.index(blocks[0].text) < markdown.index(blocks[1].text)


def test_a_block_is_cut_at_the_next_heading_and_not_at_the_blank_line_before_it():
    markdown = CORPUS.compact_markdown(MULTI_CHAIN_NAMES[0])

    first = split_blocks(markdown)[0]

    assert first.endswith("\n")
    assert not first.endswith("\n\n")
    assert HEADING_PATTERN.match(first.splitlines()[0]) is not None


# --- fences are content, not structure ------------------------------------------


def test_a_longer_fence_keeps_its_inner_fence_and_hash_lines_in_the_section():
    markdown = CORPUS.compact_markdown(VARIABLE_FENCE)

    blocks = parse_blocks(markdown)
    solution = blocks[0].body(SECTION_HEADERS[2])

    assert len(blocks) == 1
    assert solution.startswith("- `docs/install.md`\n````\n")
    assert "\n```\nuv sync --all-groups\n" in solution
    assert "\n# Installing\n" in solution


def test_a_section_header_inside_a_fence_is_content():
    fenced = "- `notes.md`\n```\n### Rejected approaches\n- (none)\n```"
    text = built(bodies={SECTION_HEADERS[2]: fenced})

    block = parse_blocks(text)[0]

    assert [section.header for section in block.sections] == list(SECTION_HEADERS)
    assert block.body(SECTION_HEADERS[2]) == f"{fenced}\n"


def test_a_chain_heading_inside_a_fence_does_not_open_a_block():
    fenced = "- `notes.md`\n```\n## [C9] Intent: not a heading\n```"
    text = built(bodies={SECTION_HEADERS[2]: fenced})

    blocks = parse_blocks(text)

    assert len(blocks) == 1
    assert blocks[0].chain_id == "C1"


def test_a_fence_may_carry_a_language_tag():
    fenced = "- `src/billing/export.py`\n```python\nreturn 1\n```"
    text = built(bodies={SECTION_HEADERS[2]: fenced})

    block = parse_blocks(text)[0]

    assert block.body(SECTION_HEADERS[2]) == f"{fenced}\n"


# --- prose sections and language tags, the shape T6.2 will write ----------------


def test_the_hand_written_prose_block_round_trips():
    content = prose_block()

    assert join_blocks([chain.compacted_text for chain in parse_chains(content)]) == content


def test_the_hand_written_prose_block_reads_its_two_chains():
    content = prose_block()

    chains = parse_chains(content)

    assert [chain.id for chain in chains] == ["C1", "C2"]
    assert chains[0].label == "Make the CSV export emit UTC timestamps"
    assert chains[1].label == "Add a per-token rate limiter to the API"


def test_the_hand_written_prose_block_keeps_its_narrative_sections():
    block = parse_blocks(prose_block())[1]

    decisions = block.body(SECTION_HEADERS[0])
    rejected = block.body(SECTION_HEADERS[1])

    assert decisions.startswith("- The limiter keys on the API token")
    assert decisions.endswith("one node.\n")
    assert rejected.count("\n- ") == 1
    assert "`slowapi`: rejected" in rejected


# --- unknown sections ----------------------------------------------------------


def test_an_unknown_section_is_kept_in_document_order():
    notes = "### Notes"
    text = built(headers=(*SECTION_HEADERS, notes), bodies={notes: "- hand-added, keep me"})

    block = parse_blocks(text)[0]

    assert [section.header for section in block.sections] == [*SECTION_HEADERS, notes]
    assert block.body(notes) == "- hand-added, keep me\n"


def test_an_unknown_section_survives_the_round_trip_verbatim():
    notes = "### Notes"
    text = built(headers=(SECTION_HEADERS[0], notes, *SECTION_HEADERS[1:]), bodies={notes: "- x"})

    chains = parse_chains(text)

    assert chains[0].compacted_text == text
    assert join_blocks([chain.compacted_text for chain in chains]) == text


def test_asking_for_a_section_the_block_does_not_hold_is_the_callers_mistake():
    block = parse_blocks(built())[0]

    with pytest.raises(ValueError, match="### Notes") as caught:
        block.body("### Notes")

    assert not isinstance(caught.value, ParseError)


# --- the chunk ids a block does not state --------------------------------------


def test_a_parsed_chain_names_one_placeholder_chunk_derived_from_its_id():
    chains = parse_chains(CORPUS.compact_markdown(MULTI_CHAIN_NAMES[0]))

    assert [chain.chunk_ids for chain in chains] == [["compacted:C1"], ["compacted:C2"]]
    assert placeholder_chunk_id("C2") == f"{PLACEHOLDER_CHUNK_PREFIX}C2"


def test_two_parsed_chains_never_share_a_placeholder_chunk_id():
    chains = parse_chains(CORPUS.compact_markdown(MULTI_CHAIN_NAMES[1]))

    named = [chunk_id for chain in chains for chunk_id in chain.chunk_ids]

    assert len(set(named)) == len(named)


@pytest.mark.parametrize("name", COMPACT_NAMES)
def test_a_placeholder_chunk_id_is_no_chunk_id_the_chunker_hands_out(name):
    produced = {chunk.id for chunk in chunk_session(CORPUS.session(name))}

    assert produced
    assert not any(chunk_id.startswith(PLACEHOLDER_CHUNK_PREFIX) for chunk_id in produced)


# --- malformed input -----------------------------------------------------------


def test_a_text_without_any_heading_is_refused():
    text = f"{SECTION_HEADERS[0]}\n{EMPTY_SECTION}\n"

    with pytest.raises(NoHeadingError) as caught:
        parse_blocks(text)

    assert caught.value.line_count == 2
    assert "## [C" in str(caught.value)


def test_an_empty_text_is_refused_as_holding_no_heading():
    with pytest.raises(NoHeadingError) as caught:
        parse_blocks("")

    assert caught.value.line_count == 0


def test_a_heading_whose_id_is_not_a_chain_id_is_refused():
    text = "## [chain-1] Intent: wrong id shape\n" + built().partition("\n")[2]

    with pytest.raises(MalformedHeadingError) as caught:
        parse_blocks(text)

    assert caught.value.line == 1
    assert caught.value.text == "## [chain-1] Intent: wrong id shape"
    assert "line 1" in str(caught.value)
    assert "chain-1" in str(caught.value)


def test_a_heading_numbered_from_zero_is_refused():
    text = "## [C0] Intent: chains are numbered from one\n" + built().partition("\n")[2]

    with pytest.raises(MalformedHeadingError) as caught:
        parse_blocks(text)

    assert caught.value.line == 1
    assert "C0" in str(caught.value)


def test_a_duplicate_chain_id_in_one_sequence_is_refused():
    text = join_blocks([built(chain_id="C1"), built(chain_id="C1", label="again")])

    with pytest.raises(DuplicateChainError) as caught:
        parse_blocks(text)

    assert caught.value.chain_id == "C1"
    assert caught.value.first_line == 1
    assert caught.value.line == 11
    assert "line 1" in str(caught.value)
    assert "line 11" in str(caught.value)


def test_content_before_the_first_heading_is_refused():
    text = f"stray note a hand left here\n{built()}"

    with pytest.raises(StrayContentError) as caught:
        parse_blocks(text)

    assert caught.value.line == 1
    assert caught.value.chain_id is None
    assert caught.value.text == "stray note a hand left here"
    assert "line 1" in str(caught.value)


def test_content_between_a_heading_and_its_first_section_is_refused():
    heading, newline, rest = built().partition("\n")
    text = f"{heading}{newline}stray note under the heading\n{rest}"

    with pytest.raises(StrayContentError) as caught:
        parse_blocks(text)

    assert caught.value.line == 2
    assert caught.value.chain_id == "C1"
    assert caught.value.text == "stray note under the heading"


def test_a_blank_line_before_the_first_heading_is_not_content():
    text = f"\n{built()}"

    assert [block.chain_id for block in parse_blocks(text)] == ["C1"]


@pytest.mark.parametrize("dropped", range(len(SECTION_HEADERS)))
def test_a_missing_section_header_is_refused(dropped):
    headers = tuple(h for h in SECTION_HEADERS if h != SECTION_HEADERS[dropped])

    with pytest.raises(MissingSectionError) as caught:
        parse_blocks(built(headers=headers))

    assert caught.value.chain_id == "C1"
    assert caught.value.header == SECTION_HEADERS[dropped]
    assert caught.value.present == headers
    assert SECTION_HEADERS[dropped] in str(caught.value)
    assert "line 1" in str(caught.value)


def test_a_block_with_no_section_at_all_is_refused():
    heading = chain_heading(Chain(id="C1", label="nothing but a heading", chunk_ids=["K1"]))

    with pytest.raises(MissingSectionError) as caught:
        parse_blocks(f"{heading}\n")

    assert caught.value.present == ()
    assert caught.value.header == SECTION_HEADERS[0]


def test_an_unterminated_fence_is_refused():
    text = built(bodies={SECTION_HEADERS[2]: "- `notes.md`\n```\nnever closed"})

    with pytest.raises(UnterminatedFenceError) as caught:
        parse_blocks(text)

    assert caught.value.line == 8
    assert caught.value.fence == "```"
    assert "line 8" in str(caught.value)


def test_a_short_closing_fence_does_not_close_a_long_one():
    text = built(bodies={SECTION_HEADERS[2]: "- `notes.md`\n````\n```\nstill open"})

    with pytest.raises(UnterminatedFenceError) as caught:
        parse_blocks(text)

    assert caught.value.fence == "````"


@pytest.mark.parametrize("refusal", REFUSALS)
def test_every_refusal_is_one_catchable_family(refusal):
    assert issubclass(refusal, ParseError)
    assert not issubclass(refusal, ValueError)


def test_the_refusal_table_is_the_packages_own():
    exported = {name for name in idc.parse.__all__ if name.endswith("Error")}

    assert REFUSALS
    assert {refusal.__name__ for refusal in REFUSALS} | {"ParseError"} == exported


# --- indentation: what is structure and what is content -------------------------


@pytest.mark.parametrize("closing", ["````", "`````"])
def test_a_long_closing_fence_closes_a_short_one(closing):
    # The rule is "at least as long", not "exactly as long": `UnterminatedFenceError`
    # promises a line of *at least* N backticks closes a fence, and nothing used to
    # hold the parser to the longer half of that promise.
    fenced = f"- `notes.md`\n```\nline a\n{closing}"
    text = built(bodies={SECTION_HEADERS[2]: fenced})

    block = parse_blocks(text)[0]

    assert [section.header for section in block.sections] == list(SECTION_HEADERS)
    assert block.body(SECTION_HEADERS[2]) == f"{fenced}\n"
    assert block.body(SECTION_HEADERS[3]) == f"{EMPTY_SECTION}\n"


@pytest.mark.parametrize("spaces", STRUCTURAL_INDENTS)
def test_a_chain_heading_indented_like_structure_opens_a_block(spaces):
    indent = " " * spaces
    text = join_blocks([built(chain_id="C1"), f"{indent}{built(chain_id='C2')}"])

    blocks = parse_blocks(text)

    assert [block.chain_id for block in blocks] == ["C1", "C2"]
    assert blocks[1].text.startswith(f"{indent}## [C2]")


def test_a_chain_heading_indented_like_content_stays_in_its_section():
    indent = " " * CONTENT_INDENT
    swallowed = f"{EMPTY_SECTION}\n{indent}## [C2] Intent: indented out of structure"
    text = built(bodies={SECTION_HEADERS[3]: swallowed})

    blocks = parse_blocks(text)

    # Four spaces make an indented code block, so the line is content — but it is
    # content of the section it stands in, and the block still holds it verbatim.
    # Nothing is dropped, which is the half that used to be missing.
    assert [block.chain_id for block in blocks] == ["C1"]
    assert blocks[0].body(SECTION_HEADERS[3]) == f"{swallowed}\n"
    assert blocks[0].text == text


@pytest.mark.parametrize("spaces", STRUCTURAL_INDENTS)
def test_a_section_header_indented_like_structure_is_a_section_header(spaces):
    header = SECTION_HEADERS[2]
    text = built().replace(f"\n{header}\n", f"\n{' ' * spaces}{header}\n")

    block = parse_blocks(text)[0]

    assert [section.header for section in block.sections] == list(SECTION_HEADERS)
    assert block.body(header) == f"{EMPTY_SECTION}\n"
    assert block.text == text


def test_a_section_header_indented_like_content_leaves_its_section_missing():
    header = SECTION_HEADERS[2]
    text = built().replace(f"\n{header}\n", f"\n{' ' * CONTENT_INDENT}{header}\n")

    with pytest.raises(MissingSectionError) as caught:
        parse_blocks(text)

    assert caught.value.header == header
    assert caught.value.present == (SECTION_HEADERS[0], SECTION_HEADERS[1], SECTION_HEADERS[3])


@pytest.mark.parametrize("spaces", STRUCTURAL_INDENTS)
def test_an_opening_fence_indented_like_structure_opens_one(spaces):
    indent = " " * spaces
    fenced = f"- `notes.md`\n{indent}```\n{SECTION_HEADERS[1]}\n{indent}```"
    text = built(bodies={SECTION_HEADERS[2]: fenced})

    block = parse_blocks(text)[0]

    assert [section.header for section in block.sections] == list(SECTION_HEADERS)
    assert block.body(SECTION_HEADERS[2]) == f"{fenced}\n"


def test_an_opening_fence_indented_like_content_opens_none():
    indent = " " * CONTENT_INDENT
    notes = "### Notes"
    fenced = f"- `notes.md`\n{indent}```\n{notes}\n{indent}```"
    text = built(bodies={SECTION_HEADERS[2]: fenced})

    block = parse_blocks(text)[0]

    # Not a fence, so what it tried to hide is structure: the header inside it
    # opens a section of its own. The block is still accepted, because all four
    # written sections are there, and its bytes are untouched.
    assert [section.header for section in block.sections] == [
        *SECTION_HEADERS[:3],
        notes,
        SECTION_HEADERS[3],
    ]
    assert block.text == text


@pytest.mark.parametrize("spaces", STRUCTURAL_INDENTS)
def test_a_closing_fence_indented_like_structure_closes_the_fence(spaces):
    fenced = f"- `notes.md`\n```\nline a\n{' ' * spaces}```"
    text = built(bodies={SECTION_HEADERS[2]: fenced})

    block = parse_blocks(text)[0]

    assert block.body(SECTION_HEADERS[2]) == f"{fenced}\n"
    assert block.body(SECTION_HEADERS[3]) == f"{EMPTY_SECTION}\n"


def test_a_closing_fence_indented_like_content_leaves_the_fence_open():
    fenced = f"- `notes.md`\n```\nline a\n{' ' * CONTENT_INDENT}```"
    text = built(bodies={SECTION_HEADERS[2]: fenced})

    with pytest.raises(UnterminatedFenceError) as caught:
        parse_blocks(text)

    assert caught.value.line == 8
    assert caught.value.fence == "```"
    assert f"at most {STRUCTURAL_INDENTS[-1]} space(s)" in str(caught.value)


def test_a_section_cannot_be_escaped_through_an_indented_closing_fence():
    # The escape a review demonstrated: with any indentation closing a fence, the
    # four-space run below ended the fence, the `### Verification` inside the
    # Solution body became a second Verification section, `body()` returned
    # `- ESCAPED` for it, and the block's real Verification was shadowed — with
    # nothing raised. The fence is now genuinely open to the end of the block, so
    # the block is refused instead of being read with its sections rearranged.
    indent = " " * CONTENT_INDENT
    escape = f"- `x.py`\n```\nline a\n{indent}```\n{SECTION_HEADERS[3]}\n- ESCAPED"
    text = built(bodies={SECTION_HEADERS[2]: escape})

    with pytest.raises(UnterminatedFenceError) as caught:
        parse_blocks(text)

    assert caught.value.line == 8


@pytest.mark.parametrize(
    "malformed",
    [
        "## [c1] Intent: a lower-case id",
        "## [C0] Intent: chains are numbered from one",
        "## [chain-1] Intent: wrong id shape",
    ],
)
@pytest.mark.parametrize("spaces", STRUCTURAL_INDENTS)
def test_a_malformed_heading_indented_like_structure_is_never_swallowed(spaces, malformed):
    line = f"{' ' * spaces}{malformed}"
    text = join_blocks([built(chain_id="C1"), f"{line}\n"])

    with pytest.raises(MalformedHeadingError) as caught:
        parse_blocks(text)

    assert caught.value.line == 11
    assert caught.value.text == line


def test_the_heading_shape_the_messages_state_is_the_one_the_writer_renders():
    # `HEADING_SHAPE` is prose in two refusals, so no round trip touches it; its
    # two placeholders filled in have to give back exactly what the writer emits.
    label = "Keep the error prose and the writer on one heading"
    rendered = chain_heading(Chain(id="C1", label=label, chunk_ids=["K1"]))

    assert HEADING_SHAPE.replace("C<n>", "C1").replace("<label>", label) == rendered
    assert HEADING_PATTERN.match(HEADING_SHAPE) is None


# --- what ends a line, and what only Python thinks ends one ---------------------


NOT_LINE_TERMINATORS = (
    ("vertical tab", "\x0b"),
    ("form feed", "\x0c"),
    ("file separator", "\x1c"),
    ("group separator", "\x1d"),
    ("record separator", "\x1e"),
    ("next line", "\x85"),
    ("line separator", " "),
    ("paragraph separator", " "),
)
"""The seven characters `str.splitlines()` breaks on and CommonMark does not.

Written as code points because they are invisible. CommonMark knows three line
endings — LF, CR and CRLF — so a line that holds one of these holds a character
and nothing more, and the structure on either side of it is one line's worth.
A scanner that disagreed read the run after a form feed as a line of its own,
starting at column zero, which is how a closing fence written inside a Solution
body ended a fence opened flush and handed the headers beneath it to the
splitter — the very escape the indentation rule exists to refuse.
"""

CONTENT_WHITESPACE = (
    ("tab", "\t"),
    ("space and tab", " \t"),
    ("two spaces and tab", "  \t"),
    ("no-break space", " "),
    ("em space", " "),
    ("ideographic space", "　"),
    ("zero-width space", "​"),
)
"""Whitespace that is not a space, and therefore not a structural indent.

The parser's indent allowance is counted in literal spaces: a tab is not
expanded to the next tab stop and an exotic space is not folded to one. Both
readings agree with CommonMark on every line here — a tab within three spaces
reaches column four, so CommonMark reads it as indented code, and the others are
not Unicode whitespace CommonMark strips — but they agree only because the
arithmetic works out, which is a reason to pin the result rather than restate it.
"""


@pytest.mark.parametrize(("name", "character"), NOT_LINE_TERMINATORS)
def test_a_non_terminator_before_a_closing_fence_does_not_close_it(name, character):
    # The escape of the previous round through a second door: with the scanner
    # splitting on everything Python calls a line break, a form feed before the
    # run made it look flush, the fence closed, `### Verification` inside the
    # Solution body became a section of its own, `body()` returned `- ESCAPED`
    # and the block's real Verification was shadowed with nothing raised.
    escape = f"- `x.py`\n```\nline a\n{character}```\n{SECTION_HEADERS[3]}\n- ESCAPED"
    text = built(bodies={SECTION_HEADERS[2]: escape})

    with pytest.raises(UnterminatedFenceError) as caught:
        parse_blocks(text)

    assert caught.value.line == 8


@pytest.mark.parametrize(("name", "character"), NOT_LINE_TERMINATORS)
def test_a_non_terminator_before_a_chain_heading_opens_no_block(name, character):
    swallowed = f"{EMPTY_SECTION}\n{character}## [C2] Intent: smuggled past the line split"
    text = built(bodies={SECTION_HEADERS[3]: swallowed})

    blocks = parse_blocks(text)

    # One line, so one piece of content: the heading does not begin a block and
    # does not cut the first one short of its last section.
    assert [block.chain_id for block in blocks] == ["C1"]
    assert blocks[0].body(SECTION_HEADERS[3]) == f"{swallowed}\n"
    assert blocks[0].text == text


@pytest.mark.parametrize(("name", "character"), NOT_LINE_TERMINATORS)
def test_a_non_terminator_does_not_shift_the_line_a_refusal_names(name, character):
    # Every refusal names a line number, so a character counted as a break would
    # make every message below it point one line too far.
    bodies = {
        SECTION_HEADERS[0]: f"- a{character}b",
        SECTION_HEADERS[2]: "- `notes.md`\n```\nnever closed",
    }

    with pytest.raises(UnterminatedFenceError) as caught:
        parse_blocks(built(bodies=bodies))

    assert caught.value.line == 8


def test_a_text_that_ends_in_a_newline_gains_no_trailing_empty_line():
    with pytest.raises(NoHeadingError) as caught:
        parse_blocks("one line, and the newline that ends it\n")

    assert caught.value.line_count == 1


def test_a_text_that_does_not_end_in_a_newline_still_parses():
    text = built().removesuffix("\n")

    block = parse_blocks(text)[0]

    assert [section.header for section in block.sections] == list(SECTION_HEADERS)
    assert block.text == text


def test_windows_line_endings_end_a_line():
    text = built().replace("\n", "\r\n")

    block = parse_blocks(text)[0]

    assert [section.header for section in block.sections] == list(SECTION_HEADERS)
    assert block.text == text


def test_a_bare_carriage_return_ends_a_line():
    # CommonMark's third line ending. It is a line ending here too, so a block a
    # hand saved with classic Mac endings is read as the structure it looks like
    # rather than as one very long line.
    text = built().replace("\n", "\r")

    block = parse_blocks(text)[0]

    assert [section.header for section in block.sections] == list(SECTION_HEADERS)
    assert block.text == text


@pytest.mark.parametrize(("name", "prefix"), CONTENT_WHITESPACE)
def test_whitespace_that_is_not_a_space_leaves_a_chain_heading_content(name, prefix):
    swallowed = f"{EMPTY_SECTION}\n{prefix}## [C2] Intent: indented with no space"
    text = built(bodies={SECTION_HEADERS[3]: swallowed})

    blocks = parse_blocks(text)

    assert [block.chain_id for block in blocks] == ["C1"]
    assert blocks[0].body(SECTION_HEADERS[3]) == f"{swallowed}\n"
    assert blocks[0].text == text


@pytest.mark.parametrize(("name", "prefix"), CONTENT_WHITESPACE)
def test_whitespace_that_is_not_a_space_leaves_a_section_header_content(name, prefix):
    header = SECTION_HEADERS[2]
    text = built().replace(f"\n{header}\n", f"\n{prefix}{header}\n")

    with pytest.raises(MissingSectionError) as caught:
        parse_blocks(text)

    assert caught.value.header == header
    assert caught.value.present == (SECTION_HEADERS[0], SECTION_HEADERS[1], SECTION_HEADERS[3])


@pytest.mark.parametrize(("name", "prefix"), CONTENT_WHITESPACE)
def test_whitespace_that_is_not_a_space_opens_no_fence(name, prefix):
    notes = "### Notes"
    fenced = f"- `notes.md`\n{prefix}```\n{notes}\n{prefix}```"
    text = built(bodies={SECTION_HEADERS[2]: fenced})

    block = parse_blocks(text)[0]

    # No fence, so the header between the two runs is structure: it opens a
    # section of its own, exactly as a four-space run does.
    assert [section.header for section in block.sections] == [
        *SECTION_HEADERS[:3],
        notes,
        SECTION_HEADERS[3],
    ]
    assert block.text == text


@pytest.mark.parametrize(("name", "prefix"), CONTENT_WHITESPACE)
def test_whitespace_that_is_not_a_space_closes_no_fence(name, prefix):
    fenced = f"- `notes.md`\n```\nline a\n{prefix}```"
    text = built(bodies={SECTION_HEADERS[2]: fenced})

    with pytest.raises(UnterminatedFenceError) as caught:
        parse_blocks(text)

    assert caught.value.line == 8
    assert caught.value.fence == "```"


def test_a_zero_width_space_before_a_section_header_is_the_prose_the_writer_counts_on():
    # Load-bearing for T6.1: the writer deliberately lets a narrative section
    # keep a zero-width-space-prefixed `### Solution`, on the stated assumption
    # that this parser dedents literal spaces only and then matches
    # `SECTION_HEADERS` by equality. Were the match an `in` or a looser prefix
    # test, that assumption would break here and nowhere else.
    prose = f"{EMPTY_SECTION}\n​{SECTION_HEADERS[2]}"
    text = built(bodies={SECTION_HEADERS[1]: prose})

    block = parse_blocks(text)[0]

    assert [section.header for section in block.sections] == list(SECTION_HEADERS)
    assert block.body(SECTION_HEADERS[1]) == f"{prose}\n"
    assert block.text == text


# --- what a caller built wrongly -----------------------------------------------


def test_a_block_that_states_no_text_is_the_callers_mistake():
    with pytest.raises(ValueError, match="text") as caught:
        ParsedBlock(
            chain_id="C1",
            label="built by hand",
            text="",
            line=1,
            sections=(ParsedSection(header=SECTION_HEADERS[0], body=""),),
        )

    assert not isinstance(caught.value, ParseError)


def test_a_block_whose_id_is_not_a_chain_id_is_the_callers_mistake():
    with pytest.raises(ValueError, match="nope"):
        ParsedBlock(
            chain_id="nope",
            label="built by hand",
            text="x\n",
            line=1,
            sections=(ParsedSection(header=SECTION_HEADERS[0], body=""),),
        )


def test_a_block_without_a_section_is_the_callers_mistake():
    with pytest.raises(ValueError, match="section"):
        ParsedBlock(chain_id="C1", label="built by hand", text="x\n", line=1, sections=())


def test_a_section_without_a_header_is_the_callers_mistake():
    with pytest.raises(ValueError, match="header"):
        ParsedSection(header="   ", body="")


def test_joining_blocks_that_do_not_end_in_a_newline_is_the_callers_mistake():
    with pytest.raises(ValueError, match="newline"):
        join_blocks([built().rstrip("\n"), built(chain_id="C2")])


def test_joining_one_block_that_does_not_end_in_a_newline_is_allowed():
    text = built().rstrip("\n")

    assert join_blocks([text]) == text


def test_the_block_separator_is_the_blank_line_between_two_blocks():
    assert BLOCK_SEPARATOR == "\n"
    assert join_blocks([built(), built(chain_id="C2")]) == built() + "\n" + built(chain_id="C2")


# --- determinism ---------------------------------------------------------------


def test_parsing_reads_no_file_no_clock_and_no_randomness(monkeypatch):
    markdowns = [CORPUS.compact_markdown(name) for name in COMPACT_NAMES]

    def refuse(*args, **kwargs):
        raise AssertionError("the parser reached outside its argument")

    for target in ("time", "monotonic", "time_ns"):
        monkeypatch.setattr(time, target, refuse)
    monkeypatch.setattr(random, "random", refuse)
    monkeypatch.setattr(builtins, "open", refuse)
    monkeypatch.setattr(io, "open", refuse)

    assert [len(parse_chains(markdown)) for markdown in markdowns] == [
        len(parse_chains(markdown)) for markdown in markdowns
    ]


def test_the_parser_imports_nothing_that_could_call_a_model():
    imported: set[str] = set()
    for path in PARSE_SOURCES:
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                imported.add(node.module)

    assert PARSE_SOURCES
    assert {name for name in imported if name.startswith("idc")} <= ALLOWED_IDC_IMPORTS
    assert {name for name in imported if not name.startswith("idc")} <= ALLOWED_FOREIGN_IMPORTS
