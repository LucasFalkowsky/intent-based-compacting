"""Reading a rendered compacted block back into chains.

One module, because the format is one thing: the heading pattern, the fence
scan, the section split and the `Chain` they add up to are the same question
asked at four depths, and splitting them over four files would only let three of
them drift apart.

The format itself is **never restated here**. `idc.compact.SECTION_HEADERS` is
the list of sections a block carries, `idc.compact.chain_heading` is the heading
this module's pattern is the inverse of, and the id inside that heading is the
canonical schema's own `CHAIN_ID_PATTERN`. A second spelling of
`### Decisions & constraints` in this file is precisely the drift the round-trip
tests exist to prevent, so there is none.

Two properties carry the parent ticket's criterion that a chain nobody touched
survives `T6 → T8 → T6` byte for byte:

- **A parsed chain holds its block verbatim.** `ParsedBlock.text` is the bytes
  the block occupied in the input, and `as_chain` puts those bytes into
  `Chain.compacted_text` unchanged. Nothing is re-rendered, so nothing can be
  re-rendered differently — and an unknown section a hand added survives for the
  same reason, without this module having to know what it is.
- **The split is the inverse of the join.** `split_blocks` cuts a sequence at
  its headings and `join_blocks` puts it back one blank line apart, which is how
  `idc.compact`'s own golden sequences are written.
"""

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from idc.compact import SECTION_HEADERS
from idc.parse.errors import (
    DuplicateChainError,
    MalformedHeadingError,
    MissingSectionError,
    NoHeadingError,
    StrayContentError,
    UnterminatedFenceError,
)
from idc.schema import Chain
from idc.schema.models import CHAIN_ID_PATTERN

__all__ = [
    "BLOCK_SEPARATOR",
    "HEADING_PATTERN",
    "PLACEHOLDER_CHUNK_PREFIX",
    "ParsedBlock",
    "ParsedSection",
    "join_blocks",
    "parse_blocks",
    "parse_chains",
    "placeholder_chunk_id",
    "split_blocks",
]

BLOCK_SEPARATOR = "\n"
"""What stands between two blocks of one sequence: a blank line.

A rendered block ends in exactly one newline, so one more makes the blank line
that separates it from the next heading. This is the separator
`idc.compact`'s golden sequences are written with, and `join_blocks` is the
inverse of `split_blocks` under it.
"""

_CHAIN_ID = CHAIN_ID_PATTERN.removeprefix("^").removesuffix("$")
"""The chain-id pattern without its anchors, for embedding in the heading one.

Taken from the canonical schema rather than written out again, the way
`idc.classify.ids` takes it, so that `C<n>` has one definition in the project.
"""

HEADING_PATTERN = re.compile(rf"^## \[({_CHAIN_ID})\] Intent: (.+)$")
"""The inverse of `idc.compact.chain_heading`: the id and the label of a block.

The label group is `.+` and not `.*`, so a heading that states no label is a
`MalformedHeadingError` and not a `SchemaValidationError` two steps later — the
message that names the line is the more useful of the two. A label of blank
space does match; `idc.classify` refuses that one when it seeds the chain,
because only there is it known that the label is about to be shown to a model.
"""

PLACEHOLDER_CHUNK_PREFIX = "compacted:"
"""Prefix of the chunk id a parsed chain names, ahead of its own chain id.

A block holds a chain's *summary* and never its chunk membership, but
`Chain.chunk_ids` has `min_length=1`. The placeholder is derived from the chain
id, which makes it unique per chain — two parsed chains would otherwise collide
in `idc.classify`'s seeding with a `ChunkAlreadyAssignedError` — and the colon
keeps it clear of `idc.chunk`'s own ids, which are `K<n>`.
"""

_SECTION_START = "###"
"""What a section header line starts with; checked before `_HEADING_START`."""

_HEADING_START = "##"
"""What a chain heading line starts with, once `###` has been ruled out."""

_FENCE_LINE = re.compile(r"^ {0,3}(`{3,})(.*)$")
"""A line that may open a code fence: a run of backticks and an info string.

Up to three leading spaces, as CommonMark allows — the writer indents no fence,
but a narrative section written as a list may. A run followed by a backtick is
not an opening fence (an info string holds none), which keeps an inline span
from opening one.
"""


def placeholder_chunk_id(chain_id: str) -> str:
    """Return the chunk id a chain parsed out of a block names.

    Args:
        chain_id: The chain's id, as the heading stated it.

    Returns:
        `PLACEHOLDER_CHUNK_PREFIX` followed by `chain_id`. It stands in for the
        chunk membership the block does not hold: the schema requires at least
        one chunk id, and this one is unique per chain and cannot be mistaken
        for a chunk of the run that is reading the block.

    Raises:
        ValueError: If `chain_id` is not of the form `C<n>` — a caller's own
            mistake, since the parser reads the id off a heading that matched
            `HEADING_PATTERN`.
    """
    if re.fullmatch(_CHAIN_ID, chain_id) is None:
        raise ValueError(f"not a chain id of the form `C<n>`: {chain_id!r}")
    return f"{PLACEHOLDER_CHUNK_PREFIX}{chain_id}"


@dataclass(frozen=True)
class ParsedSection:
    """One section of a block: its header line and the lines beneath it.

    Attributes:
        header: The header line, without its line terminator — one of
            `idc.compact.SECTION_HEADERS` for a section the writer rendered, and
            anything else for one a hand added.
        body: Everything between this header and the next one (or the end of the
            block), verbatim and with its line terminators kept, so that a
            fenced block inside it is reproduced exactly.
    """

    header: str
    body: str

    def __post_init__(self) -> None:
        """Reject a section that states no header.

        Raises:
            ValueError: If `header` holds no non-blank character. The parser
                reads a header off a line that starts with `###`, so this can
                only be a caller's own construction.
        """
        if not self.header.strip():
            raise ValueError("a section states a header")


@dataclass(frozen=True)
class ParsedBlock:
    """One compacted block, as the parser read it and as it stands on disk.

    Attributes:
        chain_id: The chain's id, read off the heading.
        label: The chain's label, read off the heading.
        text: The bytes the block occupied in the parsed text, from its heading
            through its last non-blank line. This is what `as_chain` hands on as
            `Chain.compacted_text`, so an unknown section, an odd fence and any
            hand-made whitespace inside the block survive untouched.
        line: 1-based line of the heading within the parsed text — what the
            refusals in `idc.parse.errors` point at, and what tells a caller
            where a block came from.
        sections: The block's sections in document order, including any the
            writer does not render. The four of `idc.compact.SECTION_HEADERS`
            are always among them; `parse_blocks` refuses a block that is short
            of one.
    """

    chain_id: str
    label: str
    text: str
    line: int
    sections: tuple[ParsedSection, ...]

    def __post_init__(self) -> None:
        """Reject a block that states nothing the parser could have produced.

        Raises:
            ValueError: If `chain_id` is not of the form `C<n>`, if `text` is
                empty, if `line` is not positive, or if the block holds no
                section. Each of those is guaranteed by `parse_blocks`, so one
                of them failing is a caller's own construction and not a
                malformed block — the line `idc.parse.errors` draws.
        """
        if re.fullmatch(_CHAIN_ID, self.chain_id) is None:
            raise ValueError(f"not a chain id of the form `C<n>`: {self.chain_id!r}")
        if not self.text:
            raise ValueError(f"block {self.chain_id!r} states no text")
        if self.line < 1:
            raise ValueError(f"block {self.chain_id!r} states no line: {self.line}")
        if not self.sections:
            raise ValueError(f"block {self.chain_id!r} holds no section")

    def body(self, header: str) -> str:
        """Return the body of the section with that header.

        Args:
            header: The header line to look for, e.g. `idc.compact.SOLUTION_HEADER`.

        Returns:
            Everything beneath it, verbatim. The body of the *first* section
            with that header, so a block that carries one twice is read the way
            Markdown reads it, top down.

        Raises:
            ValueError: If the block carries no such section. The four headers
                the writer renders are always there, so asking for one that is
                not is a caller's own mistake and not a malformed block.
        """
        for section in self.sections:
            if section.header == header:
                return section.body
        holds = ", ".join(repr(section.header) for section in self.sections)
        raise ValueError(
            f"block {self.chain_id!r} carries no {header!r} section; it holds: {holds}"
        )

    def as_chain(self) -> Chain:
        """Return this block as the canonical `Chain` it describes.

        Returns:
            A compacted `Chain`: `compacted` is true and `compacted_text` holds
            `text` byte for byte, which is what makes a re-render of an
            untouched chain identical to the block that was read. Its single
            chunk id is `placeholder_chunk_id(chain_id)` — a block states a
            chain's summary and not its chunk membership.

        Raises:
            SchemaValidationError: If the chain the block describes does not
                satisfy the canonical schema.
        """
        return Chain(
            id=self.chain_id,
            label=self.label,
            chunk_ids=[placeholder_chunk_id(self.chain_id)],
            compacted=True,
            compacted_text=self.text,
        )


def _bare(line: str) -> str:
    """Return one line without its terminator, so a pattern can anchor at its end.

    Args:
        line: A line as `str.splitlines(keepends=True)` handed it over.

    Returns:
        The line without its line terminator, LF or CRLF. Nothing else is
        stripped: leading
        space is what tells an indented fence from a flush one, and trailing
        space is part of the text a block has to survive with.
    """
    return line.rstrip("\r\n")


def _spans(starts: Sequence[int], total: int) -> list[tuple[int, int]]:
    """Return each start paired with the line the next one begins on.

    Args:
        starts: The indices that begin a region, in ascending order.
        total: How many lines there are, which bounds the last region.

    Returns:
        One `(start, bound)` pair per start, the last bounded by `total`, and an
        empty list for no starts at all. This is the one arithmetic the block
        split and the section split share: both cut a line range at the next
        line of the same kind.
    """
    if not starts:
        return []
    return list(zip(starts, [*starts[1:], total], strict=True))


def _scan(text: str) -> tuple[list[str], list[int]]:
    """Return the lines of `text` and the indices of those no fence encloses.

    Args:
        text: The text to scan.

    Returns:
        The lines with their terminators kept — `"".join` of them is `text`
        again — and the indices of the lines that are structure: not inside a
        fenced block, and not a fence line themselves. Those are the only lines
        a heading or a section header may be read off, which is what makes a
        `###` inside a four-backtick fence content.

    Raises:
        UnterminatedFenceError: If a fence is opened and never closed. Reading
            on instead would take every heading after it for fenced content.
    """
    lines = text.splitlines(keepends=True)
    structural: list[int] = []
    fence: str | None = None
    opened = 0
    for index, line in enumerate(lines):
        bare = _bare(line)
        if fence is None:
            match = _FENCE_LINE.match(bare)
            if match is not None and "`" not in match.group(2):
                fence, opened = match.group(1), index
            else:
                structural.append(index)
            continue
        closing = bare.strip()
        if closing and set(closing) == {"`"} and len(closing) >= len(fence):
            fence = None
    if fence is not None:
        raise UnterminatedFenceError(opened + 1, fence)
    return lines, structural


def _headings(lines: Sequence[str], structural: Iterable[int]) -> list[tuple[int, re.Match[str]]]:
    """Return the index and the read heading of every line that opens a block.

    Args:
        lines: Every line of the scanned text, terminators kept.
        structural: The indices no fence encloses, in order.

    Returns:
        One `(index, match)` pair per chain heading, in document order; the
        match holds the id and the label, so no caller reads the heading twice.

    Raises:
        MalformedHeadingError: If a structural line starts a block but is not a
            chain heading. `###` is ruled out first, so a section header is
            never mistaken for a mistyped heading.
    """
    headings: list[tuple[int, re.Match[str]]] = []
    for index in structural:
        bare = _bare(lines[index])
        if bare.startswith(_SECTION_START) or not bare.startswith(_HEADING_START):
            continue
        match = HEADING_PATTERN.match(bare)
        if match is None:
            raise MalformedHeadingError(index + 1, bare)
        headings.append((index, match))
    return headings


def _first_stray(lines: Sequence[str], start: int, stop: int) -> int | None:
    """Return the index of the first non-blank line in `lines[start:stop]`.

    Args:
        lines: Every line of the scanned text, terminators kept.
        start: First index to look at.
        stop: Index to stop before.

    Returns:
        The index of the first line that holds a non-blank character, or `None`
        when the range holds none. Fenced lines are looked at too: a fence
        standing where no section can hold it is stray content like any other.
    """
    for index in range(start, stop):
        if lines[index].strip():
            return index
    return None


def _cut(text: str) -> list[tuple[int, re.Match[str], str]]:
    """Cut a sequence of rendered blocks at its headings.

    Args:
        text: One or more rendered blocks, `BLOCK_SEPARATOR`-joined.

    Returns:
        One `(line, heading, block)` triple per block, in document order: the
        1-based line of its heading, the heading as `HEADING_PATTERN` read it,
        and the block from that heading through its last non-blank line.

    Raises:
        NoHeadingError: If the text holds no chain heading.
        MalformedHeadingError: If a line opens a block without being one.
        StrayContentError: If a non-blank line stands before the first heading.
        UnterminatedFenceError: If a code fence is never closed.
    """
    lines, structural = _scan(text)
    headings = _headings(lines, structural)
    if not headings:
        raise NoHeadingError(len(lines))
    starts = [index for index, _ in headings]
    stray = _first_stray(lines, 0, starts[0])
    if stray is not None:
        raise StrayContentError(stray + 1, _bare(lines[stray]))
    cuts: list[tuple[int, re.Match[str], str]] = []
    spans = _spans(starts, len(lines))
    for (start, bound), (_, heading) in zip(spans, headings, strict=True):
        end = bound
        while end > start + 1 and not lines[end - 1].strip():
            end -= 1
        cuts.append((start + 1, heading, "".join(lines[start:end])))
    return cuts


def split_blocks(text: str) -> list[str]:
    """Split a sequence of rendered blocks into the bytes of each block.

    Args:
        text: One or more rendered blocks, `BLOCK_SEPARATOR`-joined — what
            `idc.compact` writes and what a transcript carries on the next run.

    Returns:
        Each block from its heading through its last non-blank line, verbatim
        and in document order. The blank line between two blocks belongs to
        neither, so a block that `idc.compact.render_block` wrote comes back
        exactly as it was written, and `join_blocks` puts the sequence back
        together byte for byte.

        The two things that are not reproduced are the two a rendered sequence
        does not hold: blank lines before the first heading, and a separator of
        more than one blank line — both are dropped, so `join_blocks` of the
        result normalizes them to the writer's own spacing rather than failing.

    Raises:
        ParseError: If the text is not a sequence of rendered blocks — see
            `_cut` for which refusal says what.
    """
    return [block for _, _, block in _cut(text)]


def join_blocks(texts: Sequence[str]) -> str:
    """Join block texts back into one sequence, one blank line apart.

    The inverse of `split_blocks` for a sequence of rendered blocks, and the
    same join `idc.compact`'s golden sequences are written with.

    Args:
        texts: The blocks to join, in document order.

    Returns:
        The blocks `BLOCK_SEPARATOR`-joined, so that exactly one blank line
        stands between each pair.

    Raises:
        ValueError: If a block other than the last does not end in a newline, or
            if any block is empty — the blocks would be glued into one line, and
            re-reading the result would find fewer blocks than went in. A
            caller's own mistake: every text `split_blocks` returns is non-empty,
            and only its last may lack the newline.
    """
    for position, text in enumerate(texts):
        if not text:
            raise ValueError(f"block {position + 1} of {len(texts)} is empty")
        if not text.endswith("\n") and position + 1 < len(texts):
            raise ValueError(
                f"block {position + 1} of {len(texts)} does not end in a newline, so the next "
                f"block would be glued to its last line"
            )
    return BLOCK_SEPARATOR.join(texts)


def _parse_block(line: int, heading: re.Match[str], text: str) -> ParsedBlock:
    """Read one block's sections, beneath the heading `_cut` already read.

    Args:
        line: 1-based line of the heading in the text that was cut, so that a
            refusal points into the input a caller handed over and not into the
            slice.
        heading: The heading as `HEADING_PATTERN` read it: the id and the label.
        text: The block's bytes, as `_cut` cut them.

    Returns:
        The block, with its sections in document order.

    Raises:
        MissingSectionError: If one of `idc.compact.SECTION_HEADERS` is absent.
        StrayContentError: If a non-blank line stands between the heading and
            the first section header.
    """
    lines, structural = _scan(text)
    chain_id, label = heading.group(1), heading.group(2)
    starts = [index for index in structural if _bare(lines[index]).startswith(_SECTION_START)]
    stray = _first_stray(lines, 1, starts[0] if starts else len(lines))
    if stray is not None:
        raise StrayContentError(line + stray, _bare(lines[stray]), chain_id)
    sections = tuple(
        ParsedSection(header=_bare(lines[start]), body="".join(lines[start + 1 : bound]))
        for start, bound in _spans(starts, len(lines))
    )
    headers = {section.header for section in sections}
    missing = [header for header in SECTION_HEADERS if header not in headers]
    if missing:
        raise MissingSectionError(
            chain_id, missing[0], [section.header for section in sections], line
        )
    return ParsedBlock(chain_id=chain_id, label=label, text=text, line=line, sections=sections)


def parse_blocks(text: str) -> list[ParsedBlock]:
    """Parse a sequence of rendered blocks into one `ParsedBlock` each.

    Args:
        text: One or more rendered blocks, `BLOCK_SEPARATOR`-joined — the
            compacted summary a transcript carries into the next run.

    Returns:
        One block per `## [C<n>]` heading, in document order, each holding its
        own bytes verbatim.

    Raises:
        DuplicateChainError: If two blocks carry the same chain id.
        MalformedHeadingError: If a line opens a block without being one.
        MissingSectionError: If a block is short of a section the writer renders.
        NoHeadingError: If the text holds no chain heading.
        StrayContentError: If a non-blank line belongs to no section.
        UnterminatedFenceError: If a code fence is never closed.
    """
    blocks: list[ParsedBlock] = []
    seen: dict[str, int] = {}
    for line, heading, block_text in _cut(text):
        block = _parse_block(line, heading, block_text)
        first = seen.get(block.chain_id)
        if first is not None:
            raise DuplicateChainError(block.chain_id, first, block.line)
        seen[block.chain_id] = block.line
        blocks.append(block)
    return blocks


def parse_chains(text: str) -> list[Chain]:
    """Parse a sequence of rendered blocks into the chains it summarizes.

    This is the stage's entry point: what `idc.classify` is seeded with on the
    next run, so that a chain written by an earlier one is offered as a
    candidate instead of being summarized again.

    Args:
        text: One or more rendered blocks, `BLOCK_SEPARATOR`-joined.

    Returns:
        One compacted `Chain` per block, in document order. Each carries its
        block in `compacted_text` byte for byte, which is what lets an untouched
        chain be written back out exactly as it came in.

    Raises:
        ParseError: If the text is not a sequence of rendered blocks — see
            `parse_blocks` for which refusal says what.
        SchemaValidationError: If a block describes a chain the canonical schema
            rejects.
    """
    return [block.as_chain() for block in parse_blocks(text)]
