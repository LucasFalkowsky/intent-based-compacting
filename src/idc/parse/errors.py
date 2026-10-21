"""The failures of the block parser, as one catchable family.

The parser is handed the one thing in this pipeline that a *hand* can have
touched: a Markdown block that a previous run wrote into a transcript file and
that anybody may since have edited, re-indented or truncated. So every refusal
here is about the text it was given, every one of them is a `ParseError` — a
caller (and T10's CLI) catches the whole stage with one `except` — and each
subclass carries the values its message is built from, so a test asserts on an
attribute instead of on prose. That is the shape `idc.classify.errors`
established and `idc.resolve.errors` followed.

Each message has to name **what** was wrong and **where**, because a block that
a person edited is a block a person has to repair: the refusals therefore carry
the 1-based line number of the offending line within the text the parser was
handed, and echo the line itself (or the id, header or fence at fault).

What is *not* here: a value object a caller built wrongly. `ParsedSection`,
`ParsedBlock`, `placeholder_chunk_id` and `join_blocks` raise a plain
`ValueError` the way `SolutionFile` does in `idc.resolve.writes` and
`SubagentSpan` in `idc.chunk.subagents`. The line is drawn by **where the data
comes from**: text the parser is handed is in the family, a bad literal in a
caller's own construction is a programming error. None of these refusals is a
`ValueError` as well, so the two sides never overlap — unlike
`idc.classify.errors.AdapterContractError`, which is deliberately both because
a model's answer is at once a bad value and a failure of that stage.

A block the parser *accepts* but whose chain the canonical schema rejects is
neither: `ParsedBlock.as_chain` builds a `Chain` through the schema, so an empty
label or an id the pattern refuses surfaces as a `SchemaValidationError`, which
is the contract `CLAUDE.md` fixes for every stage.
"""

from collections.abc import Sequence

__all__ = [
    "HEADING_SHAPE",
    "MAX_STRUCTURE_INDENT",
    "DuplicateChainError",
    "MalformedHeadingError",
    "MissingSectionError",
    "NoHeadingError",
    "ParseError",
    "StrayContentError",
    "UnterminatedFenceError",
]

HEADING_SHAPE = "## [C<n>] Intent: <label>"
"""How a chain heading has to read, for the messages that ask for one.

Prose, not a pattern: the pattern the parser matches with is built from
`idc.compact.chain_heading`'s own format and the schema's `CHAIN_ID_PATTERN` in
`idc.parse.blocks`, and this is the human-readable rendering of it that a
message can put in front of somebody who has to repair a block by hand.
"""


MAX_STRUCTURE_INDENT = 3
"""How many leading spaces a line may carry and still be structure.

CommonMark's own allowance, and the parser applies it to all four kinds of line
that carry structure: a chain heading, a section header, and the opening and the
closing run of a code fence. A line indented further is content — four spaces
open an indented code block — so a `### Solution` a hand pushed out that far is
text of the section it stands in and not a header of its own.

It lives here, beside `HEADING_SHAPE`, because the refusals have to be able to
say so: somebody repairing a block by hand needs to be told why their closing
fence did not count. `idc.parse.blocks` imports it from here rather than the
other way round — `blocks` already depends on `errors` — so the rule the parser
applies and the rule the messages state are one number.
"""


class ParseError(Exception):
    """Base class of every failure the block parser raises."""


class NoHeadingError(ParseError):
    """The text the parser was handed holds no chain heading at all.

    Every block starts with one, so a text without one is not a sequence of
    blocks: the usual causes are a native summary that was never compacted by
    this tool, and a slice that cut the heading off its sections.

    Attributes:
        line_count: How many lines the text held, so that the message
            distinguishes an empty input from a long one that lost its heading.
    """

    def __init__(self, line_count: int) -> None:
        """Build the error from the length of the text that held no heading.

        Args:
            line_count: How many lines the text held.
        """
        self.line_count = line_count
        super().__init__(
            f"no chain heading (`{HEADING_SHAPE}`) in the {line_count} line(s) handed to the "
            f"parser; this is not a compacted block"
        )


class MalformedHeadingError(ParseError):
    """A line opens a block but is not a chain heading.

    Any line that starts with `##` — after at most `MAX_STRUCTURE_INDENT`
    leading spaces — and is not a section header is read as an attempt at a
    heading, so that a mistyped id (`[c1]`, `[chain-1]`, `[C0]`) is named as the
    mistake it is instead of being taken for stray prose. The id is what a chain
    is recognized by on the next run, so a wrong one would split a chain's
    history in two. A heading indented further is content of the section it
    stands in, and the block keeps it verbatim; it is never dropped.

    Attributes:
        line: 1-based line number of the offending line.
        text: The line itself, without its line terminator.
    """

    def __init__(self, line: int, text: str) -> None:
        """Build the error from the line that is not a heading.

        Args:
            line: 1-based line number of the offending line.
            text: The line itself, without its line terminator.
        """
        self.line = line
        self.text = text
        super().__init__(
            f"line {line} opens a block but is not a chain heading: {text!r}; expected "
            f"`{HEADING_SHAPE}`, where `<n>` is a number from 1"
        )


class StrayContentError(ParseError):
    """A non-blank line belongs to no section.

    Two places can hold such a line, and the message says which: before the
    first chain heading, and between a heading and its first section header.
    Both are refused rather than dropped, because the parser's whole promise is
    that a chain's block survives it byte for byte — text that no section owns
    would be text the round trip loses.

    Attributes:
        line: 1-based line number of the offending line.
        text: The line itself, without its line terminator.
        chain_id: The chain whose heading the line follows, or `None` when the
            line stands before the first heading.
    """

    def __init__(self, line: int, text: str, chain_id: str | None = None) -> None:
        """Build the error from the line that no section owns.

        Args:
            line: 1-based line number of the offending line.
            text: The line itself, without its line terminator.
            chain_id: The chain whose heading the line follows, or `None` when
                it stands before the first heading.
        """
        self.line = line
        self.text = text
        self.chain_id = chain_id
        where = (
            "before the first chain heading"
            if chain_id is None
            else f"between the heading of chain {chain_id!r} and its first section header"
        )
        super().__init__(f"line {line} belongs to no section — it stands {where}: {text!r}")


class DuplicateChainError(ParseError):
    """Two blocks of one sequence carry the same chain id.

    A chain id is what a later run recognizes a chain by, so two blocks sharing
    one would make a single chain out of two and silently merge their summaries.
    The usual cause is a block copied by hand.

    Attributes:
        chain_id: The id both blocks carry.
        first_line: 1-based line of the heading that carried it first.
        line: 1-based line of the heading that repeats it.
    """

    def __init__(self, chain_id: str, first_line: int, line: int) -> None:
        """Build the error from the repeated id and the two headings.

        Args:
            chain_id: The id both blocks carry.
            first_line: 1-based line of the heading that carried it first.
            line: 1-based line of the heading that repeats it.
        """
        self.chain_id = chain_id
        self.first_line = first_line
        self.line = line
        super().__init__(
            f"the chain id {chain_id!r} is carried by two blocks, at line {first_line} and at "
            f"line {line}; one id names one chain"
        )


class MissingSectionError(ParseError):
    """A block does not carry one of the sections the writer always renders.

    `idc.compact` renders all four of `SECTION_HEADERS` for every chain and
    fills an empty one with `EMPTY_SECTION`, so a rendered section is never
    absent. A missing header therefore means the block was truncated or edited,
    and a parser that accepted it would hand on a chain whose summary has a hole
    in it.

    Attributes:
        chain_id: The chain whose block is short of a section.
        header: The first header, in the writer's order, that is missing.
        present: The section headers the block does carry, in document order.
        line: 1-based line of the block's heading.
    """

    def __init__(self, chain_id: str, header: str, present: Sequence[str], line: int) -> None:
        """Build the error from the chain, the missing header and what is there.

        Args:
            chain_id: The chain whose block is short of a section.
            header: The first header, in the writer's order, that is missing.
            present: The section headers the block does carry.
            line: 1-based line of the block's heading.
        """
        self.chain_id = chain_id
        self.header = header
        self.present = tuple(present)
        self.line = line
        holds = ", ".join(repr(name) for name in self.present) if self.present else "no section"
        super().__init__(
            f"the block of chain {chain_id!r} at line {line} carries no {header!r} section; "
            f"it holds: {holds}"
        )


class UnterminatedFenceError(ParseError):
    """A code fence is opened and never closed.

    The writer closes every fence it opens, and it sizes the fence so that the
    content cannot close it early. An open one at the end of the text means the
    block was cut short — and it is refused rather than read to the end,
    because everything after it, headings included, is inside the fence and
    would be read as content.

    A run of backticks that is long enough but indented by more than
    `MAX_STRUCTURE_INDENT` spaces closes nothing, so it raises this too. The
    message states both halves, because that is the case a hand produces: an
    over-indented closing fence in a Solution body used to end the fence and let
    the headers beneath it be read as sections of their own.

    Attributes:
        line: 1-based line the fence was opened on.
        fence: The run of backticks that opened it, so the message says how
            long the closing run has to be.
    """

    def __init__(self, line: int, fence: str) -> None:
        """Build the error from the fence that was left open.

        Args:
            line: 1-based line the fence was opened on.
            fence: The run of backticks that opened it.
        """
        self.line = line
        self.fence = fence
        super().__init__(
            f"the code fence opened at line {line} with {fence!r} is never closed; a line of at "
            f"least {len(fence)} backtick(s), indented by at most {MAX_STRUCTURE_INDENT} "
            f"space(s), has to close it"
        )
