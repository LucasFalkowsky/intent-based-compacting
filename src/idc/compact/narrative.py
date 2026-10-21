"""The two narrative sections, and the contract a model has to answer in.

`Decisions & constraints` and `Rejected approaches` say why a chain went the way
it did. No reading of the transcript yields that, so a model writes them — and
with them the chain's `Intent` label, which is a provisional truncation until
this stage replaces it. Everything else in the block stays code's:
`idc.compact.template` writes `Solution` and `Verification` from the
`idc.resolve.Resolution`, and this module never touches them.

```python
from idc.compact import compact_chain

block = compact_chain(chain, resolution, client)   # one model call
```

The client is injected, exactly as a classification backend is
(`idc.classify.adapters`): `NarrativeClient` is a `Protocol` with one method,
it is handed a `NarrativeRequest` that carries what any backend can see and
nothing that would let it reach into the transcript, and every test of this
stage runs on `idc.compact.fake.FakeNarrativeClient`. Nothing in this module
calls a model.

**Nothing a model returns is trusted.** A response is three things — a label and
the two section bodies — and `as_narrative` reads exactly that shape or raises
`NarrativeContractError`. The rule it enforces is narrow on purpose:

- The response holds the intent marker and the two narrative headers, each
  exactly once, in the template's order, with the intent marker first and
  nothing before it.
- **No other line may be read as structure.** A heading, a line that opens or
  closes a code fence, a Setext underline, an HTML comment left open — each of
  them with *any* indentation in front of it, since a parser keyed on the header
  literal reads an indented `### Solution` as one. That is what keeps a model
  from writing the deterministic sections itself: a `### Solution` of its own, a
  second `## [C…]` heading naming a chain the classifier never opened, or a
  fence that would swallow the headers the renderer appends below.
  A near miss (`### Decisions and constraints`) is refused for the same reason
  the headers are pinned character for character in T6.1 — a block is parsed
  back by its headers on the next run (T6.3), and a section lost there is a
  chain the tool then summarizes twice.
- A section with **nothing to say is not malformed**. A chain that rejected no
  approach is the normal case, and it renders `EMPTY_SECTION` — the same line
  `idc.compact.template` writes for an empty deterministic section. An absent
  body and a literal `- (none)` are the same thing, so the two render the same
  bytes; what is refused is a *missing header*, which is a response that did not
  answer rather than a chain that had nothing to say.
"""

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Protocol, Self, runtime_checkable

from idc.compact.errors import NarrativeContractError
from idc.compact.instruction import Instruction
from idc.compact.instruction import instruction as packaged_instruction
from idc.compact.template import (
    DECISIONS_HEADER,
    EMPTY_SECTION,
    REJECTED_HEADER,
    chain_heading,
    solution_section,
    verification_section,
)
from idc.resolve import Resolution
from idc.schema import Chain

__all__ = [
    "INTENT_MARKER",
    "Narrative",
    "NarrativeClient",
    "NarrativeRequest",
    "as_narrative",
    "compact_chain",
    "decisions_section",
    "rejected_section",
    "render_narrative_block",
]

INTENT_MARKER = "## Intent:"
"""What the intent line of a response begins with — the label follows it.

The block's heading is `## [<id>] Intent: <label>` and the id is the tool's own,
so a model is asked for the label alone and the renderer puts the id around it.
A response that writes an id of its own therefore does not match this marker and
is refused: a model may not choose the identifier that a later run recognizes
the chain by.
"""

_HEADING_LINE = re.compile(r"#{1,6}(\s|$)")
"""A Markdown ATX heading — one to six `#` followed by whitespace or nothing.

Matched against a line whose leading whitespace has been stripped, with no
indentation allowance at all, and that is deliberately stricter than Markdown:
CommonMark reads a line indented by four spaces as an indented code block
rather than as a heading. The consumer of this format is not a CommonMark
renderer, though — it is a parser that reads a block back on the next run and
finds its sections by the header literal (T6.3). To that parser `    ###
Solution` holds `### Solution`, so an indented copy of a header is a copy, and
four leading spaces are refused like one. `#hashtag` is still not a heading,
which is the rule Markdown itself applies.
"""

_FENCE_LINE = re.compile(r"(```|~~~)")
"""A line that opens or closes a fenced code block, indentation stripped.

Refused in a narrative body for the same reason a stray header is: the renderer
appends the deterministic sections *below* the narrative, and a fence left open
in it would enclose them — a `Solution` section that is no longer read as one.
Indented fences are refused with the unindented ones, and a tilde fence is why:
nothing the renderer appends closes one, so a `  ~~~` in the last narrative
section encloses the whole deterministic tail of the block.
"""

_SETEXT_UNDERLINE = re.compile(r"(-+|=+)$")
"""A line of nothing but dashes or nothing but equals signs.

Markdown reads it as the underline that turns the line above it into a heading,
so `Solution` and `---` on two body lines are a `Solution` heading in the
rendered block. The underline is refused rather than the pair: it is the line
that carries the meaning, a body line is a bullet and never a rule, and a
single-line rule needs no state in the reader. `---` as a thematic break and as
a front-matter delimiter go with it, which is no loss for a bullet list.
"""

_CLOSED_COMMENT = re.compile(r"<!--.*?-->")
"""One complete HTML comment, for removing the closed ones from a line.

What is left is an opener that the line does not close, and that comments out
everything below it — including the deterministic sections. A comment opened and
closed on one line encloses nothing and is not this module's business.
"""


def _refusal(line: str) -> str | None:
    """Return why `line` would be read as structure in a block, or `None`.

    Args:
        line: One line of a response or of a narrative body, right-stripped.

    Returns:
        A reason, naming the line, when a parser of the rendered block could
        read it as the block's own structure — an ATX heading, a line that
        opens or closes a code fence, a Setext underline, or an HTML comment
        left open. Leading whitespace is stripped before any of that is
        decided, so no amount of indentation hides one. `None` for every other
        line, indented bullets included.
    """
    bare = line.lstrip()
    if _HEADING_LINE.match(bare):
        return (
            f"the body line {line!r} is a header, and the only headers a narrative holds are "
            f"{DECISIONS_HEADER!r} and {REJECTED_HEADER!r}"
        )
    if _FENCE_LINE.match(bare):
        return (
            f"the body line {line!r} opens or closes a code fence; a narrative is bullet "
            f"lines, and a fence would enclose the sections rendered below it"
        )
    if _SETEXT_UNDERLINE.fullmatch(bare):
        return (
            f"the body line {line!r} underlines the line above it into a header, which is "
            f"a header the block does not have"
        )
    if "<!--" in _CLOSED_COMMENT.sub("", line):
        return (
            f"the body line {line!r} opens an HTML comment it does not close, which would "
            f"comment out the sections rendered below it"
        )
    return None


def _body_lines(lines: Sequence[str]) -> tuple[str, ...]:
    """Return the lines of a section body, blank lines and the placeholder removed.

    Args:
        lines: The lines between two markers of a response, already right-stripped.

    Returns:
        The non-blank lines, in order — and `()` when the body holds nothing, or
        holds nothing but `EMPTY_SECTION`. The placeholder is what the
        instruction asks a model to write for a section with nothing in it, and
        collapsing it to the empty body means the two ways of saying "nothing"
        render the same bytes instead of one of them rendering the placeholder
        twice.

        Blank lines are dropped rather than kept: a body is a bullet list, and
        the block it goes into is measured in tokens.
    """
    held = tuple(line for line in lines if line.strip())
    return () if held == (EMPTY_SECTION,) else held


@dataclass(frozen=True, slots=True)
class Narrative:
    """What a model contributes to one block: a label and the two section bodies.

    The validated form of a response. `as_narrative` is how one is built from
    what a client returned; constructing one directly is a caller stating the
    three parts itself, and a part that cannot go into a block is a plain
    `ValueError` — the line `idc.compact.errors` draws, and the one
    `idc.classify.adapters.Decision` draws for a bad target.

    Attributes:
        label: The chain's intent, one line, as the heading will carry it.
        decisions: The body of `Decisions & constraints`, one entry per line,
            `()` for a section with nothing in it.
        rejected: The body of `Rejected approaches`, the same way.
    """

    label: str
    decisions: tuple[str, ...] = ()
    rejected: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Reject a label without a word and a body line a block cannot hold.

        Raises:
            ValueError: If `label` holds no word — it becomes the chain's
                heading, and `## [C1] Intent: ` names no intent — or if a body
                line is blank, spans more than one line, or is one of the
                shapes `_refusal` names: a heading, a code fence, a Setext
                underline, an HTML comment left open, each of them with any
                indentation in front of it. All but the blank line are what
                would survive into the block and be read there as its
                structure, which is the same guard `as_narrative` applies to a
                model's response.
        """
        if not self.label.strip():
            raise ValueError("a narrative's label must hold a word; it names the chain's intent")
        for line in (*self.decisions, *self.rejected):
            if not line.strip():
                raise ValueError("a narrative's body holds no blank line")
            if "\n" in line or "\r" in line:
                raise ValueError(f"a narrative's body line spans one line; got {line!r}")
            reason = _refusal(line)
            if reason is not None:
                raise ValueError(
                    f"a narrative holds no line a block would read as its structure: {reason}"
                )


def _marker_of(line: str) -> str | None:
    """Return which of the three response markers `line` is, or `None`.

    Args:
        line: One right-stripped line of a response.

    Returns:
        `INTENT_MARKER`, `DECISIONS_HEADER` or `REJECTED_HEADER` when the line
        is that marker, and `None` for every other line — including a line that
        is a heading but not one of the three, which the caller refuses.
    """
    if line.startswith(INTENT_MARKER):
        return INTENT_MARKER
    if line == DECISIONS_HEADER:
        return DECISIONS_HEADER
    if line == REJECTED_HEADER:
        return REJECTED_HEADER
    return None


_EXPECTED_MARKERS = (INTENT_MARKER, DECISIONS_HEADER, REJECTED_HEADER)
"""The three markers a response holds, each once, in this order."""


def _trimmed(value: str) -> list[str]:
    """Return a response as its lines, right-stripped and without blank ends.

    Args:
        value: The response, as the client returned it.

    Returns:
        One entry per line, each without its trailing whitespace, with the blank
        lines at either end dropped. A model's answer that arrives wrapped in
        newlines is the same answer, and trailing whitespace on a header would
        otherwise make it a line that merely looks like one.
    """
    lines = [line.rstrip() for line in value.splitlines()]
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return lines


def _found_markers(lines: Sequence[str]) -> list[tuple[int, str]]:
    """Return where each response marker is, refusing every other structural line.

    Args:
        lines: The response's lines, as `_trimmed` returns them.

    Returns:
        One `(position, marker)` pair per marker found, in the order the lines
        hold them — which is what the caller compares against
        `_EXPECTED_MARKERS`.

    Raises:
        NarrativeContractError: If a line is one of the shapes `_refusal`
            names and is none of the three markers. They are refused here
            rather than where the bodies are sliced, because each is a line
            that would be read as *structure* in the rendered block: a
            `### Solution` of the model's own — indented or not — a second
            `## [C…]` heading, a misspelled narrative header, a fence or an
            HTML comment enclosing the sections the renderer appends below, or
            a Setext underline making a header out of the line above it.
    """
    found: list[tuple[int, str]] = []
    for position, line in enumerate(lines):
        marker = _marker_of(line)
        if marker is not None:
            found.append((position, marker))
            continue
        reason = _refusal(line)
        if reason is not None:
            raise NarrativeContractError(
                f"a narrative client wrote a line a block would read as its structure: "
                f"{reason}. The other sections of a block are written from the transcript, "
                f"not by a model"
            )
    return found


def as_narrative(value: object) -> Narrative:
    """Read a client's response as a `Narrative`.

    Args:
        value: What the client returned. A string of the shape the instruction
            asks for:

            ```markdown
            ## Intent: <label>
            ### Decisions & constraints
            - <one per line>
            ### Rejected approaches
            - <one per line>
            ```

            Trailing whitespace on a line, blank lines, and blank lines around
            the whole response are ignored; a body that holds nothing, or holds
            nothing but `EMPTY_SECTION`, is an empty section and not an error.

    Returns:
        The narrative: the label, and the two bodies as the lines the model
        wrote.

    Raises:
        NarrativeContractError: If the value is not a string, holds no word, or
            does not hold exactly the three markers in that order with the
            intent marker first — a header missing, repeated, misspelled, out of
            order, or added (`### Solution`, `### Verification`, a second
            `## [C…]` heading), and prose before the first marker. Also if any
            other line would be read as the block's structure — `_refusal`
            names the shapes, and indenting one does not help — or if the label
            holds no word. The message names the line it refused, so a backend
            that answered in prose fails where it answered.
    """
    if not isinstance(value, str):
        raise NarrativeContractError(
            f"a narrative client must answer with text; got {value!r} "
            f"of type {type(value).__name__}"
        )
    lines = _trimmed(value)
    if not lines:
        raise NarrativeContractError(
            f"a narrative client answered with nothing; it has to answer with "
            f"{INTENT_MARKER!r} and the two narrative headers"
        )
    found = _found_markers(lines)
    if tuple(marker for _, marker in found) != _EXPECTED_MARKERS:
        written = ", ".join(repr(marker) for _, marker in found) or "none"
        raise NarrativeContractError(
            f"a narrative client has to write {INTENT_MARKER!r}, {DECISIONS_HEADER!r} and "
            f"{REJECTED_HEADER!r}, each once and in that order; it wrote: {written}"
        )
    if found[0][0] != 0:
        raise NarrativeContractError(
            f"a narrative client answered with {lines[0]!r} before {INTENT_MARKER!r}; "
            f"a response begins with its intent line and holds no preamble"
        )
    decisions_at = found[1][0]
    rejected_at = found[2][0]
    label = lines[0][len(INTENT_MARKER) :].strip()
    if not label:
        raise NarrativeContractError(
            f"a narrative client wrote {INTENT_MARKER!r} without a label; the label is what "
            f"names the chain's intent"
        )
    return Narrative(
        label=label,
        decisions=_body_lines(lines[decisions_at + 1 : rejected_at]),
        rejected=_body_lines(lines[rejected_at + 1 :]),
    )


def _section(header: str, body: Sequence[str]) -> str:
    """Return one narrative section: its header and its body, line-terminated.

    Args:
        header: `DECISIONS_HEADER` or `REJECTED_HEADER`, verbatim.
        body: The lines of the body, `()` for a section with nothing in it.

    Returns:
        The header and one line per body entry, or the header and
        `EMPTY_SECTION` — never a header on its own, which is the rule T6.1
        states for all four sections: a blank section cannot be told from a
        block that was truncated in transit.
    """
    return f"{header}\n" + ("".join(f"{line}\n" for line in body) or f"{EMPTY_SECTION}\n")


def decisions_section(narrative: Narrative) -> str:
    """Return the `Decisions & constraints` section: its header and its body.

    Args:
        narrative: What the model wrote for this chain.

    Returns:
        `DECISIONS_HEADER` and the lines of `narrative.decisions`, or
        `EMPTY_SECTION` when there are none, ending in a newline.
    """
    return _section(DECISIONS_HEADER, narrative.decisions)


def rejected_section(narrative: Narrative) -> str:
    """Return the `Rejected approaches` section: its header and its body.

    Args:
        narrative: What the model wrote for this chain.

    Returns:
        `REJECTED_HEADER` and the lines of `narrative.rejected`, or
        `EMPTY_SECTION` when there are none, ending in a newline.
    """
    return _section(REJECTED_HEADER, narrative.rejected)


def render_narrative_block(chain: Chain, resolution: Resolution, narrative: Narrative) -> str:
    """Return the whole block of one chain, its narrative sections filled.

    Args:
        chain: The chain the block is about; its `id` becomes the heading's.
        resolution: What the chain resolved to — the two deterministic sections
            are this written out, by `idc.compact.template` and untouched here.
        narrative: The validated response: the heading's label and the two
            narrative bodies.

    Returns:
        The heading and the four sections in the order of `SECTION_HEADERS`,
        ending in exactly one newline — the same shape
        `idc.compact.template.render_block` returns, with the two narrative
        sections carrying what the model wrote instead of `EMPTY_SECTION`. The
        heading carries the model's label; the chain's own provisional label is
        what it replaces.

        A narrative whose two bodies are empty renders exactly what
        `render_block` renders for that chain, which is what keeps the one
        template in one place.

        Everything outside the model's three contributions is computed from the
        two other arguments, so a chain, a resolution and a response render the
        same bytes in every run.
    """
    return (
        f"{chain_heading(chain, narrative.label)}\n"
        f"{decisions_section(narrative)}"
        f"{rejected_section(narrative)}"
        f"{solution_section(resolution)}"
        f"{verification_section(resolution)}"
    )


@dataclass(frozen=True, slots=True)
class NarrativeRequest:
    """What a narrative client is shown of one chain, and nothing besides.

    The counterpart of `idc.classify.adapters.ChainCandidate`, and a value
    object for the same two reasons: a backend sees only what every backend can
    see — no session, no chunk indices, no tool handle — and a later ticket can
    offer one more fact without changing the signature every backend is written
    against.

    What it carries is the chain's **user messages** and the chain's own
    `Solution` and `Verification` sections as the renderer writes them. The
    chain's *path* — the superseded writes and the runs the resolver discarded,
    which `idc.resolve.Resolution.path_call_ids` names — is **not** offered: the
    ids alone say nothing, and resolving them to text needs the session, which
    is a seam this ticket does not open. `Rejected approaches` therefore rests
    on what the user messages and the achieved solution say, and offering the
    path is a field added here rather than a second interface.

    Attributes:
        instruction: The one fixed instruction, as `idc.compact.instruction`
            read it from its data file — carried along so that a client has the
            prompt *and* the digest of the prompt it used.
        chain_id: The chain's stable id, `C<n>`. For the record and the error
            messages; the client does not write it (see `INTENT_MARKER`).
        label: The chain's current label, provisional, which the response
            replaces.
        user_messages: The user messages of the chain's chunks, in transcript
            order, verbatim. Empty for a chain whose chunks hold none.
        solution: The rendered `Solution` section of the chain.
        verification: The rendered `Verification` section of the chain.
    """

    instruction: Instruction
    chain_id: str
    label: str
    solution: str
    verification: str
    user_messages: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Reject a request that names no chain or no intent.

        Raises:
            ValueError: If `chain_id` or `label` holds no word. A request is
                attributable to one chain, and the label is what the model is
                asked to improve on.
        """
        if not self.chain_id.strip():
            raise ValueError("a narrative request must carry its chain id")
        if not self.label.strip():
            raise ValueError(f"the request for chain {self.chain_id!r} must carry a label")

    @classmethod
    def for_chain(
        cls,
        chain: Chain,
        resolution: Resolution,
        *,
        user_messages: Iterable[str] = (),
        instruction: Instruction | None = None,
    ) -> Self:
        """Return the request for one chain, with the deterministic sections rendered.

        Args:
            chain: The chain to compact.
            resolution: What it resolved to.
            user_messages: The user messages of its chunks, in transcript order,
                as the caller resolved them. A chain carried over from an
                earlier run has none in this transcript.
            instruction: The instruction to carry; the packaged one when
                omitted.

        Returns:
            The request. The two sections are rendered by the same functions
            that write them into the block, so a client is shown exactly what
            the block will say and cannot be shown a second version of it.
        """
        return cls(
            instruction=packaged_instruction() if instruction is None else instruction,
            chain_id=chain.id,
            label=chain.label,
            solution=solution_section(resolution),
            verification=verification_section(resolution),
            user_messages=tuple(user_messages),
        )


@runtime_checkable
class NarrativeClient(Protocol):
    """The one method a narrative backend has to provide.

    A protocol rather than a base class, for the reason
    `idc.classify.adapters.IntentAdapter` gives: a client is anything with a
    `write` method of this shape, so a real backend inherits nothing and the
    fake is not a special case. It is `runtime_checkable`, which checks the
    *presence* of `write` and not its signature — nothing here uses
    `isinstance` as a guard, and `as_narrative` is what refuses an answer that
    is not one.
    """

    def write(self, request: NarrativeRequest) -> str:
        """Return the narrative for one chain, as text.

        Args:
            request: The chain's facts and the instruction to follow. A backend
                assembles its own prompt from `request.instruction.text` and the
                fields; it is given no way to reach into the transcript.

        Returns:
            The response, in the shape `as_narrative` reads. A backend does not
            validate its own answer — the stage does, and it refuses rather than
            repairs.
        """
        ...


def compact_chain(
    chain: Chain,
    resolution: Resolution,
    client: NarrativeClient,
    *,
    user_messages: Iterable[str] = (),
) -> str:
    """Compact one chain into its block, with exactly one model call.

    Args:
        chain: The chain to compact.
        resolution: What it resolved to.
        client: The narrative backend to ask.
        user_messages: The user messages of the chain's chunks, in transcript
            order.

    Returns:
        The whole Markdown block, as `render_narrative_block` writes it.

    Raises:
        NarrativeContractError: If the client's answer is not a narrative. The
            chain fails rather than being compacted with a block the model's
            answer only partly filled.

    One call per chain, and the call is made once: the request is built, the
    client is asked, the answer is validated, the block is rendered. The study
    pays for one call per new or extended chain, and
    `tests/test_compact_narrative.py` pins the count off the fake's record so a
    later refactor cannot quietly double it.
    """
    request = NarrativeRequest.for_chain(chain, resolution, user_messages=user_messages)
    return render_narrative_block(chain, resolution, as_narrative(client.write(request)))
