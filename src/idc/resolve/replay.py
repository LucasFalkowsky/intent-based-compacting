"""Replaying a file's edits into the content the chain left it with.

`idc.resolve.writes` resolves *which* writes a file's solution consists of; this
module answers the next question — what the file then **holds** — by replaying
the kept edits onto a base and handing out the resulting text:

```python
from idc.resolve import ReplayStatus, replay_files

for replay in replay_files(session, chunks, chain):
    if replay.status is ReplayStatus.REPLAYED:
        replay.content          # what the file holds after the chain
    else:
        replay.status           # why the file is left as its edit sequence
```

It is offered **beside** the resolution rather than inside `Resolution`: T5.4
(#24) is `optional`, and the study can run on the edit-sequence representation
the entries already are. A consumer that wants the content asks for it.

**The base is content the transcript states verbatim, and that is a full write.**
`idc.resolve.calls.content_of` reads the whole new content out of a
`WriteKind.FULL` call's own arguments, so taking it as the file's content after
that call infers nothing. Everything after it is textual: each kept edit names
the text it replaces and the text it replaces it with, and the replay applies
them in transcript order.

**A read is never a base, and that is the deliberate narrowing of this ticket.**
The ticket asks for the content to be reconstructed "from the last `read` plus
subsequent edits, where the read covered the whole file". Whether a read covered
the whole file is not decidable here. The tool has no filesystem, so a read's
*result* is the only evidence, and every read result the fixture corpus records
— for all three agents — is the file rendered behind a line-number gutter.
`rendered_extent` decodes that gutter, and what it yields is an asymmetric
answer:

- A gutter that begins past line 1 **proves** the read began inside the file, so
  the read is `PARTIAL_READ` and replaying on it would silently drop everything
  above its first line.
- A gutter that begins at line 1 proves only where the read *began*. No record
  this build reads states a file's total length, so a read cut off after its
  first lines cannot be told from one that reached the end — the status is
  `UNPROVEN_READ`, and the honest answer is the fallback.

Nothing in the corpus ever makes a read a provable base, so the reads are read
to *name the reason* a file falls back, not to replay from. The day a transcript
is recorded whose read result states a file's extent, that statement is what
would turn `UNPROVEN_READ` into a base — and `rendered_extent` is where it would
be read.

**A replay is whole or it is nothing.** An edit the replay cannot place — one
that names no anchor, one whose anchor the base does not hold, one whose anchor
the base holds twice — makes the whole file fall back (`ANCHOR_NOT_FOUND`,
`ANCHOR_AMBIGUOUS`) rather than yielding the text as it stood before that edit.
A partly replayed file is not the file, and presenting it as one would put
content into the study's resolved solution that the transcript does not show.

Nothing here reads a clock, a model, the network or the filesystem: the content
is computed from the transcript alone, so the same chain always replays to the
same text.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from itertools import pairwise

from idc.resolve.calls import REPLACEMENT_KEYS, LocatedCall, chain_calls, content_of
from idc.resolve.writes import SolutionFile, solution_files
from idc.schema import Chain, Chunk, Session, ToolCategory

__all__ = [
    "FileReplay",
    "ReplayStatus",
    "rendered_extent",
    "replay_file",
    "replay_files",
]

_REPLACEMENT_PAIRS = tuple(zip(REPLACEMENT_KEYS[::2], REPLACEMENT_KEYS[1::2], strict=True))
"""`REPLACEMENT_KEYS` in the pairs the agents send them in.

Derived from that list rather than written out again, so that a spelling added
there is replayed without a second edit here. The pairing is the order
`REPLACEMENT_KEYS` documents — `old_string`/`new_string`, then
`oldString`/`newString` — and `tests/test_resolve_replay.py` pins the pairs
against a hand-written copy in `tests/fixture_corpus.py`.

Why the stage needs both shapes of the same knowledge: *one* replacement key is
enough to know a call patches rather than overwrites, which is all
`write_kind` asks, and that is why the exported list is flat. Replaying needs
the two halves of one pair together, which is this.
"""

_GUTTER = re.compile(r"^ *(\d+)\t")
"""The line-number gutter a read result renders each file line behind.

Every read result of the fixture corpus renders the file with its line numbers
in front — the shape `    14\\tclass TokenBucket:`, and for all three agents.
The gutter is the only thing in a read result that says *which* lines of the
file the agent saw, which is what makes a partial read detectable at all.

The pattern is deliberately narrow: leading spaces, digits, one tab. It
recognizes the recorded rendering and nothing beyond it, because every shape it
matches becomes an extent this stage reports as read, and a shape matched
loosely would report an extent the output does not state.
"""


class ReplayStatus(StrEnum):
    """Whether a file's content could be replayed, and if not, why not.

    Attributes:
        REPLAYED: The content was reconstructed; `FileReplay.content` holds it.
        NO_BASE: Nothing states a base for the file — the solution holds no
            accepted full write, and the chain read the file neither before its
            first kept edit nor successfully at all.
        PARTIAL_READ: The only candidate base is a read whose own gutter proves
            it began past the file's first line, so the file's head was never
            in the transcript.
        UNPROVEN_READ: The only candidate base is a read nothing proves covered
            the whole file — its result carries no readable gutter, or a gutter
            that begins at line 1 and therefore says nothing about the end.
        ANCHOR_NOT_FOUND: A base exists, but a kept edit names no anchor the
            base holds: no replacement pair at all, an anchor holding no word,
            or an anchor the base does not contain.
        ANCHOR_AMBIGUOUS: A base exists, but a kept edit's anchor occurs in it
            more than once, so which occurrence the agent patched is not
            decidable.
    """

    REPLAYED = "replayed"
    NO_BASE = "no-base"
    PARTIAL_READ = "partial-read"
    UNPROVEN_READ = "unproven-read"
    ANCHOR_NOT_FOUND = "anchor-not-found"
    ANCHOR_AMBIGUOUS = "anchor-ambiguous"


def rendered_extent(output: str) -> tuple[int, int] | None:
    """Return the first and last file line a read result's gutter names.

    Args:
        output: A read result's output, verbatim.

    Returns:
        `(first, last)` when every line of the output is rendered behind a
        line-number gutter and the numbers run consecutively upwards —
        the lines of the file the agent was shown. `None` when the output is
        not such a rendering at all: no line carries a gutter, only some do,
        the numbering has a gap or runs backwards, or the output is empty.

        `None` is "this output states no extent", not "the read was partial":
        both are read as `ReplayStatus.UNPROVEN_READ`, because a rendering this
        build cannot decode says nothing either way. One corpus read is of that
        shape — `write_then_two_edits` renders a blank file line as an
        unnumbered empty line, which breaks the numbering — and reading an
        extent off it anyway would claim lines the output does not name.

        Note what a `(1, n)` answer does **not** say: that the file has `n`
        lines. It says the agent was shown lines 1 to `n`, and no record this
        build reads states how many the file has, so the end is open. See the
        module docstring.
    """
    numbers: list[int] = []
    for line in output.splitlines():
        match = _GUTTER.match(line)
        if match is None:
            return None
        numbers.append(int(match.group(1)))
    if not numbers:
        return None
    if any(later != earlier + 1 for earlier, later in pairwise(numbers)):
        return None
    return numbers[0], numbers[-1]


@dataclass(frozen=True, slots=True)
class FileReplay:
    """What one file of a chain holds afterwards, or why that is not decidable.

    Attributes:
        path: The file, exactly as the agent spelled it — the `path` of the
            `SolutionFile` this replay is about.
        content: What the file holds after the chain, when `status` is
            `ReplayStatus.REPLAYED`; `None` otherwise, and never a partial
            reconstruction.
        status: Whether the content was replayed, and if not, why not.
        base_call_id: `ToolCall.id` of the call the base came from — the
            accepted full write when the file was replayed. For a file that
            fell back on a read it is the read that was examined and rejected,
            so the reason names the call it is about; `None` when no call was a
            candidate at all.
        replayed_call_ids: `ToolCall.id` of the edits that were applied to the
            base, in the order they were applied. Empty whenever the file was
            not replayed, and empty for a file that was written once and never
            edited.
    """

    path: str
    content: str | None
    status: ReplayStatus
    base_call_id: str | None
    replayed_call_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        """Reject a replay whose content and status do not say the same thing.

        Raises:
            ValueError: If the path holds no word, if a content is stated
                without the status `REPLAYED` (or the other way round), if a
                replayed file names no base, or if edits are named as replayed
                by a file that was not replayed. Each of those would make a
                consumer read content, or a provenance for content, that the
                replay does not stand behind. A plain `ValueError` like
                `SolutionFile`'s: the mistake is in a caller's own construction
                and not in input the stage was handed (see
                `idc.resolve.errors`).
        """
        if not self.path.strip():
            raise ValueError("a file replay must name the file it is about")
        replayed = self.status is ReplayStatus.REPLAYED
        if (self.content is not None) is not replayed:
            raise ValueError(
                f"the replay of {self.path!r} states a content exactly when its status is "
                f"{ReplayStatus.REPLAYED!s}; it is {self.status!s} and the content is "
                f"{'set' if self.content is not None else 'unset'}"
            )
        if replayed and self.base_call_id is None:
            raise ValueError(
                f"the replay of {self.path!r} was replayed, so it names the call its base "
                f"content came from"
            )
        if not replayed and self.replayed_call_ids:
            raise ValueError(
                f"the replay of {self.path!r} names replayed edits while its status is "
                f"{self.status!s}; a file that fell back replayed nothing"
            )

    @property
    def replayed(self) -> bool:
        """Whether `content` holds the file's content after the chain."""
        return self.status is ReplayStatus.REPLAYED


def _replacement_of(call: LocatedCall) -> tuple[str, str] | None:
    """Return the `(anchor, replacement)` one edit names, or `None`.

    Args:
        call: A kept write of a file, located.

    Returns:
        The first pair of `_REPLACEMENT_PAIRS` the call answers with an anchor
        holding a word and a replacement that is a string. `None` when it
        answers none — a write tool whose patch arguments are spelled
        differently (`NotebookEdit`'s `new_source`), or a pair with only one
        half.

        The two halves are tested differently on purpose: an anchor holding no
        word names no place in the file, the way a blank path names no file in
        `idc.resolve.calls.file_of`, while an **empty replacement** is a value —
        it is how an edit deletes the text it anchors on, and refusing it would
        make every deletion fall back.
    """
    for anchor_key, replacement_key in _REPLACEMENT_PAIRS:
        anchor = call.call.args.get(anchor_key)
        replacement = call.call.args.get(replacement_key)
        if isinstance(anchor, str) and anchor.strip() and isinstance(replacement, str):
            return anchor, replacement
    return None


def _write_base(entry: SolutionFile) -> tuple[str, str] | None:
    """Return the `(content, call id)` the file's accepted write states, or `None`.

    `None` for a file the chain only ever edited, and for an entry whose
    `write` carries no whole content. The second case cannot arise from
    `solution_files`, which only ever puts a `WriteKind.FULL` call there and
    `content_of` answers every one of those with a string — but `SolutionFile`
    does not check it, so an entry a caller built from an edit falls through
    here instead of raising.
    """
    if entry.write is None:
        return None
    content = content_of(entry.write.call)
    if content is None:
        return None
    return content, entry.write.id


def _read_fallback(entry: SolutionFile, reads: Sequence[LocatedCall]) -> FileReplay:
    """Return the fallback for a file with no accepted write, with its reason.

    The read that is examined is the **last successful read of the file before
    its first kept write**: a read the agent saw fail shows nothing of the file,
    and a read that followed an edit shows the file as that edit left it rather
    than the base a replay would start from. A file whose solution is empty has
    no such boundary, so every successful read of it is a candidate.
    """
    boundary = entry.kept[0].message_index if entry.kept else None
    candidates = [
        located
        for located in reads
        if located.path == entry.path
        and not located.failed
        and (boundary is None or located.message_index < boundary)
    ]
    if not candidates:
        return FileReplay(
            path=entry.path,
            content=None,
            status=ReplayStatus.NO_BASE,
            base_call_id=None,
            replayed_call_ids=(),
        )
    read = sorted(candidates, key=lambda located: located.message_index)[-1]
    extent = rendered_extent(read.result.output) if read.result is not None else None
    partial = extent is not None and extent[0] > 1
    return FileReplay(
        path=entry.path,
        content=None,
        status=ReplayStatus.PARTIAL_READ if partial else ReplayStatus.UNPROVEN_READ,
        base_call_id=read.id,
        replayed_call_ids=(),
    )


def replay_file(entry: SolutionFile, reads: Sequence[LocatedCall] = ()) -> FileReplay:
    """Replay one file's kept edits onto its base content.

    Args:
        entry: The file's solution, as `idc.resolve.writes` resolved it.
        reads: The chain's `read`-category calls, located and in transcript
            order. They are never a base (see the module docstring); they are
            what lets a file that fell back name *why*. Omitting them makes
            every file without an accepted write report
            `ReplayStatus.NO_BASE`.

    Returns:
        The file's `FileReplay`. `ReplayStatus.REPLAYED` with the content when
        the entry holds an accepted full write and every edit behind it could
        be placed in it exactly once; a fallback naming its reason otherwise.
        Computed from the arguments alone, so the same entry always replays to
        the same value.
    """
    base = _write_base(entry)
    if base is None:
        return _read_fallback(entry, reads)
    content, base_call_id = base
    replayed: list[str] = []
    for edit in entry.edits:
        replacement = _replacement_of(edit)
        if replacement is None:
            return _unplaceable(entry, base_call_id, ReplayStatus.ANCHOR_NOT_FOUND)
        anchor, text = replacement
        occurrences = content.count(anchor)
        if occurrences == 0:
            return _unplaceable(entry, base_call_id, ReplayStatus.ANCHOR_NOT_FOUND)
        if occurrences > 1:
            return _unplaceable(entry, base_call_id, ReplayStatus.ANCHOR_AMBIGUOUS)
        content = content.replace(anchor, text, 1)
        replayed.append(edit.id)
    return FileReplay(
        path=entry.path,
        content=content,
        status=ReplayStatus.REPLAYED,
        base_call_id=base_call_id,
        replayed_call_ids=tuple(replayed),
    )


def _unplaceable(entry: SolutionFile, base_call_id: str, status: ReplayStatus) -> FileReplay:
    """Return the fallback for a base whose edits could not all be placed.

    The edits applied before the one that could not be placed are dropped with
    it: a file replayed up to its third of four edits is not the file, and
    handing it out as content would state a text the chain never left behind.
    """
    return FileReplay(
        path=entry.path,
        content=None,
        status=status,
        base_call_id=base_call_id,
        replayed_call_ids=(),
    )


def replay_files(session: Session, chunks: Sequence[Chunk], chain: Chain) -> tuple[FileReplay, ...]:
    """Replay the content of every file of a chain.

    Args:
        session: The transcript the chunks range over.
        chunks: The chunks of that transcript.
        chain: The chain to replay.

    Returns:
        One `FileReplay` per `SolutionFile` of the chain, in the same order —
        so the two lists line up by position as well as by path. A file whose
        content is not decidable is in the list with its reason, never missing
        from it.

        Like `resolve`, this walks the chain's calls more than once: the reads
        are collected here and `solution_files` collects the writes again. Both
        are computed from these three arguments alone, so the repetition costs
        time and nothing else.

    Raises:
        ResolveError: If the chain and the chunks or the chunks and the session
            do not fit together; see `idc.resolve.errors`.
    """
    reads = tuple(
        located
        for located in chain_calls(session, chunks, chain)
        if located.category is ToolCategory.READ
    )
    return tuple(replay_file(entry, reads) for entry in solution_files(session, chunks, chain))
