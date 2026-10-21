"""The last write per file of a chain — the file half of the accepted solution.

One rule, applied per file and computed from the transcript alone: **the last
full write of a file, plus the edits that follow it.** A full write replaces the
whole file, so everything written to it before is gone and is path; an edit
replaces a part, so the write it patched still shows and is kept with it. A file
the chain only ever edited therefore keeps its whole edit sequence — there is no
full write to anchor it to, and dropping an earlier edit would lose the part of
the file it changed.

```python
from idc.resolve import solution_files

for entry in solution_files(session, chunks, chain):
    entry.path        # the file, spelled as the agent spelled it
    entry.write       # the last full write, or None if it was only edited
    entry.edits       # the edits after it, in transcript order
    entry.superseded  # the writes it overwrote — path, not solution
    entry.failed      # the writes the agent saw fail — path either way
```

Three things are deliberately *not* decided here:

- **Nothing is summarized.** A `SolutionFile` holds `LocatedCall`s pointing at
  the session's own `ToolCall` and `ToolResult` objects, so a kept write carries
  the arguments and the output verbatim (the parent's criterion, #20). What the
  compactor writes out of them is T6's question.
- **What the file now contains** is not reconstructed: the entries are the write
  *calls*, which is the edit-sequence representation. `idc.resolve.replay`
  (T5.4, #24) replays an entry into the file's content where the transcript
  states a base to replay onto, and leaves it as this edit sequence otherwise.
- **An edit that another edit overwrote** is not detected. Two edits to the same
  region are two kept edits here, because telling them apart needs the file's
  content and this stage has no filesystem. *This* non-detection only ever
  keeps too much, never too little — unlike the non-detections `idc.resolve`'s
  own docstring records, where a file is missing from the solution entirely.

A write the agent saw fail (`ToolResult.is_error`) is reported as `failed` and is
neither accepted nor superseded: it changed nothing, so it can be neither the
file's last write nor the overwriting of an earlier one. The corpus holds one
such write — the refused edit in
`tests/fixtures/rewritten_after_failed_edit.json` — and there the rule decides
only which tuple it lands in, not the solution: a successful rewrite follows it,
so without the rule it would count as superseded instead and `kept` would be
`['c04']` either way. The outcome the rule exists for needs a failed write that
is its file's *last* write, where reading it as the accepted one would present
an empty change as the solution; no transcript of the corpus has that shape, so
what is pinned by a test is the entry such a file produces (an empty `kept`,
built on the value object) and the tuple the refused edit lands in.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from idc.resolve.calls import LocatedCall, WriteKind, chain_calls
from idc.schema import Chain, Chunk, Session, ToolCategory

__all__ = [
    "SolutionFile",
    "solution_files",
    "write_calls",
]


@dataclass(frozen=True, slots=True)
class SolutionFile:
    """What one file of a chain ended up as, and what it took to get there.

    The four tuples partition every write of the chain that names this file:
    `write` and `edits` are the solution, `superseded` and `failed` are path.

    Attributes:
        path: The file, exactly as the agent spelled it. Not normalized — the
            resolver has no filesystem, so two spellings of one file are two
            entries (see `idc.resolve.calls.file_of`).
        write: The last full write of the file, or `None` when the chain only
            edited it (or when every write to it failed).
        edits: The edits after that write, in transcript order; all edits of
            the chain when there is no full write.
        superseded: The writes the accepted one overwrote, in transcript order
            — earlier full writes and the edits that patched them. Empty when
            there is no full write, because an edit overwrites nothing whole.
        failed: The writes the agent saw fail, in transcript order. They took
            no effect, so they are neither the solution nor part of what the
            accepted write replaced.
    """

    path: str
    write: LocatedCall | None
    edits: tuple[LocatedCall, ...]
    superseded: tuple[LocatedCall, ...]
    failed: tuple[LocatedCall, ...]

    def __post_init__(self) -> None:
        """Reject an entry that states nothing about its file.

        Raises:
            ValueError: If the path holds no word, or if the entry holds no
                write at all — neither accepted, nor superseded, nor failed. A
                file nothing ever wrote to is not a file of the chain, and an
                empty entry would make the partition of the writes look
                complete while naming a file no call names. A plain
                `ValueError` like `SubagentSpan`'s in `idc.chunk.subagents`:
                the mistake is in a caller's own construction, not in input the
                stage was handed (see `idc.resolve.errors`).
        """
        if not self.path.strip():
            raise ValueError("a solution file must name the file it is about")
        if self.write is None and not self.edits and not self.failed:
            raise ValueError(
                f"the solution file {self.path!r} holds no write at all; an entry exists "
                f"because the chain wrote to that file"
            )

    @property
    def kept(self) -> tuple[LocatedCall, ...]:
        """The writes that are the solution, in transcript order.

        The last full write followed by the edits behind it, or just the edits
        when the chain never wrote the file whole. Empty for a file every write
        to which failed.
        """
        if self.write is None:
            return self.edits
        return (self.write, *self.edits)

    @property
    def call_ids(self) -> tuple[str, ...]:
        """The ids of the kept writes — what the solution is expressed in."""
        return tuple(located.id for located in self.kept)


def write_calls(session: Session, chunks: Sequence[Chunk], chain: Chain) -> tuple[LocatedCall, ...]:
    """Return every write-category call of a chain, in transcript order.

    Args:
        session: The transcript the chunks range over.
        chunks: The chunks of that transcript.
        chain: The chain to read.

    Returns:
        One `LocatedCall` per call of category `write`, in transcript order —
        including the ones whose arguments name no file (`path is None`), which
        `solution_files` cannot attribute and which are therefore reported only
        here.

    Raises:
        ResolveError: If the chain and the chunks or the chunks and the session
            do not fit together; see `idc.resolve.errors`.
    """
    return tuple(
        located
        for located in chain_calls(session, chunks, chain)
        if located.category is ToolCategory.WRITE
    )


def _entry(path: str, writes: Sequence[LocatedCall]) -> SolutionFile:
    """Apply the last-write rule to the writes of one file.

    Args:
        path: The file the writes name.
        writes: Its write calls, in transcript order; at least one.

    Returns:
        The file's solution and the writes that did not make it.
    """
    failed = tuple(located for located in writes if located.failed)
    effective = [located for located in writes if not located.failed]
    last_full = None
    for position, located in enumerate(effective):
        if located.kind is WriteKind.FULL:
            last_full = position
    if last_full is None:
        return SolutionFile(
            path=path, write=None, edits=tuple(effective), superseded=(), failed=failed
        )
    return SolutionFile(
        path=path,
        write=effective[last_full],
        edits=tuple(effective[last_full + 1 :]),
        superseded=tuple(effective[:last_full]),
        failed=failed,
    )


def solution_files(
    session: Session, chunks: Sequence[Chunk], chain: Chain
) -> tuple[SolutionFile, ...]:
    """Resolve what every file of a chain ended up as.

    Args:
        session: The transcript the chunks range over.
        chunks: The chunks of that transcript.
        chain: The chain to resolve.

    Returns:
        One `SolutionFile` per file the chain wrote to, ordered by the first
        write that touched it — so the entries are in transcript order, and the
        same chain always yields the same list. A write whose arguments name no
        file is attributed to none and does not appear (`write_calls` reports
        it); a file every write to which failed appears with an empty `kept`.

    Raises:
        ResolveError: If the chain and the chunks or the chunks and the session
            do not fit together; see `idc.resolve.errors`.
    """
    grouped: dict[str, list[LocatedCall]] = {}
    for located in write_calls(session, chunks, chain):
        if located.path is None:
            continue
        grouped.setdefault(located.path, []).append(located)
    return tuple(_entry(path, writes) for path, writes in grouped.items())
