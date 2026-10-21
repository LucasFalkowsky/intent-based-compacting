"""The partition of a chain's tool calls, and what the chain resolves to.

This is where the stage's two halves meet. `idc.resolve.writes` resolves the
accepted writes of a chain and `idc.resolve.verification` the run that proves
it; everything else the chain did is **path** — the reads and searches it
needed to understand the code, the commands that failed, the writes a later
write overwrote, the writes the agent saw fail, the commands that verify
nothing, and every call of a category the resolver has no rule for. The two
sides together are a partition of the chain's calls: one call, one side,
always.

```python
from idc.resolve import resolve

resolution = resolve(session, chunks, chain)
resolution.solution_files    # the accepted writes, per file
resolution.verification      # the run that proves them, or None
resolution.path_call_ids     # every other call of the chain, by id
```

**The path is ids, not calls.** A tool result is never summarized (#20): a
call is either in the solution, where it is carried verbatim as the session's
own `ToolCall` and `ToolResult` objects, or in the path, where only its id is
named so that a consumer can drop it. Rewriting or truncating a path call's
content would be the third state the parent's criterion rules out, and an id
cannot be a half-kept result.

**The partition is over one chain, not over the transcript.** The calls it
ranges over are the calls of the chain's own chunks, so a call in a chunk this
chain does not name is on neither side — it belongs to the chain that does name
it, or to no chain at all, and compaction is per chain. A chain whose chunks
carry no tool call resolves to three empty answers (no files, no verification,
no path) and is not a refusal: a chain that only talked is a chain that wrote
nothing, which is a result and not a disagreement between the inputs (see
`idc.resolve.errors`).

**Known limitation — an edit made outside a tool call is not detected, and the
partition is where it is lost.** A file changed by a shell command (`sed -i`, a
patch piped into `git apply`) is changed by an `exec` call whose arguments name
no file, so no `SolutionFile` is about that file; the command runs no
verification either, so the call is not the chain's proof. Everything not in
the solution is path, so the call lands in `path_call_ids` and a consumer that
drops the path drops the only record of the change, while the file itself looks
untouched by the chain. The alternative — reading file effects out of shell
command lines — would have to be right about redirections, here-documents and
every editor's flags, and a wrong guess would put a file into the solution that
the transcript does not show being written. The limitation is accepted and
recorded (#20) rather than guessed at.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from idc.resolve.calls import chain_calls
from idc.resolve.verification import SolutionVerification, solution_verification
from idc.resolve.writes import SolutionFile, solution_files
from idc.schema import Chain, Chunk, Session

__all__ = [
    "Resolution",
    "path_call_ids",
    "resolve",
    "solution_call_ids",
]


def _unique(call_ids: Iterable[str]) -> tuple[str, ...]:
    """Return the ids in the order they first occur, each one once.

    One call can be located twice: two chunks whose ranges overlap both carry
    it, and the stage is handed its chunks rather than chunking itself (the
    chunker produces no such list). The sides of the partition are ids, so the
    second sighting is the same call and not a second one. Order-preserving
    rather than sorted, because the order the ids are collected in is the
    transcript's.
    """
    seen: dict[str, None] = {}
    for call_id in call_ids:
        seen.setdefault(call_id, None)
    return tuple(seen)


def _solution_ids(
    files: Sequence[SolutionFile], verification: SolutionVerification | None
) -> tuple[str, ...]:
    """Return the ids the solution of a chain is expressed in."""
    ids = [call_id for entry in files for call_id in entry.call_ids]
    if verification is not None:
        ids.append(verification.run.id)
    return _unique(ids)


@dataclass(frozen=True, slots=True)
class Resolution:
    """What one chain resolved to: its solution, and the path that led there.

    The three fields are the parent ticket's answer for a chain (#20) and are
    what T6 compacts: the accepted writes per file, the run that proves them,
    and the ids of every call that is a candidate for removal.

    Attributes:
        solution_files: One entry per file the chain wrote to, in transcript
            order — see `idc.resolve.writes.SolutionFile`. Empty for a chain
            that wrote no file.
        verification: The run that proves the chain, or `None` when it ran no
            verification — see `idc.resolve.verification`.
        path_call_ids: `ToolCall.id` of every call of the chain that is not in
            the solution, in transcript order and each id once. Ids and not
            calls, because a path call is a candidate for removal and never a
            shortened result.
    """

    solution_files: tuple[SolutionFile, ...]
    verification: SolutionVerification | None
    path_call_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        """Reject a resolution whose two sides are not a partition.

        Raises:
            ValueError: If an id is named twice in `path_call_ids`, or if a
                path id is also a solution id. Either would make the two sides
                count one call twice, which is the one thing the partition
                promises they do not — and a caller building a `Resolution`
                from a call list it forgot to subtract the solution from is
                exactly how that happens. A plain `ValueError` like
                `SolutionFile`'s and `SolutionVerification`'s: the mistake is
                in a caller's own construction and not in input the stage was
                handed (see `idc.resolve.errors`).
        """
        duplicates = sorted(
            {
                call_id
                for position, call_id in enumerate(self.path_call_ids)
                if call_id in self.path_call_ids[:position]
            }
        )
        if duplicates:
            raise ValueError(
                f"the path names {', '.join(repr(call_id) for call_id in duplicates)} more than "
                f"once; it is one id per call of the chain"
            )
        both = sorted(set(self.path_call_ids) & set(self.solution_call_ids))
        if both:
            raise ValueError(
                f"{', '.join(repr(call_id) for call_id in both)} is both solution and path; the "
                f"two sides are the partition of the chain's calls"
            )

    @property
    def solution_call_ids(self) -> tuple[str, ...]:
        """The ids the solution is expressed in: the kept writes, then the kept run.

        The other side of `path_call_ids`, and in the same currency, so a
        consumer can decide per call id which calls to keep without walking
        the entries. The kept writes of every file in the order the entries
        hold them, followed by the verification run when there is one.
        """
        return _solution_ids(self.solution_files, self.verification)


def solution_call_ids(session: Session, chunks: Sequence[Chunk], chain: Chain) -> tuple[str, ...]:
    """Return the ids of the calls that are a chain's solution.

    Args:
        session: The transcript the chunks range over.
        chunks: The chunks of that transcript.
        chain: The chain to resolve.

    Returns:
        `ToolCall.id` of every kept write of every file of the chain, followed
        by the id of the verification run that proves it when there is one.
        Each id once.

    Raises:
        ResolveError: If the chain and the chunks or the chunks and the session
            do not fit together; see `idc.resolve.errors`.

            One input shape escapes this family and is tracked as #79: an
            *overlapping* chunk list that covers a verification run twice
            reaches `SolutionVerification` with that run on both sides, and
            raises a plain `ValueError` from there. `chunk_session` produces no
            overlapping list, so the pipeline cannot reach it — but this clause
            does not hold for a caller that builds one by hand.
    """
    return _solution_ids(
        solution_files(session, chunks, chain),
        solution_verification(session, chunks, chain),
    )


def path_call_ids(session: Session, chunks: Sequence[Chunk], chain: Chain) -> tuple[str, ...]:
    """Return the ids of the calls of a chain that are not its solution.

    Args:
        session: The transcript the chunks range over.
        chunks: The chunks of that transcript.
        chain: The chain to resolve.

    Returns:
        `ToolCall.id` of every call of the chain's chunks that is not in
        `solution_call_ids`, in transcript order and each id once — the reads,
        the searches, the superseded and failed writes, the discarded and
        unrecognized commands, the writes whose arguments name no file, and
        every call of a category this stage has no rule for. Empty for a chain
        whose every call is solution, and for a chain that made none.

    Raises:
        ResolveError: If the chain and the chunks or the chunks and the session
            do not fit together; see `idc.resolve.errors`.

            One input shape escapes this family and is tracked as #79: an
            *overlapping* chunk list that covers a verification run twice
            reaches `SolutionVerification` with that run on both sides, and
            raises a plain `ValueError` from there. `chunk_session` produces no
            overlapping list, so the pipeline cannot reach it — but this clause
            does not hold for a caller that builds one by hand.
    """
    kept = set(solution_call_ids(session, chunks, chain))
    walked = _unique(located.id for located in chain_calls(session, chunks, chain))
    return tuple(call_id for call_id in walked if call_id not in kept)


def resolve(session: Session, chunks: Sequence[Chunk], chain: Chain) -> Resolution:
    """Resolve one chain into its solution and its path.

    The stage's entry point: it asks the three questions of the resolver once
    each and returns their answers as one value, so a caller never has to know
    that the files, the verification and the path are computed by three
    different modules.

    Args:
        session: The transcript the chunks range over.
        chunks: The chunks of that transcript, as the chunker produced them.
        chain: The chain to resolve.

    Returns:
        The chain's `Resolution`. Computed from the transcript alone — no git,
        no filesystem, no clock, no model — so the same inputs always resolve
        to the same value.

    Raises:
        ResolveError: If the chain and the chunks or the chunks and the session
            do not fit together; see `idc.resolve.errors`.

            One input shape escapes this family and is tracked as #79: an
            *overlapping* chunk list that covers a verification run twice
            reaches `SolutionVerification` with that run on both sides, and
            raises a plain `ValueError` from there. `chunk_session` produces no
            overlapping list, so the pipeline cannot reach it — but this clause
            does not hold for a caller that builds one by hand.
    """
    return Resolution(
        solution_files=solution_files(session, chunks, chain),
        verification=solution_verification(session, chunks, chain),
        path_call_ids=path_call_ids(session, chunks, chain),
    )
