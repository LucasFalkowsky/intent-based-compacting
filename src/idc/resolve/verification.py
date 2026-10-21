"""The verification run of a chain — the proof half of the accepted solution.

One rule, computed from the transcript alone: **the last `exec` call of the
chain whose command runs a verification and whose result says it passed.** If
the chain ran verifications but none of them passed, the last of those runs is
kept instead and the chain is `verification_failed` — a chain that tried to
prove itself and did not is a different state from one that never tried.

```python
from idc.resolve import solution_verification

verification = solution_verification(session, chunks, chain)
if verification is None:
    ...                              # the chain ran no verification at all
else:
    verification.status              # verification_passed / verification_failed
    verification.command             # the command line, as the agent sent it
    verification.output              # its output, verbatim
    verification.discarded           # the runs before it — path, not solution
```

**What counts as a verification is not decided here.** The command patterns are
data in `idc.importers`' tool registry (`tool_registry.toml`, #6), and this
module asks `ToolRegistry.match_verification` once per `exec` call. That is
where the prefix stripping lives (`uv run pytest -q` is the same run as
`pytest -q`), where the matching is anchored on the command word (so
`git commit -m "add pytest"` is no run), and where the two deliberate
narrownesses live: a linter or a type checker passing is **not** a verification,
and a false negative is preferred over a false positive, because a missed
verification costs one line of the summary while a wrongly accepted one puts a
proof into the solution that the transcript does not show. This stage widens
neither.

Three states, and they are three values:

- **No verification.** `solution_verification` returns `None` — for a chain that
  made no `exec` call at all, and for one whose `exec` calls ran nothing this
  build recognizes as a verification. The two are told apart by `exec_calls`,
  which reports the chain's `exec` calls whether they verified anything or not.
- **`verification_passed`.** A run with `exit_code == 0` that the agent did not
  see as an error.
- **`verification_failed`.** Verification runs, none of which passed. A run
  whose exit code the transcript does not hold counts as not passing: nothing
  says it succeeded, and a chain must not claim a proof it cannot show.

Nothing is summarized: a `VerificationRun` points at the session's own
`ToolCall` and `ToolResult` objects, so the kept command line and the kept
output are verbatim by construction (the parent's criterion, #20).
"""

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

from idc.importers import VerificationMatch, tool_registry
from idc.resolve.calls import LocatedCall, chain_calls, command_of
from idc.schema import Chain, Chunk, Session, ToolCategory

__all__ = [
    "SolutionVerification",
    "VerificationRun",
    "VerificationStatus",
    "exec_calls",
    "solution_verification",
    "verification_runs",
]


class VerificationStatus(StrEnum):
    """Whether a chain's verification passed or failed.

    The absence of a verification is **not** a member: it is the absence of a
    `SolutionVerification` (`solution_verification` returns `None`), so no
    caller can read "nothing was run" as an outcome of a run.

    Attributes:
        PASSED: The kept run exited 0 and the agent did not see it as an error.
        FAILED: The chain ran verifications and none of them passed. The value
            is the name the ticket gives the state (#22), so a later stage can
            write it out instead of inventing a second spelling.
    """

    PASSED = "verification_passed"
    FAILED = "verification_failed"


@dataclass(frozen=True, slots=True)
class VerificationRun:
    """One `exec` call of a chain whose command runs a verification.

    Attributes:
        call: The located `exec` call, with the result the agent received back
            — the session's own objects, never a copy or a summary.
        match: What the tool registry matched: the pattern, and the segment of
            the command line it matched after the prefix stripping. Kept so a
            summary can say *which* runner proved the chain, and so the
            matching stays inspectable from outside this stage.
    """

    call: LocatedCall
    match: VerificationMatch

    @property
    def id(self) -> str:
        """`ToolCall.id` of the run — what the solution is expressed in."""
        return self.call.id

    @property
    def command(self) -> str | None:
        """The command line, exactly as the agent sent it; see `command_of`.

        `None` is unreachable for a run `verification_runs` built, which only
        builds one for a call whose command it has already read. It is in the
        type because a caller may construct a run itself, and because the
        alternative — asserting here — would make a value object refuse data
        that `idc.resolve.errors` does not cover.
        """
        return command_of(self.call.call)

    @property
    def pattern_id(self) -> str:
        """Id of the registry pattern that matched, e.g. `pytest` or `make-test`."""
        return self.match.pattern.id

    @property
    def exit_code(self) -> int | None:
        """The process exit status, or `None` when the transcript holds none."""
        return None if self.call.result is None else self.call.result.exit_code

    @property
    def output(self) -> str | None:
        """The run's output verbatim, or `None` when the transcript holds no result."""
        return None if self.call.result is None else self.call.result.output

    @property
    def passed(self) -> bool:
        """Whether this run proved anything.

        `True` only for a result that exists, that exited 0 and that the agent
        itself did not see as an error. The two fields are independent in the
        schema — `exit_code` is what the process reported and is `null` when
        the tool reports none, `is_error` is what the agent acted on (#3) — and
        the corpus holds a failed call that carries no exit code at all, so
        neither field implies the other. Requiring both is the reading that
        prefers a false negative: a chain reported as unverified costs one line
        of the summary, a chain claiming a proof it cannot show corrupts the
        resolved solution.
        """
        result = self.call.result
        return result is not None and not result.is_error and result.exit_code == 0

    @property
    def status(self) -> VerificationStatus:
        """This run's outcome as the status a chain would carry for it."""
        return VerificationStatus.PASSED if self.passed else VerificationStatus.FAILED


@dataclass(frozen=True, slots=True)
class SolutionVerification:
    """The run that stands for a chain, and the runs that do not.

    Attributes:
        run: The kept run — the last one that passed, or the last one at all
            when none passed.
        discarded: Every other verification run of the chain, in transcript
            order. They are path, not solution: an earlier failing run is what
            the chain had to get past, and a run that failed *after* the kept
            one is kept here rather than dropped, so a consumer can see that
            the chain did not end green.
    """

    run: VerificationRun
    discarded: tuple[VerificationRun, ...]

    def __post_init__(self) -> None:
        """Reject an object that counts its kept run twice.

        Raises:
            ValueError: If the kept run is also in `discarded`. The two fields
                partition the chain's runs, and a run on both sides would make
                that partition count it twice — a caller building the object
                from a list it forgot to subtract from. A plain `ValueError`
                like `SolutionFile`'s: the mistake is in a caller's own
                construction, not in input the stage was handed (see
                `idc.resolve.errors`).
        """
        if any(other.id == self.run.id for other in self.discarded):
            raise ValueError(
                f"the kept verification run {self.run.id!r} is also among the discarded runs; "
                f"the two are the partition of the chain's runs"
            )

    @property
    def status(self) -> VerificationStatus:
        """`verification_passed` or `verification_failed` — the kept run's outcome."""
        return self.run.status

    @property
    def passed(self) -> bool:
        """Whether the chain's verification passed."""
        return self.run.passed

    @property
    def command(self) -> str | None:
        """The kept run's command line, as the agent sent it."""
        return self.run.command

    @property
    def output(self) -> str | None:
        """The kept run's output, verbatim."""
        return self.run.output

    @property
    def runs(self) -> tuple[VerificationRun, ...]:
        """Every verification run of the chain, kept and discarded.

        Ordered by the message each run sits in. Within one message the kept
        run comes last, because the field pair holds no position inside a
        message — `verification_runs` is the transcript order itself and is
        what to ask when that difference matters. No transcript of the corpus
        issues two verification runs from one message.
        """
        return tuple(sorted((*self.discarded, self.run), key=lambda run: run.call.message_index))


def exec_calls(session: Session, chunks: Sequence[Chunk], chain: Chain) -> tuple[LocatedCall, ...]:
    """Return every `exec`-category call of a chain, in transcript order.

    The counterpart of `write_calls`, and the way the two "no verification"
    states are told apart: a chain with no `exec` call never tried to prove
    itself, while a chain whose `exec` calls are all reported here and in no
    run ran commands this build does not recognize as verifications.

    Args:
        session: The transcript the chunks range over.
        chunks: The chunks of that transcript.
        chain: The chain to read.

    Returns:
        One `LocatedCall` per call of category `exec`, in transcript order —
        including the ones whose arguments name no command.

    Raises:
        ResolveError: If the chain and the chunks or the chunks and the session
            do not fit together; see `idc.resolve.errors`.
    """
    return tuple(
        located
        for located in chain_calls(session, chunks, chain)
        if located.category is ToolCategory.EXEC
    )


def verification_runs(
    session: Session, chunks: Sequence[Chunk], chain: Chain
) -> tuple[VerificationRun, ...]:
    """Return every verification run of a chain, in transcript order.

    Args:
        session: The transcript the chunks range over.
        chunks: The chunks of that transcript.
        chain: The chain to read.

    Returns:
        One `VerificationRun` per `exec` call whose command line the packaged
        tool registry matches against a verification pattern, in transcript
        order. Failing runs are included — which of them is kept is
        `solution_verification`'s question, not this one's.

        **A call two chunks both cover is one run.** The stage is handed its
        chunks rather than chunking itself, so an overlapping list can locate
        the same `exec` call twice (`chunk_session` produces no such list); a
        run is a call, so the second sighting is the same run and is dropped
        here — the de-duplication `idc.resolve.resolution` does for the ids of
        the partition, one layer down. Kept because without it the doubly
        located run reaches `SolutionVerification` as the kept *and* a
        discarded run, which that object refuses with a plain `ValueError`
        (#79).

    Raises:
        ResolveError: If the chain and the chunks or the chunks and the session
            do not fit together; see `idc.resolve.errors`.
    """
    registry = tool_registry()
    runs: list[VerificationRun] = []
    seen: set[str] = set()
    for located in exec_calls(session, chunks, chain):
        if located.id in seen:
            continue
        command = command_of(located.call)
        if command is None:
            continue
        match = registry.match_verification(command)
        if match is None:
            continue
        seen.add(located.id)
        runs.append(VerificationRun(call=located, match=match))
    return tuple(runs)


def solution_verification(
    session: Session, chunks: Sequence[Chunk], chain: Chain
) -> SolutionVerification | None:
    """Resolve which run, if any, proves a chain.

    Args:
        session: The transcript the chunks range over.
        chunks: The chunks of that transcript.
        chain: The chain to resolve.

    Returns:
        The last run that passed, or — when the chain ran verifications and
        none of them passed — the last run at all, together with the other runs
        as `discarded`. `None` when the chain ran no verification: it made no
        `exec` call, or none of its `exec` calls ran a command this build
        recognizes (see `exec_calls` for telling those two apart).

        A run that failed *after* the kept one does not unseat it: the rule is
        the last run that **passed** (#20, #22), and the later failure is in
        `discarded` with its output rather than dropped.

    Raises:
        ResolveError: If the chain and the chunks or the chunks and the session
            do not fit together; see `idc.resolve.errors`.
    """
    runs = verification_runs(session, chunks, chain)
    if not runs:
        return None
    passing = [position for position, run in enumerate(runs) if run.passed]
    kept = passing[-1] if passing else len(runs) - 1
    return SolutionVerification(run=runs[kept], discarded=(*runs[:kept], *runs[kept + 1 :]))
