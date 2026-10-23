"""How much of a session was exploration, and what its writes did.

```python
from idc.measure import tool_statistics

stats = tool_statistics(session)
stats.exploration_rate    # read calls / all tool calls
stats.files_edited         # the paths the accepted writes name, verbatim
stats.lines_added          # the `+` lines of every measurable write
stats.exit_code_counts     # {None: 9, 0: 3}
```

Every number here is read off the transcript by code. Nothing is estimated,
nothing is inferred by a model, and a write whose effect this build cannot
count is **reported as uncounted** rather than given a plausible number.

**The denominator — the visible context, not the file.** `tool_statistics`
counts over `idc.importers.visible_context`, exactly as `idc.measure.tokens`
does, because that function is the one place that decides what counts as
context. A reader comparing an exploration rate against a token count is then
comparing two figures over the same set of messages. The two reasons a message
is left out — the agent's own compaction replaced it, or it is an injected
harness notice — are that function's and are not re-derived here. No fixture of
the shared corpus carries a tool call on a hidden message, so the published
tables are the same under either reading; the rule is pinned on a synthetic
transcript instead, in `tests/test_measure_statistics.py`.

**The exploration rate — `read_calls / total_calls`, with no undefined region.**
`total_calls` is every call of the visible context whatever its category,
`other` included: a `Task` call is not exploration, and leaving it out of the
denominator would raise the rate of every session that delegates. A session
that made **no** tool call has no exploration rate, and the rate refuses
(`NoToolCallsError`) instead of answering `0.0` or a `nan` — the decision
`CompressionRate` took for its zero baseline, for the same reason: a figure the
study reports may have no undefined region. The refusal sits on the *property*
and not on the constructor, because the other three metrics are perfectly
defined at zero calls (no file was edited, no line was added, the histogram is
empty) and refusing to build the object would throw them away with it.

**Files edited — distinct path strings, verbatim.** The path is
`idc.resolve.file_of`, which returns what the agent wrote and normalizes
nothing, and this count does not normalize either: `src/a.py` and `./src/a.py`
are **two** files here. That is the decision the resolver took and the
compactor's merge took again — whether two spellings are one file is a question
only a filesystem can answer, and this tool has none — so `file_count` is an
upper bound on the number of distinct files whenever a session spells one path
two ways. A write the agent saw fail (`ToolResult.is_error`) is **not** a file
edited and is listed in `failed_write_ids`: the file it names may never have
changed at all. A write the transcript holds no result for *is* counted — a
truncated transcript is not a failed call — and `calls_without_result` says how
many such calls there were.

**Lines added — the `+` lines of a diff, per write shape.** For a whole write
(`WriteKind.FULL`) every line of the content is an addition:

```
lines_added(full write) = len(content_of(call).splitlines())
```

For an edit, the two strings the agent sent are diffed line by line and the
additions are the lines the diff inserts or replaces with:

```
lines_added(edit) = sum(j2 - j1 for tag, i1, i2, j1, j2 in opcodes
                        if tag in {"insert", "replace"})
```

with `opcodes` from `difflib.SequenceMatcher(None, old.splitlines(),
new.splitlines(), autojunk=False)`. That is the `+` count a unified diff of the
two strings would print. `autojunk` is off because it only engages from 200
lines of replacement upward (`difflib` tests `len(b) >= 200`), and a
measurement must not change its rule at a size threshold.

What the figure reports and what it does not, concretely:

- A whole write **over-reports** whenever it overwrites a file that already
  held some of those lines. `rewritten_after_failed_edit` writes
  `docs/install.md` whole twice, so the fixture's 23 added lines are nearly
  twice what the file ever held: the transcript does not say what was in the
  file before a write, because a `Write` call carries the new content and
  nothing else.
- A **changed** line counts as an added line. A one-line-for-one-line
  replacement is one addition and one deletion, which is what `git` reports
  too, so `lines_added` is not "lines of new code" and must not be read as
  one.
- An edit's number is the addition count of **one** diff. `difflib` is
  deterministic, so the number is reproducible, but another diff algorithm can
  distribute the same change differently between additions and deletions.

**The shapes that are excluded rather than guessed.** `lines_added_of` answers
`None`, and `tool_statistics` lists the call in `unmeasured_write_ids`, when:

- the edit says it applies to **every** occurrence (`REPEAT_KEYS`), because the
  arguments carry one replacement and not how many times it was applied. One
  application's worth would be right only for a file that held the text once,
  and `interleaved_intents` carries exactly such a call — a `replace_all`
  rename — which is why that fixture's total is 8 and not 9;
- the write's patch arguments are spelled in a way this build has never seen
  (`NotebookEdit` sends `new_source`, not `new_string`) — the new text is in
  the arguments, but which text it replaces is not, so no addition count
  follows from them;
- the write resolves to no file (`file_of` is `None`), in which case the lines
  cannot be attributed to a file and `lines_added` is the sum over `per_file`
  by construction.

A call of the first two shapes that *does* name its file still appears in
`files_edited`: the file is known even when the lines are not, and the two gaps
are reported separately. There is no fourth shape — a `WriteKind.FULL` write
always carries its content, because `write_kind` answers `FULL` only for a call
that holds a string under a content key.

**The exit-code histogram — keyed by the exit code, over every answered call.**
One bucket per distinct `ToolResult.exit_code`, with `None` its own bucket for
a result that records none. It covers every category and not only `exec`:
`exit_code` sits on every `ToolResult` in the canonical schema, and an
exec-only histogram would silently drop a code an importer recorded for
another tool. A call the transcript holds no result for is in **no** bucket and
in `calls_without_result`, so the histogram reconciles —
`sum(counts) + calls_without_result == total_calls`.

`ToolResult.is_error` is a second, independent axis and gets **no** bucket of
its own: a bucket keyed on something that is not an exit code would make one
key space mean two things. A failing result lands in the bucket its own exit
code names — the `None` bucket when it records none, which is where the one
failed `Edit` of the corpus lands — and `error_results` counts the failures on
their own axis.

Nothing here reads a clock, a model, the network or the filesystem, and two
calls over the same transcript are equal objects.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher

from idc.importers import ImportedSession, visible_context
from idc.measure.errors import NoToolCallsError
from idc.resolve import REPLACEMENT_KEYS, WriteKind, content_of, file_of, write_kind
from idc.schema import Message, Session, ToolCall, ToolCategory, ToolResult

__all__ = [
    "REPEAT_KEYS",
    "REPLACEMENT_PAIRS",
    "ExitCodeCount",
    "FileEdits",
    "ToolStatistics",
    "lines_added_of",
    "tool_statistics",
]

REPLACEMENT_PAIRS = tuple(zip(REPLACEMENT_KEYS[::2], REPLACEMENT_KEYS[1::2], strict=True))
"""`idc.resolve.REPLACEMENT_KEYS` in the pairs the agents send them in.

Derived from that list rather than written out again, so a spelling added there
is counted here without a second edit. Counting an edit needs the **two halves
of one pair** together — the text that was there and the text that replaced it —
while `write_kind` only asks whether *any* replacement key is present, which is
why the exported list is flat. `idc.resolve.replay` derives the same pairs for
the same reason and keeps its copy private, so this is a second derivation of
one public list rather than a second list; `tests/test_measure_statistics.py`
pins it against the hand-written copy in `tests/fixture_corpus.py`.

Both halves are required to be strings and **neither** is required to hold a
word, which is where this differs from the replay's reading. Replaying has to
*locate* the anchor in the file, so a blank anchor names no place and falls
back; counting only needs the two strings, and an empty anchor is how an agent
inserts text — every line of the replacement is then an addition, which is
exactly what the diff reports.
"""

REPEAT_KEYS = ("replace_all", "replaceAll")
"""Argument keys an edit uses to say it replaces **every** occurrence.

Claude Code's `Edit` sends `replace_all` and OpenCode's `edit` sends
`replaceAll`; both are recorded in the agent-native transcripts under
`tests/native/`, and `tests/fixtures/interleaved_intents.json` carries the one
corpus call that sets the flag. Only observed spellings are listed, the
evidence rule `idc.resolve.calls` states.

A call whose flag is true has **no** computable addition count: the arguments
carry one replacement and never how many times it was applied, and the file's
earlier content is not in the transcript either. Such a call is reported in
`ToolStatistics.unmeasured_write_ids` instead of being counted as a single
application. The flag is read only to *exclude*; a spelling this build has
never seen therefore leaves a repeated edit counted once, which is the known
edge of this exclusion and not a claim about it.
"""


def _lines(text: str) -> list[str]:
    """Return the lines of `text`, a trailing newline terminating the last one."""
    return text.splitlines()


def _repeats(call: ToolCall) -> bool:
    """Return whether the call says it replaces every occurrence in the file.

    Args:
        call: A write-category call.

    Returns:
        `True` when one of `REPEAT_KEYS` holds `True`. Any other value — the
        flag absent, `False`, or something that is not a boolean — is read as a
        single application, which is what every recorded `false` means.
    """
    return any(call.args.get(key) is True for key in REPEAT_KEYS)


def _replacement_of(call: ToolCall) -> tuple[str, str] | None:
    """Return the `(old, new)` strings an edit replaces, or `None`.

    Args:
        call: A write-category call.

    Returns:
        The first pair of `REPLACEMENT_PAIRS` whose two halves are both
        strings. `None` when the call answers none — a patch spelled in a way
        this build has never seen, or a pair with only one half, both of which
        leave the addition count undefined rather than zero.
    """
    for old_key, new_key in REPLACEMENT_PAIRS:
        old = call.args.get(old_key)
        new = call.args.get(new_key)
        if isinstance(old, str) and isinstance(new, str):
            return old, new
    return None


def _inserted_lines(old: str, new: str) -> int:
    """Return how many lines a line-level diff from `old` to `new` adds.

    Args:
        old: The text the edit replaces.
        new: The text it replaces it with.

    Returns:
        The `+` line count of a unified diff of the two — see this module's
        docstring for the formula and for what it over- and under-reports.
    """
    matcher = SequenceMatcher(None, _lines(old), _lines(new), autojunk=False)
    return sum(
        stop - start
        for tag, _, _, start, stop in matcher.get_opcodes()
        if tag in {"insert", "replace"}
    )


def lines_added_of(call: ToolCall) -> int | None:
    """Return how many lines a write call added, or `None` if that is not computable.

    Args:
        call: Any tool call, of any category.

    Returns:
        For a `WriteKind.FULL` write, every line of its content. For an edit,
        the `+` lines of a line-level diff between the text it replaces and the
        text it replaces it with. `None` for a call that is not a write, and
        for a write whose shape leaves the count undefined: an edit that
        applies to every occurrence (`REPEAT_KEYS`) or one whose patch
        arguments are spelled in a way this build has never seen. `None` is
        never silently read as `0` — `tool_statistics` reports such a call in
        `unmeasured_write_ids`, because a write that added no line and a write
        whose lines cannot be counted are different findings.

    The `content is None` branch below is unreachable from a transcript and
    kept as a type guard: `write_kind` answers `FULL` only for a call that
    carries a string under a content key, and `content_of` returns that string
    — the same reasoning `idc.resolve.replay` gives for its own such branch.
    """
    kind = write_kind(call)
    if kind is WriteKind.FULL:
        content = content_of(call)
        return None if content is None else len(_lines(content))
    if kind is WriteKind.EDIT:
        if _repeats(call):
            return None
        replacement = _replacement_of(call)
        return None if replacement is None else _inserted_lines(*replacement)
    return None


@dataclass(frozen=True, slots=True)
class FileEdits:
    """What the accepted writes of one file did to it.

    Attributes:
        path: The file as the agent spelled it, verbatim and not normalized.
        writes: Accepted write calls naming that spelling — a write the agent
            saw fail is not one of them.
        lines_added: Their added lines, summed; the writes whose count is
            undefined contribute nothing and are reported by
            `ToolStatistics.unmeasured_write_ids`.
    """

    path: str
    writes: int = 0
    lines_added: int = 0

    def __post_init__(self) -> None:
        """Reject a path naming no file and a negative count.

        Raises:
            ValueError: Always a plain one and never a `MeasureError`: a blank
                path and a count below zero can only come from a caller's own
                construction. `idc.resolve.file_of` answers a blank path
                argument with `None`, so no transcript reaches here with one.
        """
        if not self.path.strip():
            raise ValueError("a file breakdown has to name a file; got a blank path")
        if self.writes < 0 or self.lines_added < 0:
            raise ValueError(
                f"a count cannot be negative; got writes={self.writes}, "
                f"lines_added={self.lines_added} for {self.path!r}"
            )

    def __str__(self) -> str:
        """Return the entry as one diagnostic line."""
        return f"{self.path}: {self.writes} write(s), {self.lines_added} line(s) added"


@dataclass(frozen=True, slots=True)
class ExitCodeCount:
    """One bucket of the exit-code histogram.

    Attributes:
        exit_code: The code the results in this bucket carry, or `None` for the
            bucket of the results that record none — see this module's
            docstring for why that is a bucket and not an omission.
        calls: How many answered calls are in it.
    """

    exit_code: int | None
    calls: int = 0

    def __post_init__(self) -> None:
        """Reject a negative count.

        Raises:
            ValueError: If `calls` is negative. A plain one: nothing here
                counts below zero.
        """
        if self.calls < 0:
            raise ValueError(f"a bucket cannot hold {self.calls} call(s)")

    def __str__(self) -> str:
        """Return the bucket as one diagnostic line."""
        code = "no exit code" if self.exit_code is None else f"exit {self.exit_code}"
        return f"{code}: {self.calls} call(s)"


@dataclass(frozen=True, slots=True)
class ToolStatistics:
    """What a transcript's tool calls were, counted over its visible context.

    Attributes:
        session_id: `Session.id` of the transcript that was counted.
        read_calls: `read`-category calls.
        write_calls: `write`-category calls, failed and unmeasurable ones
            included — it is what the agent did, not what could be counted.
        exec_calls: `exec`-category calls.
        other_calls: `other`-category calls.
        per_file: One entry per distinct path an accepted write named, in the
            order the first of those writes appears in the transcript.
        exit_codes: The histogram, `None` first and then by ascending code.
        error_results: Results whose `is_error` is true, whatever their exit
            code — the second axis of this module's docstring.
        failed_write_ids: `ToolCall.id` of the writes the agent saw fail, in
            transcript order. Excluded from `per_file`.
        unmeasured_write_ids: `ToolCall.id` of the writes whose added lines are
            not computable from the transcript, in transcript order.
        calls_without_result: Calls the transcript holds no result for; they
            are in no bucket of `exit_codes`.
    """

    session_id: str
    read_calls: int = 0
    write_calls: int = 0
    exec_calls: int = 0
    other_calls: int = 0
    per_file: tuple[FileEdits, ...] = ()
    exit_codes: tuple[ExitCodeCount, ...] = ()
    error_results: int = 0
    failed_write_ids: tuple[str, ...] = ()
    unmeasured_write_ids: tuple[str, ...] = ()
    calls_without_result: int = 0

    def __post_init__(self) -> None:
        """Reject a negative count and a file counted twice.

        Raises:
            ValueError: Always a plain one and never a `MeasureError`: nothing
                in this stage counts below zero and `tool_statistics` groups
                the writes by path, so both can only come from a caller's own
                construction. The duplicate check is what makes `files_edited`
                a list of *distinct* files rather than a claim about one.
        """
        counts = {
            "read_calls": self.read_calls,
            "write_calls": self.write_calls,
            "exec_calls": self.exec_calls,
            "other_calls": self.other_calls,
            "error_results": self.error_results,
            "calls_without_result": self.calls_without_result,
        }
        negative = {name: value for name, value in counts.items() if value < 0}
        if negative:
            spelled = ", ".join(f"{name}={value}" for name, value in sorted(negative.items()))
            raise ValueError(f"a count cannot be negative; got {spelled}")
        paths = [entry.path for entry in self.per_file]
        if len(set(paths)) != len(paths):
            raise ValueError(
                f"every file is counted once in a breakdown; got {len(paths)} entries for "
                f"{len(set(paths))} file(s)"
            )

    def __str__(self) -> str:
        """Return the statistics as one diagnostic line."""
        return (
            f"{self.session_id}: {self.read_calls}/{self.total_calls} read calls, "
            f"{self.file_count} file(s) edited, {self.lines_added} line(s) added"
        )

    @property
    def total_calls(self) -> int:
        """Return every tool call of the visible context, whatever its category."""
        return self.read_calls + self.write_calls + self.exec_calls + self.other_calls

    @property
    def exploration_rate(self) -> float:
        """Return `read_calls / total_calls` — the figure the study reports.

        `1.0` for a session that only ever read and `0.0` for one that never
        did. Never clamped, because it cannot leave `[0, 1]`: the numerator is
        one of the four summands of the denominator.

        Raises:
            NoToolCallsError: If the session made no tool call, which leaves
                the rate undefined. Refused rather than answered with a `0.0`
                that would read as "this session explored nothing" for a
                session that used no tool at all.
        """
        if self.total_calls == 0:
            raise NoToolCallsError(self.session_id)
        return self.read_calls / self.total_calls

    @property
    def files_edited(self) -> tuple[str, ...]:
        """Return the distinct paths the accepted writes named, verbatim.

        In the order the first accepted write of each appears in the
        transcript. Two spellings of one file are two entries — see this
        module's docstring.
        """
        return tuple(entry.path for entry in self.per_file)

    @property
    def file_count(self) -> int:
        """Return how many distinct path spellings the accepted writes named."""
        return len(self.per_file)

    @property
    def lines_added(self) -> int:
        """Return the added lines of every measurable write, summed over the files."""
        return sum(entry.lines_added for entry in self.per_file)

    @property
    def exit_code_counts(self) -> Mapping[int | None, int]:
        """Return the histogram as a mapping, `None` first and then by ascending code."""
        return {entry.exit_code: entry.calls for entry in self.exit_codes}


@dataclass
class _FileTally:
    """Mutable accumulator behind one `FileEdits`."""

    writes: int = 0
    lines_added: int = 0


def _results_by_call_id(messages: Sequence[Message]) -> dict[str, ToolResult]:
    """Return the result of every answered call, keyed by `ToolCall.id`."""
    return {
        message.tool_result.call_id: message.tool_result
        for message in messages
        if message.tool_result is not None
    }


def _sorted_buckets(counts: Mapping[int | None, int]) -> tuple[ExitCodeCount, ...]:
    """Return the histogram as buckets, the missing code first and then ascending.

    Args:
        counts: How many answered calls carry each exit code.

    Returns:
        One bucket per key. The order is fixed rather than the transcript's so
        that two sessions' histograms read the same way; `None` sorts first
        because it is not a code and has no place among them.
    """
    ordered: list[ExitCodeCount] = []
    if None in counts:
        ordered.append(ExitCodeCount(exit_code=None, calls=counts[None]))
    ordered.extend(
        ExitCodeCount(exit_code=code, calls=counts[code])
        for code in sorted(code for code in counts if code is not None)
    )
    return tuple(ordered)


def tool_statistics(session: ImportedSession | Session) -> ToolStatistics:
    """Return the exploration rate, the write counts and the exit-code histogram.

    Counted over the **visible context** and never over the raw session, the
    way `idc.measure.count_context` counts: `idc.importers.visible_context` is
    the one place that decides what counts as context, and this function
    re-derives none of it.

    Args:
        session: An `ImportedSession`, or a canonical `Session` on its own —
            including one loaded back from a file.

    Returns:
        The statistics, with the per-file breakdown and the identifiers of the
        writes that were not counted, so a number a reader doubts can be traced
        to the calls that produced it.

    Raises:
        TranscriptImportError: If the session was compacted by the agent itself
            but does not locate its summary. Raised by `visible_context` and
            passed through unchanged, exactly as in `idc.measure.tokens`:
            measuring a transcript whose dropped history cannot be located is
            the one thing this must not do.
        SchemaValidationError: If the visible messages do not satisfy the
            canonical schema, likewise from `visible_context`.
    """
    context = visible_context(session)
    results = _results_by_call_id(context.messages)
    categories = dict.fromkeys(ToolCategory, 0)
    tallies: dict[str, _FileTally] = {}
    exit_codes: dict[int | None, int] = {}
    failed: list[str] = []
    unmeasured: list[str] = []
    without_result = 0
    errors = 0
    for message in context.messages:
        for call in message.tool_calls:
            categories[call.category] += 1
            result = results.get(call.id)
            if result is None:
                without_result += 1
            else:
                exit_codes[result.exit_code] = exit_codes.get(result.exit_code, 0) + 1
                errors += int(result.is_error)
            if call.category is not ToolCategory.WRITE:
                continue
            if result is not None and result.is_error:
                failed.append(call.id)
                continue
            # Two independent gaps: a write can name no file, and a write
            # can name one whose added lines do not follow from the arguments.
            # Either makes the call unmeasured; only the first keeps it out of
            # the breakdown.
            path = file_of(call)
            added = lines_added_of(call)
            if path is None or added is None:
                unmeasured.append(call.id)
            if path is not None:
                tally = tallies.setdefault(path, _FileTally())
                tally.writes += 1
                tally.lines_added += 0 if added is None else added
    return ToolStatistics(
        session_id=context.session.id,
        read_calls=categories[ToolCategory.READ],
        write_calls=categories[ToolCategory.WRITE],
        exec_calls=categories[ToolCategory.EXEC],
        other_calls=categories[ToolCategory.OTHER],
        per_file=tuple(
            FileEdits(path=path, writes=tally.writes, lines_added=tally.lines_added)
            for path, tally in tallies.items()
        ),
        exit_codes=_sorted_buckets(exit_codes),
        error_results=errors,
        failed_write_ids=tuple(failed),
        unmeasured_write_ids=tuple(unmeasured),
        calls_without_result=without_result,
    )
