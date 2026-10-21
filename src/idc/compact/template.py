"""The compaction template, and the two sections of it that code can write.

One chain compacts to one Markdown block of four sections under one heading:

```markdown
## [C1] Intent: <label>
### Decisions & constraints
### Rejected approaches
### Solution
### Verification
```

`Decisions & constraints` and `Rejected approaches` are **narrative**: they say
why the chain went the way it did, which no reading of the transcript yields, so
a model writes them (T6.2). This module emits their headers and the
empty-section placeholder and nothing else. `Solution` and `Verification` are
**deterministic**: they are the `idc.resolve.Resolution` of the chain written
out — the accepted writes per file and the run that proves them, with the text
the agent sent and the output it received back, verbatim. Given the same chain
and resolution `render_block` returns byte-identical Markdown; it calls no
model, reads no file, asks no clock.

Two rules hold throughout, and both exist so that a block can be parsed back
into chains on a later run (T6.3):

- **A section is never absent and never blank.** A section with nothing to say
  holds `EMPTY_SECTION` — one line, the same in all four. A missing header would
  break the round trip, and a blank section is indistinguishable from a block
  that was truncated in transit.
- **Nothing is summarized and nothing is invented.** A bullet carries the path
  or the command verbatim, a fenced block carries the text verbatim. Where the
  transcript holds nothing — a write whose arguments carry no text, a call whose
  result never arrived — the bullet says so in a fixed phrase instead of showing
  an empty fence, which would claim the agent wrote or received nothing.

What the resolver calls *path* never appears here: the superseded writes, the
writes the agent saw fail, and the discarded verification runs are what the
chain tried, and the block states what it achieved.
"""

from idc.resolve import (
    REPLACEMENT_KEYS,
    LocatedCall,
    Resolution,
    SolutionVerification,
    content_of,
)
from idc.schema import Chain

__all__ = [
    "DECISIONS_HEADER",
    "EMPTY_SECTION",
    "ERROR_NOTE",
    "NO_COMMAND_NOTE",
    "NO_CONTENT_NOTE",
    "NO_RESULT_NOTE",
    "REJECTED_HEADER",
    "SECTION_HEADERS",
    "SOLUTION_HEADER",
    "UNKNOWN_EXIT_NOTE",
    "VERIFICATION_HEADER",
    "chain_heading",
    "render_block",
    "solution_section",
    "verification_section",
]

DECISIONS_HEADER = "### Decisions & constraints"
"""Header of the first narrative section — a model's (T6.2), not this module's."""

REJECTED_HEADER = "### Rejected approaches"
"""Header of the second narrative section — a model's (T6.2), not this module's."""

SOLUTION_HEADER = "### Solution"
"""Header of the accepted writes per file: `Resolution.solution_files`."""

VERIFICATION_HEADER = "### Verification"
"""Header of the run that proves the chain: `Resolution.verification`."""

SECTION_HEADERS = (DECISIONS_HEADER, REJECTED_HEADER, SOLUTION_HEADER, VERIFICATION_HEADER)
"""The four section headers, in the order a block holds them.

Exported as the template's own order so that the renderer, the parser (T6.3)
and any test state it once. The literal strings are pinned individually in
`tests/test_compact_template.py`: a header is the block's structure, and a near
miss in one of them is a round trip that silently loses a section.
"""

EMPTY_SECTION = "- (none)"
"""The body of a section that has nothing to say — the same line in all four.

A bullet rather than a sentence, because every section body is a bullet list and
a parser that reads one shape reads them all. One constant rather than four, so
that a section cannot be empty in two different ways.
"""

NO_CONTENT_NOTE = "no content recorded"
"""What a kept write whose arguments carry no text says instead of a fence.

Reachable: `idc.resolve.write_kind` falls back to `EDIT` for a write shape it
does not know, and a call of that shape answers neither `CONTENT_KEYS` nor the
replacement keys — `NotebookEdit`'s `new_source` is the recorded example. The
file was written and saying so with an empty fence would claim the agent sent
an empty file.
"""

NO_RESULT_NOTE = "no result recorded"
"""What a verification bullet says when the transcript holds no result for it.

`LocatedCall.result` is `None` for a transcript that ends before the answer
arrived. There is then no exit code and no output to show, and an empty fence
would read as a command that printed nothing.
"""

UNKNOWN_EXIT_NOTE = "exit unknown"
"""What a verification bullet says in place of an exit code the result lacks.

`ToolResult.exit_code` is `null` whenever the tool reports none, which the
corpus holds for a failed call, so this is not a hypothetical branch.
"""

ERROR_NOTE = "error"
"""Appended to a verification bullet whose result the agent saw as a failure.

`ToolResult.is_error` and `exit_code` are independent fields (#3): a result can
carry `is_error` without an exit code, and `exit 0` alone would read as a run
that proved the chain when the agent itself did not accept it.
"""

NO_COMMAND_NOTE = "(no command recorded)"
"""What a verification bullet names the run by when its arguments hold no command.

Unreachable for a run `idc.resolve.verification_runs` built — it only builds one
for a call whose command it has already read — but `VerificationRun.command` is
`str | None` because a caller may construct a run itself. Without backticks: it
is this module speaking, not a command line to be copied.
"""

_MIN_FENCE = 3
"""The shortest code fence, in backticks.

The opening fence grows past this when the text it encloses holds a longer run
of backticks of its own; see `_fence`.
"""

_REPLACEMENT_TEXT_KEYS = REPLACEMENT_KEYS[1::2]
"""The second half of each pair of `REPLACEMENT_KEYS`: the text an edit inserts.

The agents send the keys in pairs — `old_string`/`new_string` and
`oldString`/`newString` — and what an edit *added* is the second of each pair.
Sliced out of the exported list rather than written out again, so that a
spelling added there is rendered without a second edit here; the slice is the
pairing that list documents, which is also how `idc.resolve.replay` derives its
pairs.
"""


def _fence(text: str) -> str:
    """Return `text` enclosed in a code fence, verbatim and never re-indented.

    Args:
        text: The text to enclose — a file's content, an edit's replacement or a
            command's output, exactly as the transcript holds it.

    Returns:
        The opening fence, the text, and the closing fence, each on its own
        line and the whole ending in a newline.

        The fence is `_MIN_FENCE` backticks, or one more than the longest run of
        backticks in `text` when that run is as long or longer. A write whose
        content is itself Markdown with a fenced block in it is the case: three
        backticks inside a three-backtick fence would close the block early and
        the rest of the file would be read as the block's sections. A text whose
        line merely *looks* like a header needs nothing — inside a fence a `###`
        is text, which is what the fence is for.

        A `text` that does not end in a newline gets one, so the closing fence
        is never glued to its last line; an empty `text` yields the two fence
        lines and nothing between them, which is what a command that printed
        nothing produced.
    """
    longest = run = 0
    for character in text:
        run = run + 1 if character == "`" else 0
        longest = max(longest, run)
    ticks = "`" * max(_MIN_FENCE, longest + 1)
    tail = "" if not text or text.endswith("\n") else "\n"
    return f"{ticks}\n{text}{tail}{ticks}\n"


def _written_text(located: LocatedCall) -> str | None:
    """Return the text a kept write sent, or `None` when it sent none.

    Args:
        located: One of a file's kept writes — see
            `idc.resolve.SolutionFile.kept`.

    Returns:
        The whole content for a full write (`idc.resolve.content_of`), the
        replacement text for an edit, and `None` for a write that carries
        neither. The text the agent itself sent, so the block reproduces what
        was written rather than a reading of it — the resolver's edit replay
        (`idc.resolve.replay_file`) is what reconstructs a file's resulting
        content, and it is a question of its own.
    """
    whole = content_of(located.call)
    if whole is not None:
        return whole
    for key in _REPLACEMENT_TEXT_KEYS:
        value = located.call.args.get(key)
        if isinstance(value, str):
            return value
    return None


def chain_heading(chain: Chain) -> str:
    """Return the block's heading line for one chain.

    Args:
        chain: The chain the block is about. Its `id` is already of the form
            `C<n>` — the schema enforces it — and carries the stability the
            round trip rests on, so it is written out as it stands.

    Returns:
        `## [<id>] Intent: <label>`, without a trailing newline.
    """
    return f"## [{chain.id}] Intent: {chain.label}"


def _solution_lines(resolution: Resolution) -> list[str]:
    """Return the body of the `Solution` section, line-terminated.

    Args:
        resolution: What the chain resolved to.

    Returns:
        One bullet per kept write, in the transcript order of the files and of
        the writes within a file, each followed by the fenced text that write
        sent. A file is named once per kept write rather than once in total:
        every bullet then stands for exactly one call of the transcript, and a
        file written whole and then corrected shows the correction as a block of
        its own instead of as a second fence under a shared heading.

        `[EMPTY_SECTION]` when no file of the chain has a kept write — a chain
        that wrote nothing, and a chain every write of which failed. A file
        whose `kept` is empty contributes no bullet at all: an empty bullet
        would name a file the chain left as it found it.
    """
    lines: list[str] = []
    for entry in resolution.solution_files:
        for located in entry.kept:
            text = _written_text(located)
            if text is None:
                lines.append(f"- `{entry.path}` → {NO_CONTENT_NOTE}\n")
                continue
            lines.append(f"- `{entry.path}`\n")
            lines.append(_fence(text))
    return lines or [f"{EMPTY_SECTION}\n"]


def _verification_lines(verification: SolutionVerification | None) -> list[str]:
    """Return the body of the `Verification` section, line-terminated.

    Args:
        verification: The chain's kept run and the runs it discarded, or `None`
            when the chain ran no verification.

    Returns:
        One bullet for the kept run — its command, what came back, and the
        output fenced below it — or `[EMPTY_SECTION]` when there is no run. The
        discarded runs are path and are not rendered: an earlier failing run is
        what the chain had to get past.

        The bullet states only what the result holds: `exit <n>` when it carries
        an exit code and `UNKNOWN_EXIT_NOTE` when it does not, `ERROR_NOTE`
        appended when the agent saw the result as a failure, and
        `NO_RESULT_NOTE` with no fence at all when the transcript holds no
        result. A kept run that ended red is rendered exactly like one that
        passed, as a red run is the resolver's answer when none of the chain's
        runs passed, and hiding it would make the chain look proven.
    """
    if verification is None:
        return [f"{EMPTY_SECTION}\n"]
    run = verification.run
    command = run.command
    named = NO_COMMAND_NOTE if command is None else f"`{command}`"
    result = run.call.result
    if result is None:
        return [f"- {named} → {NO_RESULT_NOTE}\n"]
    outcome = UNKNOWN_EXIT_NOTE if result.exit_code is None else f"exit {result.exit_code}"
    if result.is_error:
        outcome = f"{outcome}, {ERROR_NOTE}"
    return [f"- {named} → {outcome}\n", _fence(result.output)]


def solution_section(resolution: Resolution) -> str:
    """Return the `Solution` section of a block: its header and its body.

    Args:
        resolution: What the chain resolved to.

    Returns:
        `SOLUTION_HEADER` and the lines of `_solution_lines`, ending in a
        newline.
    """
    return f"{SOLUTION_HEADER}\n" + "".join(_solution_lines(resolution))


def verification_section(resolution: Resolution) -> str:
    """Return the `Verification` section of a block: its header and its body.

    Args:
        resolution: What the chain resolved to.

    Returns:
        `VERIFICATION_HEADER` and the lines of `_verification_lines`, ending in a
        newline.
    """
    return f"{VERIFICATION_HEADER}\n" + "".join(_verification_lines(resolution.verification))


def render_block(chain: Chain, resolution: Resolution) -> str:
    """Return the whole Markdown block of one chain, narrative sections empty.

    Args:
        chain: The chain the block is about; its id and label become the
            heading.
        resolution: What that chain resolved to — the two deterministic
            sections are this written out.

    Returns:
        The heading and the four sections in the order of `SECTION_HEADERS`,
        ending in exactly one newline. `Decisions & constraints` and
        `Rejected approaches` carry `EMPTY_SECTION`: they are a model's to write
        (T6.2), and a block whose headers are already in place is what that
        ticket fills in.

        Nothing in here depends on anything outside its two arguments, so the
        same chain and resolution render the same bytes in every run — which is
        what makes the study's before-and-after token counts repeatable.
    """
    return (
        f"{chain_heading(chain)}\n"
        f"{DECISIONS_HEADER}\n{EMPTY_SECTION}\n"
        f"{REJECTED_HEADER}\n{EMPTY_SECTION}\n"
        f"{solution_section(resolution)}"
        f"{verification_section(resolution)}"
    )
