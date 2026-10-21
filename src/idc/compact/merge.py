"""Merging an extended chain's earlier `Solution` section with its new writes.

A chain an earlier run summarized and this run gave a new chunk to is written
again (`idc.compact.incremental`), and a rewrite from the new chunks'
`idc.resolve.Resolution` alone states only what *those* chunks wrote: every file
the chain's earlier `Solution` section named would fall out of the summary. This
module is what keeps them — the union of the two sides, with the last write
winning per file.

**The merge is at the text level, and that is a decision.** The new side is a
`Resolution`: objects, each write with the `ToolCall` behind it. The old side is
the Markdown an earlier run rendered, reachable only as the body of the block's
`Solution` section. The two are brought to the one form they can share, the
**rendered bullet**: the old body is cut into one segment per flush `- ` bullet,
`idc.compact.solution_section` renders the new resolution and is cut the same
way, and the two are overlaid by path. The cost is that this module reads the
writer's own output, so the cut has to agree with how `idc.parse` reads a block
— see `_bullet_starts`, and the two tests named there. The alternative, turning
a bullet back into something `idc.resolve.SolutionFile`-shaped, was rejected: a
rendered bullet carries no `ToolCall`, so it would be a second, lossy type with
no transcript behind it, and the conversion would throw away the very prose a
hand-written block states.

What the merge is pinned to, in `tests/test_compact_merge.py`:

- **A file only the old section names keeps its bullet byte for byte** — prose,
  info string, fence width and all. A hand-written block states things the
  writer never emits (`tests/fixtures/already_compacted_block.json` has
  ``- `src/billing/export.py` — `_format_row` converts before formatting:``),
  and dropping the file would be worse still: the chain did write it.
- **An overwritten file loses its old bullet entirely**, prose included. The
  prose described content the new write replaced, so carrying it over to the new
  fence would state a reading of the file that no longer holds.
- **A path is a string.** `idc.resolve.SolutionFile.path` is the path exactly as
  the agent spelled it, so `./src/a.py` and `src/a.py` are two files here, and
  both survive side by side. Deciding that two spellings name one file belongs
  to the stage that reads the transcript, not to the one that writes Markdown;
  inventing it here would also make the merged section disagree with the
  `Solution` section of every chain that is *not* extended.
- **Order: the old section is the spine.** A file the old section names keeps
  its place, whether it was overwritten or left alone, and a file only the new
  resolution names is appended after all of them, in the resolution's own order.
  A reader who saw the earlier block finds its files where they were, and the
  diff between the two blocks is the changed content plus an addition at the
  end. "Last write wins" settles a file's content and not its position.

`EMPTY_SECTION` names no file and is not carried over: an old section that held
it contributes nothing, and a merge with nothing on either side renders it
again, so a section is still never blank.
"""

import re

from idc.compact.template import EMPTY_SECTION, SOLUTION_HEADER, solution_section
from idc.resolve import Resolution
from idc.schema import Chain

__all__ = [
    "merge_solution_bodies",
    "merged_solution_section",
]

_BULLET = "- "
"""What a bullet line starts with, flush and with no indent allowance.

Deliberately stricter than the indentation rule `idc.parse` applies to a header
or a fence: the writer emits its bullets flush, and an indented `- ` line is a
hand's nested list inside the bullet above it rather than a file of its own —
`tests/test_compact_merge.py` pins that it stays with that bullet.
"""

_BULLET_PATH = re.compile(r"^- `([^`]+)`")
"""The path a bullet names: the first backtick span on the line.

What both sides spell the path as — `` - `<path>` ``, on its own or followed by
`NO_CONTENT_NOTE` or by a hand's prose. A bullet with no backtick span names no
file: `EMPTY_SECTION` is the one the writer emits, and a hand's own bullet is
carried over where it stands rather than being read as a path.
"""

_FENCE_LINE = re.compile(r"^(`{3,})(.*)$")
"""A line that may open a code fence: a run of backticks and an info string.

The same pattern `idc.parse` matches, for the same reason and with the same two
conditions — no indent allowance of its own, and no backtick in the info string,
so an inline span opens nothing.
"""

_LINE_ENDING = re.compile(r"(?<=\n)|(?<=\r)(?!\n)")
"""Where one line ends and the next begins: after an LF, a CRLF or a bare CR.

CommonMark's three endings and nothing else, as `idc.parse` splits them, and for
the reason its own copy of this pattern gives: `str.splitlines` also breaks on
the form feed and six other characters, and a fence run written behind a form
feed would then look like a line of its own and close a fence it is inside of.
Here that truncates a kept file's content at the form feed and overlays the rest
away — `test_a_fence_run_behind_a_form_feed_closes_nothing` constructs that body
and counts the path, because the truncated text is a prefix of what is overlaid
in and a substring check passes either way.
"""


def _lines(text: str) -> list[str]:
    """Return the lines of `text`, their terminators kept.

    Args:
        text: The text to cut into lines.

    Returns:
        One entry per line, each ending in its own terminator except a last line
        the text does not terminate, so that `"".join` of them is `text` again —
        which is what lets a kept bullet be copied byte for byte. An empty text
        holds no line.
    """
    lines = _LINE_ENDING.split(text)
    if lines and not lines[-1]:
        lines.pop()
    return lines


def _dedented(line: str, allowance: int) -> str | None:
    """Return a line without its structural indent, or `None` when it has too much.

    Args:
        line: One line as `_lines` returned it, terminator included.
        allowance: How many leading spaces still count as structure —
            `idc.parse.MAX_STRUCTURE_INDENT`, read from there rather than
            written out again.

    Returns:
        The line without its terminator and without its leading spaces, when
        there are at most `allowance` of them; `None` when it is indented
        further, which makes it content and no kind of structure. Only the
        literal space is counted, which is the rule `idc.parse` applies to a
        fence line.
    """
    bare = line.rstrip("\r\n")
    stripped = bare.lstrip(" ")
    if len(bare) - len(stripped) > allowance:
        return None
    return stripped


def _bullet_starts(lines: list[str]) -> list[int]:
    """Return the indices of the lines that open a bullet, fences skipped.

    Args:
        lines: The lines of a `Solution` body, as `_lines` cut them.

    Returns:
        The index of every flush `- ` line that no fence encloses, ascending.
        The fence scan is the one `idc.parse._scan` runs when it finds a block's
        section headers: a fence opens on `_FENCE_LINE` with no backtick in its
        info string and closes on a run of at least as many backticks, indented
        by at most the allowance and followed by nothing but whitespace. It is
        spelled a second time here because that scan is the reader's own
        private one, and the two are held together by test, not by trust:
        `test_a_bullet_line_inside_a_fence_is_content` and
        `test_a_fence_run_behind_a_form_feed_closes_nothing` both feed a body
        whose fenced content holds a bullet line naming a file the new
        resolution writes. Without the scan the first of them splices the new
        bullets into that file's fenced content: its `markdown` fence closes
        around the new path line, the file content beneath lands outside any
        fence, and the rest of the checklist is gone. Checked by dropping the
        scan and reading the result, not assumed.

        A fence that is opened and never closed encloses the rest of the body,
        and this raises nothing of its own: `idc.parse` is what refuses an
        unterminated fence, so a body that came out of a parsed block has none.
    """
    allowance = _indent_allowance()
    starts: list[int] = []
    fence: str | None = None
    for index, line in enumerate(lines):
        dedented = _dedented(line, allowance)
        if fence is None:
            match = None if dedented is None else _FENCE_LINE.match(dedented)
            if match is not None and "`" not in match.group(2):
                fence = match.group(1)
            elif line.startswith(_BULLET):
                starts.append(index)
            continue
        if dedented is None:
            continue
        closing = dedented.rstrip()
        if closing and set(closing) == {"`"} and len(closing) >= len(fence):
            fence = None
    return starts


def _indent_allowance() -> int:
    """Return how many leading spaces still count as structure.

    Returns:
        `idc.parse.MAX_STRUCTURE_INDENT`, so the number has one definition.

    The import is deferred for the reason `idc.compact.incremental`'s module
    docstring gives: `idc.parse` imports this package's format at module level,
    so importing it here at module level is a cycle.
    """
    from idc.parse import MAX_STRUCTURE_INDENT

    return MAX_STRUCTURE_INDENT


def _segments(body: str) -> list[tuple[str | None, str]]:
    """Cut a rendered `Solution` body into one segment per bullet.

    Args:
        body: A `Solution` section's body — the old block's, or what
            `idc.compact.solution_section` rendered for the new resolution.

    Returns:
        One `(path, text)` pair per bullet, in document order, where `text` runs
        from the bullet line to just before the next one and `path` is what
        `_BULLET_PATH` read off it, or `None` for a bullet that names none.
        `"".join` of the texts is `body` again, except that text before the
        first bullet is dropped when it is blank — so nothing a bullet carries
        is lost in the cut. Text before the first bullet that is *not* blank is
        a pair of its own with no path: a hand wrote it, and this module is not
        the one to decide it away.
    """
    lines = _lines(body)
    starts = _bullet_starts(lines)
    segments: list[tuple[str | None, str]] = []
    if not starts:
        return [(None, "".join(lines))] if body.strip() else []
    lead = "".join(lines[: starts[0]])
    if lead.strip():
        segments.append((None, lead))
    for start, bound in zip(starts, [*starts[1:], len(lines)], strict=True):
        match = _BULLET_PATH.match(lines[start])
        segments.append((match.group(1) if match else None, "".join(lines[start:bound])))
    return segments


def _pathless(text: str) -> list[str]:
    """Return what a segment that names no file contributes to the merge.

    Args:
        text: The segment's text — a bullet with no backtick span, or the text a
            side holds before its first bullet.

    Returns:
        `[text]` for anything but the empty-section placeholder, which
        contributes nothing: `EMPTY_SECTION` names no file, and carrying it over
        would put `- (none)` above the bullets of the other side. Anything else
        is a hand's and is kept where it stands — this module is not the one to
        decide a sentence away.

        The placeholder is matched on the whole segment and not on its first
        line, so a fence somebody put beneath an `EMPTY_SECTION` bullet is not
        dropped with it.
    """
    return [] if text.strip() == EMPTY_SECTION else [text]


def merge_solution_bodies(old: str, new: str) -> str:
    """Return the union of two `Solution` bodies, the new write winning per file.

    Args:
        old: The body of the `Solution` section an earlier run rendered, as
            `idc.parse.ParsedBlock.body` returns it.
        new: The body `idc.compact.solution_section` renders for the chunks this
            run added — the section without its header line.

    Returns:
        The merged body, ending in a newline. The bullets of `old` in their own
        order, each file the new body also names replaced by its new bullets at
        the position the old one had, each file only `old` names kept byte for
        byte, and the files only `new` names appended after all of them in its
        order. `EMPTY_SECTION` when neither side names a file, so the section is
        never blank.

        A path is compared as the string it is spelled with. The overlay is
        per **path**, not per bullet: the writer renders one bullet per kept
        write, so a file written and then corrected states both calls, and a
        hand-edited block can name one file twice — when `new` rewrites such a
        file, **every** old bullet for it goes, and when `new` does not name it,
        all of them are kept where they stand. The sentence here once promised
        that a side's repeated bullets are all kept, which is true only of the
        file nothing overwrote; a review disproved it.
    """
    old_segments = _segments(old)
    new_segments = _segments(new)
    new_bullets: dict[str, list[str]] = {}
    for path, text in new_segments:
        if path is not None:
            new_bullets.setdefault(path, []).append(text)

    merged: list[str] = []
    overlaid: set[str] = set()
    for path, text in old_segments:
        if path is None:
            merged.extend(_pathless(text))
        elif path not in new_bullets:
            merged.append(text)
        elif path not in overlaid:
            overlaid.add(path)
            merged.extend(new_bullets[path])
    for path, text in new_segments:
        if path is None:
            merged.extend(_pathless(text))
        elif path not in overlaid:
            merged.append(text)
    return "".join(merged) or f"{EMPTY_SECTION}\n"


def merged_solution_section(chain: Chain, resolution: Resolution) -> str:
    """Return the `Solution` section of a chain's rewritten block.

    Args:
        chain: The chain being written again. A chain an earlier run summarized
            carries that run's block in `compacted_text`, and its `Solution`
            section is the old side of the merge.
        resolution: What the chunks this run added resolved to — the new side.

    Returns:
        `SOLUTION_HEADER` and the merged body of `merge_solution_bodies`. For a
        chain this run opened — `compacted_text` is `None` — exactly what
        `idc.compact.solution_section` renders: there is no earlier solution,
        and nothing about such a chain changes.

    Raises:
        ParseError: If the chain's `compacted_text` is not a compacted block.
            The refusal is `idc.parse`'s and is not wrapped: a chain that came
            out of `idc.parse.parse_chains` carries a block that parsed, so this
            is a caller's own construction.

    The block a chain carries is the chain's own, so the first block of
    `compacted_text` is the one read.
    """
    new = solution_section(resolution)
    if chain.compacted_text is None:
        return new

    # Deferred for the reason `_indent_allowance` gives: the reader imports this
    # package's format, so this package cannot import it at module level.
    from idc.parse import parse_blocks

    old = parse_blocks(chain.compacted_text)[0].body(SOLUTION_HEADER)
    body = merge_solution_bodies(old, new.removeprefix(f"{SOLUTION_HEADER}\n"))
    return f"{SOLUTION_HEADER}\n{body}"
