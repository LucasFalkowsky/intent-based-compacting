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
writer's own output, so the cut has to agree with how `idc.parse` reads a block.
It agrees by **running the reader's own scan**: `idc.parse.split_lines`,
`idc.parse.fence_opening` and `idc.parse.closes_fence` are what `_bullet_starts`
and `_segments` are written with, imported inside those functions because the
reader imports this package at module level — see `_bullet_starts`. The alternative, turning
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

Deliberately stricter than the indentation rule `idc.parse.MAX_STRUCTURE_INDENT`
allows a header or a fence: the writer emits its bullets flush, and an indented
`- ` line is a hand's nested list inside the bullet above it rather than a file
of its own — `tests/test_compact_merge.py` pins that it stays with that bullet.
"""

_BULLET_PATH = re.compile(r"^- `([^`]+)`")
"""The path a bullet names: the first backtick span on the line.

What both sides spell the path as — `` - `<path>` ``, on its own or followed by
`NO_CONTENT_NOTE` or by a hand's prose. A bullet with no backtick span names no
file: `EMPTY_SECTION` is the one the writer emits, and a hand's own bullet is
carried over where it stands rather than being read as a path.
"""


def _bullet_starts(lines: list[str]) -> list[int]:
    """Return the indices of the lines that open a bullet, fences skipped.

    Args:
        lines: The lines of a `Solution` body, as `_lines` cut them.

    Returns:
        The index of every flush `- ` line that no fence encloses, ascending.
        The fence scan is **the reader's own**, not a second spelling of its
        rules: `idc.parse.fence_opening` is what opens a fence here and
        `idc.parse.closes_fence` is what ends it, which are the two functions
        `idc.parse._scan` finds a block's section headers with. They carry the
        indent bound, the info-string condition and the "at least as long"
        closing rule, so no number and no pattern is written out here.

        That the cut needs a fence scan at all is pinned by
        `test_a_bullet_line_inside_a_fence_is_content` and
        `test_a_fence_run_behind_a_form_feed_closes_nothing`, which both feed a
        body whose fenced content holds a bullet line naming a file the new
        resolution writes. Without the scan the first of them splices the new
        bullets into that file's fenced content: its `markdown` fence closes
        around the new path line, the file content beneath lands outside any
        fence, and the rest of the checklist is gone. Checked by dropping the
        scan and reading the result, not assumed. That the scan is the reader's
        and not a copy is pinned by stubbing the public names.

        A fence that is opened and never closed encloses the rest of the body,
        and this raises nothing of its own: `idc.parse` is what refuses an
        unterminated fence, so a body that came out of a parsed block has none.

    The import is deferred for the reason `idc.compact.incremental`'s module
    docstring gives: `idc.parse` imports this package's format at module level,
    so importing it here at module level is a cycle.
    """
    from idc.parse import closes_fence, fence_opening

    starts: list[int] = []
    fence: str | None = None
    for index, line in enumerate(lines):
        if fence is None:
            opening = fence_opening(line)
            if opening is not None:
                fence = opening
            elif line.startswith(_BULLET):
                starts.append(index)
            continue
        if closes_fence(line, fence):
            fence = None
    return starts


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

    What a line is here is `idc.parse.split_lines` — LF, CRLF and a bare CR and
    nothing else — so that `"".join` of the cut is the body again and a kept
    bullet is copied byte for byte. `str.splitlines` would break on seven
    further characters, and a fence run written behind a form feed would then
    look like a line of its own and close a fence it is inside of: that
    truncates a kept file's content at the form feed and overlays the rest away,
    which `test_a_fence_run_behind_a_form_feed_closes_nothing` constructs and
    counts the path for, because the truncated text is a prefix of what is
    overlaid in and a substring check passes either way. The import is deferred
    for the reason `_bullet_starts` gives.
    """
    from idc.parse import split_lines

    lines = split_lines(body)
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
