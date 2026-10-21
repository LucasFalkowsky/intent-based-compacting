"""Resolution of the accepted solution out of the path that led to it.

Within one chain, this stage separates the **solution** — the last write per
file, and the last passing verification command — from the **path**: the reads,
the searches, the failed attempts and the writes a later write overwrote. It is
the step that makes the compaction intent-driven rather than chronological: what
a chain *achieved* is kept, what it *tried* becomes a candidate for removal.

```python
from idc.chunk import chunk_session
from idc.resolve import solution_files

chunks = chunk_session(session)
for entry in solution_files(session, chunks, chain):
    entry.path, entry.write, entry.edits   # the solution
    entry.superseded, entry.failed         # the path
```

Everything here is computed by code, never inferred by a model: the stage reads
`Session`, `Chunk` and `Chain` and nothing else — no git, no filesystem, no
clock, no network. The same transcript therefore always resolves to the same
solution, which is what makes the study's measurements repeatable.

Two properties hold throughout:

- **A tool result is never summarized.** A resolved write points at the
  session's own `ToolCall` and `ToolResult` objects, so a kept result is kept
  verbatim and a dropped one is dropped whole. There is no third state.
- **Nothing is quietly incomplete.** Where the three inputs can disagree — a
  chain naming a chunk nobody offered, a chunk reaching past the transcript —
  the stage refuses with a `ResolveError` instead of resolving to fewer writes
  than the chain made. `idc.resolve.errors` draws the line between those
  refusals and a value object a caller built wrongly.

The modules are one question each: `idc.resolve.calls` locates a chain's tool
calls and reads what a single call did (which file, whole write or patch),
`idc.resolve.writes` applies the last-write-per-file rule to them, and
`idc.resolve.errors` holds the refusals.

**Known limitation — an edit made outside a tool call is not detected.** The
stage sees tool calls, so a file changed by a shell command (`sed -i`, a patch
piped into `git apply`, a script the agent wrote and ran) is invisible to it:
that `exec` call names no file, so the change belongs to no file's solution and
the file looks untouched by the chain. The same holds for a file an agent edits
through a tool this build's registry does not categorize as `write`. This is
accepted rather than guessed at: parsing shell commands for file effects would
have to be right about redirections, here-documents and every editor's flags,
and a wrong guess would put a file into the solution that the transcript does
not show being written. It is recorded as a limitation of the method (#20), not
as a bug.
"""

from idc.resolve.calls import (
    CONTENT_KEYS,
    FILE_PATH_KEYS,
    REPLACEMENT_KEYS,
    LocatedCall,
    WriteKind,
    chain_calls,
    chain_chunks,
    content_of,
    file_of,
    write_kind,
)
from idc.resolve.errors import ForeignChunkError, ResolveError, UnknownChunkError
from idc.resolve.writes import SolutionFile, solution_files, write_calls

# T5.1 (#21) opened this surface; the sibling sub-tickets of T5 append a block
# of their own at the end of this file rather than editing the sorted literal
# below. Two sub-tickets of one parent have already collided in a package
# `__init__.py` here (see the comment at the end of `idc.classify.__init__`),
# and each block being self-contained is the whole resolution: T5.2 (#22) adds
# the verification names, T5.3 (#23) the path classification, T5.4 (#24) the
# edit replay.
__all__ = [
    "CONTENT_KEYS",
    "FILE_PATH_KEYS",
    "REPLACEMENT_KEYS",
    "ForeignChunkError",
    "LocatedCall",
    "ResolveError",
    "SolutionFile",
    "UnknownChunkError",
    "WriteKind",
    "chain_calls",
    "chain_chunks",
    "content_of",
    "file_of",
    "solution_files",
    "write_calls",
    "write_kind",
]
