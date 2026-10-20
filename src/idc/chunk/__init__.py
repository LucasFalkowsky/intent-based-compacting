"""Chunking of a canonical transcript at user-message boundaries.

A chunk starts at a user message and runs up to (but not including) the next
one, so that every chunk carries exactly one user turn plus the agent work it
triggered. System noise is reduced here and subagent transcripts become chunks
of their own.

A chunk never copies a message: it is a half-open range over `Session.messages`
(`Chunk.start_index` included, `Chunk.end_index` excluded), and the ranges of a
chunked session partition the transcript — contiguous, no gap, no overlap — so
the stage is verifiably lossless:

```python
from idc.chunk import chunk_session, chunk_with_subagents, reduce_session
from idc.schema import Session, load

session = load(Session, "session.json")
chunks = chunk_session(session)          # the boundaries only
reduced = reduce_session(session)        # the boundaries plus the reduction
lifted = chunk_with_subagents(           # the boundaries plus the subagents
    session, registry.is_subagent_tool
)
```

What a *user turn* is, and what happens to the messages a transcript may carry
before its first one, is decided in `idc.chunk.boundaries`. Because a chunk
holds indices and no messages, reducing the noise inside one cannot rewrite a
message in place: `idc.chunk.reduction` returns a `ReducedSession` — a new
transcript, the same chunks re-indexed onto it, and a `ReductionNote` per
removal. `idc.chunk.subagents` works on the chunk list instead: it splits the
chunk that delegated work to a subagent and puts that work into a chunk of its
own, so the two stages compose in either order over the same session.
"""

from idc.chunk.boundaries import (
    CHUNK_ID_PREFIX,
    chunk_boundaries,
    chunk_id,
    chunk_session,
    is_user_turn,
    user_turn_indices,
)
from idc.chunk.reduction import (
    ReducedChunk,
    ReducedSession,
    ReductionNote,
    ReductionReason,
    reduce_chunks,
    reduce_session,
)
from idc.chunk.subagents import (
    SubagentSpan,
    chunk_with_subagents,
    extract_subagents,
    subagent_spans,
)

__all__ = [
    "CHUNK_ID_PREFIX",
    "ReducedChunk",
    "ReducedSession",
    "ReductionNote",
    "ReductionReason",
    "SubagentSpan",
    "chunk_boundaries",
    "chunk_id",
    "chunk_session",
    "chunk_with_subagents",
    "extract_subagents",
    "is_user_turn",
    "reduce_chunks",
    "reduce_session",
    "subagent_spans",
    "user_turn_indices",
]
