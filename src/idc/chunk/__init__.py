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
from idc.chunk import chunk_session
from idc.schema import Session, load

chunks = chunk_session(load(Session, "session.json"))
```

What a *user turn* is, and what happens to the messages a transcript may carry
before its first one, is decided in `idc.chunk.boundaries`. The noise reduction
inside a chunk (T3.2) and the subagent chunks (T3.3) are added on top of it.
"""

from idc.chunk.boundaries import (
    CHUNK_ID_PREFIX,
    chunk_boundaries,
    chunk_id,
    chunk_session,
    is_user_turn,
    user_turn_indices,
)

__all__ = [
    "CHUNK_ID_PREFIX",
    "chunk_boundaries",
    "chunk_id",
    "chunk_session",
    "is_user_turn",
    "user_turn_indices",
]
