"""Putting a run's output together: the summary block, then the latest chunk.

This is the first half of the renderer (T7.1, #31) and it is deliberately the
dull one. It decides **what** goes into the output and in **what order**; the
target agent's native tags, the role the summary is attributed to and the
metadata that goes with it are T7.2's, and nothing here branches on `Agent` or
writes a file.

Two rules are the whole stage:

- **The chains are written in the order the block holds them.** Nothing here
  sorts and nothing re-derives an order. `CompactedBlock.chains` is documented
  as chronological, and `idc.classify` produces exactly that order — the
  carried-over chains in the order their blocks had, then the chains this run
  opened — which is also the order `idc.compact.incremental_blocks` renders in,
  with the same "the order is the caller's" rule. A second opinion in this
  module could only disagree with the first, and the disagreement would show up
  as a chain whose summary moves between runs. A block whose chains are *not*
  chronological is therefore passed through as it stands: this stage cannot tell
  a wrong order from an intended one, because nothing it is handed carries a
  timestamp for a chain.
- **The latest chunk is not touched.** Its messages are handed on as the
  `Session`'s own objects, sliced out of `Session.messages` by the chunk's
  half-open range. Nothing is copied, re-ordered, re-timed or re-identified, and
  no tool result is shortened — that verbatim turn is what the resumed agent
  works from, and the study's claim rests on it being the input.

What this stage *does* refuse is a block it cannot render at all: one naming a
chunk that is not there, and one holding a chain that was never compacted. See
`idc.render.errors` for which refusal is about the input and which is a caller's
own mistake.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from idc.parse import join_blocks
from idc.render.errors import UncompactedChainError, UnknownLatestChunkError
from idc.schema import Chunk, CompactedBlock, Message, Session

__all__ = [
    "AssembledTranscript",
    "assemble",
    "latest_messages",
    "summary_text",
]


@dataclass(frozen=True, slots=True)
class AssembledTranscript:
    """The two parts of a rendered run, in the order they are written.

    Kept as two parts rather than one text on purpose: T7.2 has to attribute the
    summary to a role and write metadata beside it, and it has to keep the
    latest chunk's messages *as messages* — a single pre-joined string would
    force it to split the one from the other again, and the only separator it
    could split on would be one this stage invented.

    Attributes:
        summary: The chains' summary blocks, in the block's own order, joined by
            `idc.parse.BLOCK_SEPARATOR` — so the text that is written is the
            text `idc.parse.parse_chains` reads back on the next run. The empty
            string for a block that holds no chains, which is what a first run
            that compacted nothing produces.
        messages: The messages of the chunk the block named as the one to keep
            verbatim, in transcript order and as the `Session`'s own objects.
    """

    summary: str
    messages: tuple[Message, ...]


def summary_text(block: CompactedBlock) -> str:
    """Return the block's chains as one Markdown text, in the block's own order.

    Args:
        block: The run's compacted block. Its `chains` are written in the order
            they stand in; nothing here sorts them.

    Returns:
        The chains' `compacted_text`, `idc.parse.BLOCK_SEPARATOR`-joined, and
        the empty string for a block without chains.

    Raises:
        UncompactedChainError: If a chain of the block is not compacted and so
            carries no text to write.
        ValueError: If a chain's `compacted_text` other than the last does not
            end in a newline — propagated from `idc.parse.join_blocks`, which
            refuses to glue two blocks into one line. Every text this pipeline
            produces does end in one: `idc.compact.render_block` writes it and
            `idc.parse.split_blocks` keeps it.
    """
    texts = []
    for position, chain in enumerate(block.chains, start=1):
        if not chain.compacted or chain.compacted_text is None:
            raise UncompactedChainError(chain.id, position, len(block.chains))
        texts.append(chain.compacted_text)
    return join_blocks(texts)


def _named(chunk_id: str, chunks: Sequence[Chunk]) -> Chunk:
    """Return the chunk carrying an id, or refuse because none does.

    Args:
        chunk_id: The id to look for.
        chunks: The chunks to look in, searched in their own order. `idc.chunk`
            assigns each id once, so the first match is the only one.

    Returns:
        The chunk whose `id` is `chunk_id`.

    Raises:
        UnknownLatestChunkError: If no chunk carries it.
    """
    for chunk in chunks:
        if chunk.id == chunk_id:
            return chunk
    raise UnknownLatestChunkError(chunk_id, [chunk.id for chunk in chunks])


def latest_messages(
    block: CompactedBlock, session: Session, chunks: Sequence[Chunk]
) -> tuple[Message, ...]:
    """Return the messages of the chunk the block keeps verbatim.

    Args:
        block: The run's compacted block, whose `latest_chunk_id` names the
            chunk. That id is the authority on which chunk stays verbatim — the
            chunker's own `Chunk.is_latest` is not consulted, so a block that
            deliberately names an earlier chunk renders that one.
        session: The session the chunk ranges over.
        chunks: The session's chunks, as `idc.chunk.chunk_session` returned
            them. Searched in order; the first chunk carrying the id wins, and
            `idc.chunk` assigns each id once.

    Returns:
        The messages `session.messages` holds over the chunk's half-open range,
        in transcript order and as the session's own objects — not copies.

    Raises:
        UnknownLatestChunkError: If no chunk of `chunks` carries the block's
            `latest_chunk_id`.
        ValueError: If the named chunk's range reaches past `session.messages`.
            A chunk sequence belongs to the session it was computed from, so
            pairing one with another session is a caller's own mistake.
    """
    chunk = _named(block.latest_chunk_id, chunks)
    if chunk.end_index > len(session.messages):
        raise ValueError(
            f"chunk {chunk.id!r} spans messages [{chunk.start_index}, {chunk.end_index}) but "
            f"session {session.id!r} holds {len(session.messages)}; these chunks were computed "
            f"from another session"
        )
    return tuple(session.messages[chunk.start_index : chunk.end_index])


def assemble(
    block: CompactedBlock, *, session: Session, chunks: Sequence[Chunk]
) -> AssembledTranscript:
    """Assemble a run's output: the summary block, then the latest chunk verbatim.

    The stage's entry point, and the input of T7.2, which wraps the two parts in
    the target agent's native form. Deterministic in the strict sense: it reads
    no file, calls no model, asks no clock and draws no random number — the
    result is a function of the three arguments alone.

    Args:
        block: The run's compacted block. Its chains are written in the order it
            holds them, and its `latest_chunk_id` names the chunk kept verbatim.
        session: The session the chunks range over, and the one whose messages
            are handed on unchanged.
        chunks: The session's chunks, as `idc.chunk.chunk_session` returned them.

    Returns:
        An `AssembledTranscript` whose `summary` is the chains' blocks in order
        and whose `messages` are the named chunk's, in that order.

    Raises:
        RenderError: If the block cannot be rendered — `UnknownLatestChunkError`
            when its `latest_chunk_id` names no chunk of `chunks`,
            `UncompactedChainError` when one of its chains carries no summary.
        ValueError: If `chunks` does not belong to `session`; see
            `latest_messages`.
    """
    return AssembledTranscript(
        summary=summary_text(block),
        messages=latest_messages(block, session, chunks),
    )
