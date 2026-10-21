"""The failures of the renderer, as one catchable family.

Two kinds of thing can be wrong when a run is rendered, and they are kept apart
the way `idc.parse.errors` keeps them apart — by **where the data comes from**:

- A `CompactedBlock` and a `Session` are canonical documents: both are in
  `idc.schema.CANONICAL_MODELS`, so either can be loaded from disk through
  `idc.schema.load`, and the schema accepts documents this stage cannot render
  (a block naming a chunk the session does not hold, a chain that was never
  compacted). Those are refusals about the input, every one of them is a
  `RenderError`, and T10's CLI catches the whole stage with one `except`.
- A `Chunk` sequence that does not belong to the session it is handed with is a
  caller's own construction — the caller produced it with `chunk_session` — and
  raises a plain `ValueError`, the way `SolutionFile` does in
  `idc.resolve.writes` and `ParsedSection` in `idc.parse.blocks`.

No refusal here is both, so the two sides never overlap, and each subclass
carries the values its message is built from so that a test asserts on an
attribute instead of on prose.
"""

from collections.abc import Iterable, Sequence

from idc.schema import Agent

__all__ = [
    "MissingWriterError",
    "RenderError",
    "UnanchoredToolResultError",
    "UncompactedChainError",
    "UnknownLatestChunkError",
]


class RenderError(Exception):
    """Base class of every failure the renderer raises about its input."""


class UnknownLatestChunkError(RenderError):
    """The block names a chunk the chunk sequence does not hold.

    The latest chunk is the one part of the output that is not summarized, so a
    block whose `latest_chunk_id` matches nothing cannot be rendered at all: the
    alternative would be to render a summary with no verbatim turn behind it,
    which is the one thing the resumed session needs.

    Attributes:
        chunk_id: The `latest_chunk_id` that matched no chunk.
        known: The ids the sequence did hold, in its own order, so that the
            message distinguishes a typo from a chunking that never ran.
    """

    def __init__(self, chunk_id: str, known: Sequence[str]) -> None:
        """Build the error from the unmatched id and the ids that were there.

        Args:
            chunk_id: The `latest_chunk_id` that matched no chunk.
            known: The ids the chunk sequence held, in its own order.
        """
        self.chunk_id = chunk_id
        self.known = tuple(known)
        holds = ", ".join(repr(name) for name in self.known) if self.known else "no chunk"
        super().__init__(
            f"the block's latest_chunk_id {chunk_id!r} names no chunk of this session; "
            f"the chunking holds: {holds}"
        )


class UncompactedChainError(RenderError):
    """A chain of the block carries no summary to write.

    `Chain.compacted_text` is set exactly when `compacted` is true, so a chain
    the compaction stage has not reached yet carries `None`. Both halves of what
    `str(None)` in the summary would do were checked against this repository's
    own writer and parser:

    - As the **last** chain, the four characters land after the preceding
      block's last line, and `parse_chains` reads the whole text back as one
      chain fewer whose `compacted_text` now ends in a blank line and the
      word `None`. The carried
      chain's bytes have changed, which is the one thing a copied block exists
      to prevent.
    - Anywhere **else**, `idc.parse.join_blocks` refuses the text because
      `'None'` does not end in a newline, with a message that names a block
      position and not the chain that is at fault.

    So the run is refused on the chain itself, before anything is joined.

    Attributes:
        chain_id: The chain that holds no block.
        position: 1-based position of that chain within the block's chains, so
            the message points at one of several chains that may share a label.
        total: How many chains the block held.
    """

    def __init__(self, chain_id: str, position: int, total: int) -> None:
        """Build the error from the chain that is not compacted and where it sits.

        Args:
            chain_id: The chain that holds no block.
            position: 1-based position of that chain within the block's chains.
            total: How many chains the block held.
        """
        self.chain_id = chain_id
        self.position = position
        self.total = total
        super().__init__(
            f"chain {chain_id!r}, {position} of {total} in this block, is not compacted and "
            f"carries no compacted_text to render"
        )


class MissingWriterError(RenderError):
    """No native writer is registered for the agent the block names.

    `CompactedBlock.rendered_for` is a canonical field, so the agent it names
    comes from the input — which is why this is a `RenderError` and not a
    caller's mistake. `Agent` has three members and `idc.render.WRITERS` holds
    three entries, so **the refusal cannot be reached today**: it exists so that
    a fourth agent added to the enum fails here by name instead of being
    rendered silently in one of the three other formats. Its own test patches
    the table, because an unpatched one cannot produce it, and a second test
    asserts that the table is complete.

    Attributes:
        agent: The agent no writer was found for.
        known: The agents the table did hold, in the order it held them, so that
            the message distinguishes an unregistered agent from an empty table.
    """

    def __init__(self, agent: Agent, writers: Iterable[Agent]) -> None:
        """Build the error from the agent and the table that did not hold it.

        Args:
            agent: The agent no writer was found for.
            writers: The keys of the writer table.
        """
        self.agent = agent
        self.known = tuple(writers)
        holds = ", ".join(one.value for one in self.known) if self.known else "no agent"
        super().__init__(f"no native writer for {agent.value}; the writer table holds: {holds}")


class UnanchoredToolResultError(RenderError):
    """A tool result cannot be written because its call is not in the chunk.

    Raised by the OpenCode writer alone. OpenCode records a result in the
    `state` of the tool part that made the call — `idc.importers.opencode`
    documents that there is no separate result record — so a result whose call
    is in no message of the chunk has nowhere to go. Writing it with a
    reconstructed call would put a tool call in the transcript that the session
    never made, and `idc.resolve` reads a session's tool calls as the steps of
    the path it resolves the accepted solution from. The
    other two agents record a result as a message of its own and render the same
    chunk without complaint.

    Attributes:
        message_id: The canonical message carrying the result.
        call_id: The call the result answers, which no message of the chunk
            issued.
        agent: The agent whose format has no record for it.
    """

    def __init__(self, message_id: str, call_id: str, agent: Agent) -> None:
        """Build the error from the result, its call and the target agent.

        Args:
            message_id: The canonical message carrying the result.
            call_id: The call the result answers.
            agent: The agent whose format cannot hold it.
        """
        self.message_id = message_id
        self.call_id = call_id
        self.agent = agent
        super().__init__(
            f"message {message_id!r} reports tool call {call_id!r}, which no message of this "
            f"chunk issued; {agent.value} carries a result inside its own call's record, so "
            f"there is no record to write it as"
        )


class SummaryIdInUseError(RenderError):
    """A kept message already carries the identifier the renderer invents.

    `idc.render.native.SUMMARY_MESSAGE_ID` is a constant rather than a derived
    name, so a message of the latest chunk can hold it. The clash used to rest
    on the canonical schema refusing a duplicated message id on re-import, and
    that cover is **not** uniform: Claude Code and OpenCode do refuse, but
    Gemini CLI's importer has append-log semantics, so the later of two records
    sharing an id wins and the `<state_snapshot>` record is **dropped**. A
    review constructed it and the whole compacted block vanished with no error,
    leaving a `raw` session — a summary silently lost, which is the one outcome
    this format exists to prevent.

    So the refusal is the renderer's and it happens before anything is written:
    the renderer invents the identifier, so it owns the clash, and one refusal
    across the three agents beats cover that two of them happen to provide.
    """

    def __init__(self, summary_id: str, message_ids: Sequence[str]) -> None:
        """Build the error from the invented id and the ids it clashed with.

        Args:
            summary_id: The identifier the renderer would have written.
            message_ids: Identifiers of the messages that are being kept.
        """
        self.summary_id = summary_id
        self.message_ids = tuple(message_ids)
        super().__init__(
            f"the kept chunk already holds a message called {summary_id!r}, which is the "
            f"identifier the summary record is written under; rendering it would put two "
            f"records under one identity, and Gemini CLI's importer would drop the summary "
            f"without a word"
        )
