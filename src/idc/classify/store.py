"""The chain store: the open chains of one transcript, in the order they opened.

The store is the half of the stage that holds no model: it assigns the chain
ids, it keeps each chain's chunks, it builds the candidates an adapter is shown,
and it is the only place that could let a chunk end up in two chains or in none.
Everything it can compute from the transcript it computes itself, and everything
it cannot carry out it refuses:

```python
from idc.classify import ChainStore

store = ChainStore(session)                  # optionally seeded with earlier chains
store.subject(chunks[0])                     # -> ChunkUnderDecision(id="K1", …)
store.open_chain(chunks[0])                  # -> Chain C1, provisionally labelled
store.candidates()                           # -> [ChainCandidate(id="C1", …)]
store.append("C1", chunks[2])                # -> Chain C1 with two chunks
store.chains()                               # -> the canonical chains, in order
```

Four decisions are the store's own:

- **Ids are positions, not hashes.** A new chain gets `C<n>` one past the
  highest chain number the store holds, so a re-run over a transcript that only
  grew keeps the ids of the chains that were already there — T6.3 (#28) writes
  them into the summary block and T8 (#33) reads them back (see
  `idc.classify.ids`).
- **Every chain stays a candidate, and the order is opening order.** T4's goal
  says the first half: a chain is never closed while the transcript runs, so the
  candidate list only grows. The second half is *checked*, not assumed — a chain
  id is its opening position, so a seeded chain whose number goes backwards
  would state two contradicting orders, and the store refuses it rather than
  offering `C3` before `C1`. Every refusal of a *seeded* chain is a
  `ClassifyError` like the rest of the stage, because that chain was parsed out
  of a transcript file (#33) and is input, not a programmer's literal.
- **A chunk belongs to exactly one chain.** A second assignment is
  `ChunkAlreadyAssignedError`, and so is a chunk a seeded chain already claims —
  but the error says which of the two it is looking at: the chunk ids of a chain
  carried over from an earlier run are positions in *that* run's transcript and
  can collide with this run's, which has to be re-mapped where the block is
  parsed (#33) rather than papered over here.
- **A label is provisional and deterministic.** It is the chunk's own user
  message, collapsed and cut at a word boundary — never a model call, because
  the real label is written by the compactor (T6) from the compaction template.
  A **seeded** chain arrives with such a real label, and that label is also
  what its candidate offers as its last user message (#35): its chunks are not
  ranges over this transcript, so there is no turn to read, and the label is the
  only thing the chain has to say what it is about. The first appended chunk
  that *carries a user turn* replaces it; a chunk of subagent work carries none
  and leaves the label standing.

The store copies no message: a `Chunk` is a half-open range over
`Session.messages`, and the store only reads through those ranges — which is why
a range that reaches past the transcript is `ForeignChunkError` instead of a
range that silently resolves to nothing. That is also
why both sides of an adapter's question are built here — the candidates *and*
the `ChunkUnderDecision` the adapter decides about — and why the chunk's user
message is resolved once: the subject and the provisional label are the same
resolution.
"""

from collections.abc import Iterable
from dataclasses import dataclass, field

from idc.chunk import is_user_turn
from idc.classify.adapters import ChainCandidate, ChunkUnderDecision
from idc.classify.errors import (
    BlankChainLabelError,
    ChainAlreadySeededError,
    ChainOrderError,
    ChunkAlreadyAssignedError,
    ChunkOrderError,
    ForeignChunkError,
    UnknownChainError,
)
from idc.classify.ids import chain_id, chain_position
from idc.schema import Chain, Chunk, Session

PROVISIONAL_LABEL_MAX_LENGTH = 60
"""How long a provisional label may be, ellipsis included.

Long enough to carry the first clause of a real user turn, short enough to stay
one line in a summary block.
"""

PROVISIONAL_LABEL_ELLIPSIS = "…"
"""What marks a label that was cut short."""

PROVISIONAL_LABEL_TRIM = " ,.;:"
"""Characters dropped from the end of a cut label, before the ellipsis.

A cut falls at a word boundary, so the character before it is as often a comma
or a colon as a letter — and `with tests:…` reads as a truncated sentence twice
over.
"""


def provisional_label(text: str, position: int) -> str:
    """Return the provisional label of a chain, derived from its user message.

    The label is the user's own words, not a summary: whitespace is collapsed to
    single spaces (a turn can span paragraphs), and anything longer than
    `PROVISIONAL_LABEL_MAX_LENGTH` is cut at the last word boundary that fits
    and marked with an ellipsis. The compactor replaces it with a real label in
    T6; until then it is what tells a human — and an adapter, through
    `ChainCandidate.label` — which intent a chain is.

    Args:
        text: Content of the chain's user message; may be empty.
        position: The chain's position, used for the fallback label when there
            is no user message to take words from.

    Returns:
        A non-empty, single-line label of at most
        `PROVISIONAL_LABEL_MAX_LENGTH` characters. `intent <position>` when
        `text` holds no word at all — a chunk need not carry a user turn (a
        subagent chunk does not), and `Chain.label` may not be empty.
    """
    collapsed = " ".join(text.split())
    if not collapsed:
        return f"intent {position}"
    if len(collapsed) <= PROVISIONAL_LABEL_MAX_LENGTH:
        return collapsed
    head = collapsed[: PROVISIONAL_LABEL_MAX_LENGTH - len(PROVISIONAL_LABEL_ELLIPSIS)]
    if " " in head:
        head = head.rsplit(" ", 1)[0]
    return head.rstrip(PROVISIONAL_LABEL_TRIM) + PROVISIONAL_LABEL_ELLIPSIS


def require_chunk_of(session: Session, chunk: Chunk) -> None:
    """Raise unless `chunk`'s range fits the session's messages.

    This is the whole of what is checkable: a `Chunk` carries indices and no
    session id (#3), so a chunk of another run whose range happens to fit is
    accepted and resolves against *this* session's messages. Only the caller
    can keep that from happening; what the guard catches is the range that
    reaches past the transcript and would otherwise resolve to no message at
    all.

    Args:
        session: The transcript the chunk is supposed to range over.
        chunk: The chunk to check. Its range is non-empty and non-negative by
            the schema's own invariant, so only its end has to be checked.

    Raises:
        ForeignChunkError: If the chunk reaches past the session's messages.
    """
    if chunk.end_index > len(session.messages):
        raise ForeignChunkError(chunk.id, session.id, chunk.end_index, len(session.messages))


def chunk_user_message(session: Session, chunk: Chunk) -> str | None:
    """Return the content of the chunk's last real user turn, or `None`.

    What a *user turn* is, is not restated here: `idc.chunk.is_user_turn` is the
    one definition (a `user` message the harness neither injected nor wrote as
    the echo of a slash command), and the classifier must not disagree with the
    chunker about it.

    A chunk the chunker produced holds **at most one** real user turn, because
    that is where T3.1 splits. This function does not rely on that invariant —
    it takes any range, including one a later ticket builds by hand — so it
    scans backwards and answers with the range's *last* user turn.

    Args:
        session: The transcript the chunk ranges over.
        chunk: The chunk to read.

    Returns:
        The user turn's content verbatim — empty string included, when the turn
        carried only tool calls — or `None` when the chunk holds no user turn at
        all, which is the case for a chunk of subagent work.

    Raises:
        ForeignChunkError: If the chunk is not a range over this session.
    """
    require_chunk_of(session, chunk)
    for index in reversed(range(chunk.start_index, chunk.end_index)):
        message = session.messages[index]
        if is_user_turn(message):
            return message.content
    return None


@dataclass(slots=True)
class _OpenChain:
    """One chain while the transcript is still being classified.

    The mutable counterpart of `Chain`: a chain grows by one chunk at a time, and
    `as_chain` is where it becomes a canonical object again — validated, so an
    invariant the schema owns (a non-empty label, no repeated chunk id) is never
    only promised here.

    Attributes:
        id: The chain's stable id, `C<n>`.
        label: Provisional label, or the label a seeded chain came with.
        chunk_ids: The chain's chunks, in transcript order.
        compacted: Whether `compacted_text` holds this chain's summary block;
            true only for a chain seeded from an earlier run.
        compacted_text: That summary block, or `None`.
        last_user_message: Content of the last user turn seen in this chain —
            or, for a seeded chain, its label, which stands in for the turn its
            carried-over chunks cannot offer and stands until a chunk with a
            user turn is appended (#35).
        last_start_index: Where this chain's last chunk starts, or `-1` when the
            chain was seeded and its chunks are not ranges over this transcript.
        seeded: Whether the chain was carried over from an earlier run. Only
            used to say so in a message: its chunk ids are positions in that
            run's transcript, which is a different cause for the same clash.
    """

    id: str
    label: str
    chunk_ids: list[str] = field(default_factory=list)
    compacted: bool = False
    compacted_text: str | None = None
    last_user_message: str = ""
    last_start_index: int = -1
    seeded: bool = False

    def as_chain(self) -> Chain:
        """Return the canonical `Chain` this chain currently is.

        Returns:
            A new `Chain`, so that a caller cannot grow the store's chain by
            mutating the list it got.

        Raises:
            SchemaValidationError: If the chain does not satisfy the canonical
                schema.
        """
        return Chain(
            id=self.id,
            label=self.label,
            chunk_ids=list(self.chunk_ids),
            compacted=self.compacted,
            compacted_text=self.compacted_text,
        )

    def as_candidate(self) -> ChainCandidate:
        """Return the candidate an adapter is shown for this chain."""
        return ChainCandidate(
            id=self.id, label=self.label, last_user_message=self.last_user_message
        )


class ChainStore:
    """The open chains of one transcript, ordered by when they were opened.

    Args:
        session: The transcript being classified. The store reads through the
            chunks' ranges over `session.messages` and copies no message.
        chains: Chains carried over from an earlier run — what T8 (#33) parses
            out of an existing summary block, so that they can be extended
            instead of being summarized a second time. They keep their ids,
            their labels and their summary text, and new chains are numbered
            past the highest of them. Each one's label is also what its
            candidate offers as its last user message, until a chunk carrying a
            user turn is appended to it. They have to arrive in the order they
            were opened, which is the order the summary block holds them in;
            the numbers need not be contiguous.

    Raises:
        BlankChainLabelError: If a seeded chain's label holds no word.
        ChainAlreadySeededError: If two seeded chains share an id.
        ChainOrderError: If their ids do not increase.
        ChunkAlreadyAssignedError: If two seeded chains claim the same chunk.
    """

    def __init__(self, session: Session, chains: Iterable[Chain] = ()) -> None:
        """Build the store, seeded with the chains of an earlier run."""
        self._session = session
        self._messages: dict[tuple[str, int, int], str | None] = {}
        self._chains: dict[str, _OpenChain] = {}
        self._order: list[str] = []
        self._owner: dict[str, str] = {}
        for chain in chains:
            self._seed(chain)

    def __repr__(self) -> str:
        """Return a representation naming the session and the chains held."""
        return f"{type(self).__name__}(session={self._session.id!r}, chains={self._order})"

    def __len__(self) -> int:
        """Return how many chains the store holds."""
        return len(self._order)

    def __contains__(self, chain_identifier: object) -> bool:
        """Return whether the store holds a chain of that id."""
        return isinstance(chain_identifier, str) and chain_identifier in self._chains

    def subject(self, chunk: Chunk) -> ChunkUnderDecision:
        """Return `chunk` as an adapter gets to see it.

        The other half of what `candidates` offers: a chunk carries indices and
        no text, so the chunk's own last user turn is resolved here — once per
        chunk, because opening a chain for it needs the same string for the
        provisional label.

        Args:
            chunk: The chunk about to be classified.

        Returns:
            The chunk's id together with its last real user turn, empty when it
            holds none.

        Raises:
            ForeignChunkError: If the chunk is not a range over this session.
        """
        return ChunkUnderDecision.from_chunk(chunk, self._message_of(chunk))

    def candidates(self) -> list[ChainCandidate]:
        """Return every open chain as a candidate, in the order they opened.

        Returns:
            A new list of candidates — every chain is still a candidate until
            the transcript ends, and the list is the store's own copy, so an
            adapter cannot change the store by changing what it was handed.
        """
        return [self._chains[identifier].as_candidate() for identifier in self._order]

    def chains(self) -> list[Chain]:
        """Return the chains as canonical objects, in the order they opened.

        Returns:
            A new list of new `Chain` objects, validated by the schema.
        """
        return [self._chains[identifier].as_chain() for identifier in self._order]

    def chain_of(self, chunk_id: str) -> str | None:
        """Return the id of the chain a chunk was assigned to, or `None`.

        Args:
            chunk_id: `Chunk.id` to look up.

        Returns:
            The chain's id, or `None` when the chunk is not assigned.
        """
        return self._owner.get(chunk_id)

    def open_chain(self, chunk: Chunk, label: str | None = None) -> Chain:
        """Open a new chain for `chunk` and return it.

        Args:
            chunk: The chunk that starts the chain.
            label: The chain's label. Omitted, it is derived from the chunk's
                user message (see `provisional_label`).

        Returns:
            The new chain, holding that one chunk.

        Raises:
            ValueError: If `label` is given and holds no word — empty or
                whitespace only, which `provisional_label` reads as no text.
            ForeignChunkError: If the chunk is not a range over this session.
            ChunkAlreadyAssignedError: If the chunk is already in a chain.
        """
        if label is not None and not label.strip():
            raise ValueError("a chain's label must hold a word; omit it for a provisional one")
        message = self._message_of(chunk)
        position = self._next_position()
        chain = _OpenChain(
            id=chain_id(position),
            label=label if label is not None else provisional_label(message or "", position),
        )
        self._claim(chain, chunk, message)
        self._chains[chain.id] = chain
        self._order.append(chain.id)
        return chain.as_chain()

    def append(self, chain_id: str, chunk: Chunk) -> Chain:
        """Append `chunk` to an open chain and return the extended chain.

        Args:
            chain_id: Id of the chain to extend, `C<n>`.
            chunk: The chunk that continues it.

        Returns:
            The chain, now holding that chunk as its last one.

        Raises:
            UnknownChainError: If the store holds no chain of that id.
            ForeignChunkError: If the chunk is not a range over this session.
            ChunkAlreadyAssignedError: If the chunk is already in a chain.
            ChunkOrderError: If the chunk precedes the chain's last chunk.
        """
        chain = self._chains.get(chain_id)
        if chain is None:
            raise UnknownChainError(chain_id, self._order)
        self._claim(chain, chunk, self._message_of(chunk))
        return chain.as_chain()

    def _claim(self, chain: _OpenChain, chunk: Chunk, message: str | None) -> None:
        """Add `chunk` to `chain`, refusing anything that would break a chain.

        Args:
            chain: The chain that takes the chunk.
            chunk: The chunk to add.
            message: The chunk's user message, already resolved by the caller
                (which has to resolve it anyway to label a new chain).

        Raises:
            ChunkAlreadyAssignedError: If the chunk is already in a chain.
            ChunkOrderError: If the chunk precedes the chain's last chunk.
        """
        owner = self._owner.get(chunk.id)
        if owner is not None:
            raise ChunkAlreadyAssignedError(
                chunk.id, owner, carried_over=self._chains[owner].seeded
            )
        if chunk.start_index < chain.last_start_index:
            raise ChunkOrderError(chunk.id, chain.id, chunk.start_index, chain.last_start_index)
        chain.chunk_ids.append(chunk.id)
        chain.last_start_index = chunk.start_index
        if message is not None:
            chain.last_user_message = message
        self._owner[chunk.id] = chain.id

    def _message_of(self, chunk: Chunk) -> str | None:
        """Return the chunk's last user message, resolving it at most once.

        The same string answers two questions — what the adapter is shown and
        what a new chain is labelled with — and the transcript cannot change
        under the store, so it is read once and kept.

        Args:
            chunk: The chunk to resolve.

        Returns:
            The content of its last real user turn, or `None` when it has none.

        Raises:
            ForeignChunkError: If the chunk is not a range over this session.
        """
        key = (chunk.id, chunk.start_index, chunk.end_index)
        if key not in self._messages:
            self._messages[key] = chunk_user_message(self._session, chunk)
        return self._messages[key]

    def _next_position(self) -> int:
        """Return the position of the next chain to open.

        Returns:
            One past the highest chain number the store holds, so the ids of the
            chains already there never move.
        """
        return max((chain_position(identifier) for identifier in self._order), default=0) + 1

    def _seed(self, chain: Chain) -> None:
        """Take over a chain from an earlier run.

        A seeded chain is the one thing the store is *handed* rather than
        computing, and it is handed data a model never saw and a programmer
        never wrote: #33 parses it out of a compacted block in a transcript
        file. Every refusal here is therefore a `ClassifyError` like the rest of
        the stage, so that a hand-edited block reaches T10's CLI as a message
        and not as a traceback.

        Args:
            chain: The chain as the parser read it back.

        Raises:
            BlankChainLabelError: If the chain's label holds no word. The
                schema accepts any non-empty string, so `"   "` arrives here
                and would otherwise be what an adapter is shown.
            ChainAlreadySeededError: If another seeded chain has the same id.
            ChainOrderError: If this chain is numbered before the one seeded
                ahead of it.
            ChunkAlreadyAssignedError: If another seeded chain already claims
                one of its chunks — the same refusal, and the same cause, as a
                chunk of *this* run running into a carried-over chunk id.
        """
        if not chain.label.strip():
            raise BlankChainLabelError(chain.id, chain.label)
        if chain.id in self._chains:
            raise ChainAlreadySeededError(chain.id)
        if self._order and chain_position(chain.id) < chain_position(self._order[-1]):
            raise ChainOrderError(chain.id, self._order[-1])
        for chunk_id in chain.chunk_ids:
            owner = self._owner.get(chunk_id)
            if owner is not None:
                raise ChunkAlreadyAssignedError(chunk_id, owner, carried_over=True)
            self._owner[chunk_id] = chain.id
        # The chunk ids of a carried-over chain are positions in the earlier
        # run's transcript, so neither a user turn nor the position of its last
        # chunk can be *read* here; `-1` leaves the order check open for the
        # first chunk this run appends, and the chain's label stands in for the
        # turn it does not have (#35). The label is not a guess: the compactor
        # wrote it into the block T8 parses, and `_claim` replaces it with the
        # first appended chunk that carries a user turn — a chunk of subagent
        # work carries none, and the label stays.
        self._chains[chain.id] = _OpenChain(
            id=chain.id,
            label=chain.label,
            chunk_ids=list(chain.chunk_ids),
            compacted=chain.compacted,
            compacted_text=chain.compacted_text,
            last_user_message=chain.label,
            seeded=True,
        )
        self._order.append(chain.id)
