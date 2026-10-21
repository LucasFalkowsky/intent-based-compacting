"""Compacting one run's chains: a model call for what changed, a copy for the rest.

This is what makes the method cheap on the second run. A chain that an earlier
run already summarized and that this run did not touch is **copied**: its block
goes back out byte for byte, and no model is asked about it. Only a chain this
run gave a new chunk to, and a chain this run opened, cost a call.

```python
from idc.compact import compact_incrementally

text = compact_incrementally(
    chains,                                                   # what T4 classified
    resolution_of=lambda chain: resolve(session, chunks, chain),
    client=client,
)
```

**Which chains have new chunks, and how it is decided.** A chain that came out
of an earlier run's block carries `compacted_text` but *not* its chunk
membership: a block states none, and `Chain.chunk_ids` needs at least one id, so
`idc.parse` puts a placeholder there and `idc.classify` seeds the chain with it.
Comparing a seeded chain's `chunk_ids` with a freshly classified chain's
therefore answers nothing — the two lists are not the same kind of thing. What
*is* decidable from a single chain, and is what this module decides on, is
whether the chain holds a chunk id **of the transcript being compacted**: the
chunker names its chunks `K<n>` (`idc.chunk.CHUNK_ID_PREFIX`), and every other
id a chain carries was put there in place of a membership this run cannot see.
So:

- `compacted` and no `K<n>` → the earlier run's chain, untouched: **copied**.
- `compacted` and at least one `K<n>` → extended: **compacted again**, one call.
- not `compacted` → opened by this run: **compacted**, one call.

Reading the ids from the chunker's side rather than by the placeholder's prefix
keeps this package independent of `idc.parse`, which imports *its* vocabulary
(`SECTION_HEADERS`) from here; the two spellings of the line agree on every id
this tool produces, because a chunk id comes from `idc.chunk.chunk_id` and a
carried one from `idc.parse.placeholder_chunk_id`. `idc.classify.gold` reads a
chunk id the same way.

**A copy is a copy.** The block that goes out is the one that came in, not a
re-render that agrees with it: `compacted_text` is returned unchanged, so a
section an earlier version of the writer emitted — or a hand added — survives,
and a chain that cost nothing is visibly a chain that cost nothing in the
client's call record.

**What this module does not know.** It does not resolve and it does not read the
session: `resolution_of` and `user_messages_of` are injected, and are asked only
about the chains that are actually compacted — which is also why a carried-over
chain may not be resolvable at all. They are handed `transcript_chain(chain)`,
the chain reduced to the chunks this transcript holds, so that
`idc.resolve.resolve` can be given it directly: the placeholder id would
otherwise raise `UnknownChunkError`. Nothing here re-derives the chunk
membership a block does not state.

The one import this module defers is `idc.parse.join_blocks`, inside
`compact_incrementally`. The dependency between the two packages runs the other
way — the reader imports the writer's format — so importing the reader here at
module level is a cycle (`idc.parse.blocks` does `from idc.compact import
SECTION_HEADERS` and would find a half-built package). Deferring it to the call
keeps the one definition of how two blocks are separated, instead of spelling
the blank line a second time.
"""

from collections.abc import Callable, Sequence

from idc.chunk import CHUNK_ID_PREFIX
from idc.compact.errors import CarriedChainError
from idc.compact.narrative import NarrativeClient, compact_chain
from idc.resolve import Resolution
from idc.schema import Chain

__all__ = [
    "compact_incrementally",
    "incremental_blocks",
    "is_carried_over",
    "transcript_chain",
    "transcript_chunk_ids",
]


def transcript_chunk_ids(chain: Chain) -> tuple[str, ...]:
    """Return the chain's chunk ids that name chunks of the transcript at hand.

    Args:
        chain: The chain to look at, as `idc.classify` returned it.

    Returns:
        Its `K<n>` ids, in the order the chain lists them. Empty for a chain
        carried over from an earlier run and not extended by this one: the ids
        such a chain carries stand in for a membership its block does not
        state.

    The test is `startswith` and deliberately **looser** than
    `idc.classify.gold`'s, which also requires the remainder to be digits. The
    schema admits any non-empty string as an `Identifier`, so the two can
    disagree — on `"K"`, `"Kompakt1"`, `"Knope"` — and the looser test is the
    one that fails safely. Counting such an id as a chunk makes a carried chain
    look *extended*, so it is rewritten and `idc.resolve` refuses loudly;
    the stricter test would make it look *carried*, so its block would be
    copied and a chunk the chain really gained would be **lost from the
    summary** with the suite green. No id this tool produces reaches either
    reading; the direction of the disagreement is what is chosen here.
    """
    return tuple(chunk_id for chunk_id in chain.chunk_ids if chunk_id.startswith(CHUNK_ID_PREFIX))


def is_carried_over(chain: Chain) -> bool:
    """Say whether the chain is an earlier run's, with nothing new in it.

    Args:
        chain: The chain to judge.

    Returns:
        True when the chain holds a summary from an earlier run and no chunk of
        the transcript at hand — the one case in which its block is copied
        rather than written again. False for a chain this run extended and for
        one this run opened.
    """
    return chain.compacted and not transcript_chunk_ids(chain)


def transcript_chain(chain: Chain) -> Chain:
    """Return the chain reduced to the chunks the transcript at hand holds.

    What `resolution_of` and `user_messages_of` are asked about: the same chain,
    minus the ids that name no chunk of this transcript, so that
    `idc.resolve.resolve` and a caller's own lookup can be handed it without
    tripping over a carried-over placeholder.

    Args:
        chain: The chain to reduce. Its summary, if it has one, is kept — the
            block it stands for is the whole chain's, not this run's part of it.

    Returns:
        A new `Chain` with the same id, label and summary, whose `chunk_ids` are
        `transcript_chunk_ids(chain)`.

    Raises:
        CarriedChainError: If the chain holds no chunk of this transcript.
            There is no such chain to resolve, and `Chain.chunk_ids` could not
            hold it: `is_carried_over` is the question to ask first. A
            `ValueError` too, so a caller that catches that still works.
    """
    chunk_ids = transcript_chunk_ids(chain)
    if not chunk_ids:
        raise CarriedChainError(
            f"chain {chain.id!r} holds no chunk of this transcript, so it has no part in it to "
            f"resolve; it is carried over from an earlier run and its block is copied"
        )
    return Chain(
        id=chain.id,
        label=chain.label,
        chunk_ids=list(chunk_ids),
        compacted=chain.compacted,
        compacted_text=chain.compacted_text,
    )


def _no_user_messages(chain: Chain) -> tuple[str, ...]:
    """Return no user messages, the default when a caller offers no lookup.

    Args:
        chain: The chain that would have been looked up; unused.

    Returns:
        The empty tuple — `idc.compact.NarrativeRequest`'s own default, so a
        caller that has not wired the transcript up offers a model nothing
        rather than something invented.
    """
    return ()


def incremental_blocks(
    chains: Sequence[Chain],
    *,
    resolution_of: Callable[[Chain], Resolution],
    client: NarrativeClient,
    user_messages_of: Callable[[Chain], Sequence[str]] = _no_user_messages,
) -> list[str]:
    """Return one block per chain: copied when nothing is new, written when it is.

    Args:
        chains: This run's chains, in the order they are to be written — which
            is the order `idc.classify` returns them in: the carried-over chains
            in the order their blocks had, then the chains this run opened. The
            order is the caller's and nothing here sorts.
        resolution_of: What a chain resolved to, asked only for the chains that
            are compacted and handed `transcript_chain(chain)`.
        client: The narrative backend, asked exactly once per compacted chain
            and not at all for a copied one.
        user_messages_of: The user messages to offer the client for a chain,
            asked under the same condition as `resolution_of`. Offers none by
            default.

    Returns:
        The blocks, in the order of `chains`. A copied block is the bytes of
        `Chain.compacted_text`; a written one is what
        `idc.compact.compact_chain` renders.

    Raises:
        NarrativeContractError: If the client's answer to one chain is not a
            narrative. The run fails on that chain rather than writing a block
            the answer only partly filled.
    """
    blocks: list[str] = []
    for chain in chains:
        if is_carried_over(chain):
            # The canonical schema ties `compacted_text` to `compacted`, so a
            # carried-over chain always carries its block.
            blocks.append(str(chain.compacted_text))
            continue
        current = transcript_chain(chain)
        blocks.append(
            compact_chain(
                chain,
                resolution_of(current),
                client,
                user_messages=user_messages_of(current),
            )
        )
    return blocks


def compact_incrementally(
    chains: Sequence[Chain],
    *,
    resolution_of: Callable[[Chain], Resolution],
    client: NarrativeClient,
    user_messages_of: Callable[[Chain], Sequence[str]] = _no_user_messages,
) -> str:
    """Compact a run's chains into one block sequence, writing only what changed.

    Args:
        chains: This run's chains, in the order they are to be written.
        resolution_of: What a chain resolved to; see `incremental_blocks`.
        client: The narrative backend, asked once per new or extended chain.
        user_messages_of: The user messages to offer the client for a chain.

    Returns:
        The blocks of `incremental_blocks`, one blank line apart — the sequence
        `idc.parse` reads back on the next run. A run in which no chain changed
        returns the bytes its blocks came in as, separator included, and the
        empty text for no chains at all.

    Raises:
        NarrativeContractError: If the client's answer to one chain is not a
            narrative.
    """
    # Deferred on purpose: `idc.parse` imports this package's format, so this
    # package cannot import it at module level — see the module docstring.
    from idc.parse import join_blocks

    return join_blocks(
        incremental_blocks(
            chains,
            resolution_of=resolution_of,
            client=client,
            user_messages_of=user_messages_of,
        )
    )
