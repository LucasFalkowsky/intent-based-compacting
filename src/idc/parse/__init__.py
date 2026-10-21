"""Reading an already-compacted block back into the chains it summarizes.

This is the stage that makes the method incremental. A run writes one Markdown
block per chain into the transcript; the next run finds those blocks, reads them
back here, and offers the chains they describe to the classifier as candidates —
so a chain that was summarized once is *extended* rather than summarized again,
and the tokens spent on it are spent once.

```python
from idc.parse import parse_chains

for chain in parse_chains(markdown):
    chain.id, chain.label        # read off the `## [C<n>] Intent: …` heading
    chain.compacted              # always true: a parsed chain is a summarized one
    chain.compacted_text         # its block, byte for byte as it was read
```

The format is **the writer's**, not this package's: `idc.compact` owns the
heading, the four section headers and the empty-section placeholder, and
`idc.parse.blocks` imports them instead of spelling them a second time. That is
what makes the round trip a property of the two packages together — a block
`idc.compact.render_block` wrote, parsed here and written back out, is the same
bytes — and it is why there is no second definition of
`### Decisions & constraints` anywhere in here to drift from the first.

Two things this stage does **not** do:

- **It re-renders nothing.** A chain keeps its block verbatim in
  `compacted_text`, so a section the writer never emitted — one a hand added, or
  one a later version of the writer will add — survives untouched without this
  package knowing what it is.
- **It guesses nothing.** No model is asked, no file is opened and no clock is
  read: the only input is the text a caller hands over, so the same block always
  parses to the same chains. Everything the text cannot answer is refused in
  `idc.parse.errors`, with the line it went wrong on, rather than filled in —
  except the one thing a block structurally cannot state, which is the chunk
  membership of its chain: `Chain.chunk_ids` needs at least one id, so a parsed
  chain names `placeholder_chunk_id(chain.id)` and nothing else. That id is
  deliberately not a `K<n>`, so no later stage can take it for a chunk of the
  transcript being read; a parsed chain is carried by its summary, not re-walked
  through chunks it no longer has.
"""

from idc.parse.blocks import (
    BLOCK_SEPARATOR,
    HEADING_PATTERN,
    PLACEHOLDER_CHUNK_PREFIX,
    ParsedBlock,
    ParsedSection,
    join_blocks,
    parse_blocks,
    parse_chains,
    placeholder_chunk_id,
    split_blocks,
)
from idc.parse.errors import (
    HEADING_SHAPE,
    DuplicateChainError,
    MalformedHeadingError,
    MissingSectionError,
    NoHeadingError,
    ParseError,
    StrayContentError,
    UnterminatedFenceError,
)

# T8.1 (#34) opened this surface. The sibling sub-ticket of T8 appends a block
# of its own at the end of this file with `__all__ += [...]` rather than editing
# the sorted literal below — the convention `idc.resolve.__init__` states at its
# own `__all__`, where two sub-tickets of one parent had already collided in a
# package `__init__.py`.
__all__ = [
    "BLOCK_SEPARATOR",
    "HEADING_PATTERN",
    "HEADING_SHAPE",
    "PLACEHOLDER_CHUNK_PREFIX",
    "DuplicateChainError",
    "MalformedHeadingError",
    "MissingSectionError",
    "NoHeadingError",
    "ParseError",
    "ParsedBlock",
    "ParsedSection",
    "StrayContentError",
    "UnterminatedFenceError",
    "join_blocks",
    "parse_blocks",
    "parse_chains",
    "placeholder_chunk_id",
    "split_blocks",
]
