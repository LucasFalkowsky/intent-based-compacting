"""Compaction of new or extended intent chains into Markdown blocks.

Each chain is compacted with one fixed instruction into a block with a stable
chain ID, so that a block written by an earlier run can be recognized, parsed
back and extended instead of being summarized again. Model calls are confined to
the narrative sections of the template.

```python
from idc.compact import compact_chain, compact_incrementally, render_block

print(compact_chain(chain, resolve(session, chunks, chain), client))  # one call
print(render_block(chain, resolve(session, chunks, chain)))           # no call
print(compact_incrementally(chains, resolution_of=..., client=client))  # one run
```

`idc.compact.template` holds the template itself and the two sections a
transcript determines — `Solution` and `Verification`; `render_block` writes
those and leaves the narrative sections as the empty-section placeholder, which
is the whole block for a caller that has no model.
`idc.compact.incremental` is the stage above the two: handed a whole run's
chains, it writes a block for every chain this run opened or gave a new chunk
to, and copies the block of every other chain byte for byte — so a chain that
was summarized once is paid for once. An extended chain's `Solution` section is
the merge of its earlier block's and this run's writes — `idc.compact.merge`,
last write winning per file, so a rewrite does not drop the files the chain
already had. `idc.compact.narrative` fills the
narrative sections and the `Intent` label from
a model: the client is injected, the response is validated against the section
headers, and the instruction is read from the versioned data file
`idc.compact.instruction` loads — whose digest identifies the prompt a run used.
"""

from idc.compact.errors import (
    CarriedChainError,
    CompactError,
    InstructionError,
    NarrativeContractError,
    NarrativeScriptExhaustedError,
)
from idc.compact.fake import FakeNarrativeCall, FakeNarrativeClient
from idc.compact.incremental import (
    compact_incrementally,
    incremental_blocks,
    is_carried_over,
    transcript_chain,
    transcript_chunk_ids,
)
from idc.compact.instruction import (
    INSTRUCTION_PATH,
    INSTRUCTION_VERSION,
    Instruction,
    instruction,
    instruction_digest,
    load_instruction,
)
from idc.compact.merge import merge_solution_bodies, merged_solution_section
from idc.compact.narrative import (
    INTENT_MARKER,
    Narrative,
    NarrativeClient,
    NarrativeRequest,
    as_narrative,
    compact_chain,
    decisions_section,
    rejected_section,
    render_narrative_block,
)
from idc.compact.template import (
    DECISIONS_HEADER,
    EMPTY_SECTION,
    ERROR_NOTE,
    NO_COMMAND_NOTE,
    NO_CONTENT_NOTE,
    NO_RESULT_NOTE,
    REJECTED_HEADER,
    SECTION_HEADERS,
    SOLUTION_HEADER,
    UNKNOWN_EXIT_NOTE,
    VERIFICATION_HEADER,
    chain_heading,
    render_block,
    solution_section,
    verification_section,
)

# T6.1 (#26) opened this surface; the sibling sub-tickets of T6 append a block
# of their own at the end of this file rather than editing the literal below —
# the convention `idc.resolve.__init__` states at its own `__all__`, where two
# sub-tickets of one parent had already collided in a package `__init__.py`.
__all__ = [
    "DECISIONS_HEADER",
    "EMPTY_SECTION",
    "ERROR_NOTE",
    "NO_COMMAND_NOTE",
    "NO_CONTENT_NOTE",
    "NO_RESULT_NOTE",
    "REJECTED_HEADER",
    "SECTION_HEADERS",
    "SOLUTION_HEADER",
    "UNKNOWN_EXIT_NOTE",
    "VERIFICATION_HEADER",
    "chain_heading",
    "render_block",
    "solution_section",
    "verification_section",
]

# T6.2 (#27) — the two narrative sections, the instruction that produces them
# and its digest, appended as a block of its own for the reason the comment
# above gives. The imports stay in the one sorted block at the top, which is
# how `idc.resolve.__init__` carries its four tickets' worth of them.
#
# `instruction_digest` is exposed for T9.3's run manifest (#39), which is its
# one consumer: `idc.measure.manifest.build_manifest` records it together with
# `INSTRUCTION_VERSION`, so a run's numbers are attributable to the exact
# prompt that produced them.
__all__ += [
    "INSTRUCTION_PATH",
    "INSTRUCTION_VERSION",
    "INTENT_MARKER",
    "CompactError",
    "FakeNarrativeCall",
    "FakeNarrativeClient",
    "Instruction",
    "InstructionError",
    "Narrative",
    "NarrativeClient",
    "NarrativeContractError",
    "NarrativeRequest",
    "NarrativeScriptExhaustedError",
    "as_narrative",
    "compact_chain",
    "decisions_section",
    "instruction",
    "instruction_digest",
    "load_instruction",
    "rejected_section",
    "render_narrative_block",
]

# T6.3 (#28) — the stage that writes a whole run: a model call for a chain this
# run opened or extended, and the earlier run's own bytes for every other one.
# Appended as a block of its own for the reason the comment above `__all__`
# gives.
__all__ += [
    "compact_incrementally",
    "incremental_blocks",
    "is_carried_over",
    "transcript_chain",
    "transcript_chunk_ids",
]

# T6.3 (#28) — the refusal `transcript_chain` raises, appended as a block of its
# own for the reason the comment above gives. It was a bare `ValueError` until a
# review showed it escaping the stage's public entry point, where a caller could
# not tell it from a built-in failure.
__all__ += ["CarriedChainError"]

# T6.4 (#29) — the merge an extended chain's rewrite goes through: the earlier
# block's `Solution` section and this run's writes, last write winning per file.
# Appended as a block of its own for the reason the comment above `__all__`
# gives.
__all__ += [
    "merge_solution_bodies",
    "merged_solution_section",
]
