"""Compaction of new or extended intent chains into Markdown blocks.

Each chain is compacted with one fixed instruction into a block with a stable
chain ID, so that a block written by an earlier run can be recognized, parsed
back and extended instead of being summarized again. Model calls are confined to
the narrative sections of the template.

```python
from idc.compact import compact_chain, render_block

print(compact_chain(chain, resolve(session, chunks, chain), client))  # one call
print(render_block(chain, resolve(session, chunks, chain)))           # no call
```

`idc.compact.template` holds the template itself and the two sections a
transcript determines — `Solution` and `Verification`; `render_block` writes
those and leaves the narrative sections as the empty-section placeholder, which
is the whole block for a caller that has no model.
`idc.compact.narrative` fills the narrative sections and the `Intent` label from
a model: the client is injected, the response is validated against the section
headers, and the instruction is read from the versioned data file
`idc.compact.instruction` loads — whose digest identifies the prompt a run used.
"""

from idc.compact.errors import (
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
# one consumer and does not exist yet: nothing on this branch wires it up.
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
