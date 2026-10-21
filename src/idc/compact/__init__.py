"""Compaction of new or extended intent chains into Markdown blocks.

Each chain is compacted with one fixed instruction into a block with a stable
chain ID, so that a block written by an earlier run can be recognized, parsed
back and extended instead of being summarized again. Model calls are confined to
the narrative sections of the template.

```python
from idc.compact import render_block

print(render_block(chain, resolve(session, chunks, chain)))
```

`idc.compact.template` holds the template itself and the two sections a
transcript determines — `Solution` and `Verification`; the narrative sections
are rendered as the empty-section placeholder until T6.2 fills them.
"""

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
