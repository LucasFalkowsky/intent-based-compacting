"""Rendering of the compacted transcript in a target agent's native form.

The output is one summary block in the agent's own tags plus the latest chunk
verbatim, written back as a transcript file the agent can resume from. The
counterpart that reads such a file back in lives in `idc.importers`.

T7.1 adds the assembly: which parts the output holds and in which order. The
native tags, the role the summary is written under and the metadata beside it
are T7.2's, so nothing here branches on `Agent` yet.
"""

from idc.render.assembly import AssembledTranscript, assemble, latest_messages, summary_text
from idc.render.errors import RenderError, UncompactedChainError, UnknownLatestChunkError

# T7.1 (#31) — the assembly of the two parts and its two refusals. The sibling
# sub-ticket T7.2 appends a block of its own at the end of this file with
# `__all__ += [...]` rather than editing the sorted literal below — the
# convention `idc.resolve.__init__` states at its own `__all__`, where two
# sub-tickets of one parent had already collided in a package `__init__.py`.
__all__ = [
    "AssembledTranscript",
    "RenderError",
    "UncompactedChainError",
    "UnknownLatestChunkError",
    "assemble",
    "latest_messages",
    "summary_text",
]
