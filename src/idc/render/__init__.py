"""Rendering of the compacted transcript in a target agent's native form.

The output is one summary block in the agent's own tags plus the latest chunk
verbatim, written back as a transcript file the agent can resume from. The
counterpart that reads such a file back in lives in `idc.importers`.

The renderer and the block parser are added in tickets T7 and T8; this package
carries no implementations yet.
"""
