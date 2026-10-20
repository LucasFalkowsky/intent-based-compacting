"""Compaction of new or extended intent chains into Markdown blocks.

Each chain is compacted with one fixed instruction into a block with a stable
chain ID, so that a block written by an earlier run can be recognized, parsed
back and extended instead of being summarized again. Model calls are confined to
the narrative sections of the template.

The compactor is added in ticket T6; this package carries no implementations
yet.
"""
