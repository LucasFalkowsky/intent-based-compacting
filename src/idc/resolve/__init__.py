"""Resolution of the accepted solution out of the path that led to it.

Everything resolvable from the transcript is computed by code, not by a model:
the last write per file, the last passing verification command, and which reads,
failed attempts and overwritten edits belong to the path rather than to the
solution.

The resolver is added in ticket T5; this package carries no implementations yet.
"""
