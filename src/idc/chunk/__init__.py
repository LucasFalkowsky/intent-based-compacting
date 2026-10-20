"""Chunking of a canonical transcript at user-message boundaries.

A chunk starts at a user message and runs up to (but not including) the next
one, so that every chunk carries exactly one user turn plus the agent work it
triggered. System noise is reduced here and subagent transcripts become chunks
of their own.

The chunker is added in ticket T3; this package carries no implementations yet.
"""
