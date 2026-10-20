"""Importers that read agent transcripts into the canonical format.

One importer per supported agent (Claude Code, Gemini CLI, OpenCode). An
importer is responsible for the agent's on-disk shape — JSON or JSONL, tool-call
encoding, role names — and for nothing else: everything downstream sees only
canonical objects.

The importers are added in ticket T2; this package carries no implementations
yet.
"""
