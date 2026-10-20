"""Intent-driven context compaction for coding-agent transcripts.

`idc` is an offline tool that compacts a coding-agent transcript (Claude Code,
Gemini CLI, OpenCode) along the *user intents* it contains instead of
summarizing the whole history in one pass.

The package is split into one sub-package per pipeline stage, in the order the
stages run:

- `idc.importers` — read an agent transcript into the canonical format.
- `idc.chunk` — split the canonical transcript at user-message boundaries.
- `idc.classify` — assign each chunk to an intent chain.
- `idc.resolve` — derive the accepted solution from the path that led to it.
- `idc.compact` — turn new or extended chains into Markdown summary blocks.
- `idc.render` — emit the summary in the target agent's native tags.
- `idc.measure` — count tokens before and after compaction.

`idc.schema` holds the canonical schema that is the contract between those
stages, and `idc.cli` the command surface. The tool reads and writes transcript
files only: it never calls an agent and never touches git.
"""
