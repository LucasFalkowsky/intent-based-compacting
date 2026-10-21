# Agent-native transcript fixtures

Hand-written transcripts **in an agent's own on-disk format** — the input side of the
importers (T2). They are deliberately *not* part of `tests/fixtures/`, which is the shared
**canonical** corpus (one `Session` document per file) that every later stage is tested
against and whose content is fixed.

One directory per agent, named after the agent's wire string, so the directory name is the
`Agent` value an importer is asked for:

```
tests/native/claude-code/     # T2.2 — Claude Code session JSONL
```

Load a transcript **by agent and name**, never by path (see `tests/native_transcripts.py`):

```python
from native_transcripts import NATIVE

path = NATIVE.path("claude-code", "golden_session")  # the .jsonl
expected = NATIVE.expected_text("claude-code", "golden_session")  # the canonical golden
```

## Conventions

- The **keys** of every record are the ones real transcripts of that agent carry; the
  **values** are invented, short and readable. A key that no real transcript shows would
  pin behaviour that was never observed, so it does not appear here. The Claude Code
  records below were modelled on recorded Claude Code sessions.
- Record identifiers are readable (`u01`, `a01`, `t01`, `toolu_01`) rather than UUIDs, so
  that a canonical golden file shows at a glance that the importer **takes** the
  transcript's identifiers instead of renumbering them.
- A fixture named `<name>.jsonl` may be accompanied by `<name>.expected.json`: the exact
  bytes `idc.schema.dump` writes for the canonical session it imports to — the golden file.
  Only the one full-coverage fixture per agent carries one; the edge-case fixtures are
  asserted on the single property they exist for.
- No fixture is larger than it has to be, and none of them is a recording of a real
  session: a real transcript carries a user's source code and prompts, which has no place
  in this repository.

## `claude-code/`

- `golden_session.jsonl` — the full-coverage session: bookkeeping records that carry no
  message (`mode`, `permission-mode`, `file-history-snapshot`, `ai-title`), a typed user
  prompt, an assistant turn with a `thinking`, a `text` and a `tool_use` block, tool
  results as a string and as a list of content blocks, two results in **one** record, a
  failed result (`is_error`), an `isMeta` user record, a `system` record without content
  and an `attachment` record. `golden_session.expected.json` is its canonical form.
- `truncated_front.jsonl` — a transcript cut off at the front: the first record is a
  `tool_result` whose `tool_use` is not in the file, and the only timestamps carry
  milliseconds. The canonical schema deliberately does not require referential
  completeness, so this must import as a valid session.
- `interrupted_turn.jsonl` — an assistant `tool_use` whose result never arrives, followed
  by the interruption notice the harness records as a user message.
- `unknown_tool.jsonl` — one declared subagent call (`Agent`) next to two tools the
  registry does not declare: an invented one and an MCP tool, whose `mcp__<server>__<tool>`
  shape is observed but whose server is a placeholder.
- `native_compaction.jsonl` — the native compaction flow: the `/compact` slash command,
  the summary record Claude Code writes with `isCompactSummary`, and the turn that follows
  it.
