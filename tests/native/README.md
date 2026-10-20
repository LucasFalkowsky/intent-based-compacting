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

## `opencode/`

OpenCode stores a session as one record per **message** (`session/message/<id>.json`) with
that message's **parts** beside it (`session/part/<message id>/<part id>.json`); a fixture
here is that stream, one message record per line, with its parts nested under `parts` —
which is also the shape `idc.importers.jsonl` reads for Claude Code. A record names a
`role` (`user`, `assistant`) or, for the summary OpenCode's own compaction writes, a
`type` (`compaction`). Instants are **epoch milliseconds** under `time.created`, not
ISO-8601 strings, and a tool call carries its own result in the same part
(`state.output` / `state.error`).

Unlike the Claude Code fixtures, these record-level keys were **not** modelled on a
recorded session: OpenCode is not installed in this environment, so they come from its
documented storage model (see the known limitations in `src/idc/importers/opencode.py`),
and the same holds for the six tool names T2.3 added to the `opencode` table of
`tool_registry.toml`. Corroborated from inside the repository are only the lower-case
tool spellings `read` / `edit` / `bash` with their argument keys, the JSONL input shape
and the `compaction` record. **Before the study records its OpenCode condition, one real
session has to be checked against `golden_session.jsonl` below** — if a key is spelled
differently, the fix touches only that importer and this directory.

- `golden_session.jsonl` — the full-coverage session: a user prompt with an attached
  `file` part, an assistant turn with `step-start`, `reasoning`, `text` and `tool` parts,
  a `completed` tool state, an `error` one, two tool parts in **one** record, a
  `synthetic` user record (the harness wrote it, not the user), and an assistant record
  that carries a tool call and no text. `golden_session.expected.json` is its canonical
  form.
- `truncated_front.jsonl` — a transcript cut off at the front: it begins mid-turn with an
  assistant record whose tool part already carries its result, and no user turn is
  invented in front of it. Its instants carry milliseconds. (OpenCode cannot show a
  *result without its call* the way Claude Code can: the call and its result are the same
  part, so front truncation drops both together.)
- `interrupted_turn.jsonl` — a tool part still in `running` state, whose result therefore
  never arrives, followed by the user's next turn.
- `unknown_tool.jsonl` — one declared delegating call (`task`) next to a tool the registry
  does not declare (`imaginary_tool`, an invented name: an undeclared tool has to resolve
  to `other` and be logged).
- `native_compaction.jsonl` — the native compaction flow: a user turn, the `compaction`
  record carrying the `summary` and the `recent` messages it kept, and the turn that
  follows it.
## `gemini-cli/`

Gemini CLI appends one JSON object per line to
`~/.gemini/tmp/<project_hash>/chats/session-<timestamp>-<short id>.jsonl`: one metadata
record, then one record per message, plus the two log operations `$set` (a metadata update,
which may replace the whole message list) and `$rewindTo`. The records below were modelled
on that writer — `packages/core/src/services/chatRecordingService.ts` of the installed
Gemini CLI 0.46.0 — and on its tool reference (`docs/reference/tools.md`), which is where
the argument spellings come from.

- `golden_session.jsonl` — the full-coverage session: the metadata record, the injected
  `<session_context>` turn, a typed prompt, a `gemini` record appended **twice** (once
  while its call is still running, once with the result — the later record is the
  message), a record with two calls and two results, a `$set` metadata update, an `info`
  record, a model turn in its raw-history shape (a `thought` part, text, and the call as a
  `functionCall` part), a `user` record that only reports a `functionResponse`, a pasted
  image, and a closing model turn. `golden_session.expected.json` is its canonical form.
- `truncated_front.jsonl` — a transcript cut off at the front: the first record reports a
  `functionResponse` whose call is not in the file. The canonical schema deliberately does
  not require referential completeness, so this must import as a valid session.
- `interrupted_turn.jsonl` — a call whose result never arrives: the file ends with the
  record that still has `result: null`.
- `unknown_tool.jsonl` — one declared subagent call (`invoke_agent`) next to a tool the
  registry does not declare.
- `native_compaction.jsonl` — the native compaction flow: the `<state_snapshot>` the
  compression writes back as a user turn, its acknowledgement, and the `$set.messages`
  record that replaces the history with what was kept.
- `rewound_turn.jsonl` — `$rewindTo`, which drops the message it names and everything
  recorded after it.
