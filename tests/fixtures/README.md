# Shared transcript fixture corpus

Nine hand-written **canonical** transcripts — one `Session` document per file, not an
agent-native transcript (importing those is T2). Every later ticket's tests are written
against this corpus instead of against transcripts of their own, so a fixture's structure
is a contract: `tests/test_fixture_cases.py` pins the property each one exists for, and
`tests/test_fixture_corpus.py` validates every `*.json` file in this directory against the
canonical schema.

Load a fixture **by name**, never by path (see `tests/fixture_corpus.py`):

```python
from fixture_corpus import CORPUS

session = CORPUS.session("interleaved_intents")  # validated on load
```

## The fixtures

- `single_intent.json` — one user intent carried to a green test run: two reads, one
  accepted edit, one passing verification command. The minimal chain of one chunk.
- `two_sequential_intents.json` — two user turns pursuing unrelated intents, over
  **disjoint** file sets (`src/billing/*` then `README.md`), so a classifier must open a
  second chain rather than extend the first.
- `interleaved_intents.json` — intents in the order **A, B, A**: a flaky-test fix, an
  unrelated config-key rename, then a return to the first intent that re-edits
  `src/rates/sampler.py`. The first and third turn share files, the middle one shares none
  with either — the discriminating case for chain assignment and the end-to-end test.
- `slash_command.json` — two slash commands (`is_command`) around a single real user turn:
  one opens the session before any turn, one sits inside the turn it belongs to. Three
  user-role messages, one user boundary.
- `system_injected_message.json` — three harness-injected messages (`is_system_injected`),
  one of them in the **user** role in the middle of the agent's work: the subtle case the
  chunker must not treat as a new turn.
- `subagent_result.json` — a `Task` call whose subagent trace is the span of messages
  between the call and its result, with the subagent's own calls resolving inside that
  span, followed by the main agent acting on the report.
- `failed_then_passing_command.json` — the same verification command run twice, first with
  exit code 1 and `is_error`, then passing: the last passing verification is resolvable and
  the failed run is path, not solution.
- `write_then_two_edits.json` — one file created and then edited twice (`write_file`,
  `replace`, `replace`) plus a second file written once, so the last write **per file** is
  resolvable and the two superseded writes are distinguishable from it.
- `already_compacted_block.json` — a session with `source = intent-compaction` whose first
  message holds a rendered summary block for the chains `C1` and `C2`, followed by one new
  turn that rewrites a file of `C2`'s solution and adds another: the re-entry input for the
  parser, the candidate feed and the solution merge.

## Conventions the corpus follows

- Each file is byte-identical to what `idc.schema.dump` writes (indent 2, trailing
  newline); a test enforces it, so the corpus stays readable in diffs.
- `Session.id` is `sess-<file name with hyphens>`, message ids are `m01…`, tool-call ids
  `c01…`; every call has exactly one later result and every message is timestamped in
  transcript order.
- All three agents appear (`claude-code`, `gemini-cli`, `opencode`) with their own tool
  names and their own argument spellings — `file_path`, `filePath`, `absolute_path` — because
  the schema keeps a call's arguments exactly as the agent sent them. Two fixtures are
  recorded in a `+02:00` offset rather than UTC.
- Tool-result outputs are realistic but short (file excerpts with line numbers, pytest
  output, edit confirmations). `exit_code` is set for `exec` calls only.

## Adding a fixture

Drop a `<name>.json` canonical `Session` into this directory and add one `- \`<name>.json\` —
…` line here. Discovery is automatic: `CORPUS.names()` and the corpus-wide tests pick the
file up without any list to update. If the fixture exists for a specific structural
property, pin that property in `tests/test_fixture_cases.py`.
