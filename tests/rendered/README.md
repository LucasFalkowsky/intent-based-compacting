# Rendered transcript golden files (T7.2)

One file per agent, byte-compared by `tests/test_render_native.py`:

| File | What it is |
| --- | --- |
| `claude-code.jsonl` | A Claude Code session JSONL: the summary as a `user` record flagged `isCompactSummary`, then one record per kept message. |
| `gemini-cli.jsonl` | A Gemini CLI chat recording in its append-log shape: the metadata record, the summary as a `user` turn wrapped in `<state_snapshot>`, then one record per kept message. |
| `opencode.jsonl` | An OpenCode message stream: the summary as a `compaction` record, then one record per kept message, a tool result inside the tool part of its own call. |

All three are `idc.render.render_transcript`'s output for the **same** input —
the corpus fixture `interleaved_intents`, its golden chains from
`tests/fixtures/compact/`, and the last of its three chunks. Regenerate them
only when a writer's output is meant to change, and review the diff: these files
are the only place a changed field name, field order or line break shows up as a
failing test.

Why a directory of its own, next to `tests/fixtures/` and `tests/native/`:
those two hold **inputs** and both are discovered by globbing
(`fixture_corpus.Corpus`, `native_transcripts.NativeTranscripts`), so a rendered
*output* dropped into either would be picked up as a transcript some other
stage's test is expected to read.
