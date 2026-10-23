# End-to-end golden files (T10.2)

The two files one whole `idc run` leaves behind, byte-compared by
`tests/test_cli_end_to_end.py`:

| File | What it is |
| --- | --- |
| `interleaved_intents.jsonl` | The compacted Claude Code transcript the run wrote: the summary of both chains as a `user` record flagged `isCompactSummary`, then the third chunk verbatim. |
| `interleaved_intents.jsonl.manifest.json` | The run manifest, at the path `run` puts one when it is given no `--manifest`: `<--out>.manifest.json`. |

Both come out of **one** invocation —
`main(["run", "--agent", "claude-code", "--in", …, "--out", …])` — with the four
seams of `idc.cli.Runtime` injected as fakes: `FakeAdapter` playing
`idc.classify.gold_script` of the fixture's gold labelling, `FakeNarrativeClient`
with one answer per chain, `FakeTokenizer` behind every model family, and
`FakeClock` with two instants. The input is the shared corpus fixture
`interleaved_intents` written out in Claude Code's own record shape; the test's
`native_input` says how and why, and two of its tests pin that what the run
imports is that fixture.

Why a directory of its own, next to `tests/fixtures/`, `tests/native/` and
`tests/rendered/`: the first two hold **inputs** and are both discovered by
globbing, so an output dropped into either would be read as a transcript by
some other stage's test; `tests/rendered/` holds T7.2's three *rendered* files,
which are the output of one stage and not of a run. These two are the output of
the whole command, and they are a **pair** — see "they have to agree" below.

## What the manifest golden pins, and what it does not

It pins every value in the document except two, with the timestamps included:
the clock is injected, so `started_at` and `finished_at` are the two instants
the test hands the run and not the time of day it ran at. Pinned, therefore:
both file digests and sizes, the configuration (agent, adapter name, the
`adapter_requested`/`reduce`/`source_agent` options), the instruction identity,
the tokenizer identity and every measured number down to `saved_fraction`.

The two exceptions are `input_file.path` and `output_file.path`. They are the
paths the run was handed, they live under pytest's temporary directory, and they
are the only values in the document that differ between two identical runs —
the test masks them as `<input>` and `<output>` before comparing, and it asserts
that both placeholders really appear, so a golden can never quietly hold
somebody's home directory.

**They have to agree.** `output_file.sha256` and `output_file.size_bytes` are
the digest and length of `interleaved_intents.jsonl`'s bytes, and
`test_the_two_golden_files_agree_about_the_output_transcript` checks exactly
that with `hashlib`, reading nothing but the two committed files. A transcript
updated without its manifest therefore fails even before a run happens.

**`instruction.digest` is the sha256 of `src/idc/compact/narrative_instruction.toml`.**
Editing that file breaks the manifest comparison, and that is deliberate: the
digest is in the manifest so that a changed instruction is visible in a run's
provenance, and a golden that held it loosely would be the one place where
changing the prompt costs nothing. If you changed the instruction on purpose,
regenerate below and check that `instruction.version` moved too.

## Regenerating them

There is no switch that rewrites a golden in place, on purpose: the way a golden
stops protecting anything is a regression accepted because accepting it was one
keystroke. The failing test prints a unified diff and the path of the file the
run wrote under pytest's temporary directory, so:

1. `make unit` (or `uv run pytest tests/test_cli_end_to_end.py`) and **read the
   diff in the failure message**. It is rendered field by field, with the
   summary's own line breaks expanded, so the change is readable rather than
   "line 1 differs".
2. Only if every line of it is a change you meant to make: copy the file the
   message names over the golden here. For the manifest it names a copy that is
   already masked and already has this file name, so nothing has to be edited
   by hand.
3. Run the suite again. If `test_the_two_golden_files_agree_about_the_output_transcript`
   now fails, you updated one of the pair and not the other.

A diff that you cannot explain is not a golden to update; it is the failure the
file exists for.
