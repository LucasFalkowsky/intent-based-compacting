# End-to-end golden files (T10.2, T10.3)

The files two whole `idc run` invocations leave behind — a first run, and the
**second** run that continues it — byte-compared by
`tests/test_cli_end_to_end.py` and `tests/test_cli_second_run.py`:

| File | What it is |
| --- | --- |
| `interleaved_intents.jsonl` | The compacted Claude Code transcript the first run wrote: the summary of both chains as a `user` record flagged `isCompactSummary`, then the third chunk verbatim. |
| `interleaved_intents.jsonl.manifest.json` | Its run manifest, at the path `run` puts one when it is given no `--manifest`: `<--out>.manifest.json`. |
| `interleaved_intents_second_run.jsonl` | What the second run wrote, reading the file above with two further chunks appended to it: a summary of **three** chains — one rewritten, one copied, one new — then the last of the new chunks verbatim. |
| `interleaved_intents_second_run.jsonl.manifest.json` | Its manifest, the one that records the first run's output digest in `previous_output_sha256`. |

The first pair comes out of **one** invocation —
`main(["run", "--agent", "claude-code", "--in", …, "--out", …])` — with the four
seams of `idc.cli.Runtime` injected as fakes: `FakeAdapter` playing
`idc.classify.gold_script` of the fixture's gold labelling, `FakeNarrativeClient`
with one answer per chain, `FakeTokenizer` behind every model family, and
`FakeClock` with two instants. The input is the shared corpus fixture
`interleaved_intents` written out in Claude Code's own record shape; the test's
`native_input` says how and why, and two of its tests pin that what the run
imports is that fixture.

The second pair comes out of a second invocation of the same `main`, in
`tests/test_cli_second_run.py`, which reuses all of that: the first run is
literally the `produced` fixture of the module above, its output file is the
start of the second run's input, and the two new chunks are eight hand-written
canonical messages written into Claude Code's records by the same writer. Its
own fakes differ in the two places the run differs: the adapter's script is
`("C1", "C1", NEW)` — the chunk the first run kept, one new chunk continuing
that chain, one new chunk on a new intent — and the narrative client holds
**two** answers for three chains, because `C2`, the one nothing in this run
touches, is copied and costs no call.
The clock's two instants are an hour after the first run's.

Why a directory of its own, next to `tests/fixtures/`, `tests/native/` and
`tests/rendered/`: the first two hold **inputs** and are both discovered by
globbing, so an output dropped into either would be read as a transcript by
some other stage's test; `tests/rendered/` holds T7.2's three *rendered* files,
which are the output of one stage and not of a run. These four are the output of
the whole command, and they are two **pairs** — see "they have to agree" below.

## What the manifest goldens pin, and what they do not

Each pins every value in the document except two, with the timestamps included:
the clock is injected, so `started_at` and `finished_at` are the two instants
the test hands the run and not the time of day it ran at. Pinned, therefore:
both file digests and sizes, the configuration (agent, adapter name, the
`adapter_requested`/`reduce`/`source_agent` options), the instruction identity,
the tokenizer identity and every measured number down to `saved_fraction`.

The second run's manifest additionally pins `previous_output_sha256`: the
digest of the file the first run wrote, and therefore of the first transcript
golden in this directory.

The two exceptions are `input_file.path` and `output_file.path`. They are the
paths the run was handed, they live under pytest's temporary directory, and they
are the only values in the document that differ between two identical runs —
the test masks them as `<input>` and `<output>` before comparing, and it asserts
that both placeholders really appear, so a golden can never quietly hold
somebody's home directory.

**They have to agree.** `output_file.sha256` and `output_file.size_bytes` are
the digest and length of the transcript next to them, and
`test_the_two_golden_files_agree_about_the_output_transcript` —
`test_the_second_runs_golden_files_agree_about_its_output_transcript` for the
second pair — checks exactly that with `hashlib`, reading nothing but the
committed files. A transcript updated without its manifest therefore fails even
before a run happens.

**And the two pairs have to agree with each other.** The second run's
`previous_output_sha256` is the sha256 of `interleaved_intents.jsonl`, checked
by `test_the_second_runs_golden_manifest_names_the_first_runs_golden_transcript`
off the four committed files alone. Regenerating the first pair without
regenerating the second therefore fails here, which is the point: the four files
are one story about two runs, not two stories.

**`instruction.digest` is the sha256 of `src/idc/compact/narrative_instruction.toml`.**
Editing that file breaks the manifest comparison, and that is deliberate: the
digest is in the manifest so that a changed instruction is visible in a run's
provenance, and a golden that held it loosely would be the one place where
changing the prompt costs nothing. If you changed the instruction on purpose,
regenerate below and check that `instruction.version` moved too.

## What the second run's goldens do **not** pin

They do not pin `continues_previous`, beyond its being `False`. That field asks
whether this run's input *is* the file the predecessor wrote, and the second
run's input is that file **with two chunks appended** — so a continued session
answers `False` by construction, and the digest above is what records the
predecessor. Handing a run the output file unchanged does not produce a `True`
either: such a transcript is a single chunk that holds the summary record, and
`idc.render` refuses to write a second record under that identity, so the
command exits `EXIT_INPUT_ERROR`. Both halves are pinned in
`tests/test_cli_second_run.py`, the second by a run that is expected to be
refused.

They also do not pin any ordering the classifier chose: the chain ids and the
order of the three blocks are the ones `("C1", "C1", NEW)` dictates, and the
module asserts the decisions and the candidates it was offered separately, so a
golden diff is never the only evidence of what the run decided.

## Regenerating them

There is no switch that rewrites a golden in place, on purpose: the way a golden
stops protecting anything is a regression accepted because accepting it was one
keystroke. The failing test prints a unified diff and the path of the file the
run wrote under pytest's temporary directory, so:

1. `make unit` (or `uv run pytest tests/test_cli_end_to_end.py
   tests/test_cli_second_run.py`) and **read the diff in the failure
   message**. It is rendered field by field, with the summary's own line breaks
   expanded, so the change is readable rather than "line 1 differs".
2. Only if every line of it is a change you meant to make: copy the file the
   message names over the golden here. For the manifest it names a copy that is
   already masked and already has this file name, so nothing has to be edited
   by hand.
3. Run the suite again. If an `..._agree_about_the_output_transcript` test now
   fails, you updated one of a pair and not the other; if
   `test_the_second_runs_golden_manifest_names_the_first_runs_golden_transcript`
   fails, you updated the first pair and left the second run's manifest naming
   the transcript you replaced — regenerate that pair too.

A diff that you cannot explain is not a golden to update; it is the failure the
file exists for.
