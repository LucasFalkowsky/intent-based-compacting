# idc — Intent-Driven Context Compaction

An **offline command-line tool** that compacts coding-agent transcripts (Claude Code,
Gemini CLI, OpenCode) along **user intents** instead of summarizing the whole history in
one pass. It is the method under evaluation in the master's study *"The Impact of
Intent-Driven Context Compaction on Code Generation Conversations"*.

The tool only reads and writes transcript files: no agent integration, no git access, no
network at test time.

## Pipeline

| Stage | Package | What it does |
|---|---|---|
| `import` | `idc.importers` | read an agent transcript into the canonical format |
| `chunk` | `idc.chunk` | split the transcript at user-message boundaries, reduce the harness noise inside a chunk, lift a subagent's work into a chunk of its own |
| `classify` | `idc.classify` | assign each chunk to an intent chain |
| `resolve` | `idc.resolve` | derive the accepted solution from the path |
| `compact` | `idc.compact` | render new or extended chains into Markdown blocks |
| `render` | `idc.render` | emit the summary in the target agent's native tags |
| `measure` | `idc.measure` | count tokens before and after |

`idc.schema` holds the canonical schema that is the contract between all stages, and
`idc.cli` the command surface.

## Canonical schema

Every stage reads and writes the seven canonical types in `idc.schema`; the agents' own
transcript shapes exist only inside `idc.importers` and `idc.render`.

| Type | Holds |
|---|---|
| `Session` | a whole transcript: agent, model, provenance (`raw`, `native-compaction`, `intent-compaction`), messages |
| `Message` | role, content, timestamp, and whether it was system-injected or a slash command |
| `ToolCall` / `ToolResult` | a call with its category (`read`, `write`, `exec`, `other`) and arguments, and its output verbatim |
| `Chunk` | one user input plus the agent's response, under a stable ID `K<n>`, as a half-open range over `Session.messages` |
| `Chain` | the chunks of one user intent, under a stable ID `C<n>`, with its summary block |
| `CompactedBlock` | the chains of a transcript plus the chunk that is kept verbatim |

Load and dump canonical documents through the package, never with `json` directly — every
load validates and raises a single `SchemaValidationError` naming the type and the
offending field path:

```python
from idc.schema import Session, dump, load

session = load(Session, "session.json")  # validated on load
dump(session, "copy.json")  # lossless round trip
```

Unknown fields are rejected, timestamps must carry a timezone, and identifiers that are
referenced elsewhere must be unique. The same contract is published as a JSON Schema
document in `src/idc/schema/canonical.schema.json`, which is **generated** from the models
and kept in sync by a test:

```
make schema   # regenerate it after changing a model
```

Changing the schema is a ticket of its own, never a side effect of another change.

## Requirements

- Python 3.13 (see `.python-version`)
- [uv](https://docs.astral.sh/uv/) for dependency and virtualenv management

## Setup

```
make setup    # uv sync --all-groups — creates .venv and installs all dependencies
```

## Test

```
make test     # lint (ruff check + ruff format --check) + the full pytest suite
make lint     # lint only
make unit     # unit tests only
```

The suite is offline and deterministic: no network, no real model calls. Anything that
would call a model is exercised through a fake adapter.

Shared transcript fixtures live in `tests/fixtures/` — nine hand-written canonical
transcripts, one per case (single intent, sequential intents, interleaved intents, a slash
command, a system-injected message, a subagent result, a failed command followed by a
passing one, a file written then edited twice, a session that already contains a compacted
block). Every test loads them **by name** through `tests/fixture_corpus.py`, never through
a relative path:

```python
from fixture_corpus import CORPUS

session = CORPUS.session("interleaved_intents")  # validated on load
```

`tests/fixtures/README.md` states what each fixture exercises; the directory is discovered
on every call, so a new fixture is picked up without changing any test.

## Run

```
make run ARGS="compact --agent opencode --in session.jsonl --out compacted.jsonl"
```

`make run` forwards `ARGS` to `python -m idc`. The command surface itself is implemented in
ticket T10.1; until then the entry point exits with code 2 and says so.

The classifier evaluation harness is a development utility outside that surface,
next to `make schema`:

```
python -m idc.classify tests/fixtures/gold tests/fixtures
python -m idc.classify GOLD_DIR TRANSCRIPT_DIR --decisions RUN_DIR
```

It reports how far a classifier run agrees with the gold chain labels of
`tests/fixtures/gold/` — per-chunk accuracy and chain-level Rand index, per transcript and
over the corpus. Without `--decisions` it runs the gold baseline, which scores 1.000 by
construction; with it, the recorded decisions of a real run are replayed and scored. No
invocation calls a model: the real backends take their model as an injected dependency.

Every report **names the source of its numbers** — a `source:` line, and the same label on
the corpus line, which is the line a reader would paste. The baseline and a real run that
scored 1.000 are otherwise indistinguishable, so a number without its source says nothing:

```
source: gold baseline, 1.000 by construction — the gold labels played back through gold_script, …
corpus [gold baseline, 1.000 by construction]: 9 transcripts, 12 chunks, 4 chunk pairs; accuracy 1.000, …
floor: accuracy 0.750 — 9 of 12 chunks are a transcript's first chunk, where NEW is the only …
floor: the rand index rests on 4 chunk pairs; 7 of 9 transcripts hold a single chunk and form no …
```

The two `floor:` lines are the second thing to read before any corpus number: both metrics
have a floor that belongs to the **corpus**, not to the classifier. A transcript's first
chunk is offered no candidate chain, so `NEW` is the only decision the classifier accepts
and that chunk is scored right in every run that completes — and a transcript of a single
chunk therefore admits no wrong decision at all. The Rand index, in turn, rests only on
the pairs the multi-chunk transcripts form. Only the distance between `accuracy` and the
accuracy floor measures a backend. The figures above are the ones the command printed for
the corpus as it stands; they are **derived from the cases on every run**, so read them
from the output rather than from this paragraph — a fixture added to `tests/fixtures/`
moves all of them.

## Layout

```
src/idc/              # the package (src layout)
  schema/             # the canonical types + the generated canonical.schema.json
  importers/ chunk/ classify/ resolve/ compact/ render/ measure/ cli/
tests/                # pytest suite
  fixtures/           # shared transcript fixture corpus
.github/workflows/    # CI: make test on every pull request
```

## Contributing

Work is organized entirely through this repository's GitHub Issues and its board; see
`CLAUDE.md` for the ticket structure, the branching model and the mandatory TDD loop.
