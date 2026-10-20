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
| `chunk` | `idc.chunk` | split the transcript at user-message boundaries |
| `classify` | `idc.classify` | assign each chunk to an intent chain |
| `resolve` | `idc.resolve` | derive the accepted solution from the path |
| `compact` | `idc.compact` | render new or extended chains into Markdown blocks |
| `render` | `idc.render` | emit the summary in the target agent's native tags |
| `measure` | `idc.measure` | count tokens before and after |

`idc.schema` holds the canonical schema that is the contract between all stages, and
`idc.cli` the command surface.

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
would call a model is exercised through a fake adapter. Shared transcript fixtures live in
`tests/fixtures/` and are resolved through the `fixtures_dir` pytest fixture, never through
a relative path.

## Run

```
make run ARGS="compact --agent opencode --in session.jsonl --out compacted.jsonl"
```

`make run` forwards `ARGS` to `python -m idc`. The command surface itself is implemented in
ticket T10.1; until then the entry point exits with code 2 and says so.

## Layout

```
src/idc/              # the package (src layout)
  schema/ importers/ chunk/ classify/ resolve/ compact/ render/ measure/ cli/
tests/                # pytest suite
  fixtures/           # shared transcript fixture corpus
.github/workflows/    # CI: make test on every pull request
```

## Contributing

Work is organized entirely through this repository's GitHub Issues and its board; see
`CLAUDE.md` for the ticket structure, the branching model and the mandatory TDD loop.
