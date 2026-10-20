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

## Canonical schema

Every stage reads and writes the seven canonical types in `idc.schema`; the agents' own
transcript shapes exist only inside `idc.importers` and `idc.render`.

| Type | Holds |
|---|---|
| `Session` | a whole transcript: agent, model, provenance (`raw`, `native-compaction`, `intent-compaction`), messages |
| `Message` | role, content, timestamp, and whether it was system-injected or a slash command |
| `ToolCall` / `ToolResult` | a call with its category (`read`, `write`, `exec`, `other`) and arguments, and its output verbatim |
| `Chunk` | one user input plus the agent's response, as a half-open range over `Session.messages` |
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

## Tool registry

What a tool call does to the workspace, and which commands count as a verification run,
are **data, not code**: `src/idc/importers/tool_registry.toml` holds one tool table per
agent and the list of verification patterns. Adding a tool or a pattern is a change to
that file alone.

```python
from idc.importers import tool_registry

registry = tool_registry()
registry.category("claude-code", "Edit")  # ToolCategory.WRITE
registry.category("opencode", "Read")  # ToolCategory.OTHER — OpenCode spells it `read`
registry.knows("claude-code", "SomeMcpTool")  # False — resolves to `other`, importers log it
registry.is_verification("uv run pytest -q")  # True
registry.is_verification('git commit -m "add pytest"')  # False
```

A tool the file does not declare resolves to `other`; the file is deliberately incomplete
where no transcript has shown a spelling yet, because an invented entry would resolve
wrongly instead of merely being coarse. Verification matching is **anchored** on the
command word (after `NAME=value` assignments and runner prefixes such as `uv run` are
stripped, and a leading path is reduced to its file name), so a command that only mentions
a test runner in an argument never counts as one — not in a quoted string, not behind a
`#` comment, and not when the runner is merely asked for its version, its help or what it
*would* run (`pytest --collect-only`, `make test -n`: the data file's `non_running_flags`).

A command the matcher cannot read — a runner inside a shell construct, a Windows path, an
unbalanced quote — is **missed**, never mis-matched; those accepted limitations are listed
in the data file next to the patterns.

The file is validated when it is read — an unknown agent, an unknown category, an unknown
field or a missing section raises one `ToolRegistryError` naming the file and the offending
path, rather than letting every tool resolve to `other` unnoticed.

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
