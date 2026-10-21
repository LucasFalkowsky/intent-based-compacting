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
| `Session` | a whole transcript: agent, model, provenance (`raw`, `native-compaction`, `intent-compaction`), which message the agent's own summary is, messages |
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

## Importers

An importer reads one agent's transcript into a canonical `Session`. It returns that
session together with **notes**: one line per thing the transcript held that the canonical
schema has no place for, so that nothing is lost silently.

```python
from idc.importers import import_transcript

imported = import_transcript("claude-code", "~/.claude/projects/repo/session.jsonl")
imported.session.source  # SessionSource.RAW, or NATIVE_COMPACTION for a compacted one
imported.unknown_tools  # ('mcp__server__thing',) — resolved to `other`, and logged
for note in imported.notes:
    print(note)  # 'skipped-record: attachment (x19)', 'dropped-block: thinking (x7)', …
```

All three agents are implemented: **Claude Code** (ticket T2.2), **OpenCode** (T2.3) and
**Gemini CLI** (T2.4). What is shared sits in `idc.importers.base` (assembling and
validating the session, resolving tool categories, logging the unknown ones, reading
timestamps, recording which message the agent's own compaction summary is),
`idc.importers.jsonl` (reading a line-delimited transcript) and
`idc.importers.provenance` (which of the imported messages are context — see below); an
agent module is only that agent's format.

The three formats differ in more than their field names. A Claude Code record carries one
tool result per `tool_result` block of a *user* record, while an OpenCode tool call carries
its own result in the same part of the *assistant* record, and OpenCode records an instant
as epoch milliseconds rather than as an ISO-8601 string. Gemini CLI reads two shapes: the
single JSON document a recorded session has, and a newer append log whose `$set` and
`$rewindTo` operations are **replayed** rather than concatenated — without that, a
natively compacted transcript would import the history the agent threw away.

Four properties hold for every importer:

- **Identifiers come from the transcript.** A message keeps the record's own identifier and
  a tool call the one its result references, so a canonical transcript can still be read
  against the file it came from. A record that yields several canonical messages — the
  schema holds one tool result per message — derives them as `<id>#<block index>`.
- **Nothing is repaired.** A transcript cut off at the front, whose first tool result has
  no call, and a call whose result never arrived because the turn was interrupted, both
  import as they are: the canonical schema deliberately does not require referential
  completeness, and the measurement has to count them.
- **No message content is dropped without a note.** A block the schema has no field for, a
  record that carries no message, and a value an importer cannot read at all are each
  reported: `dropped-block: thinking`, `skipped-record: attachment`, and — for something that
  has no kind to name — the shape it had, as in `dropped-block: tool_use/Bash/<input: array>`,
  `dropped-block: text/<text: object>` or `skipped-record: <null type>`. Nothing is coerced
  either: a block whose text is not a string is dropped and noted rather than turned into a
  repr of itself, because inventing content is worse than losing it. A record whose content
  *is* a block list the schema has no field for — a pasted screenshot — does import as the
  turn it was, with empty content and the note, because that turn happened.

  A record whose content cannot be read **at all** — a number, an object — is the one case
  where the importers deliberately differ, and each of them notes what it did. Claude Code
  **and Gemini CLI** yield **no message** and note the record both for the content
  (`dropped-block: <content: object>`) and as skipped: one of their `user` records is a
  typed prompt and the next is a tool result, so an unreadable content means the record's
  kind is unknown too, and an invented user turn would be an invented intent boundary.
  OpenCode is the outlier: it **keeps the user turn**, with empty content and a note for the
  unreadable `parts`: a `role = user` record there is always the user's turn and never a
  result carrier, so the turn is still attested — dropping it would fabricate the chain
  structure T3.1 splits on. Each of the three has a test of its own on this, so which side
  an agent is on is pinned rather than described.

  It is a property of the *message content*, and only of that. The record-level fields an
  importer deliberately never reads carry no note — for Claude Code `isSidechain`,
  `toolUseResult`, `parentUuid`, `cwd`, `gitBranch` and `version` — and each importer's
  module docstring lists the ones it ignores and why.
- **One error type.** Anything an importer rejects is a `TranscriptImportError` naming the
  file, and a problem that belongs to one record names its line number. A tool call that
  does not satisfy the canonical schema is reported by the schema's own
  `SchemaValidationError` — e.g. a `tool_use` identifier that occurs twice.

### The visible context

Importing faithfully is not enough to compare the three agents: a Claude Code or OpenCode
transcript still holds the history its own compaction replaced, Gemini CLI's does not
(its `$set.messages` already dropped it), and each agent records a different amount of
harness chatter. `visible_context` is the one place that decides what counts as context,
so measurement counts the same thing whichever agent recorded the session:

```python
from idc.importers import import_transcript, visible_context

context = visible_context(import_transcript("claude-code", "session.jsonl"))
context.message_ids  # the messages the agent is still being sent
[str(hidden) for hidden in context.hidden]  # ['replaced-by-compaction: u40']
context.session  # those messages as a canonical Session
```

Two reasons hide a message and nothing else does: it sits **before the summary the agent's
own compaction wrote** (`replaced-by-compaction`), or it is an injected `system` **notice**
that is not that summary — Claude Code's `system` records, Gemini CLI's `info` and `error`
lines (`harness-notice`). Injected text in the *user* role stays visible, because the model
is sent it: a `<system-reminder>`, Gemini CLI's `<session_context>`, OpenCode's `synthetic`
text. Normalisation changes exactly one thing on the way out — a harness-written notice in
the user role, Claude Code's `[Request interrupted by user for tool use]`, is returned
marked `is_system_injected`, so that the flag means *the harness wrote this* for all three
agents and the chunker does not read it as a user turn. Such a notice is recognised by its
**whole text**, and only in a spelling this repository has actually observed: a user who
pastes the notice and keeps typing has opened a real intent, and a prefix test or a guessed
spelling would reduce that intent away.

Which message the agent's own summary is travels in the document itself, as
`Session.native_summary_id`, so a `native-compaction` transcript written by one command is
still readable by the next — the pipeline is one command per stage with files in between,
and the baseline the study compares against has to survive that. The identifier and
`source` have to agree: a summary named by a session that was never natively compacted is
rejected by the schema. The converse is deliberately left to the stages, exactly as for a
`ToolResult` whose call the transcript truncated away — a `native-compaction` document that
names no summary, or one whose identifier names no message in it, still loads, and
`visible_context` refuses it rather than guessing where the agent's context starts.
Every stage that rebuilds the transcript carries the identifier on, and the message it
names is exempt from the chunker's noise reduction — the agent's summary block is an
injected message, so reducing it away would leave the document naming a message it no
longer holds.

`python -c` is not the interface: the `import` command is wired up in T10.1.

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
wrongly instead of merely being coarse. A tool a transcript *has* shown is declared even
when its category is `other`, so that the unknown-tool log keeps flagging the calls whose
category would be resolved wrongly rather than warning on every import. MCP tools are the
one group that stays undeclared: their names are `mcp__<server>__<tool>` and the server is
whatever the user configured, so neither the spelling nor the effect can be known here.
Verification matching is **anchored** on the
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

The importers need the other side as well — transcripts in an agent's **own** on-disk
format. Those live in `tests/native/<agent>/` and are resolved by agent and name through
`tests/native_transcripts.py`. Every record key they carry is one that real transcripts of
that agent show; the values are invented, so no fixture is a recording of a real session.
`tests/native/README.md` states what each one holds.

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
  fixtures/           # shared canonical transcript fixture corpus
  native/             # agent-native transcripts — the importers' input side
.github/workflows/    # CI: make test on every pull request
```

## Contributing

Work is organized entirely through this repository's GitHub Issues and its board; see
`CLAUDE.md` for the ticket structure, the branching model and the mandatory TDD loop.
