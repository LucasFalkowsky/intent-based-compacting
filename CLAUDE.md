# CLAUDE.md — Intent-Driven Context Compaction (offline CLI tool)

This repository contains an **offline command-line tool** that compacts coding-agent
transcripts (Claude Code, Gemini CLI, OpenCode) along **user intents** instead of
summarizing the whole history in one pass. It is the method under evaluation in the
master's study "The Impact of Intent-Driven Context Compaction on Code Generation
Conversations". The study design and all method decisions are fixed; this file describes
**how the tool is built and tested**, not what the method is. How the *work* is organised —
tickets, the board, branching, who spawns whom — is in `docs/orchestration.md`; see
"Who reads what" below.

## What the tool does (one paragraph)

`import` an agent transcript into a canonical format → `chunk` at user-message boundaries
→ `classify` each chunk into an intent chain (LLM or S1M adapter) → `resolve` the accepted
solution (last write per file, last passing verification command) from the path (reads,
failed attempts, overwritten edits) → `compact` every new or extended chain with a fixed
instruction into a Markdown block with stable chain IDs → `render` one summary block in
the target agent's native tags plus the latest chunk verbatim → `measure` tokens before
and after. Already-compacted blocks are parsed back into chains and offered as
classification candidates on the next run. No git access, no agent integration: the tool
only reads and writes transcript files.

## Who reads what

This file is **everything a subagent needs and nothing it does not**: what the tool is, how it is
tested, the conventions, and the rules about the user's local data. If you are implementing or
reviewing one ticket, this file plus your ticket's issue body is the whole context — do not go
looking for more.

**The orchestration lives in `docs/orchestration.md`** and is the orchestrating session's alone:
the GitHub Issues as source of truth, ticket structure, the label taxonomy, milestones, the board
and its ids, picking and closing tickets, the dependency graph and parallel execution, branching
and stacked PRs, how subagents are spawned, and the review step. A subagent is *forbidden* to
perform any of it, which is why it is no longer in its context.

**The orchestrating session reads `docs/orchestration.md` at the start of every session.** It is
not auto-loaded, so that is a deliberate step and not an optional one — the process is in that
file, and until it is read the session does not know it.

Why the split, measured rather than assumed: a subagent that makes **zero** tool calls and answers
with one word costs **33k tokens**, of which ~10k was project documentation and ~6k of that was
orchestration it must not act on. Across the thirteen subagents of one session that was ~78k tokens
spent handing agents instructions they were forbidden to follow.

## Working with local agent data

The three importers describe real transcript formats, and this machine has real transcripts on it.
They are the user's private data, and the mandate to work the tickets autonomously is a mandate over
ticket decisions — **not** over that data. An agent cannot authorise access to it, and neither can
this session on the user's behalf.

- A **structure-only survey** of a locally installed agent's sessions is permitted: key names,
  record types, tool names, argument keys.
- **Message content is never read.**
- **Nothing about the user's own usage is written into the repository** — no session counts, no
  concrete paths to their files, no project names, no identifiers. What a reader needs is the
  *evidence class* ("observed in recorded sessions" vs. "read off the installed version"), and that
  is what gets written down. A documented path layout with placeholders is the software's own and
  stays.
- Fixtures are **hand-written**, with synthetic ids and invented content — never excerpts.
- Before merging, the orchestrating session scans the whole branch **diff and every commit message**
  for home paths, names, addresses and statements about usage.

The cost of this is accepted openly: T2.3 (OpenCode) carries a documented limitation precisely
because OpenCode is not installed here, so its record-level field names rest on the documented
storage model rather than on a readable file. That is the honest trade-off, and it is recorded on the
ticket instead of being papered over.

## Test-driven development (mandatory)

Every ticket is a **self-contained, working building block with tests**. The loop runs inside the
ticket's implementation subagent:

1. **Red** — write the failing test(s) that express the acceptance criteria. Commit:
   `test(T3.1): add failing tests for user-boundary chunking`.
2. **Green** — write the minimal code that makes them pass. Commit:
   `feat(T3.1): split transcript at user-message boundaries`.
3. **Refactor** — clean up with tests green. Commit: `refactor(T3.1): …`.

Rules:

- No production code without a test that required it.
- Tests use the shared fixture corpus (`tests/fixtures/`, created in T1.3). Add fixtures
  there when an edge case is missing; never embed large transcripts inline in tests.
- Anything that calls a model is tested through a **fake adapter**; real model calls are
  never part of the test suite.
- The full suite must be green before a PR is opened: `make test` (lint + unit tests).
- Keep each test independent and deterministic. No network, no clock, no randomness
  without a seed.
- **A test parametrized over a table that may legitimately empty needs an assertion on the table
  itself.** An empty parametrization is a *skip*, not a pass. `PENDING_IMPORTERS` emptied when T2.4
  landed, which would have silently removed the dispatcher's pending path from the suite without one
  red test; it took a monkeypatched table plus an explicit `PENDING_IMPORTERS == {}` assertion to
  keep the coverage. The same applies to any fixture-derived parameter list.
- When a verification depends on bytes on disk rather than on the test suite (mutation runs, golden
  comparisons), clear stale caches first: `PYTHONDONTWRITEBYTECODE=1` and a removed `__pycache__`.
  A stale `.pyc` once produced a false mutation verdict for two mutants of equal byte length.

## Definition of done (per ticket)

- [ ] All acceptance criteria in the issue demonstrably met.
- [ ] Tests exist for every criterion and for the listed edge cases; `make test` is green.
- [ ] Public functions/classes have docstrings; `README.md` or the module docstring is
      updated when behavior visible to the CLI user changed.
- [ ] No TODOs left that belong to this ticket's scope.
- [ ] Decisions made on the way are reported, with their reasons, in the handover report.

The rest of done — subagent discipline, posting the decisions, the PR, the review, the merge and
the issue bookkeeping — is the orchestrating session's and lives in `docs/orchestration.md` under
"Definition of done — the orchestrating session's half".

## Conventions

- **Language:** all persisted artifacts (code, comments, docstrings, tests, issues,
  commit messages, docs) are **English**. Conversation with the user may be German.
- **Commits:** `<type>(T<id>): <imperative summary>` with `type ∈ {test, feat, fix,
  refactor, docs, chore}`.
- **Stack:** **Python 3.13** with `uv` (dependencies, venv), `pytest`, `pydantic` v2 (canonical
  schema) and `ruff` (lint + format) — decided in T1.1 (#2), recorded in the "Decision log" (#44).
  A new dependency is justified on the ticket that introduces it, as a `**Decision:**` comment.
- **Canonical schema** (T1.2) is the contract between all building blocks. Changing it is a
  ticket of its own, never a side effect.
- **Validate through the package API:** stages load and validate with `loads` / `load` /
  `validate_document`, **never** with pydantic's `model_validate`. Only the former raise
  `SchemaValidationError`; `model_validate` surfaces pydantic's own error and would put a pydantic
  traceback on the CLI. Construction and attribute assignment are contained too (#3).
- **No agent integration, no git calls** inside the tool. The tool reads transcript files
  and writes transcript files. If a ticket seems to need either, stop and ask.
- **Determinism first:** everything that can be computed from the transcript (solution
  files, verification commands, path classification, token counts) is computed by code,
  not by a model. Model calls are confined to the classifier adapters and the three
  narrative sections of the compaction template.

## Commands

```
make setup    # install dependencies (uv sync --all-groups)
make test     # lint + full test suite — the gate before every PR
make run ARGS="compact --agent opencode --in session.jsonl --out compacted.jsonl"
```

Convenience targets that exist but are not the gate: `make help`, `make lint`, `make unit`,
`make clean`, and `make schema` (regenerates `src/idc/schema/canonical.schema.json` from the
models — a dev utility, deliberately not part of the CLI command surface).

`make test` runs `ruff` with `--no-cache`: a warm `.ruff_cache` once reported a false green to two
independent subagents while the lint stage was actually failing (#45, PR #50). A gate whose verdict
depends on cache state is not a gate.
