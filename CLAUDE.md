# CLAUDE.md — Intent-Driven Context Compaction (offline CLI tool)

This repository contains an **offline command-line tool** that compacts coding-agent
transcripts (Claude Code, Gemini CLI, OpenCode) along **user intents** instead of
summarizing the whole history in one pass. It is the method under evaluation in the
master's study "The Impact of Intent-Driven Context Compaction on Code Generation
Conversations". The study design and all method decisions are fixed; this file describes
**how work on the tool is organized**, not what the method is.

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

## Source of truth: GitHub Issues

**The GitHub Issues of this repository are the single source of truth** for what is to be
built, in which order, what has been decided, and what changed. Nothing about tickets,
priorities, decisions or process lives only in a file or only in a conversation.

Use the `gh` CLI for every interaction with issues (`gh issue list/view/create/edit/comment/close`)
and for the board (`gh project item-list/item-edit`). The tickets are **#1–#43**; the two pinned
`meta` issues are **#44 "Decision log"** and **#45 "Process"**.

### Ticket structure

- **Parent ticket** — one per pipeline building block. Title `T<n> — <name>`, label `epic`.
  Body: goal, acceptance criteria, a task list of its sub-tickets (`- [ ] #<issue> — <name>`), and a
  `## Relations` section. The sub-tickets are additionally attached as **native GitHub sub-issues**,
  so the parent shows a sub-issue progress bar; the task list and the sub-issue list must agree.
- **Sub-ticket** — title `T<n>.<m> — <name>`, label `task`. Body: scope, the required tests as a
  checklist, and a `## Relations` section with `Parent: #<issue>`, `Depends on: #<issue>, …`
  (or `—`) and `Blocks: #<issue>, …`. Every sub-ticket also carries the branch it is cut from in
  its header line.
- **The ticket ID encodes priority.** Parents are prioritized by `n` (T1 before T2); sub-tickets
  within a parent by `m` (T3.1 before T3.2). Never renumber. A new sub-ticket gets the next free
  `m`; if it must run earlier, say so in its body (`Priority: before T3.2`) — the ID stays.
- Titles use an em dash (`T3.1 — Subagent extraction`); **commit messages and PR titles use the
  short form** `<type>(T3.1): …` and `T3.1: <name>`.

### Label taxonomy

| Group | Labels | Meaning |
|---|---|---|
| Kind | `epic`, `task`, `meta` | parent ticket · sub-ticket · process/decision-log issue (not a ticket) |
| Status | `in-progress`, `review`, `blocked` | see the status table below; at most one per ticket |
| Phase | `priority:critical`, `priority:high`, `priority:medium`, `priority:low` | coarse filter mirroring the milestones (T1–T3 / T4–T6 / T7–T9 / T10). **Never** a tie-break: the ticket ID decides order. |
| Area | `area:infra:scaffold`, `area:schema`, `area:importers`, `area:chunker`, `area:classifier`, `area:resolver`, `area:compactor`, `area:renderer`, `area:parser`, `area:measurement`, `area:cli`, `area:eval` | which building block the ticket touches; a ticket may carry two |
| Flag | `tdd`, `optional`, `needs-decision` | tests-first (all tickets) · the study can run without it · an explicit project decision is part of the scope |

### Milestones

Milestones group the parents into the four phases of the study and exist only for reporting:
**M1 — Foundation (T1–T3)**, **M2 — Intent core (T4–T6)**, **M3 — Output & measurement (T7–T9)**,
**M4 — Integration (T10)**. A sub-ticket always carries its parent's milestone.

### Status

Status is expressed with labels, open/closed state, and the board column — and those three must
always agree. The column additionally answers "is this building block currently being built":
`Backlog` means the parent has not been activated yet, `Ready` means the ticket belongs to the
parent that is being implemented right now (see "Activating a parent").

| State | Meaning | Board column |
|---|---|---|
| open, no status label | todo; the next free ticket of the active parent | `Ready` if its parent is activated, else `Backlog` |
| open + `blocked`, dependency | its `Depends on` issues are not all closed — usually a sibling earlier in the same parent | `Ready` if its parent is activated, else `Backlog` |
| open + `blocked`, decision or scope problem | waiting on the user; the reason is in the latest comment | `Backlog` |
| open + `in-progress` | one agent is working on it; assignee set | `In progress` |
| open + `review` | implementation done, tests green, review pending | `In review` |
| closed | done; the closing comment names the merge commit SHA | `Done` |

`blocked` is therefore both the dependency marker and the "waiting on the user" marker; only the
second kind pulls a card out of `Ready` back to `Backlog`. **Remove the dependency kind from the
dependants as part of closing a ticket** — see "Closing a ticket".

An `epic` is `Backlog` until it is activated, then `In progress` (label `in-progress`, assignee
set) until it is closed — it stays there while its sub-tickets run, because the parent *is* the
work being done. A parent is closed only after all its sub-tickets are closed and its own
acceptance criteria hold on the parent branch.

### Board (mandatory)

The repository board is **Project #5 "Intent-Driven Context Compaction — Pipeline"**
(`https://github.com/users/LucasFalkowsky/projects/5`), columns
`Backlog · Ready · In progress · In review · Done`. All 43 tickets are on it; the `meta` issues are
not (they are pinned instead).

**Every status change moves the card in the same step as the label change** — a label without the
matching column, or the other way round, is a broken state. Concretely:

```
gh project item-list 5 --owner LucasFalkowsky --limit 100 --format json   # find the item id
gh project item-edit --id <item-id> --project-id PVT_kwHOBtVe8M4Bn5gZ \
  --field-id PVTSSF_lAHOBtVe8M4Bn5gZzhmcArU --single-select-option-id <option>
#   Backlog f75ad846 · Ready 61e4505c · In progress 47fc9ee4 · In review df73e18b · Done 98236657
```

Closing an issue does **not** move the card by itself unless the project's built-in workflow does
it; verify with `gh project item-list` and set `Done` explicitly if it did not.

New issues are picked up by the board's auto-add workflow. After creating a ticket, check that it
landed on the board and set its column; `meta` issues are removed from the board again.

#### Activating a parent

Implementation starts per **parent**, not per sub-ticket. When a parent becomes the one being
built, its whole scope is pulled onto the board in one step:

1. Cut the parent branch `ticket/T<n>-<slug>` from `main`.
2. Assign yourself the parent, add `in-progress`, move the parent card to `In progress`, and
   comment with the branch name.
3. **Move every sub-ticket of that parent from `Backlog` to `Ready`** — all of them, including the
   ones that carry `blocked` because they depend on a sibling. `Ready` is the pool for this parent;
   the labels, not the column, say which one may be picked.
4. From there the sub-tickets are worked **one at a time** in ID order (see "Picking the next
   ticket"); a ticket leaves `Ready` only into `In progress`, `In review` or `Done`.

Only **one parent is activated at a time**. The next parent is activated after the current one is
closed and merged into `main`. If a sub-ticket of the active parent turns out to need a decision,
it goes back to `Backlog` with `blocked` (see "Status") — the rest of the parent keeps running.

### Recording decisions and process changes

Claude writes to the issues — that is where information is kept:

- **Decisions** that arise while implementing a ticket (an ambiguity resolved, a format
  chosen, a limitation accepted) are recorded as a comment on that ticket, prefixed
  `**Decision:**`, with the reasoning in two or three sentences. If the decision affects
  other tickets, link them in the comment and leave a one-line pointer comment on each.
- **Cross-cutting decisions** (schema changes, stack, conventions) are additionally added
  to the pinned issue **"Decision log" (#44)** as a dated list entry linking to the
  originating comment.
- **Process changes** (anything that would change this file) are proposed as a comment on
  the pinned issue **"Process" (#45)**, then applied to `CLAUDE.md` in a
  `chore(process): …` PR that links the comment. Until merged, the old process applies.
- **Scope problems** (ticket is wrong, missing, or contradicts another) are a comment on
  the ticket plus the label `blocked` plus the board column `Backlog`, and a question to the
  user. Do not silently change scope, and never edit another ticket's body except to tick its
  task list, add a `Depends on` line, or remove a dependency that was resolved.
- **Progress notes** are short comments on the ticket when something non-obvious happened
  (an approach abandoned, a test that needed a new fixture). Routine progress is visible in
  the commits; do not narrate it.

## Picking the next ticket

1. `gh issue list --label epic --state open` → the parent with the lowest `n`. If it is not
   activated yet (card still in `Backlog`, no `in-progress`), **activate it first** — see
   "Activating a parent"; its sub-tickets all move to `Ready` before any of them is touched.
2. Within the active parent, `gh issue list --label task --state open --search "T<n>."` → the
   sub-ticket with the lowest `m` that carries **no** status label (`in-progress`, `review`,
   `blocked`). The `blocked` label already encodes "dependencies not closed"; if you suspect it is
   stale, verify the `Depends on` issues in the body and remove the label with a one-line comment.
3. If every sub-ticket of the parent is closed but the parent is open, the next job is the
   parent's own acceptance criteria on the parent branch, then its merge into `main`.
4. Before writing code: assign yourself, add `in-progress`, move the card from `Ready` to
   `In progress`, and comment with the branch name. One ticket at a time.
5. Hand the ticket to a **fresh implementation subagent** — see "One subagent per ticket". The
   orchestrating session does not write the ticket's code itself.

## One subagent per ticket

Every ticket is implemented in its **own subagent with a fresh context**, and reviewed in a second
one. Nothing but the ticket's own scope ever enters that context, and the exploration, the failed
attempts and the tool output of one ticket never leak into the next — that is the same reason this
tool exists, applied to its own development.

**Roles**

- **Orchestrating session** (this one): activates parents, picks tickets, sets labels, moves cards,
  creates and edits issues, records decisions, opens/merges PRs, and closes tickets. It reads the
  issue and the handover reports — it does **not** write the ticket's production code or run its
  test loop.
- **Implementation subagent**: one per ticket, started at step 5 of "Picking the next ticket",
  general-purpose, full tool access. It does the Red/Green/Refactor loop on the ticket's branch and
  returns a handover report.
- **Review subagent**: one per ticket, started after the implementation reported green — see
  "Review step". Never the context that wrote the code.

**Starting the implementation subagent** — the prompt names the ticket and nothing else; the
subagent fetches the scope itself:

> *"Implement ticket T<id> (issue #<n>) in this repository. Read `CLAUDE.md` and
> `gh issue view <n>` first — the issue body is the scope and the acceptance criteria. Work on
> branch `<branch>`, which is already cut from `<base>`. Follow the mandatory TDD loop (failing
> tests first, commits `test(T<id>): …` / `feat(T<id>): …` / `refactor(T<id>): …`), use the shared
> fixture corpus, no network and no real model calls in tests. Do not touch any other ticket's
> scope, do not edit issues, labels or the board, and do not open the PR. When `make test` is green,
> return the handover report described in `CLAUDE.md` → "One subagent per ticket" and nothing
> else."*

**Handover report** (what the subagent returns — a compact summary, never raw file dumps):

```
Ticket:        T<id> (#<n>)        Branch: <branch> (base <base>)
Commits:       <sha> <subject>  … (one line each)
Tests:         make test → <pass/fail>, <n> tests, <relevant new fixtures>
Criteria:      <criterion> → <test that covers it>   (one line per acceptance criterion)
Decisions:     <decision + one-sentence reason>      (or: none)
Scope/blocked: <what is wrong, missing or needs a user decision>  (or: none)
Files:         <path> — <what changed in one phrase>
```

The orchestrating session then does what only it may do: post the `**Decision:**` comments on the
issue (and the "Decision log" #44 entry for cross-cutting ones), open the PR, start the review
subagent, and run "Closing a ticket". A `Scope/blocked` entry means: `blocked` label, card back to
`Backlog`, question to the user — never a silent scope change.

**Rules**

- One implementation subagent at a time. Sub-tickets of a parent are not implemented in parallel:
  they share the parent branch, they are stacked, and `Depends on` is sequential by design.
- A subagent is never reused for a second ticket, and never for the review of its own work. If a
  ticket comes back `changes-requested`, the fixes may run in a **new** implementation subagent
  with the review findings in the prompt.
- If a subagent needs something outside its ticket (a missing fixture is fine, another ticket's
  scope is not), it stops and reports it under `Scope/blocked`.

## Closing a ticket

In this order, so that the board, the labels and the dependency graph stay consistent:

1. Merge the PR into its base branch, delete the branch.
2. Close the issue with a comment naming the merge commit SHA; remove `review`.
3. Move the card to `Done` (verify, see "Board").
4. Tick the entry in the parent's task list.
5. For every issue listed under `Blocks:` in the closed ticket's `## Relations`: if all of its
   `Depends on` issues are now closed, remove its `blocked` label. Siblings of the active parent
   are already in `Ready` and stay there; tickets of a parent that is not activated yet keep their
   card in `Backlog` until that parent is activated.
## Branching and stacked PRs

```
main
└── ticket/T3-chunker                 ← parent branch, cut from main
    ├── ticket/T3.1-user-boundaries   ← sub-ticket branch, cut from the parent branch
    ├── ticket/T3.2-system-reduction  ← cut from T3.1 (stacked), or from parent if independent
    └── ticket/T3.3-subagent-chunks
```

- **Parent ticket** → branch `ticket/T<n>-<slug>` from `main`.
- **Sub-ticket** → branch `ticket/T<n>.<m>-<slug>` from the **parent branch** (or from the
  preceding sub-ticket branch when it builds on it — that is the stack).
- Every sub-ticket opens a **PR against its parent branch** (stacked PR) whose body starts
  with `Closes #<issue>`. The parent opens **one PR against `main`** once all its
  sub-tickets are merged into it.
- Never commit directly on `main`. Never commit directly on a parent branch except merge
  commits from sub-ticket PRs.
- Rebase a stacked branch onto its base after the base changed; do not merge the base
  downwards.

## Test-driven development (mandatory)

Every ticket is a **self-contained, working building block with tests**. The loop runs inside the
ticket's implementation subagent (see "One subagent per ticket"):

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

## Definition of done (per ticket)

- [ ] All acceptance criteria in the issue demonstrably met.
- [ ] Tests exist for every criterion and for the listed edge cases; `make test` is green.
- [ ] Public functions/classes have docstrings; `README.md` or the module docstring is
      updated when behavior visible to the CLI user changed.
- [ ] No TODOs left that belong to this ticket's scope.
- [ ] Decisions made on the way are recorded on the issue (see above) — taken from the
      implementation subagent's handover report.
- [ ] Implementation ran in its own subagent, the review in a second one.
- [ ] PR opened against the correct base branch, title `T3.1: <name>`, body `Closes #<issue>`.
- [ ] Review passed (see below) and merged into the parent branch (sub-ticket) or `main`
      (parent ticket).
- [ ] Issue closed with a comment naming the merge commit SHA; card moved to `Done`; parent task
      list ticked; the `blocked` label removed from the tickets this one unblocks (see
      "Closing a ticket").

## Review step (second agent, lightweight)

When the implementation subagent reports green, replace `in-progress` with `review`, move the card
to `In review`, and run the review as a **second, separate subagent call** — never the context that
wrote the code:

- If the `tyler-review` skill is available, invoke it with the ticket ID (e.g.
  `review T3.1`) and the branch name. It is read-only and returns a verdict.
- Otherwise spawn a general-purpose subagent with this prompt:
  *"Review branch `<branch>` against the acceptance criteria of issue #<n>
  (`gh issue view <n>`). Re-run `make test`. Report: verdict (approved /
  changes-requested), findings with severity (blocker / major / minor) and file:line, and
  which criteria are met. Do not edit anything."*

The review is deliberately **quick**: it checks that tests are green, that every
acceptance criterion is covered by a test, and that nothing is obviously wrong or out of
scope. It is not a design review. The verdict is posted as a comment on the issue
(prefix `**Review:**`).

- **approved** → merge the PR into its base branch (squash for sub-tickets, merge commit
  for parent → `main`), then run "Closing a ticket" above.
- **changes-requested** → fix the `developer`-owned findings on the same branch, re-run
  the review. Findings owned by `user` set the label `blocked` and move the card back to
  `Backlog`; ask the user.

## Bootstrap (done — 2026-10-20)

The one-time bootstrap from the `ISSUES.md` seed is **complete**: labels, milestones M1–M4, the
tickets #1–#43 (parents first, then sub-tickets with `Parent:`/`Depends on:` resolved to issue
numbers and the parents' task lists filled in), the native sub-issue links, the board (Project #5),
and the pinned `meta` issues #44/#45. `ISSUES.md` is **not** part of the repository; the issues are
the only ticket source. Do not re-create it — a new ticket is created directly with
`gh issue create` following "Ticket structure".


## Conventions

- **Language:** all persisted artifacts (code, comments, docstrings, tests, issues,
  commit messages, docs) are **English**. Conversation with the user may be German.
- **Commits:** `<type>(T<id>): <imperative summary>` with `type ∈ {test, feat, fix,
  refactor, docs, chore}`.
- **Stack:** decided in T1.1 (#2) and recorded in the "Decision log" issue (#44). Until then, do not
  add dependencies.
- **Canonical schema** (T1.2) is the contract between all building blocks. Changing it is a
  ticket of its own, never a side effect.
- **No agent integration, no git calls** inside the tool. The tool reads transcript files
  and writes transcript files. If a ticket seems to need either, stop and ask.
- **Determinism first:** everything that can be computed from the transcript (solution
  files, verification commands, path classification, token counts) is computed by code,
  not by a model. Model calls are confined to the classifier adapters and the three
  narrative sections of the compaction template.

## Commands (filled in by T1.1)

```
make setup    # install dependencies
make test     # lint + full test suite
make run ARGS="compact --agent opencode --in session.jsonl --out compacted.jsonl"
```