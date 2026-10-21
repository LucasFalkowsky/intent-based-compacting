# Orchestration — Intent-Driven Context Compaction

**Audience: the orchestrating session only.** This file is not auto-loaded into any context; the
orchestrating session reads it at the start of every session, and `CLAUDE.md` says so.

It was split out of `CLAUDE.md` because every subagent paid for it: the project instructions are
injected into each agent's context, so all thirteen subagents of one session carried ~6k tokens of
board ids, label taxonomy, merge order and closing procedure that a subagent is explicitly
forbidden to perform. `CLAUDE.md` now carries only what a subagent needs; everything an agent must
*not* do is here.

Nothing in the process changed with the split — only where it is written down.

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
4. From there the sub-tickets are picked in ID order (see "Picking the next ticket"); a ticket
   leaves `Ready` only into `In progress`, `In review` or `Done`.

**More than one parent may be activated at a time**, and independent tickets run concurrently — see
"Parallel execution". If a sub-ticket of an active parent turns out to need a decision, it goes back
to `Backlog` with `blocked` (see "Status") — the rest of the parent keeps running.

### Recording decisions and process changes

Claude writes to the issues — that is where information is kept:

- **Decisions** that arise while implementing a ticket (an ambiguity resolved, a format
  chosen, a limitation accepted) are recorded as a comment on that ticket, prefixed
  `**Decision:**`, with the reasoning in two or three sentences. If the decision affects
  other tickets, link them in the comment and leave a one-line pointer comment on each.
- **Cross-cutting decisions** (schema changes, stack, conventions) are additionally added
  to the pinned issue **"Decision log" (#44)** as a dated list entry linking to the
  originating comment.
- **Process changes** (anything that would change this file or `CLAUDE.md`) are proposed as a
  comment on the pinned issue **"Process" (#45)**, then applied in a `chore(process): …` PR that
  links the comment — to **this** file when the rule is orchestration, to `CLAUDE.md` when it is
  something a subagent must follow. Until merged, the old process applies.
- **Scope problems** (ticket is wrong, missing, or contradicts another) are a comment on
  the ticket plus the label `blocked` plus the board column `Backlog`, and a question to the
  user. Do not silently change scope, and never edit another ticket's body except to tick its
  task list, add a `Depends on` line, or remove a dependency that was resolved.
- **Progress notes** are short comments on the ticket when something non-obvious happened
  (an approach abandoned, a test that needed a new fixture). Routine progress is visible in
  the commits; do not narrate it.

## Picking the next ticket

1. `gh issue list --label epic --state open` → the parent with the lowest `n` that still has open
   sub-tickets. If it is not activated yet (card still in `Backlog`, no `in-progress`),
   **activate it first** — see "Activating a parent"; its sub-tickets all move to `Ready` before
   any of them is touched. Lower `n` is picked first, but a higher parent whose dependencies are
   satisfied may be activated alongside it — see "Parallel execution".
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

## Parallel execution

The ordering authority is the **`Depends on` graph over all tickets**, not the ID alone: the IDs give
the priority within a ready set, the graph says what is ready at all. Many edges cross parents
(#12→#4, #14→#6, #26→#20, #32→#5, #41→#36), so once several parents are open the graph is the only
thing that keeps the order correct.

- Two tickets may be implemented **concurrently exactly when neither is in the other's transitive
  `Depends on` closure**. Everything else is sequential.
- Each concurrent implementation runs in its **own `git worktree`** on its own ticket branch. Two
  subagents in one working directory is a broken state, not a slow one.
- Independent sub-tickets of the same parent are each cut from the **parent branch** and merged back
  independently. A sub-ticket with a `Depends on` on a sibling is cut from **that sibling's branch**
  (the stack) and rebased onto the parent branch after the sibling merges.
- A ticket that depends on a *parent* ticket (e.g. #26 → #20) waits for that parent to be closed and
  merged into `main`, not merely for its sub-tickets.
- **A cross-parent `Depends on` edge on a *sub-ticket* gives ordering, not code.** The edge is
  satisfied for ordering as soon as the dependency closes — but closing a sub-ticket merges it into
  **its own parent branch**, so its code is reachable only once *that parent* is merged into `main`.
  "Concurrently implementable" and "can import each other's code" are different questions and only
  the first one is in the graph (this bit #14 → #6). When the code is genuinely needed, in this
  order of preference: **inject** the dependency at the stage boundary and defer the wiring;
  **sequence** the two parents; or — worst — pull the other branch in, which duplicates its merge.
  Where injection is chosen, record the still-open **call site** as a pointer on the ticket that
  wires the stages: an injected predicate that is never wired fails *silently*, with the suite green.
- The per-ticket rules do not change: one implementation subagent and one separate review subagent
  per ticket, labels and board column moved in the same step, and "Closing a ticket" run in order.

The orchestrating session works the graph **autonomously** through to merge. It asks the user only
when a ticket is genuinely unimplementable as written — a subagent's `Scope/blocked`, or a
contradiction between tickets. Everything else is decided and recorded as a `**Decision:**` comment
on the ticket (see "Recording decisions and process changes").

**Work in progress is driven to closed before new work starts.** Parallelism is bounded by the
graph, but the number of *in-flight* tickets is bounded by this rule: never start a ticket while an
already-started one is waiting on nothing but this session's own next step — opening the PR, reading
the review, merging, running "Closing a ticket". Four sub-tickets once sat in `in-progress`/`review`
at the same time because the orchestrating session kept activating new work instead of finishing
what it had; the user stopped it. An open ticket is not progress.

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

**The process lives in the agent definitions, not in the prompt.** `.claude/agents/idc-implementer.md`
and `.claude/agents/idc-reviewer.md` carry the TDD loop, the non-negotiable rules, the context
discipline and the report format. Use `subagent_type: idc-implementer` / `idc-reviewer` and keep the
prompt to what is specific to *this* ticket:

> *"Implement ticket T<id> (issue #<n>). Worktree: `<path>`. Branch: `<branch>`, already cut from
> `<base>`. <Any fact from another ticket that this one needs, stated here rather than named as an
> issue to go and read.> <Anything the issue body does not say but you need decided.>"*

**Why so short — this is measured, and it was bad.** Naming two to four issues for a subagent to
read pulled in *this session's own* review and decision comments: on #19 the body is 543 bytes and
the comments are **26 KB**. The result was 50k tokens spent adding a single assertion and 75k on a
two-line guard.

**What the measurement actually showed, once it was done properly.** A subagent that makes **zero**
tool calls and answers with one word costs **33k tokens**. Shortening the *prompt* could never have
fixed that, because the project instructions are injected into every agent's context whether the
prompt mentions them or not — which is why the orchestration was moved out of `CLAUDE.md` into this
file rather than merely left unmentioned. `CLAUDE.md` went from ~8.0k tokens to ~2.3k, so each
subagent carries ~5.7k less, and across the thirteen subagents of one session that is ~75k that
used to be spent handing agents rules they were forbidden to act on. The remaining ~23k of the 33k
is the harness's own system prompt and tool schemas and is not ours to shrink.

The consequence for prompting is the opposite of what it looks like: since the fixed cost per agent
dominates, **not spawning an agent is worth more than any prompt economy**. A pinpoint change is
cheaper done by the orchestrating session than handed to a fresh 33k context — see the rules below.

**Context budget, which the orchestrating session is responsible for:**

- The subagent reads **the issue body only** — `gh issue view <n> --json body --jq .body`, never
  plain `gh issue view` and never `--comments`.
- **Never name a second issue for a subagent to read.** If a decision from another ticket matters,
  the orchestrating session — which has already read it — states the fact in the prompt. One
  sentence beats ten kilobytes of thread.
- Never ask for a repository survey or for sibling stages to be read "for orientation". If one
  sibling's conventions genuinely matter, name that one file.
- A **pinpoint change** (one assertion, one guard, one docstring clause) gets a pinpoint prompt:
  the exact `file:line`, the exact change, and an explicit instruction not to explore. It does not
  get a ticket-shaped prompt.

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

- One implementation subagent **per ticket**; several may run at once when the tickets are
  independent (see "Parallel execution"). Wherever a `Depends on` edge exists, the tickets are
  stacked and strictly sequential.
- Concurrent implementation subagents never share a working directory: each one runs in its own
  `git worktree`, so one ticket's uncommitted state can never leak into another's.
- **Preparing the worktree is the orchestrating session's job, and it is sequential.** Run
  `make setup` in every fresh worktree *before* launching a subagent into it, one worktree after the
  other. Two subagents once stalled at the 600 s watchdog having done no work at all: both ran
  `uv sync` at the same moment and blocked on the shared uv cache lock.
- Implementation subagents are told **not to use WebSearch or WebFetch**. The scope is in the
  repository and in the issues; the one time a subagent went researching, that was its whole budget.
- A subagent is never reused for a second ticket, and never for the review of its own work. If a
  ticket comes back `changes-requested`, the fixes may run in a **new** implementation subagent
  with the review findings in the prompt.
- **A trivial, pinpoint change may be made by the orchestrating session itself** — one assertion,
  one docstring clause, a number updated, or the correction of a claim a review proved false —
  provided that when there is behaviour to pin it writes the test, watches it fail, and says so,
  and that when the change is **prose only** it says that too, explicitly, so nobody looks for the
  missing test. The "no production code without a subagent" rule exists to keep *design* out of
  the orchestrating context, not to ritualise one-liners. Anything touching production behaviour
  still goes to an implementation subagent.

  **This is the single largest lever on the cost of a session, and it is measured.** A subagent
  that makes zero tool calls and answers with one word costs **33k tokens**; the fixed cost per
  agent therefore dominates everything prompt economy can save. Four pinpoint corrections were
  made directly in one session — three false or stale docstring sentences and one test-oracle
  guard — and not spawning an agent for them saved ~130k tokens, more than the whole
  orchestration split saves across thirteen agents. When the choice is "a fresh 33k context" or
  "three lines and an honest note", take the three lines.

  **Never merge a claim you know to be false, though.** Two of those four corrections were
  `Raises`/evidence sentences that a review had just disproved; leaving them for the follow-up
  ticket would have put a false contract on the parent branch in the meantime.
- If a subagent needs something outside its ticket (a missing fixture is fine, another ticket's
  scope is not), it stops and reports it under `Scope/blocked`.

## Definition of done — the orchestrating session's half

The engineering half (criteria met, tests for every criterion and edge case, docstrings and
`README.md` updated, no leftover TODOs, decisions reported with their reasons) is in `CLAUDE.md`,
because a subagent owes it. These five are owed by this session and by nobody else:

- [ ] Implementation ran in its own subagent, the review in a second one.
- [ ] Decisions from the handover report are posted on the issue, and cross-cutting ones added to
      the "Decision log" (#44).
- [ ] PR opened against the correct base branch, title `T3.1: <name>`, body `Closes #<issue>`.
- [ ] Review passed (see "Review step") and merged into the parent branch (sub-ticket) or `main`
      (parent ticket).
- [ ] Issue closed with a comment naming the merge commit SHA; card moved to `Done`; parent task
      list ticked; the `blocked` label removed from the tickets this one unblocks (see
      "Closing a ticket").

## Closing a ticket

In this order, so that the board, the labels and the dependency graph stay consistent:

1. Merge the PR into its base branch, then delete the branch **on `origin` as well as locally** — a
   squash merge leaves the head branch behind, and `git branch -d` only touches the local one. Watch
   the matching trap: after a squash merge the original commits are **not** ancestors of the base, so
   `git merge-base --is-ancestor` reports every merged branch as unmerged. The PR state says whether
   a branch is done; ancestry does not.
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

## Review step (second agent, lightweight)

When the implementation subagent reports green, replace `in-progress` with `review`, move the card
to `In review`, and run the review as a **second, separate subagent call** — never the context that
wrote the code.

**Hand the reviewer what you already know, and measure the test count yourself.** Anything the
orchestrating session has verified — a hash, a byte comparison, a count — goes into the prompt as
*verified*, so the review does not spend a worktree and a `make setup` re-deriving it. In
particular: **measure the base commit's test count once when you prepare the worktree** and state
it in both prompts. Two reviews each built a second worktree, ran `make setup` and ran the full
suite twice purely to reconcile a handover's base count, and both times the handover was simply
wrong — a cost the orchestrating session can remove with one command it is already positioned to
run.

**CI on the PR is the authoritative gate, not a handover report.** Run `gh pr checks <pr>` twice:
once after opening the PR and **before** launching the review subagent, and again immediately
before merging. A red run sends the ticket back to `in-progress` no matter what the handover report
or the review said — this has already happened once (#45, PR #50), where two independent subagents
reported `make test` green on a commit whose lint stage was failing. For the same reason, prefer
verifying a claim in a **fresh clone or `git worktree`** rather than in the working checkout, where
cache state and an editable install hide failures.

The review itself:

Spawn it with `subagent_type: idc-reviewer`, whose definition carries the method and the report
format. The prompt stays specific to the ticket and obeys the same context budget as above:

> *"Review branch `<branch>` against the acceptance criteria of issue #<n>. Worktree: `<path>`.
> Base: `<base>`. <The claims the implementation made, listed so they can be tested rather than
> read.> <Any fact from another ticket that the review needs, stated here.>"*

**List the claims to be tested.** The reviews that have found real defects on this project are the
ones handed a numbered list of assertions to disprove, not the ones asked to "review the branch".

The review is deliberately **quick**: it checks that tests are green, that every
acceptance criterion is covered by a test, and that nothing is obviously wrong or out of
scope. It is not a design review. The verdict is posted as a comment on the issue
(prefix `**Review:**`).

- **approved** → merge the PR into its base branch (squash for sub-tickets, merge commit
  for parent → `main`), then run "Closing a ticket" above. **Whenever an original commit message
  must not survive, the squash message has to be written by hand**: GitHub's default body
  concatenates the originals, so rewriting the commits (`git filter-branch --msg-filter` and a
  force-push) is only half the job — in T2.4 the removed sentences would have reappeared in the
  squash body.
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
