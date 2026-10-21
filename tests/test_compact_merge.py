"""Merging an extended chain's earlier Solution with its new writes (T6.4, #29).

T6.3 rewrites the block of a chain this run gave a new chunk to, and it rewrote
it from the new chunks' `Resolution` alone — so every file the chain's earlier
`Solution` section named fell out of the summary. These tests pin what the
rewrite holds instead: the union of the two sides, with the new write winning
per file.

The realistic input is the one the sibling stage uses,
`tests/fixtures/already_compacted_block.json`: its block's `C2` names
`src/api/limiter.py`, and the one new chunk writes that file *and*
`src/api/middleware.py`. That is the ticket's own case — old solution has A, a
new chunk writes A and B — and it is a fixture rather than an inline transcript.

The old sides that a fixture cannot provide are written out as block text here:
a hand-written block's bullets carry prose the writer never emits, two spellings
of one path, and a fence whose *content* holds a bullet line. Those are block
Markdown, not transcripts, and a few lines of it is the whole input.

No test here calls a model, and none reads a transcript of this machine.
"""

import ast
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

from fixture_corpus import CORPUS
from idc.chunk import chunk_session
from idc.compact import (
    DECISIONS_HEADER,
    EMPTY_SECTION,
    REJECTED_HEADER,
    SOLUTION_HEADER,
    VERIFICATION_HEADER,
    FakeNarrativeClient,
    compact_chain,
    compact_incrementally,
    merge_solution_bodies,
    merged_solution_section,
    solution_section,
    verification_section,
)
from idc.parse import join_blocks, parse_blocks, parse_chains, split_blocks
from idc.resolve import Resolution
from idc.resolve import resolve as resolve_chain
from idc.schema import Chain

SEEDED = "already_compacted_block"
"""The fixture whose transcript opens with an earlier run's two-chain block."""

EXTENDED = "C2"
"""The chain the fixture's one new chunk continues, per its gold note."""

UNTOUCHED = "C1"
"""The chain of the same block the new turn has nothing to do with."""

OVERWRITTEN = "src/api/limiter.py"
"""The file the block's `C2` already named and the new chunk writes again."""

ADDED = "src/api/middleware.py"
"""The file only the new chunk writes."""

EMPTY_RESOLUTION = Resolution(solution_files=(), verification=None, path_call_ids=())
"""What a chain resolves to when it wrote no file and ran no command."""


def narrative(label: str) -> str:
    """Return a well-formed narrative response for a chain with that intent."""
    return (
        f"## Intent: {label}\n"
        f"{DECISIONS_HEADER}\n"
        f"- The limiter keeps its in-process bucket.\n"
        f"{REJECTED_HEADER}\n"
        f"- A shared Redis bucket, deferred again.\n"
    )


def block_text() -> str:
    """Return the compacted block the fixture's transcript carries."""
    session = CORPUS.session(SEEDED)
    blocks = [m.content for m in session.messages if m.content.startswith("## [C")]
    assert len(blocks) == 1, f"{SEEDED} is used as the one-block fixture, found {len(blocks)}"
    return blocks[0]


def seeded_run() -> tuple[list[Chain], Callable[[Chain], Resolution]]:
    """Return the fixture run's chains — `C1` carried, `C2` extended — and its resolver.

    The chains are built from the block and the chunk list rather than through
    `idc.classify`: which chain the new chunk continues is the fixture's gold
    note, and asserting the classifier agrees is the sibling stage's test.
    """
    session = CORPUS.session(SEEDED)
    chunks = chunk_session(session)
    carried, extended = parse_chains(block_text())
    chains = [
        carried,
        Chain(
            id=extended.id,
            label=extended.label,
            chunk_ids=[*extended.chunk_ids, *(chunk.id for chunk in chunks)],
            compacted=True,
            compacted_text=extended.compacted_text,
        ),
    ]

    def resolution_of(chain: Chain) -> Resolution:
        return resolve_chain(session, chunks, chain)

    return chains, resolution_of


def new_resolution() -> Resolution:
    """Return what the fixture's one new chunk resolves to: the new side of the merge."""
    session = CORPUS.session(SEEDED)
    chunks = chunk_session(session)
    return resolve_chain(
        session,
        chunks,
        Chain(
            id=EXTENDED,
            label="Add a per-token rate limiter to the API",
            chunk_ids=[chunk.id for chunk in chunks],
        ),
    )


def body_of(section: str) -> str:
    """Return a rendered section without its header line."""
    return section.split("\n", 1)[1]


def old_block(solution_body: str, chain_id: str = EXTENDED) -> str:
    """Return a block whose `Solution` section holds exactly `solution_body`."""
    return (
        f"## [{chain_id}] Intent: Add a per-token rate limiter to the API\n"
        f"{DECISIONS_HEADER}\n{EMPTY_SECTION}\n"
        f"{REJECTED_HEADER}\n{EMPTY_SECTION}\n"
        f"{SOLUTION_HEADER}\n{solution_body}"
        f"{VERIFICATION_HEADER}\n{EMPTY_SECTION}\n"
    )


def extended_chain(block: str, chain_id: str = EXTENDED) -> Chain:
    """Return a chain carrying that block and one chunk of the transcript at hand."""
    return Chain(
        id=chain_id,
        label="Add a per-token rate limiter to the API",
        chunk_ids=[f"compacted:{chain_id}", "K1"],
        compacted=True,
        compacted_text=block,
    )


def merged_body(solution_body: str, resolution: Resolution | None = None) -> str:
    """Return the merged `Solution` body of that old section and the new resolution."""
    section = merged_solution_section(
        extended_chain(old_block(solution_body)),
        new_resolution() if resolution is None else resolution,
    )
    assert section.startswith(f"{SOLUTION_HEADER}\n")
    return body_of(section)


KEPT_BULLET = (
    "- `src/billing/export.py` — `_format_row` converts before formatting:\n"
    "```python\n"
    '"booked_at": entry.booked_at.astimezone(UTC).isoformat(),\n'
    "```\n"
)
"""An old bullet of a file no new chunk touches — prose and a tagged fence.

The shape `tests/fixtures/already_compacted_block.json` carries, which is the
realistic one: a hand or an earlier version of the writer states on the bullet
line what the writer itself never emits.
"""

NOTES_BULLET = (
    "- `docs/notes.md` — the checklist the reviewer asked for:\n"
    "```markdown\n"
    f"- `{OVERWRITTEN}`\n"
    f"- `{ADDED}`\n"
    "```\n"
)
"""An old bullet whose fenced content is itself a list of the two new paths.

Inside a fence a `- ` line is content, which is how `idc.parse` reads a block. A
merge that cut the body on every flush bullet instead would cut this one into
three, overlay the two inner lines away and leave the fence unbalanced.
"""


def test_the_fixture_is_the_tickets_own_case() -> None:
    """Old solution names A; the new chunk writes A and B. Pin the input itself."""
    carried, extended = parse_chains(block_text())
    old_body = parse_blocks(str(extended.compacted_text))[0].body(SOLUTION_HEADER)

    assert f"- `{OVERWRITTEN}`" in old_body
    assert ADDED not in old_body
    assert [entry.path for entry in new_resolution().solution_files] == [OVERWRITTEN, ADDED]
    assert carried.id == UNTOUCHED
    assert extended.id == EXTENDED


def test_an_extended_block_holds_the_new_a_and_the_new_b() -> None:
    """The ticket's own test, through the stage that writes the run.

    It is the criterion as the ticket states it and it is **not** on its own a
    test of the merge: the block's `C2` names one file and the new chunk writes
    that same file, so a rewrite from the new resolution alone — what T6.3 did —
    satisfies it too, as the stub this test first ran against confirmed. What
    fails without the merge is the sibling below, where the old section also
    names a file the new chunk never touches.
    """
    chains, resolution_of = seeded_run()
    client = FakeNarrativeClient([narrative("Give the rate limiter a Retry-After header")])

    written = compact_incrementally(chains, resolution_of=resolution_of, client=client)

    rewritten = {chain.id: str(chain.compacted_text) for chain in parse_chains(written)}[EXTENDED]
    solution = parse_blocks(rewritten)[0].body(SOLUTION_HEADER)
    assert f"- `{OVERWRITTEN}`\n" in solution
    assert f"- `{ADDED}`\n" in solution
    assert "@property" in solution
    assert "class TokenBucket:" not in solution


def test_the_rewritten_block_states_both_files_with_the_new_content() -> None:
    """Both paths are present, and the overwritten one carries this run's text."""
    resolution = new_resolution()
    body = merged_body(f"- `{OVERWRITTEN}`\n```\nclass TokenBucket:\n```\n")

    assert f"- `{OVERWRITTEN}`\n" in body
    assert f"- `{ADDED}`\n" in body
    assert "class TokenBucket:" not in body
    assert body == body_of(solution_section(resolution))


def test_a_file_only_the_old_section_names_survives_prose_and_all() -> None:
    """A kept file's bullet is copied byte for byte — the prose is not the writer's."""
    body = merged_body(KEPT_BULLET)

    assert KEPT_BULLET in body
    assert f"- `{OVERWRITTEN}`\n" in body
    assert f"- `{ADDED}`\n" in body


def test_a_file_only_the_new_resolution_names_appears() -> None:
    """B is in the rewritten section even though no earlier block named it."""
    body = merged_body(KEPT_BULLET)

    assert f"- `{ADDED}`\n" in body


def test_the_old_section_is_the_spine_and_new_files_are_appended() -> None:
    """Order: a kept file holds its place, an overwritten one holds its place too."""
    old = f"- `{OVERWRITTEN}`\n```\nclass TokenBucket:\n```\n{KEPT_BULLET}"

    body = merged_body(old)

    assert body.index(f"- `{OVERWRITTEN}`\n") < body.index(KEPT_BULLET)
    assert body.index(KEPT_BULLET) < body.index(f"- `{ADDED}`\n")


def test_an_overwritten_files_old_bullet_is_gone_entirely() -> None:
    """Last write wins means the old bullet's prose goes with its content.

    The prose described the content that no longer stands, so keeping it under
    the new fence would state a reading of the file the new write contradicts.
    """
    old = f"- `{OVERWRITTEN}` — the first cut, 60 per minute:\n```\nclass TokenBucket:\n```\n"

    body = merged_body(old)

    assert "the first cut" not in body
    assert "class TokenBucket:" not in body
    assert f"- `{OVERWRITTEN}`\n" in body


def test_two_spellings_of_one_path_stay_two_files() -> None:
    """Paths are not normalized here: `./src/a.py` and `src/a.py` are two files.

    `idc.resolve.SolutionFile.path` is the path exactly as the agent spelled it,
    and deciding that two spellings are one file is a rule of the resolver's
    stage, not of the writer's. Both bullets therefore survive.
    """
    old = f"- `./{OVERWRITTEN}`\n```\nclass TokenBucket:\n```\n"

    body = merged_body(old)

    assert f"- `./{OVERWRITTEN}`\n" in body
    assert "class TokenBucket:" in body
    assert f"- `{OVERWRITTEN}`\n" in body


def test_a_bullet_line_inside_a_fence_is_content() -> None:
    """The cut agrees with how `idc.parse` reads a block: a fenced `- ` is content."""
    body = merged_body(NOTES_BULLET)

    assert NOTES_BULLET in body
    assert body.count("```markdown\n") == 1


def test_a_fence_run_behind_a_form_feed_closes_nothing() -> None:
    """The line-ending rule is the parser's: a form feed does not end a line.

    `idc.parse` splits lines on LF, CRLF and a bare CR only, and documents what
    `str.splitlines` would reopen: a closing run written behind a form feed
    would look like a line of its own, close the fence, and hand the lines
    beneath it to the splitter. Here that truncates the kept file's content at
    the form feed and overlays the rest away as the overwritten file — which
    `odd in body` alone does **not** catch, because what is overlaid in begins
    with the same two lines the cut dropped. Counting the path is what catches
    it: the right answer states it twice, once inside this file's fence and once
    as the file the new chunk wrote.
    """
    odd = f"- `docs/odd.md`\n```\n\f```\n- `{OVERWRITTEN}`\n```\n"

    body = merged_body(odd)

    assert odd in body
    assert body.count(f"- `{OVERWRITTEN}`") == 2


def test_an_indented_bullet_belongs_to_the_bullet_above_it() -> None:
    """Only a flush `- ` opens a bullet — a hand's nested list is not a file."""
    old = f"{KEPT_BULLET}  - and the column order stays as it is\n"

    body = merged_body(old)

    assert f"{KEPT_BULLET}  - and the column order stays as it is\n" in body


def test_an_empty_old_section_contributes_nothing() -> None:
    """`- (none)` names no file, so it is not carried into the merged section."""
    body = merged_body(f"{EMPTY_SECTION}\n")

    assert EMPTY_SECTION not in body
    assert body == body_of(solution_section(new_resolution()))


def test_an_extension_that_wrote_nothing_keeps_the_old_section() -> None:
    """A chain extended by a chunk that wrote no file still states its files."""
    body = merged_body(KEPT_BULLET, EMPTY_RESOLUTION)

    assert body == KEPT_BULLET


def test_two_empty_sides_are_the_empty_section() -> None:
    """A section is never blank: nothing on either side renders the placeholder."""
    body = merged_body(f"{EMPTY_SECTION}\n", EMPTY_RESOLUTION)

    assert body == f"{EMPTY_SECTION}\n"


def test_the_bodies_merge_without_a_chain() -> None:
    """`merge_solution_bodies` is the core, and takes the two bodies as they stand."""
    new = body_of(solution_section(new_resolution()))

    assert merge_solution_bodies(KEPT_BULLET, new) == merged_body(KEPT_BULLET)
    assert merge_solution_bodies("", "") == f"{EMPTY_SECTION}\n"


def test_the_merged_section_parses_back_as_a_solution_section() -> None:
    """The rewritten block is a block: it parses, and its section is the merged one."""
    section = merged_solution_section(extended_chain(old_block(NOTES_BULLET)), new_resolution())
    block = old_block(body_of(section))

    parsed = parse_blocks(block)

    assert len(parsed) == 1
    assert parsed[0].body(SOLUTION_HEADER) == body_of(section)


def test_the_rewritten_run_round_trips_through_the_parser() -> None:
    """What this run writes, the next run reads back and copies byte for byte."""
    chains, resolution_of = seeded_run()
    client = FakeNarrativeClient([narrative("Give the rate limiter a Retry-After header")])

    written = compact_incrementally(chains, resolution_of=resolution_of, client=client)
    again = compact_incrementally(
        parse_chains(written),
        resolution_of=_unexpected,
        client=FakeNarrativeClient([]),
    )

    assert again == written


def test_the_verification_is_the_new_runs_alone() -> None:
    """Only one run can prove the chain, and it is the one that saw the new writes."""
    chains, resolution_of = seeded_run()
    client = FakeNarrativeClient([narrative("Give the rate limiter a Retry-After header")])

    written = compact_incrementally(chains, resolution_of=resolution_of, client=client)

    rewritten = {chain.id: str(chain.compacted_text) for chain in parse_chains(written)}[EXTENDED]
    assert parse_blocks(rewritten)[0].body(VERIFICATION_HEADER) == body_of(
        verification_section(new_resolution())
    )
    assert "tests/api/test_limiter.py" not in rewritten


def test_an_extension_that_ran_nothing_states_no_verification() -> None:
    """The old run is dropped rather than re-stated: it never saw the new writes."""
    chain = extended_chain(split_blocks(block_text())[1])
    client = FakeNarrativeClient([narrative("Give the rate limiter a Retry-After header")])

    block = compact_incrementally(
        [chain], resolution_of=lambda _chain: EMPTY_RESOLUTION, client=client
    )

    assert parse_blocks(block)[0].body(VERIFICATION_HEADER) == f"{EMPTY_SECTION}\n"


def test_a_chain_this_run_opened_renders_from_its_resolution_alone() -> None:
    """Nothing changed for a chain that was never compacted: there is no old side."""
    opened = Chain(id="C3", label="Add a retry budget to the HTTP client", chunk_ids=["K1"])
    resolution = new_resolution()
    script = [narrative("Add a retry budget")]

    written = compact_incrementally(
        [opened], resolution_of=lambda _chain: resolution, client=FakeNarrativeClient(list(script))
    )

    assert written == compact_chain(opened, resolution, FakeNarrativeClient(list(script)))
    assert merged_solution_section(opened, resolution) == solution_section(resolution)


def test_a_carried_chain_is_still_copied_for_nothing() -> None:
    """The copy path is untouched: no model call, no resolution, the same bytes."""
    chains, resolution_of = seeded_run()
    asked: list[str] = []

    def recording(chain: Chain) -> Resolution:
        asked.append(chain.id)
        return resolution_of(chain)

    client = FakeNarrativeClient([narrative("Give the rate limiter a Retry-After header")])
    written = compact_incrementally(chains, resolution_of=recording, client=client)

    assert split_blocks(written)[0] == split_blocks(block_text())[0]
    assert asked == [EXTENDED]
    assert [call.chain_id for call in client.calls] == [EXTENDED]


def test_the_same_inputs_write_the_same_bytes() -> None:
    """Determinism: two runs over one script, through the merge, one result."""
    script = [narrative("Give the rate limiter a Retry-After header")]
    first_chains, first_resolver = seeded_run()
    second_chains, second_resolver = seeded_run()

    first = compact_incrementally(
        first_chains, resolution_of=first_resolver, client=FakeNarrativeClient(list(script))
    )
    second = compact_incrementally(
        second_chains, resolution_of=second_resolver, client=FakeNarrativeClient(list(script))
    )

    assert first == second
    assert first == join_blocks(split_blocks(first))


def _unexpected(chain: Chain) -> Resolution:
    """A resolver that fails: handed to runs in which no chain may be resolved."""
    raise AssertionError(f"chain {chain.id!r} was resolved, and it has nothing to resolve")


# --- the wiring, which the unit tests above cannot reach ------------------------


def test_the_stage_rewrites_from_the_merged_section_and_not_the_resolution() -> None:
    """The rewrite path uses the merge — the half that fails silently without a test.

    Every discriminating test above calls `merged_solution_section` directly, and
    the one that goes through the stage uses the fixture's own old section, which
    names only the file the new chunk rewrites — so the stage-level criterion is
    satisfied by rendering the new resolution alone, which is what the previous
    ticket did. A review removed the merge from the rewrite path and the whole
    suite stayed green.

    This test puts a file in the old section that the new chunk never touches, so
    `solution=None` on the rewrite path drops it and the assertion fails.
    """
    session = CORPUS.session(SEEDED)
    chunks = chunk_session(session)
    (extended,) = parse_chains(old_block(KEPT_BULLET))
    chains = [
        Chain(
            id=extended.id,
            label=extended.label,
            chunk_ids=[*extended.chunk_ids, *(chunk.id for chunk in chunks)],
            compacted=True,
            compacted_text=extended.compacted_text,
        )
    ]
    client = FakeNarrativeClient([narrative("Give the rate limiter a Retry-After header")])

    written = compact_incrementally(
        chains,
        resolution_of=lambda chain: resolve_chain(session, chunks, chain),
        client=client,
    )

    rewritten = {chain.id: str(chain.compacted_text) for chain in parse_chains(written)}[EXTENDED]
    solution = parse_blocks(rewritten)[0].body(SOLUTION_HEADER)
    assert KEPT_BULLET in solution, "the old section's untouched file was dropped by the rewrite"
    assert f"- `{ADDED}`\n" in solution
    assert solution.index("src/billing/export.py") < solution.index(ADDED)


SRC_SOURCES = sorted(Path(__file__).resolve().parents[1].joinpath("src/idc").rglob("*.py"))
"""Every source file of the tool — what the one-definition census below reads."""

READER_PACKAGE = "src/idc/parse/"
"""Where the line scan is defined; every other file has to import it from there."""

PATTERN_SPELLINGS = ("`{3,}", r"(?<=\n)", r"(?<=\r)")
"""The pieces the two patterns are spelled with, as regex source.

T8.3 made the scan public because these were spelled twice: a stale copy of
either reopens a defect three review rounds of #34 paid for. A grep is blunt and
is the point — it catches the second spelling at the moment it is written,
whereas a behavioural test only catches it once the two copies disagree.
"""

COMPACT_SOURCES = sorted(
    Path(__file__).resolve().parents[1].joinpath("src/idc/compact").glob("*.py")
)
"""Every source file of the writer — what the import-cycle census below reads."""


def module_level_imports(source: str) -> set[str]:
    """Return the modules a source file imports at import time.

    Only the top-level statements are read, because only those run when the
    module is imported: an import inside a function body is exactly the deferral
    this package uses to stay clear of the cycle.
    """
    imported: set[str] = set()
    for node in ast.parse(source).body:
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.add(node.module)
    return imported


def test_the_line_scan_has_one_definition_in_the_source_tree() -> None:
    """No second spelling of either pattern outside `idc.parse`.

    This replaces the pattern-string comparison of #29, which pinned that two
    copies agreed. There are no longer two copies to compare: `idc.parse`
    exports `split_lines`, `fence_opening` and `closes_fence`, and every other
    module imports them.
    """
    outside = [path for path in SRC_SOURCES if READER_PACKAGE not in path.as_posix()]

    assert outside, "the census found no file to search, so finding nothing proves nothing"
    respelled = sorted(
        (path.name, spelling)
        for path in outside
        for spelling in PATTERN_SPELLINGS
        if spelling in path.read_text(encoding="utf-8")
    )
    assert respelled == []


def test_the_merge_opens_its_fences_through_the_readers_public_name(monkeypatch) -> None:
    """The cut runs the reader's own fence scan, not a copy of its rule.

    Stubbing `idc.parse.fence_opening` leaves `idc.parse.blocks` untouched — the
    reader holds its own reference — so this is discriminating in the direction
    #29 could not pin: a module that re-spelled the rule would keep cutting the
    fenced bullets correctly with the public name stubbed away.
    """
    assert NOTES_BULLET in merged_body(NOTES_BULLET)

    monkeypatch.setattr("idc.parse.fence_opening", lambda line: None)

    assert NOTES_BULLET not in merged_body(NOTES_BULLET)


def test_the_merge_cuts_its_lines_through_the_readers_public_name(monkeypatch) -> None:
    """The same for the line split, pinned on the form-feed body.

    With `str.splitlines` in its place the run behind the form feed becomes a
    line of its own, closes the fence it is inside of, and the bullet beneath it
    is cut out of the kept file's content and overlaid away.
    """
    odd = f"- `docs/odd.md`\n```\n\f```\n- `{OVERWRITTEN}`\n```\n"
    assert odd in merged_body(odd)

    monkeypatch.setattr("idc.parse.split_lines", lambda text: text.splitlines(True))

    assert odd not in merged_body(odd)


def test_the_writer_imports_the_reader_nowhere_at_module_level() -> None:
    """`idc.parse.blocks` imports this package, so this direction is a cycle.

    Every import of the reader in here is deferred into the function that needs
    it, which is the precedent `idc.compact.incremental` set and states.
    """
    assert COMPACT_SOURCES, "the census found no file to read"

    offenders = sorted(
        path.name
        for path in COMPACT_SOURCES
        for module in module_level_imports(path.read_text(encoding="utf-8"))
        if module == "idc.parse" or module.startswith("idc.parse.")
    )
    assert offenders == []


@pytest.mark.parametrize("first", ["idc.compact", "idc.parse"])
def test_either_package_imports_first(first: str) -> None:
    """An import cycle is a crash at import time, not a failing assertion.

    The census above reads the source; this runs it. Each order is a fresh
    interpreter, because by the time this test module is collected both packages
    are long imported and either order would pass in-process.
    """
    second = "idc.parse" if first == "idc.compact" else "idc.compact"

    result = subprocess.run(
        [sys.executable, "-c", f"import {first}\nimport {second}\nprint('imported')"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "imported"


def test_a_path_named_twice_on_the_old_side_is_overwritten_as_one_file() -> None:
    """Last write wins per *file*, so both old bullets go, not just the first.

    The writer renders one bullet per kept write, so a hand-edited block can name
    one file twice. When the new resolution rewrites that file, **every** old
    bullet for it is replaced — the merge is per path, not per bullet. A docstring
    once promised that a side's repeated bullets are all kept, which is true only
    while the file is not overwritten.
    """
    twice = f"- `{OVERWRITTEN}`\n```\nfirst\n```\n- `{OVERWRITTEN}`\n```\nsecond\n```\n"

    merged = merged_solution_section(extended_chain(old_block(twice)), new_resolution())

    assert "first" not in merged
    assert "second" not in merged
    assert merged.count(f"- `{OVERWRITTEN}`") == 1


def test_a_path_named_twice_and_not_overwritten_keeps_both_bullets() -> None:
    """The other half of the same rule: untouched repeats are kept as they stand."""
    twice = KEPT_BULLET + KEPT_BULLET

    merged = merged_solution_section(extended_chain(old_block(twice)), new_resolution())

    assert merged.count("- `src/billing/export.py`") == 2
