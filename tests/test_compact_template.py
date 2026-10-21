"""The compaction template and its two deterministic sections (T6.1, #26).

The oracle of the golden tests is `tests/fixtures/gold/compact/<name>.md`: one
file per gold-labelled transcript, holding the blocks its chains must render to,
compared character for character. The files were authored from the fixture
documents themselves — the texts are the `content` and replacement arguments the
fixtures carry, the kept writes are the ones the resolver reports — and never
from this renderer's output, which is what makes the comparison a test rather
than a snapshot of whatever the code happens to do.

The edge cases the corpus does not hold (a chain that wrote nothing, a file
every write to which failed, a verification that ended red, a result the
transcript never recorded, content carrying a fence of its own) are built by
replacing one field of a *resolved* corpus chain with
`dataclasses.replace`, so even they keep the stage's own objects and no test
here fabricates a `VerificationMatch` or a `Chunk`.
"""

import dataclasses
import re

import pytest

from fixture_corpus import CORPUS, REPLACEMENT_PAIRS
from idc.chunk import chunk_session
from idc.classify import FakeAdapter, classify, gold_script
from idc.compact import (
    DECISIONS_HEADER,
    EMPTY_SECTION,
    ERROR_NOTE,
    NO_COMMAND_NOTE,
    NO_CONTENT_NOTE,
    NO_RESULT_NOTE,
    REJECTED_HEADER,
    SECTION_HEADERS,
    SOLUTION_HEADER,
    UNKNOWN_EXIT_NOTE,
    VERIFICATION_HEADER,
    chain_heading,
    render_block,
    solution_section,
    verification_section,
)
from idc.resolve import Resolution, content_of, resolve
from idc.schema import Chain, ToolCall, ToolCategory, ToolResult

COMPACT_NAMES = CORPUS.compact_names()
"""Every transcript the corpus holds golden compaction Markdown for."""

GOLD_NAMES = CORPUS.gold_names()
"""Every transcript the corpus holds gold chain labels for."""

EMPTY_RESOLUTION = Resolution(solution_files=(), verification=None, path_call_ids=())
"""A chain that wrote no file and ran no verification."""

ONE_CHAIN = "single_intent"
"""The one-chunk, one-edit, one-run fixture the single-case tests read."""

WRITE_THEN_EDITS = "write_then_two_edits"
"""The fixture whose second file is written whole and then corrected twice."""

ALL_WRITES_FAILED = "rewritten_after_failed_edit"
"""The fixture that holds a write the agent saw fail, next to one it kept."""

FENCE_LINE = re.compile(r"^`{3,}")
"""A line that opens or closes a fenced block — three backticks or more."""


def chains_of(name: str) -> tuple[Chain, ...]:
    """Return a fixture's chains, built by the pipeline from its gold script."""
    session = CORPUS.session(name)
    chunks = chunk_session(session)
    return tuple(classify(session, chunks, FakeAdapter(gold_script(CORPUS.gold(name)))))


def resolved(name: str) -> tuple[tuple[Chain, Resolution], ...]:
    """Return a fixture's chains with what each of them resolved to."""
    session = CORPUS.session(name)
    chunks = chunk_session(session)
    return tuple(
        (chain, resolve(session, chunks, chain))
        for chain in classify(session, chunks, FakeAdapter(gold_script(CORPUS.gold(name))))
    )


def only(name: str) -> tuple[Chain, Resolution]:
    """Return a single-chain fixture's chain and resolution."""
    pairs = resolved(name)
    assert len(pairs) == 1, f"{name} is used as a single-chain fixture"
    return pairs[0]


def rendered(name: str) -> str:
    """Return every block of a fixture, in chain order, one blank line apart."""
    return "\n".join(render_block(chain, resolution) for chain, resolution in resolved(name))


def outside_fences(block: str) -> list[str]:
    """Return the lines of a block that no fenced code block encloses."""
    lines: list[str] = []
    fence: str | None = None
    for line in block.splitlines():
        if fence is None and FENCE_LINE.match(line):
            fence = FENCE_LINE.match(line).group()
            continue
        if fence is not None:
            if line == fence:
                fence = None
            continue
        lines.append(line)
    return lines


def fenced_texts(section: str) -> list[str]:
    """Return the content of every fenced block of a section, in order."""
    texts: list[str] = []
    fence: str | None = None
    held: list[str] = []
    for line in section.splitlines(keepends=True):
        bare = line.rstrip("\n")
        if fence is None:
            if FENCE_LINE.match(bare):
                fence = bare
            continue
        if bare == fence:
            texts.append("".join(held))
            fence, held = None, []
            continue
        held.append(line)
    assert fence is None, "a fenced block was left open"
    return texts


def fenced(text: str) -> str:
    """Return `text` the way a fence holds it: with the closing newline it needs.

    The renderer completes a text that does not end in a newline so that the
    closing fence keeps a line of its own, which is pinned on its own below;
    a test comparing fenced texts states the same completion rather than
    repeating the rule as an expectation.
    """
    return text if not text or text.endswith("\n") else f"{text}\n"


def replacement_of(call: ToolCall) -> str | None:
    """Return the text an edit call replaces with, read through the corpus' own pairs."""
    for _, replacement_key in REPLACEMENT_PAIRS:
        value = call.args.get(replacement_key)
        if isinstance(value, str):
            return value
    return None


def with_run(resolution: Resolution, run_field: str, value: object) -> Resolution:
    """Return the resolution with one field of its kept verification run replaced."""
    verification = resolution.verification
    assert verification is not None, "the fixture chain is used for its verification"
    run = dataclasses.replace(verification.run, **{run_field: value})
    return dataclasses.replace(resolution, verification=dataclasses.replace(verification, run=run))


def with_write_args(resolution: Resolution, index: int, args: dict[str, object]) -> Resolution:
    """Return the resolution with the arguments of one kept write replaced."""
    entry = resolution.solution_files[index]
    kept = entry.kept[0]
    call = dataclasses.replace(
        kept, call=ToolCall(id=kept.id, name=kept.name, category=ToolCategory.WRITE, args=args)
    )
    patched = (
        dataclasses.replace(entry, write=call, edits=())
        if entry.write is not None
        else dataclasses.replace(entry, edits=(call,))
    )
    files = tuple(
        patched if position == index else other
        for position, other in enumerate(resolution.solution_files)
    )
    return dataclasses.replace(resolution, solution_files=files, path_call_ids=())


# --- the corpus, byte for byte -------------------------------------------------


def test_the_corpus_holds_golden_markdown_for_every_gold_labelled_transcript():
    # The golden table below is parametrized over this list; an empty one would
    # turn every golden test into a silent skip.
    assert COMPACT_NAMES
    assert COMPACT_NAMES == GOLD_NAMES


def test_the_compact_gold_directory_holds_nothing_but_markdown():
    unexpected = [
        path.name for path in sorted(CORPUS.compact_directory.iterdir()) if path.suffix != ".md"
    ]

    assert unexpected == []


@pytest.mark.parametrize("name", COMPACT_NAMES)
def test_every_fixture_chain_renders_its_golden_markdown(name):
    assert rendered(name) == CORPUS.compact_markdown(name)


@pytest.mark.parametrize("name", COMPACT_NAMES)
def test_rendering_a_fixture_twice_gives_the_same_bytes(name):
    assert rendered(name) == rendered(name)


# --- the template --------------------------------------------------------------


def test_the_four_section_headers_are_the_ones_the_template_names():
    assert DECISIONS_HEADER == "### Decisions & constraints"
    assert REJECTED_HEADER == "### Rejected approaches"
    assert SOLUTION_HEADER == "### Solution"
    assert VERIFICATION_HEADER == "### Verification"
    assert SECTION_HEADERS == (
        DECISIONS_HEADER,
        REJECTED_HEADER,
        SOLUTION_HEADER,
        VERIFICATION_HEADER,
    )


def test_the_empty_section_placeholder_is_one_line():
    assert EMPTY_SECTION == "- (none)"


def test_the_chain_heading_names_the_chain_id_and_its_label():
    chain, _ = only(ONE_CHAIN)

    assert chain_heading(chain) == f"## [{chain.id}] Intent: {chain.label}"


def test_a_block_opens_with_the_heading_and_holds_the_headers_in_template_order():
    chain, resolution = only(ONE_CHAIN)

    lines = outside_fences(render_block(chain, resolution))

    assert lines[0] == chain_heading(chain)
    assert [line for line in lines if line.startswith("#")] == [
        chain_heading(chain),
        *SECTION_HEADERS,
    ]


def test_a_block_ends_with_exactly_one_newline():
    chain, resolution = only(ONE_CHAIN)

    block = render_block(chain, resolution)

    assert block.endswith("\n")
    assert not block.endswith("\n\n")


def test_a_chain_that_wrote_nothing_and_ran_nothing_renders_four_placeholders():
    chain, _ = only(ONE_CHAIN)

    block = render_block(chain, EMPTY_RESOLUTION)

    assert block == (
        f"{chain_heading(chain)}\n"
        f"{DECISIONS_HEADER}\n{EMPTY_SECTION}\n"
        f"{REJECTED_HEADER}\n{EMPTY_SECTION}\n"
        f"{SOLUTION_HEADER}\n{EMPTY_SECTION}\n"
        f"{VERIFICATION_HEADER}\n{EMPTY_SECTION}\n"
    )


def test_both_deterministic_sections_use_the_same_placeholder():
    assert solution_section(EMPTY_RESOLUTION) == f"{SOLUTION_HEADER}\n{EMPTY_SECTION}\n"
    assert verification_section(EMPTY_RESOLUTION) == f"{VERIFICATION_HEADER}\n{EMPTY_SECTION}\n"


# --- the solution section ------------------------------------------------------


def test_an_edited_file_renders_the_replacement_text_verbatim():
    _, resolution = only(ONE_CHAIN)
    entry = resolution.solution_files[0]

    section = solution_section(resolution)

    assert section.startswith(f"{SOLUTION_HEADER}\n- `{entry.path}`\n")
    assert fenced_texts(section) == [fenced(replacement_of(entry.kept[0].call))]


def test_a_file_written_and_then_edited_renders_one_bullet_per_kept_write():
    _, resolution = only(WRITE_THEN_EDITS)
    entry = resolution.solution_files[1]
    assert len(entry.kept) == 3, "the fixture's second file is written and then corrected twice"

    section = solution_section(resolution)

    assert section.count(f"- `{entry.path}`\n") == 3
    assert fenced_texts(section)[1:] == [
        fenced(content_of(entry.kept[0].call)),
        fenced(replacement_of(entry.kept[1].call)),
        fenced(replacement_of(entry.kept[2].call)),
    ]


def test_the_bullets_follow_the_transcript_order_of_the_kept_writes():
    _, resolution = only(WRITE_THEN_EDITS)
    first, second = resolution.solution_files

    bullets = outside_fences(solution_section(resolution))[1:]

    assert bullets == [f"- `{first.path}`", *[f"- `{second.path}`"] * 3]


def test_a_superseded_and_a_failed_write_never_reach_the_solution():
    _, resolution = only(ALL_WRITES_FAILED)
    entry = resolution.solution_files[0]
    assert entry.superseded, "the fixture holds a write a later one overwrote"
    assert entry.failed, "and a write the agent saw fail"

    section = solution_section(resolution)

    texts = fenced_texts(section)

    # The one accepted write, and no block of its own for either path write —
    # a `not in` would not do: the write this one overwrote shares whole
    # paragraphs with it, which is what makes it the same file twice.
    assert texts == [fenced(content_of(entry.kept[0].call))]
    for located in (*entry.superseded, *entry.failed):
        text = content_of(located.call) or replacement_of(located.call)
        assert text
        assert fenced(text) not in texts


def test_a_file_whose_every_write_failed_contributes_no_solution_content():
    _, resolution = only(ALL_WRITES_FAILED)
    entry = resolution.solution_files[0]
    failed = dataclasses.replace(entry, write=None, edits=(), failed=(*entry.failed, *entry.kept))
    assert failed.kept == (), "the entry under test holds no accepted write"
    resolution = dataclasses.replace(resolution, solution_files=(failed,), path_call_ids=())

    section = solution_section(resolution)

    # Not an empty bullet naming the file: the chain changed nothing there.
    assert entry.path not in section
    assert section == f"{SOLUTION_HEADER}\n{EMPTY_SECTION}\n"


def test_a_kept_write_whose_arguments_hold_no_text_says_so_without_a_fence():
    _, resolution = only(ONE_CHAIN)
    path = resolution.solution_files[0].path
    resolution = with_write_args(resolution, 0, {"file_path": path})

    section = solution_section(resolution)

    assert section == f"{SOLUTION_HEADER}\n- `{path}` → {NO_CONTENT_NOTE}\n"
    assert "```" not in section


def test_content_carrying_a_fence_and_a_header_is_enclosed_in_a_longer_fence():
    chain, resolution = only(ONE_CHAIN)
    path = resolution.solution_files[0].path
    content = "### Solution\n```\nnot a section\n```\n"
    resolution = with_write_args(resolution, 0, {"file_path": path, "content": content})

    block = render_block(chain, resolution)

    assert f"- `{path}`\n````\n{content}````\n" in block
    assert fenced_texts(solution_section(resolution)) == [content]
    # The header line inside the content is content, not a fifth section.
    assert [line for line in outside_fences(block) if line.startswith("#")] == [
        chain_heading(chain),
        *SECTION_HEADERS,
    ]


def test_content_without_a_closing_newline_keeps_the_closing_fence_on_its_own_line():
    _, resolution = only(ONE_CHAIN)
    path = resolution.solution_files[0].path
    resolution = with_write_args(resolution, 0, {"file_path": path, "content": "one line"})

    assert solution_section(resolution) == f"{SOLUTION_HEADER}\n- `{path}`\n```\none line\n```\n"


def test_empty_content_renders_an_empty_fence():
    _, resolution = only(ONE_CHAIN)
    path = resolution.solution_files[0].path
    resolution = with_write_args(resolution, 0, {"file_path": path, "content": ""})

    assert solution_section(resolution) == f"{SOLUTION_HEADER}\n- `{path}`\n```\n```\n"


# --- the verification section --------------------------------------------------


def test_a_passing_run_renders_its_command_exit_code_and_output():
    _, resolution = only(ONE_CHAIN)
    run = resolution.verification.run

    assert verification_section(resolution) == (
        f"{VERIFICATION_HEADER}\n- `{run.command}` → exit 0\n```\n{run.output}```\n"
    )


def test_a_kept_run_that_ended_red_is_rendered_as_such():
    _, resolution = only(ONE_CHAIN)
    run = resolution.verification.run
    result = ToolResult(call_id=run.id, output="1 failed in 0.09s\n", exit_code=1, is_error=True)
    resolution = with_run(resolution, "call", dataclasses.replace(run.call, result=result))

    section = verification_section(resolution)

    assert section == (
        f"{VERIFICATION_HEADER}\n- `{run.command}` → exit 1, {ERROR_NOTE}\n"
        f"```\n1 failed in 0.09s\n```\n"
    )


def test_a_run_the_agent_saw_fail_without_an_exit_code_says_the_code_is_unknown():
    _, resolution = only(ONE_CHAIN)
    run = resolution.verification.run
    result = ToolResult(call_id=run.id, output="command not found\n", is_error=True)
    resolution = with_run(resolution, "call", dataclasses.replace(run.call, result=result))

    section = verification_section(resolution)

    assert f"- `{run.command}` → {UNKNOWN_EXIT_NOTE}, {ERROR_NOTE}\n" in section


def test_a_run_whose_result_the_transcript_does_not_hold_renders_no_fence():
    _, resolution = only(ONE_CHAIN)
    run = resolution.verification.run
    resolution = with_run(resolution, "call", dataclasses.replace(run.call, result=None))

    section = verification_section(resolution)

    assert section == f"{VERIFICATION_HEADER}\n- `{run.command}` → {NO_RESULT_NOTE}\n"
    assert "```" not in section


def test_a_run_whose_arguments_name_no_command_says_so_instead_of_an_empty_name():
    _, resolution = only(ONE_CHAIN)
    run = resolution.verification.run
    nameless = ToolCall(id=run.id, name=run.call.name, category=ToolCategory.EXEC, args={})
    resolution = with_run(resolution, "call", dataclasses.replace(run.call, call=nameless))

    section = verification_section(resolution)

    assert section.splitlines()[1] == f"- {NO_COMMAND_NOTE} → exit 0"


def test_an_empty_output_renders_an_empty_fence():
    _, resolution = only(ONE_CHAIN)
    run = resolution.verification.run
    result = ToolResult(call_id=run.id, output="", exit_code=0)
    resolution = with_run(resolution, "call", dataclasses.replace(run.call, result=result))

    assert verification_section(resolution).endswith("```\n```\n")


def test_the_discarded_runs_are_path_and_never_rendered():
    _, resolution = only(WRITE_THEN_EDITS)
    kept, discarded = resolution.verification.run, resolution.verification.discarded
    assert discarded, "the fixture holds runs that did not prove the chain"

    texts = fenced_texts(verification_section(resolution))

    # By output, not by command: the chain ran the same command again, so the
    # discarded runs are told apart from the kept one by what came back.
    assert texts == [fenced(kept.output)]
    for run in discarded:
        assert fenced(run.output) not in texts


# --- the package surface -------------------------------------------------------


def test_the_package_exports_the_template_and_nothing_more():
    import idc.compact

    assert sorted(idc.compact.__all__) == idc.compact.__all__
    assert set(idc.compact.__all__) == {
        "DECISIONS_HEADER",
        "EMPTY_SECTION",
        "ERROR_NOTE",
        "NO_COMMAND_NOTE",
        "NO_CONTENT_NOTE",
        "NO_RESULT_NOTE",
        "REJECTED_HEADER",
        "SECTION_HEADERS",
        "SOLUTION_HEADER",
        "UNKNOWN_EXIT_NOTE",
        "VERIFICATION_HEADER",
        "chain_heading",
        "render_block",
        "solution_section",
        "verification_section",
    }
    for name in idc.compact.__all__:
        assert hasattr(idc.compact, name)
