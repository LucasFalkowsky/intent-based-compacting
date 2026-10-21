"""Exploration rate, write counts and the exit-code histogram (T9.2, #38).

Every number in `CORPUS_EXPECTATIONS` is **hand-computed from the fixture
JSON**, not read off the implementation:

- the category counts by listing the `tool_calls` of each fixture and reading
  each call's `category` field,
- `files` and `per_file` by reading the path argument of every `write` call
  whose result is not an error, verbatim and in transcript order,
- `lines_added` by counting the lines of a whole write's `content` and, for an
  edit, the lines of a line-level diff between `old_string` and `new_string`
  that a unified diff would mark `+` — worked out below for the two edits that
  are not a one-line-for-one-line replacement,
- `exit_codes` by reading `exit_code` off every `tool_result`,
- `error_results` by counting the results whose `is_error` is true.

`lines_added` for the two shapes, spelled out so a reader can check the rule
itself:

- `rewritten_after_failed_edit` writes `docs/install.md` whole twice, with an
  edit between them that the agent saw fail. The failed edit contributes
  nothing; the two whole writes contribute every line of their `content`,
  11 and 12, so 23 — and that is deliberately more lines than the file ever
  held, which is the over-report `idc.measure.statistics` documents.
- `slash_command`'s `c02` inserts two lines after a line it keeps (its
  `old_string` is the first line of its `new_string`), so 2. Its `c04` moves
  that read-the-body line below the new guard: of its three new lines a diff
  can match only one against the three old ones, so 2 again — 4 for the
  fixture.

No test here calls a model, reads a clock, draws a random number or touches
the filesystem beyond the fixture corpus.
"""

from dataclasses import dataclass
from datetime import UTC, datetime

import pytest
from idc.measure.statistics import REPLACEMENT_PAIRS

from fixture_corpus import CORPUS
from fixture_corpus import REPLACEMENT_PAIRS as CORPUS_REPLACEMENT_PAIRS
from idc.importers import visible_context
from idc.measure import (
    ExitCodeCount,
    FileEdits,
    MeasureError,
    NoToolCallsError,
    ToolStatistics,
    lines_added_of,
    tool_statistics,
)
from idc.schema import (
    Agent,
    Message,
    Role,
    Session,
    SessionSource,
    ToolCall,
    ToolCategory,
    ToolResult,
)

CREATED_AT = datetime(2026, 9, 1, 9, 0, tzinfo=UTC)
"""Fixed instant; no test in this module reads the clock."""


@dataclass(frozen=True)
class Expected:
    """The hand-computed statistics of one fixture.

    Attributes:
        reads: `read`-category calls.
        writes: `write`-category calls.
        execs: `exec`-category calls.
        others: `other`-category calls.
        files: Distinct paths of the accepted writes, in first-write order.
        per_file: Path to `(accepted write calls, lines added)`.
        lines_added: The fixture's total, stated on its own so that it is not
            derived from `per_file` by the same hand that wrote `per_file`.
        exit_codes: `exit_code` to how many answered calls carry it.
        error_results: Results whose `is_error` is true.
        failed_writes: `ToolCall.id` of the writes the agent saw fail.
    """

    reads: int
    writes: int
    execs: int
    others: int
    files: tuple[str, ...]
    per_file: dict[str, tuple[int, int]]
    lines_added: int
    exit_codes: dict[int | None, int]
    error_results: int
    failed_writes: tuple[str, ...]

    @property
    def total(self) -> int:
        """Return every tool call of the fixture, whatever its category."""
        return self.reads + self.writes + self.execs + self.others


CORPUS_EXPECTATIONS = {
    "already_compacted_block": Expected(
        reads=2,
        writes=2,
        execs=1,
        others=0,
        files=("src/api/limiter.py", "src/api/middleware.py"),
        # c03 inserts 4 lines above the `def take` line it keeps; c04 replaces
        # its 2 old lines with 8 that share none of them.
        per_file={"src/api/limiter.py": (1, 4), "src/api/middleware.py": (1, 8)},
        lines_added=12,
        exit_codes={None: 4, 0: 1},
        error_results=0,
        failed_writes=(),
    ),
    "failed_then_passing_command": Expected(
        reads=2,
        writes=2,
        execs=2,
        others=0,
        files=("src/net/backoff.py", "src/net/config.py"),
        # c02 drops two lines and keeps the `def delay_for` line, rewriting the
        # `return` under it: 1 added. c05 rewrites one line: 1.
        per_file={"src/net/backoff.py": (1, 1), "src/net/config.py": (1, 1)},
        lines_added=2,
        exit_codes={None: 4, 0: 1, 1: 1},
        error_results=1,
        failed_writes=(),
    ),
    "interleaved_intents": Expected(
        reads=4,
        writes=5,
        execs=3,
        others=0,
        files=(
            "src/rates/sampler.py",
            "tests/rates/test_sampler.py",
            "config/settings.toml",
            "src/config/loader.py",
        ),
        # sampler.py: c03 keeps `self._weights = weights`, replaces the one
        # `def __init__` line with three and appends one — 4; c11 replaces 7
        # lines with 2 that share none of them — 2.
        per_file={
            "src/rates/sampler.py": (2, 6),
            "tests/rates/test_sampler.py": (1, 1),
            "config/settings.toml": (1, 1),
            "src/config/loader.py": (1, 1),
        },
        lines_added=9,
        exit_codes={None: 9, 0: 3},
        error_results=0,
        failed_writes=(),
    ),
    "rewritten_after_failed_edit": Expected(
        reads=1,
        writes=3,
        execs=0,
        others=0,
        files=("docs/install.md",),
        per_file={"docs/install.md": (2, 23)},
        lines_added=23,
        exit_codes={None: 4},
        error_results=1,
        failed_writes=("c03",),
    ),
    "single_intent": Expected(
        reads=2,
        writes=1,
        execs=1,
        others=0,
        files=("src/billing/export.py",),
        per_file={"src/billing/export.py": (1, 1)},
        lines_added=1,
        exit_codes={None: 3, 0: 1},
        error_results=0,
        failed_writes=(),
    ),
    "slash_command": Expected(
        reads=1,
        writes=2,
        execs=2,
        others=0,
        files=("src/api/orders.py",),
        per_file={"src/api/orders.py": (2, 4)},
        lines_added=4,
        exit_codes={None: 3, 0: 1, 1: 1},
        error_results=1,
        failed_writes=(),
    ),
    "subagent_result": Expected(
        reads=2,
        writes=3,
        execs=1,
        others=1,
        files=("src/billing/money.py", "src/billing/invoice.py", "src/report/totals.py"),
        per_file={
            "src/billing/money.py": (1, 1),
            "src/billing/invoice.py": (1, 1),
            "src/report/totals.py": (1, 1),
        },
        lines_added=3,
        exit_codes={None: 6, 0: 1},
        error_results=0,
        failed_writes=(),
    ),
    "system_injected_message": Expected(
        reads=1,
        writes=1,
        execs=1,
        others=0,
        files=("/repo/src/jobs/nightly.py",),
        # c02 keeps the `except Exception:` line and replaces the `pass` under
        # it with two lines.
        per_file={"/repo/src/jobs/nightly.py": (1, 2)},
        lines_added=2,
        exit_codes={None: 2, 0: 1},
        error_results=0,
        failed_writes=(),
    ),
    "two_sequential_intents": Expected(
        reads=2,
        writes=3,
        execs=1,
        others=0,
        files=("src/billing/cli.py", "src/billing/export.py", "README.md"),
        # c03 inserts a whole new function (4 lines) above the `def
        # export_report` line it keeps; c02 and c06 append one line each.
        per_file={
            "src/billing/cli.py": (1, 1),
            "src/billing/export.py": (1, 4),
            "README.md": (1, 1),
        },
        lines_added=6,
        exit_codes={None: 5, 0: 1},
        error_results=0,
        failed_writes=(),
    ),
    "write_then_two_edits": Expected(
        reads=1,
        writes=4,
        execs=3,
        others=0,
        files=("/repo/tests/text/test_slug.py", "/repo/src/text/slug.py"),
        # slug.py: c03 writes 15 lines whole, c05 inserts 2 above the line it
        # keeps, c07 rewrites 1 of its 2 lines — 18.
        per_file={
            "/repo/tests/text/test_slug.py": (1, 16),
            "/repo/src/text/slug.py": (3, 18),
        },
        lines_added=34,
        exit_codes={None: 5, 0: 1, 1: 2},
        error_results=2,
        failed_writes=(),
    ),
}
"""Hand-computed statistics of every corpus fixture; see the module docstring."""


def message(**fields: object) -> Message:
    """Return a message with the fields a test names, and defaults elsewhere."""
    return Message(id=fields.pop("id", "m01"), role=fields.pop("role", Role.ASSISTANT), **fields)  # type: ignore[arg-type]


def session(*messages: Message, **fields: object) -> Session:
    """Return a session holding `messages`."""
    return Session(
        id=fields.pop("id", "s-1"),  # type: ignore[arg-type]
        agent=fields.pop("agent", Agent.CLAUDE_CODE),  # type: ignore[arg-type]
        model="claude-opus-5",
        source=fields.pop("source", SessionSource.RAW),  # type: ignore[arg-type]
        created_at=CREATED_AT,
        messages=list(messages),
        **fields,  # type: ignore[arg-type]
    )


def write(call_id: str, path: str, **args: object) -> ToolCall:
    """Return a Claude Code `Edit`/`Write` call on `path`."""
    return ToolCall(
        id=call_id,
        name="Write" if "content" in args else "Edit",
        category=ToolCategory.WRITE,
        args={"file_path": path, **args},
    )


def edit(old: str, new: str, call_id: str = "c01") -> ToolCall:
    """Return an `Edit` call replacing `old` with `new`."""
    return write(call_id, "src/a.py", old_string=old, new_string=new)


def result(call_id: str, *, is_error: bool = False, exit_code: int | None = None) -> Message:
    """Return a tool message carrying the result of `call_id`."""
    return message(
        id=f"r-{call_id}",
        role=Role.TOOL,
        tool_result=ToolResult(
            call_id=call_id, output="done", exit_code=exit_code, is_error=is_error
        ),
    )


# --------------------------------------------------------------------------
# The table itself
# --------------------------------------------------------------------------


def test_the_expectation_table_covers_the_whole_corpus() -> None:
    assert CORPUS_EXPECTATIONS, "an empty table would make every test below a skip"
    assert set(CORPUS_EXPECTATIONS) == set(CORPUS.names())


def test_no_hidden_message_of_the_corpus_carries_a_tool_call() -> None:
    # The table is therefore the same under either denominator, which is why
    # the visible-context rule is pinned on a synthetic session below.
    for name in CORPUS.names():
        document = CORPUS.session(name)
        hidden = set(visible_context(document).hidden_ids)
        carriers = [
            msg.id
            for msg in document.messages
            if msg.id in hidden and (msg.tool_calls or msg.tool_result is not None)
        ]
        assert carriers == [], f"{name}: {carriers}"


# --------------------------------------------------------------------------
# Exploration rate
# --------------------------------------------------------------------------


@pytest.mark.parametrize(("name", "expected"), sorted(CORPUS_EXPECTATIONS.items()))
def test_category_counts_match_the_fixture(name: str, expected: Expected) -> None:
    stats = tool_statistics(CORPUS.session(name))
    assert (stats.read_calls, stats.write_calls, stats.exec_calls, stats.other_calls) == (
        expected.reads,
        expected.writes,
        expected.execs,
        expected.others,
    )
    assert stats.total_calls == expected.total


@pytest.mark.parametrize(("name", "expected"), sorted(CORPUS_EXPECTATIONS.items()))
def test_exploration_rate_is_read_calls_over_all_calls(name: str, expected: Expected) -> None:
    stats = tool_statistics(CORPUS.session(name))
    assert stats.exploration_rate == expected.reads / expected.total


def test_exploration_rate_of_one_fixture_in_full() -> None:
    # already_compacted_block: two Read calls out of two reads, two edits and
    # one Bash — 2 / 5.
    assert tool_statistics(CORPUS.session("already_compacted_block")).exploration_rate == 0.4


def test_exploration_rate_counts_other_category_calls_in_the_denominator() -> None:
    # subagent_result's `Task` call is `other`; dropping it would make the
    # rate 2/6 instead of 2/7.
    stats = tool_statistics(CORPUS.session("subagent_result"))
    assert stats.other_calls == 1
    assert stats.exploration_rate == 2 / 7


def test_statistics_count_only_the_visible_context() -> None:
    document = session(
        message(id="m01", tool_calls=[ToolCall(id="c01", name="Read", category=ToolCategory.READ)]),
        message(id="m02", role=Role.SYSTEM, content="Summary.", is_system_injected=True),
        message(id="m03", tool_calls=[write("c02", "src/a.py", content="one\ntwo\n")]),
        source=SessionSource.NATIVE_COMPACTION,
        native_summary_id="m02",
    )
    stats = tool_statistics(document)
    assert stats.total_calls == 1, "the Read before the agent's own summary is not context"
    assert stats.read_calls == 0
    assert stats.write_calls == 1
    assert stats.files_edited == ("src/a.py",)


# --------------------------------------------------------------------------
# No tool calls at all
# --------------------------------------------------------------------------


def test_a_session_without_tool_calls_refuses_an_exploration_rate() -> None:
    stats = tool_statistics(session(message(id="m01", role=Role.USER, content="Hello.")))
    assert stats.total_calls == 0
    with pytest.raises(NoToolCallsError) as raised:
        stats.exploration_rate  # noqa: B018 - the access is what refuses
    assert "no tool call" in str(raised.value)


def test_the_refusal_is_both_a_measure_error_and_a_value_error() -> None:
    assert issubclass(NoToolCallsError, MeasureError)
    assert issubclass(NoToolCallsError, ValueError)


def test_a_session_without_tool_calls_still_reports_the_defined_numbers() -> None:
    stats = tool_statistics(session(message(id="m01", role=Role.USER, content="Hello.")))
    assert stats.files_edited == ()
    assert stats.lines_added == 0
    assert stats.exit_code_counts == {}
    assert stats.error_results == 0
    assert stats.calls_without_result == 0


# --------------------------------------------------------------------------
# Files edited
# --------------------------------------------------------------------------


@pytest.mark.parametrize(("name", "expected"), sorted(CORPUS_EXPECTATIONS.items()))
def test_files_edited_match_the_fixture(name: str, expected: Expected) -> None:
    stats = tool_statistics(CORPUS.session(name))
    assert stats.files_edited == expected.files
    assert stats.file_count == len(expected.files)


@pytest.mark.parametrize(("name", "expected"), sorted(CORPUS_EXPECTATIONS.items()))
def test_per_file_writes_and_lines_match_the_fixture(name: str, expected: Expected) -> None:
    stats = tool_statistics(CORPUS.session(name))
    assert {entry.path: (entry.writes, entry.lines_added) for entry in stats.per_file} == (
        expected.per_file
    )


def test_two_spellings_of_one_path_are_two_files() -> None:
    document = session(
        message(
            id="m01",
            tool_calls=[
                write("c01", "src/a.py", content="one\n"),
                write("c02", "./src/a.py", content="two\n"),
            ],
        ),
    )
    stats = tool_statistics(document)
    assert stats.files_edited == ("src/a.py", "./src/a.py")
    assert stats.file_count == 2


def test_a_failed_write_is_not_a_file_edited() -> None:
    document = session(
        message(id="m01", tool_calls=[write("c01", "src/a.py", content="one\ntwo\n")]),
        result("c01", is_error=True),
    )
    stats = tool_statistics(document)
    assert stats.files_edited == ()
    assert stats.lines_added == 0
    assert stats.failed_write_ids == ("c01",)
    assert stats.write_calls == 1, "it is still a write the agent made"


def test_the_corpus_failed_write_is_reported_and_not_counted() -> None:
    expected = CORPUS_EXPECTATIONS["rewritten_after_failed_edit"]
    stats = tool_statistics(CORPUS.session("rewritten_after_failed_edit"))
    assert stats.failed_write_ids == expected.failed_writes
    assert stats.per_file == (FileEdits(path="docs/install.md", writes=2, lines_added=23),)


def test_a_write_without_a_result_counts_as_an_edit() -> None:
    document = session(message(id="m01", tool_calls=[write("c01", "src/a.py", content="one\n")]))
    stats = tool_statistics(document)
    assert stats.files_edited == ("src/a.py",)
    assert stats.failed_write_ids == ()
    assert stats.calls_without_result == 1


def test_a_write_naming_no_file_is_unmeasured() -> None:
    document = session(
        message(
            id="m01",
            tool_calls=[
                ToolCall(
                    id="c01",
                    name="Write",
                    category=ToolCategory.WRITE,
                    args={"content": "one\ntwo\n"},
                )
            ],
        ),
    )
    stats = tool_statistics(document)
    assert stats.files_edited == ()
    assert stats.lines_added == 0
    assert stats.unmeasured_write_ids == ("c01",)


# --------------------------------------------------------------------------
# Lines added
# --------------------------------------------------------------------------


@pytest.mark.parametrize(("name", "expected"), sorted(CORPUS_EXPECTATIONS.items()))
def test_lines_added_match_the_fixture(name: str, expected: Expected) -> None:
    assert tool_statistics(CORPUS.session(name)).lines_added == expected.lines_added


@pytest.mark.parametrize(("name", "expected"), sorted(CORPUS_EXPECTATIONS.items()))
def test_lines_added_is_the_sum_over_the_files(name: str, expected: Expected) -> None:
    assert expected.lines_added == sum(lines for _, lines in expected.per_file.values())
    stats = tool_statistics(CORPUS.session(name))
    assert stats.lines_added == sum(entry.lines_added for entry in stats.per_file)


@pytest.mark.parametrize(
    ("content", "lines"),
    [
        ("", 0),
        ("one", 1),
        ("one\n", 1),
        ("one\ntwo\n", 2),
        ("one\ntwo", 2),
        ("one\n\nthree\n", 3),
    ],
)
def test_a_whole_write_adds_every_line_of_its_content(content: str, lines: int) -> None:
    assert lines_added_of(write("c01", "src/a.py", content=content)) == lines


@pytest.mark.parametrize(
    ("old", "new", "added"),
    [
        ("one\n", "one\n", 0),
        ("one\n", "two\n", 1),
        ("one\ntwo\nthree\n", "one\n", 0),
        ("one\n", "one\ntwo\nthree\n", 2),
        ("", "one\ntwo\n", 2),
        ("one\ntwo\nthree\n", "four\n", 1),
        ("one\ntwo\n", "one\nTWO\n", 1),
    ],
)
def test_an_edit_adds_the_plus_lines_of_its_diff(old: str, new: str, added: int) -> None:
    assert lines_added_of(edit(old, new)) == added


def test_an_edit_that_only_deletes_adds_nothing() -> None:
    assert lines_added_of(edit("one\ntwo\nthree\n", "one\n")) == 0


def test_an_edit_with_an_empty_replacement_adds_nothing() -> None:
    # An empty `new_string` is how an agent deletes the text it anchors on.
    assert lines_added_of(edit("one\ntwo\n", "")) == 0


def test_a_write_shape_this_build_cannot_measure_answers_none() -> None:
    notebook = ToolCall(
        id="c01",
        name="NotebookEdit",
        category=ToolCategory.WRITE,
        args={"notebook_path": "nb.ipynb", "new_source": "one\ntwo\n"},
    )
    assert lines_added_of(notebook) is None


def test_an_unmeasurable_write_is_reported_and_not_guessed() -> None:
    notebook = ToolCall(
        id="c01",
        name="NotebookEdit",
        category=ToolCategory.WRITE,
        args={"notebook_path": "nb.ipynb", "new_source": "one\ntwo\n"},
    )
    stats = tool_statistics(session(message(id="m01", tool_calls=[notebook])))
    assert stats.unmeasured_write_ids == ("c01",)
    assert stats.lines_added == 0
    assert stats.files_edited == ("nb.ipynb",), "the file is known even when the lines are not"


def test_a_call_that_is_not_a_write_answers_none() -> None:
    assert lines_added_of(ToolCall(id="c01", name="Read", category=ToolCategory.READ)) is None
    assert (
        lines_added_of(
            ToolCall(
                id="c02", name="Bash", category=ToolCategory.EXEC, args={"command": "pytest -q"}
            )
        )
        is None
    )


def test_the_replacement_pairs_are_the_corpus_spellings() -> None:
    assert REPLACEMENT_PAIRS == CORPUS_REPLACEMENT_PAIRS
    assert len(REPLACEMENT_PAIRS) == 2, "the table the spelling test ranges over"


@pytest.mark.parametrize(("old_key", "new_key"), CORPUS_REPLACEMENT_PAIRS)
def test_every_edit_spelling_is_counted(old_key: str, new_key: str) -> None:
    call = ToolCall(
        id="c01",
        name="edit",
        category=ToolCategory.WRITE,
        args={"filePath": "src/a.py", old_key: "one\n", new_key: "one\ntwo\n"},
    )
    assert lines_added_of(call) == 1


# --------------------------------------------------------------------------
# Exit-code histogram
# --------------------------------------------------------------------------


@pytest.mark.parametrize(("name", "expected"), sorted(CORPUS_EXPECTATIONS.items()))
def test_exit_code_histogram_matches_the_fixture(name: str, expected: Expected) -> None:
    assert tool_statistics(CORPUS.session(name)).exit_code_counts == expected.exit_codes


@pytest.mark.parametrize(("name", "expected"), sorted(CORPUS_EXPECTATIONS.items()))
def test_error_results_match_the_fixture(name: str, expected: Expected) -> None:
    assert tool_statistics(CORPUS.session(name)).error_results == expected.error_results


@pytest.mark.parametrize(("name", "expected"), sorted(CORPUS_EXPECTATIONS.items()))
def test_the_histogram_reconciles_with_every_call(name: str, expected: Expected) -> None:
    stats = tool_statistics(CORPUS.session(name))
    answered = sum(entry.calls for entry in stats.exit_codes)
    assert answered + stats.calls_without_result == stats.total_calls
    assert stats.calls_without_result == 0, f"{name} answers every call it makes"


def test_the_histogram_covers_every_category_not_only_exec() -> None:
    # `single_intent` answers two reads and one edit with no exit code and one
    # Bash with 0; an exec-only histogram would report `{0: 1}`.
    assert tool_statistics(CORPUS.session("single_intent")).exit_code_counts == {None: 3, 0: 1}


def test_a_call_without_a_result_is_not_in_the_histogram() -> None:
    document = session(
        message(id="m01", tool_calls=[ToolCall(id="c01", name="Read", category=ToolCategory.READ)]),
    )
    stats = tool_statistics(document)
    assert stats.exit_code_counts == {}
    assert stats.calls_without_result == 1


def test_an_error_without_an_exit_code_lands_in_the_none_bucket() -> None:
    document = session(
        message(id="m01", tool_calls=[write("c01", "src/a.py", old_string="a", new_string="b")]),
        result("c01", is_error=True),
    )
    stats = tool_statistics(document)
    assert stats.exit_code_counts == {None: 1}
    assert stats.error_results == 1


def test_an_error_with_an_exit_code_lands_in_its_own_bucket() -> None:
    document = session(
        message(
            id="m01",
            tool_calls=[
                ToolCall(
                    id="c01", name="Bash", category=ToolCategory.EXEC, args={"command": "pytest"}
                )
            ],
        ),
        result("c01", is_error=True, exit_code=2),
    )
    stats = tool_statistics(document)
    assert stats.exit_code_counts == {2: 1}
    assert stats.error_results == 1


def test_the_histogram_is_ordered_with_the_missing_code_first() -> None:
    document = session(
        message(
            id="m01",
            tool_calls=[
                ToolCall(id="c01", name="Bash", category=ToolCategory.EXEC, args={"command": "a"}),
                ToolCall(id="c02", name="Bash", category=ToolCategory.EXEC, args={"command": "b"}),
                ToolCall(id="c03", name="Read", category=ToolCategory.READ),
            ],
        ),
        result("c01", exit_code=2),
        result("c02", exit_code=0),
        result("c03"),
    )
    stats = tool_statistics(document)
    assert stats.exit_codes == (
        ExitCodeCount(exit_code=None, calls=1),
        ExitCodeCount(exit_code=0, calls=1),
        ExitCodeCount(exit_code=2, calls=1),
    )


# --------------------------------------------------------------------------
# Determinism and construction
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(CORPUS_EXPECTATIONS))
def test_the_same_transcript_gives_the_same_numbers(name: str) -> None:
    assert tool_statistics(CORPUS.session(name)) == tool_statistics(CORPUS.session(name))


def test_a_session_and_its_visible_context_are_counted_alike() -> None:
    document = CORPUS.session("system_injected_message")
    assert tool_statistics(document) == tool_statistics(visible_context(document).session)


def test_statistics_report_the_session_they_counted() -> None:
    document = CORPUS.session("single_intent")
    assert tool_statistics(document).session_id == document.id


def test_a_negative_count_is_a_plain_value_error() -> None:
    with pytest.raises(ValueError, match="cannot be negative") as raised:
        ToolStatistics(session_id="s-1", read_calls=-1)
    assert not isinstance(raised.value, MeasureError)


def test_a_file_repeated_in_the_breakdown_is_refused() -> None:
    with pytest.raises(ValueError, match="once"):
        ToolStatistics(
            session_id="s-1",
            per_file=(
                FileEdits(path="src/a.py", writes=1, lines_added=1),
                FileEdits(path="src/a.py", writes=1, lines_added=1),
            ),
        )


def test_a_blank_path_in_the_breakdown_is_refused() -> None:
    with pytest.raises(ValueError, match="name a file"):
        FileEdits(path="  ", writes=1, lines_added=1)


def test_statistics_say_what_they_counted_in_one_line() -> None:
    stats = tool_statistics(CORPUS.session("single_intent"))
    assert str(stats) == "sess-single-intent: 2/4 read calls, 1 file(s) edited, 1 line(s) added"
