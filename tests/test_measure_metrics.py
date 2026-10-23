"""The aggregate `measure(before, after, tokenizer)` (T9.3, #39).

The parts are tested where they are implemented — `count_context` in
`test_measure_counting`, `CompressionRate` in `test_measure_compression`,
`tool_statistics` in `test_measure_statistics`. What is tested here is the
assembly: that `Metrics` holds exactly what those parts answer for the two
sessions it was handed, that **one** tokenizer counts both sides, that the tool
statistics are the *before* session's, and that a session which made no tool
call yields `exploration_rate is None` instead of a refusal.

The two hand-built sessions make the numbers checkable without the corpus:
`before` holds `please fix the slug helper` (5 words) and `done` (1), so 6
tokens; `after` holds `Summary block here` (3), so the saved fraction is
`(6 - 3) / 6 = 0.5`. The one corpus fixture used here is `write_then_two_edits`,
and only to pin that the aggregate equals the parts on a transcript with tool
calls.
"""

from datetime import UTC, datetime

import pytest

from fixture_corpus import CORPUS
from idc.measure import (
    CompressionRate,
    EmptyBaselineError,
    FakeTokenizer,
    Metrics,
    count_context,
    measure,
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
)

CREATED_AT = datetime(2026, 10, 21, 9, 0, 0, tzinfo=UTC)
"""Fixed creation instant; no test here reads a clock."""

FIXTURE = "write_then_two_edits"
"""The corpus transcript the composition test uses — it holds tool calls."""


def session(session_id: str, *messages: Message) -> Session:
    """Return a raw canonical session holding `messages`."""
    return Session(
        id=session_id,
        agent=Agent.CLAUDE_CODE,
        model="claude-opus-5",
        source=SessionSource.RAW,
        created_at=CREATED_AT,
        messages=list(messages),
    )


def message(message_id: str, role: Role, content: str, *calls: ToolCall) -> Message:
    """Return one message with `content` and, optionally, tool calls."""
    return Message(
        id=message_id,
        role=role,
        content=content,
        timestamp=CREATED_AT,
        tool_calls=list(calls),
    )


def before_session() -> Session:
    """Return the six-token "before" session; it makes no tool call."""
    return session(
        "s-before",
        message("m-1", Role.USER, "please fix the slug helper"),
        message("m-2", Role.ASSISTANT, "done"),
    )


def after_session() -> Session:
    """Return the three-token "after" session."""
    return session("s-after", message("m-1", Role.USER, "Summary block here"))


def test_metrics_hold_the_hand_counted_tokens_and_fraction():
    metrics = measure(before_session(), after_session(), FakeTokenizer())

    assert (metrics.before.tokens, metrics.after.tokens) == (6, 3)
    assert metrics.compression == CompressionRate(before=6, after=3)
    assert metrics.saved_fraction == pytest.approx(0.5)


def test_metrics_equal_the_parts_on_a_corpus_transcript():
    before = CORPUS.session(FIXTURE)
    after = after_session()
    tokenizer = FakeTokenizer()

    metrics = measure(before, after, tokenizer)

    assert metrics.before == count_context(before, FakeTokenizer())
    assert metrics.after == count_context(after, FakeTokenizer())
    assert metrics.compression == CompressionRate.between(metrics.before, metrics.after)
    assert metrics.tools == tool_statistics(before)
    assert isinstance(metrics, Metrics)


def test_one_injected_tokenizer_counts_both_sides():
    tokenizer = FakeTokenizer()

    measure(before_session(), after_session(), tokenizer)

    assert "please fix the slug helper" in tokenizer.calls
    assert "Summary block here" in tokenizer.calls
    assert tokenizer.total == 9


def test_tool_statistics_are_the_before_session_s():
    read = ToolCall(id="c-1", name="Read", category=ToolCategory.READ, args={})
    before = session(
        "s-before",
        message("m-1", Role.USER, "please fix the slug helper"),
        message("m-2", Role.ASSISTANT, "reading", read),
    )
    after = session(
        "s-after",
        message("m-1", Role.USER, "Summary block here"),
        message(
            "m-2",
            Role.ASSISTANT,
            "listing",
            ToolCall(id="c-2", name="Bash", category=ToolCategory.EXEC, args={"command": "ls"}),
        ),
    )

    metrics = measure(before, after, FakeTokenizer())

    assert metrics.tools.session_id == "s-before"
    assert (metrics.tools.read_calls, metrics.tools.exec_calls) == (1, 0)
    assert metrics.exploration_rate == pytest.approx(1.0)


def test_exploration_rate_is_none_when_the_session_made_no_tool_call():
    metrics = measure(before_session(), after_session(), FakeTokenizer())

    assert metrics.tools.total_calls == 0
    assert metrics.exploration_rate is None


def test_exploration_rate_is_the_statistics_own_rate_when_it_is_defined():
    before = CORPUS.session(FIXTURE)

    metrics = measure(before, after_session(), FakeTokenizer())

    assert metrics.exploration_rate == pytest.approx(tool_statistics(before).exploration_rate)


def test_a_baseline_without_a_word_is_refused():
    empty = session("s-empty", message("m-1", Role.USER, ""))

    with pytest.raises(EmptyBaselineError):
        measure(empty, after_session(), FakeTokenizer())


def test_metrics_render_one_diagnostic_line():
    metrics = measure(before_session(), after_session(), FakeTokenizer())

    line = str(metrics)

    assert "6" in line
    assert "3" in line
