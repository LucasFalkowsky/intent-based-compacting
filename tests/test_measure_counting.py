"""Counting tokens over the visible context (T9.1, #37).

The expectations in this module are **hand-computed from the fixtures** against
the rule the counter documents — a message contributes its `content`, the name
and the JSON of every tool call's arguments, and its tool result's `output` —
with the fake tokenizer's one-token-per-word rule applied to each. One of them
is spelled out in full below so a reader can check the rule itself, and the rest
follow it line by line.

No test here calls a real tokenizer or a model: every real tokenizer needs a
downloaded vocabulary or a network call, which the suite may not do.
"""

import json
from datetime import UTC, datetime

import pytest

from idc.importers import TranscriptImportError, visible_context
from idc.measure import (
    FakeTokenizer,
    MeasureError,
    count_context,
    count_message,
    count_messages,
    message_parts,
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

SINGLE_INTENT_PER_MESSAGE = {
    "m01": 40,  # the user's request, 40 words
    "m02": 17,  # 11 words of content + two Read calls at 1 (name) + 2 (args) each
    "m03": 42,  # the first Read's output
    "m04": 19,  # the second Read's output
    "m05": 41,  # 29 words of content + one Edit at 1 + 11
    "m06": 11,  # the edit's confirmation
    "m07": 11,  # 4 words of content + `Bash` (1) + 6 words of its argument JSON
    "m08": 5,  # `..\n2 passed in 0.14s`
    "m09": 11,  # the closing assistant turn
}
"""Per-message token counts of `single_intent`, hand-computed from the rule.

`m07` in full, as the worked example for every other line: its content is
`Running the exporter's tests.` (4 words), its single tool call contributes the
name `Bash` (1 word) and the arguments as JSON —
`{"command": "uv run pytest tests/billing/test_export.py -q"}`, which is
`{"command":`, `"uv`, `run`, `pytest`, `tests/billing/test_export.py`, `-q"}`
(6 words) — and it carries no tool result, so 4 + 1 + 6 = 11.
"""

SYSTEM_INJECTED_PER_MESSAGE = {
    "m01": 17,
    "m02": 25,
    "m03": 6,
    "m04": 19,
    "m05": 15,
    "m06": 46,
    "m07": 4,
    "m08": 11,
    "m09": 5,
    "m10": 5,
    "m11": 25,
}
"""Per-message counts of `system_injected_message`, by the same rule.

`m01` and `m10` are the two injected harness notices this transcript carries;
they are in the file and not in the agent's context, which is what makes this
fixture the one the visible-context assertions are written against.
"""


def message(**fields: object) -> Message:
    """Return a message with the fields a test names, and defaults elsewhere."""
    return Message(id=fields.pop("id", "m01"), role=fields.pop("role", Role.ASSISTANT), **fields)  # type: ignore[arg-type]


def session(*messages: Message, model: str | None = "claude-opus-5", **fields: object) -> Session:
    """Return a session holding `messages`."""
    return Session(
        id=fields.pop("id", "sess-measure"),  # type: ignore[arg-type]
        agent=fields.pop("agent", Agent.CLAUDE_CODE),  # type: ignore[arg-type]
        model=model,
        source=fields.pop("source", SessionSource.RAW),  # type: ignore[arg-type]
        native_summary_id=fields.pop("native_summary_id", None),  # type: ignore[arg-type]
        created_at=CREATED_AT,
        messages=list(messages),
    )


# --- what counts as the text of a message ------------------------------------


def test_the_counted_text_is_content_then_every_tool_call_then_the_tool_result():
    full = message(
        content="two words",
        tool_calls=[
            ToolCall(id="c01", name="Edit", category=ToolCategory.WRITE, args={"file_path": "a.py"})
        ],
        tool_result=ToolResult(call_id="c01", output="applied one edit", exit_code=0),
    )

    assert message_parts(full) == (
        "two words",
        "Edit",
        json.dumps({"file_path": "a.py"}, sort_keys=True, ensure_ascii=False),
        "applied one edit",
    )


@pytest.mark.parametrize(
    ("dropped", "expected"),
    [
        # 2 (content) + 1 (name) + 2 (args JSON) + 3 (output) = 8 for the whole
        # message; each row removes exactly one part and must change the total,
        # which is what makes a counter that reads `content` alone fail here.
        ("nothing", 8),
        ("content", 6),
        ("tool_calls", 5),
        ("tool_result", 5),
    ],
)
def test_every_part_of_a_message_contributes_to_its_count(dropped, expected):
    parts: dict[str, object] = {
        "content": "two words",
        "tool_calls": [
            ToolCall(id="c01", name="Edit", category=ToolCategory.WRITE, args={"file_path": "a.py"})
        ],
        "tool_result": ToolResult(call_id="c01", output="applied one edit", exit_code=0),
    }
    if dropped == "content":
        parts["content"] = ""
    elif dropped == "tool_calls":
        parts["tool_calls"] = []
    elif dropped == "tool_result":
        parts["tool_result"] = None

    assert count_message(message(**parts), FakeTokenizer()) == expected


def test_the_schema_s_own_bookkeeping_is_not_counted():
    # Identifiers, the tool category and the exit code are this tool's
    # canonical fields, not text the agent was sent: two messages that differ
    # only in them count the same.
    def built(call_id: str, category: ToolCategory, exit_code: int | None) -> Message:
        return message(
            content="",
            tool_calls=[
                ToolCall(id=call_id, name="Edit", category=category, args={"file_path": "a.py"})
            ],
            tool_result=ToolResult(call_id=call_id, output="ok", exit_code=exit_code),
        )

    plain = built("c01", ToolCategory.WRITE, 0)
    renamed = built("call-9999-zzz", ToolCategory.OTHER, 127)

    assert count_message(plain, FakeTokenizer()) == count_message(renamed, FakeTokenizer())


def test_an_empty_part_is_not_handed_to_the_tokenizer_at_all():
    tokenizer = FakeTokenizer()

    count_message(message(content="", tool_calls=[], tool_result=None), tokenizer)

    assert tokenizer.calls == ()


def test_a_tool_call_without_arguments_contributes_its_name_alone():
    # An empty argument object is not text the agent was sent, so it is not
    # counted as the one word `{}` would be.
    bare = message(
        content="",
        tool_calls=[ToolCall(id="c01", name="TodoWrite", category=ToolCategory.OTHER, args={})],
    )

    assert message_parts(bare) == ("TodoWrite",)
    assert count_message(bare, FakeTokenizer()) == 1


# --- the counts of the fixtures ----------------------------------------------


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("single_intent", SINGLE_INTENT_PER_MESSAGE),
        ("system_injected_message", SYSTEM_INJECTED_PER_MESSAGE),
    ],
)
def test_the_per_message_counts_of_a_fixture_are_the_hand_computed_ones(corpus, name, expected):
    fixture = corpus.session(name)

    counted = {entry.id: count_message(entry, FakeTokenizer()) for entry in fixture.messages}

    assert counted == expected


def test_the_total_of_a_sequence_of_messages_is_the_sum_of_their_counts(corpus):
    fixture = corpus.session("single_intent")

    assert count_messages(fixture.messages, FakeTokenizer()) == sum(
        SINGLE_INTENT_PER_MESSAGE.values()
    )
    assert count_messages(fixture.messages, FakeTokenizer()) == 197


def test_an_empty_sequence_of_messages_counts_zero():
    assert count_messages([], FakeTokenizer()) == 0


# --- counting is over the visible context ------------------------------------


def test_a_context_count_holds_one_entry_per_visible_message(corpus):
    fixture = corpus.session("system_injected_message")

    counted = count_context(fixture, FakeTokenizer())

    assert counted.session_id == fixture.id
    assert counted.message_ids == tuple(
        identifier for identifier in SYSTEM_INJECTED_PER_MESSAGE if identifier not in {"m01", "m10"}
    )
    assert {entry.message_id: entry.tokens for entry in counted.per_message} == {
        identifier: tokens
        for identifier, tokens in SYSTEM_INJECTED_PER_MESSAGE.items()
        if identifier not in {"m01", "m10"}
    }


def test_a_session_with_hidden_messages_counts_fewer_tokens_than_the_raw_session(corpus):
    fixture = corpus.session("system_injected_message")

    counted = count_context(fixture, FakeTokenizer())
    raw = count_messages(fixture.messages, FakeTokenizer())

    assert counted.hidden_ids == ("m01", "m10")
    assert counted.tokens < raw
    assert (counted.tokens, raw) == (156, 178)


def test_the_difference_is_exactly_the_hidden_messages_contribution(corpus):
    fixture = corpus.session("system_injected_message")
    context = visible_context(fixture)
    hidden = [entry for entry in fixture.messages if entry.id in set(context.hidden_ids)]

    counted = count_context(fixture, FakeTokenizer())

    assert count_messages(fixture.messages, FakeTokenizer()) - counted.tokens == count_messages(
        hidden, FakeTokenizer()
    )


def test_a_session_without_hidden_messages_counts_the_whole_transcript(corpus):
    fixture = corpus.session("single_intent")

    counted = count_context(fixture, FakeTokenizer())

    assert counted.hidden_ids == ()
    assert counted.tokens == count_messages(fixture.messages, FakeTokenizer()) == 197


def test_every_fixture_of_the_corpus_counts_its_visible_messages_and_no_more(corpus):
    names = corpus.names()

    assert names, "the shared corpus holds no fixture"
    for name in names:
        fixture = corpus.session(name)
        counted = count_context(fixture, FakeTokenizer())
        hidden = [entry for entry in fixture.messages if entry.id in set(counted.hidden_ids)]
        assert counted.tokens + count_messages(hidden, FakeTokenizer()) == count_messages(
            fixture.messages, FakeTokenizer()
        )


# --- the refusal of `visible_context` is not swallowed -----------------------


def test_a_compacted_session_that_names_no_summary_is_refused_by_the_counter():
    # The refusal exists so that measurement cannot count history the agent
    # dropped; a counter that caught it would count exactly that.
    compacted = session(
        message(id="m01", role=Role.USER, content="go"),
        source=SessionSource.NATIVE_COMPACTION,
    )

    with pytest.raises(TranscriptImportError, match="native summary"):
        count_context(compacted, FakeTokenizer())


def test_the_import_refusal_is_not_wrapped_into_a_measurement_error():
    compacted = session(
        message(id="m01", role=Role.USER, content="go"),
        source=SessionSource.NATIVE_COMPACTION,
        native_summary_id="m99",
    )

    with pytest.raises(TranscriptImportError) as refusal:
        count_context(compacted, FakeTokenizer())

    assert not isinstance(refusal.value, MeasureError)
    assert "m99" in str(refusal.value)


# --- determinism -------------------------------------------------------------


def test_the_same_session_and_tokenizer_count_the_same_twice(corpus):
    fixture = corpus.session("interleaved_intents")

    first = count_context(fixture, FakeTokenizer())
    second = count_context(fixture, FakeTokenizer())

    assert first == second


def test_the_order_of_the_counted_parts_is_the_transcript_s_own(corpus):
    # Determinism of the *text* handed to the tokenizer, not only of the total:
    # a dict iteration that varied would show up here and nowhere else.
    fixture = corpus.session("write_then_two_edits")
    tokenizer, again = FakeTokenizer(), FakeTokenizer()

    count_context(fixture, tokenizer)
    count_context(fixture, again)

    assert tokenizer.calls == again.calls
    assert tokenizer.calls[0] == fixture.messages[0].content


def test_a_count_does_not_record_which_tokenizer_produced_it(corpus):
    """The gap the module docstring now states, pinned so the prose stays true.

    `count_context` takes the tokenizer from its caller and never checks it
    against the document's `model`, so a `claude-*` session counted with any
    tokenizer yields a `ContextCount` whose only definition-bearing field is
    that `model`. A review found the docstring claiming the opposite. The run
    manifest (#39) is what has to record the tokenizer set a run used; this
    asserts the value object does not pretend to.
    """
    session = corpus.session("single_intent")
    count = count_context(session, FakeTokenizer())

    assert not hasattr(count, "tokenizer")
    assert count.model == session.model
