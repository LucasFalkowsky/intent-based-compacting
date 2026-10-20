"""The discriminating property of each fixture case.

Every fixture of the corpus exists because a later ticket asserts on one
specific structural property of it. Those properties are pinned here, so that a
downstream test can never be written against a fixture that quietly stopped
exercising its case.

The helpers below describe the corpus; they are not the chunker's or the
resolver's definition of a turn, a write or a verification — those are T3 and T5
and deliberately re-derive this from the canonical fields.
"""

from collections.abc import Iterable, Sequence

from fixture_corpus import CORPUS

from idc.schema import Message, Role, Session, SessionSource, ToolCall, ToolCategory, ToolResult

FILE_PATH_KEYS = ("file_path", "filePath", "absolute_path")
"""Argument keys the three agents use for the file a call works on.

Claude Code and Gemini CLI write `file_path`, OpenCode `filePath`, and Gemini's
`read_file` `absolute_path`. The canonical schema keeps a call's arguments
exactly as the agent sent them, so the corpus carries all three spellings on
purpose.
"""


def calls(messages: Iterable[Message]) -> list[ToolCall]:
    """Return every tool call of `messages`, in transcript order."""
    return [call for message in messages for call in message.tool_calls]


def file_of(call: ToolCall) -> str | None:
    """Return the file a call works on, or `None` if it names none."""
    for key in FILE_PATH_KEYS:
        value = call.args.get(key)
        if isinstance(value, str):
            return value
    return None


def touched_files(messages: Iterable[Message]) -> set[str]:
    """Return every file read or written by the calls in `messages`."""
    return {path for call in calls(messages) if (path := file_of(call)) is not None}


def written_files(messages: Iterable[Message]) -> list[str]:
    """Return the file of every write call, in transcript order (with repeats)."""
    return [
        path
        for call in calls(messages)
        if call.category is ToolCategory.WRITE and (path := file_of(call)) is not None
    ]


def user_turn_indices(session: Session) -> list[int]:
    """Return the indices of the real user turns.

    A real user turn is a `user` message that the harness did not inject and
    that is not a slash command — the two cases the chunker has to see through.
    """
    return [
        index
        for index, message in enumerate(session.messages)
        if message.role is Role.USER and not message.is_system_injected and not message.is_command
    ]


def user_turns(session: Session) -> list[Sequence[Message]]:
    """Return one slice of messages per real user turn, up to the next turn."""
    starts = user_turn_indices(session)
    bounds = [*starts, len(session.messages)]
    return [session.messages[start:end] for start, end in zip(starts, bounds[1:], strict=True)]


def results_by_call_id(session: Session) -> dict[str, ToolResult]:
    """Return every tool result of the session, keyed by the call it answers."""
    return {
        message.tool_result.call_id: message.tool_result
        for message in session.messages
        if message.tool_result is not None
    }


def exec_runs(session: Session) -> list[tuple[str, ToolResult]]:
    """Return `(command, result)` for every `exec` call, in transcript order."""
    results = results_by_call_id(session)
    return [
        (call.args["command"], results[call.id])
        for call in calls(session.messages)
        if call.category is ToolCategory.EXEC
    ]


# --- single intent ---------------------------------------------------------


def test_single_intent_has_exactly_one_user_turn_and_one_accepted_write():
    session = CORPUS.session("single_intent")

    assert len(user_turn_indices(session)) == 1
    assert written_files(session.messages) == ["src/billing/export.py"]
    assert [call.category for call in calls(session.messages)].count(ToolCategory.READ) >= 2
    assert [result.exit_code for _, result in exec_runs(session)] == [0]


# --- two sequential intents ------------------------------------------------


def test_two_sequential_intents_are_two_turns_over_disjoint_files():
    session = CORPUS.session("two_sequential_intents")
    first, second = user_turns(session)

    assert len(user_turns(session)) == 2
    assert touched_files(first) & touched_files(second) == set()
    assert written_files(second) == ["README.md"]


# --- interleaved intents (A, B, A) -----------------------------------------


def test_interleaved_intents_resume_the_first_intent_after_a_foreign_one():
    session = CORPUS.session("interleaved_intents")
    turns = user_turns(session)

    assert len(turns) == 3
    first, other, resumed = (touched_files(turn) for turn in turns)
    assert first & resumed, "the third turn must touch files of the first — that is the A, B, A"
    assert other.isdisjoint(first), "the second turn must be a foreign intent"
    assert other.isdisjoint(resumed)
    assert "src/rates/sampler.py" in first & resumed


# --- a slash command -------------------------------------------------------


def test_slash_command_fixture_holds_commands_before_and_inside_a_turn():
    session = CORPUS.session("slash_command")
    commands = [index for index, message in enumerate(session.messages) if message.is_command]
    user_role = [
        index for index, message in enumerate(session.messages) if message.role is Role.USER
    ]
    turns = user_turn_indices(session)

    assert len(turns) == 1, "the commands must not count as user turns"
    assert len(user_role) == 3, "three user-role messages, only one of them a real turn"
    assert len(commands) == 2
    assert commands[0] < turns[0], "one command opens the session, before any user turn"
    assert commands[1] > turns[0], "one command sits inside the turn it belongs to"
    assert all(session.messages[index].role is Role.USER for index in commands)


# --- a system-injected message ---------------------------------------------


def test_system_injected_fixture_injects_into_the_system_and_the_user_role():
    session = CORPUS.session("system_injected_message")
    injected = [message for message in session.messages if message.is_system_injected]
    turns = user_turn_indices(session)

    assert len(turns) == 1
    assert len(injected) >= 3
    assert {message.role for message in injected} == {Role.SYSTEM, Role.USER}
    injected_user = [
        index
        for index, message in enumerate(session.messages)
        if message.is_system_injected and message.role is Role.USER
    ]
    assert injected_user, "the subtle case: an injected message in the user role"
    assert min(injected_user) > turns[0], "it sits inside the turn, not before it"
    assert all("<system-reminder>" in message.content for message in injected)


# --- a subagent result -----------------------------------------------------


def test_subagent_result_is_a_span_of_messages_between_call_and_result():
    session = CORPUS.session("subagent_result")
    spawn = [call for call in calls(session.messages) if call.name == "Task"]

    assert len(spawn) == 1
    call_index = next(
        index for index, message in enumerate(session.messages) if spawn[0] in message.tool_calls
    )
    result_index = next(
        index
        for index, message in enumerate(session.messages)
        if message.tool_result is not None and message.tool_result.call_id == spawn[0].id
    )
    inner = session.messages[call_index + 1 : result_index]

    assert len(inner) >= 4, "the subagent's own work must be a span, not a single result blob"
    inner_calls = {call.id for call in calls(inner)}
    inner_results = {
        message.tool_result.call_id for message in inner if message.tool_result is not None
    }
    assert inner_calls == inner_results, "the subagent's calls resolve inside its own span"
    assert spawn[0].id not in inner_calls
    assert len(user_turn_indices(session)) == 1
    assert written_files(session.messages[result_index:]), "the main agent acts on the report"


# --- a failed command followed by a passing one ----------------------------


def test_failed_then_passing_command_makes_the_last_passing_run_resolvable():
    session = CORPUS.session("failed_then_passing_command")
    runs = exec_runs(session)

    assert len(runs) == 2
    (first_command, failed), (second_command, passed) = runs
    assert first_command == second_command, "the same command, run twice"
    assert failed.exit_code != 0
    assert failed.is_error
    assert passed.exit_code == 0
    assert not passed.is_error
    assert runs[-1][1].exit_code == 0, "the last verification in the session is the passing one"


# --- a file written, then edited twice -------------------------------------


def test_write_then_two_edits_keeps_the_superseded_writes_before_the_last_one():
    session = CORPUS.session("write_then_two_edits")
    target = "/repo/src/text/slug.py"
    write_calls = [call for call in calls(session.messages) if call.category is ToolCategory.WRITE]
    to_target = [call for call in write_calls if file_of(call) == target]

    assert len(to_target) == 3, "one create plus two edits of the same file"
    assert [call.name for call in to_target] == ["write_file", "replace", "replace"]
    assert write_calls[-1] is to_target[-1], "the accepted write is the last one for that file"
    assert len(set(written_files(session.messages))) == 2, "a second file makes it per-file"
    superseded = to_target[:-1]
    assert all(write_calls.index(call) < write_calls.index(to_target[-1]) for call in superseded), (
        "both superseded writes precede the accepted one"
    )
    assert {result.exit_code for _, result in exec_runs(session)} == {0, 1}


# --- a session that already contains a compacted block ---------------------

COMPACTED_SECTIONS = (
    "### Decisions & constraints",
    "### Rejected approaches",
    "### Solution",
    "### Verification",
)


def test_already_compacted_block_is_parseable_and_extendable():
    session = CORPUS.session("already_compacted_block")
    block = session.messages[0]

    assert session.source is SessionSource.INTENT_COMPACTION
    assert block.is_system_injected
    assert "## [C1] Intent:" in block.content
    assert "## [C2] Intent:" in block.content
    for section in COMPACTED_SECTIONS:
        assert block.content.count(section) == 2, f"{section} missing for one of the two chains"
    assert "src/api/limiter.py" in block.content, "C2's solution names the file the new turn edits"

    turns = user_turns(session)
    assert len(turns) == 1, "exactly one new turn, which extends a parsed chain"
    assert min(user_turn_indices(session)) > 0, "the block precedes the new turn"
    assert set(written_files(turns[0])) == {"src/api/limiter.py", "src/api/middleware.py"}, (
        "the extension rewrites a file of the old solution and adds a new one"
    )
