"""Native summary shapes and the exported transcript file (T7.2, #32).

T7.1 decided *what* goes into the output; this stage decides what it looks like
on disk for each of the three agents and writes the file. Four properties carry
the tests:

- **A golden file per agent.** One canonical input (`interleaved_intents`,
  chosen because it is the only corpus fixture whose chunking has more than two
  chunks, so the output really is "a summary plus one of three chunks") is
  rendered for all three agents and compared **byte for byte** against
  `tests/rendered/<agent>.jsonl`. A writer that changed a field name, a field
  order or a line break fails here.
- **The round trip is the only evidence that the file is loadable.** Every
  corpus fixture is rendered for every agent, written to a file and read back
  through `idc.importers.import_transcript`, which validates through the
  package API. What that proves is that *this repository's* importer reads the
  file; it does **not** prove that Claude Code, Gemini CLI or OpenCode would
  resume from it, and the three agents are not run anywhere in this suite. For
  OpenCode it proves even less, because the exported file is the line-delimited
  shape `idc.importers.opencode` reads rather than the per-message/per-part file
  tree OpenCode itself stores.
- **What survives the round trip is tabled, not hoped for.** `INJECTED_ROLE` and
  `KEEPS_IDENTIFIERS` below are the table, and every loss in it has its own test:
  `exit_code`, `is_command`, `ToolCall.category`, the role of an injected
  message, OpenCode's re-derived result identifiers and its moved results.
- **A tool result's output is never shortened.** Asserted for every fixture and
  every agent, with a guard that the fixture had a non-empty output at all, so
  the assertion cannot pass vacuously.

Nothing here calls a model, reads a transcript of this machine or touches a
clock: the inputs are the shared corpus, its golden blocks and the three golden
files this ticket adds.
"""

import json
import re
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from fixture_corpus import CORPUS
from idc.chunk import chunk_session
from idc.importers import import_transcript, tool_registry
from idc.parse import parse_chains
from idc.render import (
    SUMMARY_MESSAGE_ID,
    WRITERS,
    MissingWriterError,
    RenderError,
    SummaryIdInUseError,
    UnanchoredToolResultError,
    assemble,
    render_transcript,
    write_transcript,
)
from idc.schema import (
    Agent,
    Chain,
    CompactedBlock,
    Message,
    Role,
    Session,
    SessionSource,
    ToolCall,
    ToolCategory,
    ToolResult,
)

RENDERED_DIR = Path(__file__).parent / "rendered"
"""Where this ticket's golden output files live; see its `README.md`."""

CORPUS_NAMES = sorted(CORPUS.compact_names())
"""Every corpus fixture that has a golden block sequence to render."""

AGENTS = list(Agent)
"""The three target agents, so a fourth one added to the enum shows up here."""

GOLDEN_FIXTURE = "interleaved_intents"
"""The fixture the three golden files are rendered from: three chunks, two chains."""

INJECTED_ROLE: dict[Agent, Role | None] = {
    Agent.CLAUDE_CODE: None,
    Agent.GEMINI_CLI: Role.SYSTEM,
    Agent.OPENCODE: Role.USER,
}
"""The role an injected message comes back under, or `None` for "unchanged".

`is_system_injected` is never lost — that is the writers' rule — but only Claude
Code marks both an injected *user* turn (`isMeta`) and a harness `system` record,
so for the other two the role moves to the one shape that agent has for harness
text: a Gemini CLI `info` notice (role `system`) and an OpenCode `synthetic` text
part (role `user`).
"""

KEEPS_IDENTIFIERS = {Agent.CLAUDE_CODE, Agent.GEMINI_CLI}
"""The agents whose formats give every canonical message a record of its own.

OpenCode has no record for a tool result: it lives in the tool part of the call
that produced it, so its canonical identifier is re-derived from that part's
index and its position follows its call.
"""


def test_the_corpus_offers_fixtures_with_golden_blocks() -> None:
    """Guard the parametrization: an empty corpus would skip, not pass."""
    assert CORPUS_NAMES


def test_every_agent_has_a_writer() -> None:
    """The table is complete, so the missing-writer path needs a patched one."""
    assert set(WRITERS) == set(Agent)


def block_of(name: str, agent: Agent) -> CompactedBlock:
    """Build the `CompactedBlock` of a fixture: golden chains, last chunk, agent."""
    session = CORPUS.session(name)
    return CompactedBlock(
        chains=parse_chains(CORPUS.compact_markdown(name)),
        latest_chunk_id=chunk_session(session)[-1].id,
        rendered_for=agent,
    )


def rendered(name: str, agent: Agent) -> str:
    """Render one corpus fixture for one agent and return the file's text."""
    session = CORPUS.session(name)
    return render_transcript(block_of(name, agent), session=session, chunks=chunk_session(session))


def written(name: str, agent: Agent, directory: Path) -> Path:
    """Render one corpus fixture for one agent into a file under `directory`."""
    session = CORPUS.session(name)
    return write_transcript(
        block_of(name, agent),
        session=session,
        chunks=chunk_session(session),
        path=directory / f"{name}.jsonl",
    )


def reimported(name: str, agent: Agent, directory: Path) -> Session:
    """Render one fixture, write it, and read it back through the importer."""
    return import_transcript(agent, written(name, agent, directory)).session


def kept(name: str) -> tuple[Message, ...]:
    """Return the messages of the chunk the fixture's block keeps verbatim."""
    session = CORPUS.session(name)
    return assemble(
        block_of(name, Agent.CLAUDE_CODE), session=session, chunks=chunk_session(session)
    ).messages


EXIT_CODE_NAMES = sorted(
    name
    for name in CORPUS_NAMES
    if any(
        message.tool_result is not None and message.tool_result.exit_code is not None
        for message in kept(name)
    )
)
"""The fixtures whose kept chunk records a process status at all.

Derived rather than listed, so a corpus change cannot silently empty the
exit-code test; `test_some_fixture_records_an_exit_code` guards the table,
because an empty parametrization is a skip and not a pass.
"""


def records(text: str) -> list[dict]:
    """Return the records of a rendered file, one per line."""
    return [json.loads(line) for line in text.splitlines()]


def turns(messages: Iterable[Message], agent: Agent) -> list[tuple]:
    """Project every message that is not a tool result, in order.

    The identifier, the role an injected message comes back under, the content,
    the injected flag and the calls — everything a round trip must preserve for
    all three agents. `ToolCall.category` is left out on purpose: no native
    format carries it and the importer re-derives it from the target agent's own
    table (see `test_a_cross_agent_render_takes_the_category_from_the_target`).
    """
    return [
        (
            message.id,
            (INJECTED_ROLE[agent] or message.role) if message.is_system_injected else message.role,
            message.content,
            message.is_system_injected,
            [(call.id, call.name, call.args) for call in message.tool_calls],
        )
        for message in messages
        if message.role is not Role.TOOL
    ]


def results(messages: Iterable[Message]) -> dict[str, tuple[str, bool]]:
    """Return every tool result's output and error flag, keyed by its call id.

    The call id is the key rather than the message id because that reference is
    what every format carries: OpenCode re-derives the *message* id from the
    index of the tool part the result sits in.
    """
    return {
        message.tool_result.call_id: (message.tool_result.output, message.tool_result.is_error)
        for message in messages
        if message.tool_result is not None
    }


@pytest.mark.parametrize("agent", AGENTS)
def test_the_rendered_file_matches_the_golden_file(agent: Agent) -> None:
    """One input, three native shapes, compared byte for byte."""
    golden = RENDERED_DIR / f"{agent.value}.jsonl"

    assert rendered(GOLDEN_FIXTURE, agent) == golden.read_text(encoding="utf-8")


@pytest.mark.parametrize("agent", AGENTS)
def test_a_golden_file_exists_for_every_agent(agent: Agent) -> None:
    """Guard the comparison above: a missing file must not read as a pass."""
    assert (RENDERED_DIR / f"{agent.value}.jsonl").is_file()


@pytest.mark.parametrize("name", CORPUS_NAMES)
@pytest.mark.parametrize("agent", AGENTS)
def test_the_rendered_file_re_imports_as_a_canonical_session(
    name: str, agent: Agent, tmp_path: Path
) -> None:
    """Write, import, validate: the whole round trip, for every fixture.

    The resulting `source` is `native-compaction`, which is what actually
    happens and **not** a stand-in for `intent-compaction`: the summary is
    written in the agent's own compaction shape, and that shape is the only
    provenance signal an importer reads. `SessionSource.INTENT_COMPACTION` is
    produced nowhere in `src/` today; making this file re-import under it is a
    ticket of its own (#102).
    """
    session = reimported(name, agent, tmp_path)

    assert session.agent is agent
    assert session.source is SessionSource.NATIVE_COMPACTION
    assert session.native_summary_id == SUMMARY_MESSAGE_ID
    assert session.id == CORPUS.session(name).id


@pytest.mark.parametrize("name", CORPUS_NAMES)
@pytest.mark.parametrize("agent", AGENTS)
def test_the_summary_message_carries_the_blocks_text(
    name: str, agent: Agent, tmp_path: Path
) -> None:
    """The re-imported summary message holds the chains' text, and nothing is cut."""
    summary = assemble(
        block_of(name, agent),
        session=CORPUS.session(name),
        chunks=chunk_session(CORPUS.session(name)),
    ).summary
    session = reimported(name, agent, tmp_path)
    message = next(one for one in session.messages if one.id == SUMMARY_MESSAGE_ID)

    assert message.role is Role.SYSTEM
    assert message.is_system_injected
    assert summary in message.content
    assert parse_chains(summary) == block_of(name, agent).chains


@pytest.mark.parametrize("name", CORPUS_NAMES)
@pytest.mark.parametrize("agent", AGENTS)
def test_the_latest_chunks_turns_come_back_unchanged(
    name: str, agent: Agent, tmp_path: Path
) -> None:
    """Every non-result message returns with its id, text, flags and calls."""
    session = reimported(name, agent, tmp_path)
    back = [message for message in session.messages if message.id != SUMMARY_MESSAGE_ID]

    assert turns(back, agent) == turns(kept(name), agent)


@pytest.mark.parametrize("name", CORPUS_NAMES)
@pytest.mark.parametrize("agent", AGENTS)
def test_every_tool_result_survives_in_full(name: str, agent: Agent, tmp_path: Path) -> None:
    """A result's output is written verbatim — never shortened, never summarized."""
    before = results(kept(name))
    assert any(output for output, _ in before.values()), "fixture has no tool output to keep"

    session = reimported(name, agent, tmp_path)

    assert results(session.messages) == before


@pytest.mark.parametrize("name", CORPUS_NAMES)
@pytest.mark.parametrize("agent", AGENTS)
def test_the_message_timestamps_come_back_as_the_same_instants(
    name: str, agent: Agent, tmp_path: Path
) -> None:
    """Every kept message keeps its instant; the offset's spelling is UTC's."""
    session = reimported(name, agent, tmp_path)
    back = {
        message.id: message.timestamp
        for message in session.messages
        if message.id != SUMMARY_MESSAGE_ID and message.role is not Role.TOOL
    }
    before = {
        message.id: message.timestamp for message in kept(name) if message.role is not Role.TOOL
    }

    assert back == before


@pytest.mark.parametrize("name", CORPUS_NAMES)
@pytest.mark.parametrize("agent", AGENTS)
def test_rendering_twice_gives_byte_identical_text(name: str, agent: Agent) -> None:
    """No clock, no randomness: the text is a function of the input alone."""
    assert rendered(name, agent) == rendered(name, agent)


def test_the_agents_that_keep_every_identifier_are_the_two_named_ones() -> None:
    """Guard the table: a fourth agent must not drop out of the test below."""
    assert set(Agent) - {Agent.OPENCODE} == KEEPS_IDENTIFIERS


@pytest.mark.parametrize("name", CORPUS_NAMES)
@pytest.mark.parametrize("agent", sorted(KEEPS_IDENTIFIERS))
def test_the_identifiers_of_every_record_come_back_in_order(
    name: str, agent: Agent, tmp_path: Path
) -> None:
    """For the two agents with a record per message, ids and order are identical.

    OpenCode is not here: it has no record for a result, so a result's
    identifier is re-derived from the index of its tool part — pinned by
    `test_opencode_writes_a_result_into_the_record_of_its_own_call`.
    """
    session = reimported(name, agent, tmp_path)

    assert [message.id for message in session.messages] == [
        SUMMARY_MESSAGE_ID,
        *(message.id for message in kept(name)),
    ]


@pytest.mark.parametrize("name", CORPUS_NAMES)
def test_a_same_agent_render_keeps_the_tool_categories(name: str, tmp_path: Path) -> None:
    """Rendered for its own agent, every call's category is resolved the same way."""
    agent = CORPUS.session(name).agent
    session = reimported(name, agent, tmp_path)
    back = {call.id: call.category for message in session.messages for call in message.tool_calls}

    assert back == {call.id: call.category for message in kept(name) for call in message.tool_calls}


def test_a_cross_agent_render_takes_the_category_from_the_target(tmp_path: Path) -> None:
    """The category is not written: it is re-derived from the target's table.

    `interleaved_intents` is a Claude Code session, and OpenCode's table in
    `tool_registry.toml` declares `read` and `edit`, not `Read` and `Edit`, so
    the same calls come back under the category that table gives them.
    """
    session = reimported(GOLDEN_FIXTURE, Agent.OPENCODE, tmp_path)
    registry = tool_registry()
    calls = [call for message in session.messages for call in message.tool_calls]
    assert calls, "the fixture's latest chunk makes no tool call"

    assert [call.category for call in calls] == [
        registry.category(Agent.OPENCODE, call.name) for call in calls
    ]
    assert [call.name for call in calls] == [
        call.name for message in kept(GOLDEN_FIXTURE) for call in message.tool_calls
    ]
    assert ToolCategory.OTHER in {call.category for call in calls}


def test_some_fixture_records_an_exit_code() -> None:
    """Guard the parametrization below: an empty table would skip, not pass."""
    assert EXIT_CODE_NAMES


@pytest.mark.parametrize("name", EXIT_CODE_NAMES)
@pytest.mark.parametrize("agent", AGENTS)
def test_the_exit_code_of_a_result_does_not_survive(
    name: str, agent: Agent, tmp_path: Path
) -> None:
    """No record this renderer writes carries a process status, so it is lost.

    Claude Code and OpenCode record none at all (both importers document
    `exit_code` as always `null`). Gemini CLI reads one out of the result's own
    output, but only from a `run_shell_command` output that starts with its
    `Output: ` prefix and ends in its own `Exit Code:` block — and no corpus
    fixture's output carries that shape, which is why every fixture loses it
    here. The output itself is untouched either way.
    """
    before = [
        message.tool_result.exit_code for message in kept(name) if message.tool_result is not None
    ]
    assert any(code is not None for code in before), "fixture records no exit code"

    session = reimported(name, agent, tmp_path)

    assert all(
        message.tool_result.exit_code is None
        for message in session.messages
        if message.tool_result is not None
    )


@pytest.mark.parametrize("agent", AGENTS)
def test_a_slash_command_comes_back_unmarked(agent: Agent, tmp_path: Path) -> None:
    """`is_command` is lost: the marker it is derived from is part of the text.

    Claude Code sets the flag from the `<command-name>` tags its harness wraps a
    command in, and the other two importers never set it at all. Adding those
    tags would change the message's own text, so the renderer does not — and
    `slash_command`'s commands are recorded as the bare `/model …` line, with no
    marker to carry the flag.
    """
    before = [message.id for message in kept("slash_command") if message.is_command]
    assert before, "fixture holds no slash command"

    session = reimported("slash_command", agent, tmp_path)

    assert [message.id for message in session.messages if message.is_command] == []


@pytest.mark.parametrize("agent", AGENTS)
def test_an_injected_message_stays_marked_as_injected(agent: Agent, tmp_path: Path) -> None:
    """The flag never gets lost; only the role may move to the agent's own shape."""
    before = [message for message in kept("system_injected_message") if message.is_system_injected]
    assert before, "fixture holds no injected message"

    session = reimported("system_injected_message", agent, tmp_path)
    back = {
        message.id: message.role
        for message in session.messages
        if message.is_system_injected and message.id != SUMMARY_MESSAGE_ID
    }

    assert back == {message.id: INJECTED_ROLE[agent] or message.role for message in before}


@pytest.mark.parametrize("agent", AGENTS)
def test_an_injected_assistant_message_keeps_its_role_and_loses_the_flag(
    agent: Agent, tmp_path: Path
) -> None:
    """The one injected shape no format marks, pinned so the rule stays true.

    `is_system_injected` is kept for a `user` and a `system` message, which are
    the two roles an importer in this repository marks as injected. A model turn
    marked that way is a session no importer produces, and none of the three
    formats has a record for it, so all three write the assistant turn it is and
    the flag is the one thing lost.
    """
    session = inline_session(
        Message(id="m1", role=Role.USER, content="go"),
        Message(id="m2", role=Role.ASSISTANT, content="said", is_system_injected=True),
    )
    block, chunks = inline_block(session, agent)
    path = write_transcript(block, session=session, chunks=chunks, path=tmp_path / "a.jsonl")

    back = next(one for one in import_transcript(agent, path).session.messages if one.id == "m2")

    assert back.role is Role.ASSISTANT
    assert back.is_system_injected is False
    assert back.content == "said"


def test_the_summary_is_written_in_each_agents_own_summary_shape() -> None:
    """The one record per agent that marks the text as a compaction summary."""
    claude = records(rendered(GOLDEN_FIXTURE, Agent.CLAUDE_CODE))[0]
    assert claude["type"] == "user"
    assert claude["isCompactSummary"] is True
    assert "isMeta" not in claude

    gemini = records(rendered(GOLDEN_FIXTURE, Agent.GEMINI_CLI))[1]
    assert gemini["type"] == "user"
    assert "<state_snapshot>" in gemini["content"][0]["text"]

    opencode = records(rendered(GOLDEN_FIXTURE, Agent.OPENCODE))[0]
    assert opencode["type"] == "compaction"
    assert "role" not in opencode


@pytest.mark.parametrize("agent", AGENTS)
def test_every_timestamp_in_the_file_comes_from_the_input(agent: Agent) -> None:
    """Nothing is stamped from a clock: every time in the file is one of the input's."""
    session = CORPUS.session(GOLDEN_FIXTURE)
    instants = {session.created_at} | {
        message.timestamp for message in kept(GOLDEN_FIXTURE) if message.timestamp is not None
    }
    text = rendered(GOLDEN_FIXTURE, agent)

    if agent is Agent.OPENCODE:
        found = {int(raw) for raw in re.findall(r'"created":(\d+)', text)}
        expected = {
            (instant - datetime(1970, 1, 1, tzinfo=UTC)) // timedelta(milliseconds=1)
            for instant in instants
        }
    else:
        found = set(re.findall(r"\d{4}-\d{2}-\d{2}T[\d:.]+Z", text))
        expected = {
            instant.astimezone(UTC).isoformat().replace("+00:00", "Z") for instant in instants
        }

    assert found
    assert found <= expected


def test_a_missing_writer_fails_loudly(monkeypatch: pytest.MonkeyPatch) -> None:
    """An agent with no writer is refused by name, never served by a default.

    `WRITERS` holds all three agents (asserted by `test_every_agent_has_a_writer`),
    so this refusal is unreachable today and is exercised against a patched
    table. It exists so that a fourth agent added to `Agent` fails here instead
    of silently being rendered as one of the three.
    """
    monkeypatch.delitem(WRITERS, Agent.OPENCODE)
    session = CORPUS.session(GOLDEN_FIXTURE)

    with pytest.raises(MissingWriterError) as raised:
        render_transcript(
            block_of(GOLDEN_FIXTURE, Agent.OPENCODE),
            session=session,
            chunks=chunk_session(session),
        )

    assert raised.value.agent is Agent.OPENCODE
    assert Agent.OPENCODE.value in str(raised.value)
    assert isinstance(raised.value, RenderError)


def inline_session(*messages: Message, **fields: object) -> Session:
    """Build a two-field canonical session around hand-written messages."""
    return Session(
        id="s-inline",
        agent=Agent.CLAUDE_CODE,
        source=SessionSource.RAW,
        created_at=datetime(2026, 3, 1, 12, 0, tzinfo=UTC),
        messages=list(messages),
        **fields,
    )


def inline_block(session: Session, agent: Agent) -> tuple[CompactedBlock, Sequence]:
    """Return a one-chain block over the whole session, plus its chunking."""
    chunks = chunk_session(session)
    block = CompactedBlock(
        chains=[
            Chain(
                id="C1",
                label="inline",
                chunk_ids=[chunks[-1].id],
                compacted=True,
                compacted_text="## [C1] Intent: inline\n### Solution\n- (none)\n",
            )
        ],
        latest_chunk_id=chunks[-1].id,
        rendered_for=agent,
    )
    return block, chunks


@pytest.mark.parametrize("agent", AGENTS)
def test_a_message_without_a_timestamp_is_written_without_one(agent: Agent, tmp_path: Path) -> None:
    """A missing timestamp is omitted rather than filled in from anywhere.

    The session still loads because `Session.created_at` is required and the
    summary record carries it — which is also where that record's own timestamp
    comes from, the one time value in the output that belongs to no message.
    """
    session = inline_session(Message(id="m1", role=Role.USER, content="go"))
    block, chunks = inline_block(session, agent)
    path = write_transcript(block, session=session, chunks=chunks, path=tmp_path / "t.jsonl")

    back = import_transcript(agent, path).session

    assert back.created_at == session.created_at
    assert [one.timestamp for one in back.messages if one.id == "m1"] == [None]
    assert back.model is None


@pytest.mark.parametrize("agent", AGENTS)
def test_a_sub_millisecond_timestamp_is_kept_except_for_opencode(
    agent: Agent, tmp_path: Path
) -> None:
    """OpenCode's `time.created` is whole milliseconds, so microseconds are cut."""
    stamp = datetime(2026, 3, 1, 12, 0, 0, 123_456, tzinfo=UTC)
    session = inline_session(Message(id="m1", role=Role.USER, content="go", timestamp=stamp))
    block, chunks = inline_block(session, agent)
    path = write_transcript(block, session=session, chunks=chunks, path=tmp_path / "t.jsonl")

    back = next(one for one in import_transcript(agent, path).session.messages if one.id == "m1")

    if agent is Agent.OPENCODE:
        assert back.timestamp == stamp - timedelta(microseconds=456)
    else:
        assert back.timestamp == stamp


@pytest.mark.parametrize("agent", AGENTS)
def test_the_summary_id_is_refused_before_a_file_is_written(agent: Agent) -> None:
    """The one identifier the renderer invents is refused at the source.

    `SUMMARY_MESSAGE_ID` is a constant rather than a derived name, so a kept
    message can carry it. This used to rest on the canonical schema rejecting a
    duplicated message id on re-import — which **two** of the three formats
    provide and Gemini CLI does not: its importer has append-log semantics, so
    the later record of two sharing an id wins and the `<state_snapshot>`
    record is dropped. A review constructed that case and the whole compacted
    block vanished with no error at all, leaving a `raw` session.

    So the refusal is the renderer's, uniform across the three: it invents the
    identifier, so it owns the clash, and no file is written at all.
    """
    session = inline_session(Message(id=SUMMARY_MESSAGE_ID, role=Role.USER, content="go"))
    block, chunks = inline_block(session, agent)

    with pytest.raises(SummaryIdInUseError) as caught:
        render_transcript(block, session=session, chunks=chunks)
    assert isinstance(caught.value, RenderError)
    assert SUMMARY_MESSAGE_ID in str(caught.value)


@pytest.mark.parametrize("agent", AGENTS)
def test_nothing_is_written_when_the_summary_id_is_in_use(agent: Agent, tmp_path: Path) -> None:
    """The refusal happens before the file exists, not after it is half-written."""
    session = inline_session(Message(id=SUMMARY_MESSAGE_ID, role=Role.USER, content="go"))
    block, chunks = inline_block(session, agent)
    path = tmp_path / "clash.jsonl"

    with pytest.raises(SummaryIdInUseError):
        write_transcript(block, session=session, chunks=chunks, path=path)
    assert not path.exists()


def opencode_pair() -> Session:
    """Return a session whose tool result arrives after an unrelated message."""
    stamp = datetime(2026, 3, 1, 12, 0, tzinfo=UTC)
    return inline_session(
        Message(id="m1", role=Role.USER, content="go", timestamp=stamp),
        Message(
            id="m2",
            role=Role.ASSISTANT,
            content="delegating",
            timestamp=stamp,
            tool_calls=[ToolCall(id="c1", name="read", category=ToolCategory.READ)],
        ),
        Message(id="m3", role=Role.ASSISTANT, content="meanwhile", timestamp=stamp),
        Message(
            id="m4",
            role=Role.TOOL,
            timestamp=stamp,
            tool_result=ToolResult(call_id="c1", output="one line\n"),
        ),
    )


def test_opencode_writes_a_result_into_the_record_of_its_own_call(tmp_path: Path) -> None:
    """A result moves to its call's record, because that is the only place for it.

    OpenCode stores a result in the `state` of the tool part that made the call,
    so a result that arrived after other messages comes back at the call's
    position and under an identifier derived from that part's index. The output
    and the call reference are untouched; nothing else can be.
    """
    session = opencode_pair()
    block, chunks = inline_block(session, Agent.OPENCODE)
    path = write_transcript(block, session=session, chunks=chunks, path=tmp_path / "o.jsonl")

    back = import_transcript(Agent.OPENCODE, path).session
    tools = [one for one in back.messages if one.role is Role.TOOL]

    assert [one.id for one in back.messages] == [SUMMARY_MESSAGE_ID, "m1", "m2", "m2#1", "m3"]
    assert [one.id for one in tools] == ["m2#1"]
    assert tools[0].tool_result is not None
    assert tools[0].tool_result.call_id == "c1"
    assert tools[0].tool_result.output == "one line\n"


def test_opencode_refuses_a_result_whose_call_is_not_in_the_chunk(tmp_path: Path) -> None:
    """A result with no call to anchor it has no OpenCode record at all.

    The format carries a result inside the tool part of its own call, so writing
    one whose call is not there would mean inventing the call — a tool call the
    transcript never made, which the resolver would then read as a step of the
    path. The other two agents record a result as a message of its own and
    render the same chunk without complaint.
    """
    stamp = datetime(2026, 3, 1, 12, 0, tzinfo=UTC)
    session = inline_session(
        Message(id="m1", role=Role.USER, content="go", timestamp=stamp),
        Message(
            id="m2",
            role=Role.TOOL,
            timestamp=stamp,
            tool_result=ToolResult(call_id="gone", output="orphan\n"),
        ),
    )
    block, chunks = inline_block(session, Agent.OPENCODE)

    with pytest.raises(UnanchoredToolResultError) as raised:
        render_transcript(block, session=session, chunks=chunks)

    assert raised.value.call_id == "gone"
    assert raised.value.message_id == "m2"
    assert isinstance(raised.value, RenderError)

    for agent in (Agent.CLAUDE_CODE, Agent.GEMINI_CLI):
        other, _ = inline_block(session, agent)
        path = write_transcript(
            other, session=session, chunks=chunks, path=tmp_path / f"{agent.value}.jsonl"
        )
        back = import_transcript(agent, path).session
        assert [one.tool_result.output for one in back.messages if one.tool_result] == ["orphan\n"]


def test_write_transcript_writes_exactly_what_render_transcript_returns(
    tmp_path: Path,
) -> None:
    """The file on disk is the rendered text, in UTF-8, with no extra newline."""
    session = CORPUS.session(GOLDEN_FIXTURE)
    chunks = chunk_session(session)
    block = block_of(GOLDEN_FIXTURE, Agent.CLAUDE_CODE)

    path = write_transcript(block, session=session, chunks=chunks, path=tmp_path / "out.jsonl")

    assert path.read_bytes() == rendered(GOLDEN_FIXTURE, Agent.CLAUDE_CODE).encode("utf-8")


@pytest.mark.parametrize("agent", AGENTS)
def test_every_line_of_the_output_is_one_json_record(agent: Agent) -> None:
    """The file is JSON Lines: one record per line, and a trailing newline."""
    text = rendered(GOLDEN_FIXTURE, agent)

    assert text.endswith("\n")
    assert all(json.loads(line) for line in text.splitlines())
    assert len(text.splitlines()) == len(records(text))
