"""How a transcript this tool wrote says so, and how it is told apart (T7.3, #102).

T7.2 writes the summary in each agent's own compaction shape, which is also the
shape the agent writes when it compacts a session itself — so the shape alone
cannot say who wrote the file. This ticket adds the one signal that can:
`idc.importers.INTENT_SUMMARY_MARKER`, a Markdown comment line the renderer puts
in front of the summary text and the importers read back.

Four properties carry the tests:

- **The distinction is three-way, per agent.** `SOURCE_CASES` is the table:
  every agent crossed with every `SessionSource`, each with a transcript that
  has to read back under exactly that provenance — the agent's own
  `golden_session` fixture for `raw`, its `native_compaction` fixture for
  `native-compaction`, and a file this renderer wrote for `intent-compaction`.
  `test_the_source_table_is_complete` guards the parametrization, because an
  empty or a half-filled table would skip rather than fail.
- **The marker travels in the summary's text, so it survives a re-read.** The
  text is prose to all three formats; a key of our own beside the record would
  rest on the agent preserving a field it does not know, which no test here
  could show. What these tests do show is that each agent's own importer reads
  the marker back — the agents themselves are not run anywhere in this suite,
  exactly as for the native shapes in `tests/test_render_native.py`.
- **Naming the summary is what makes it context.** `visible_context` exempts the
  message `native_summary_id` names from the harness-notice rule *before* it
  looks at the role, and this tool writes its summary as an injected `system`
  message — so a rendered file that reported the provenance but did not name its
  summary would have the summary itself dropped out of the measured context.
  Both halves are asserted here.
- **A rendered file has nothing in front of its summary**, so the cut's effect
  is asserted on a session that does have a history: the same messages under
  `native-compaction` and under `intent-compaction` have to normalise
  identically, which is the "treated like a native one" claim stated as a test.

Nothing here calls a model, reads a transcript of this machine or touches a
clock: the inputs are the shared corpus, the agent-native fixtures and files the
renderer writes into `tmp_path`.
"""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from fixture_corpus import CORPUS
from idc.chunk import chunk_session
from idc.importers import (
    INTENT_SUMMARY_MARKER,
    HiddenMessage,
    SessionBuilder,
    import_transcript,
    summary_provenance,
    visible_context,
)
from idc.importers.provenance import HIDDEN_REPLACED_BY_COMPACTION
from idc.parse import StrayContentError, parse_chains
from idc.render import (
    SNAPSHOT_CLOSE,
    SNAPSHOT_OPEN,
    SUMMARY_MESSAGE_ID,
    assemble,
    write_transcript,
)
from idc.schema import Agent, CompactedBlock, Message, Role, Session, SessionSource
from native_transcripts import NATIVE

AGENTS = list(Agent)
"""The three target agents, so a fourth one added to the enum shows up here."""

FIXTURE = "interleaved_intents"
"""The corpus fixture rendered here: three chunks and two chains."""

RAW_FIXTURE: dict[Agent, tuple[str, str]] = {
    Agent.CLAUDE_CODE: ("golden_session", ".jsonl"),
    Agent.GEMINI_CLI: ("golden_session", ".json"),
    Agent.OPENCODE: ("golden_session", ".jsonl"),
}
"""The uncompacted transcript of each agent, with its own file suffix.

Gemini CLI's golden session is a single JSON document and the other two are
JSONL, which is why the suffix is part of the table instead of a default.
"""

SOURCE_CASES = [(agent, source) for agent in AGENTS for source in SessionSource]
"""Every agent crossed with every provenance — the three-way distinction itself."""


def block_of(agent: Agent) -> CompactedBlock:
    """Build the `CompactedBlock` of the rendered fixture for one agent."""
    session = CORPUS.session(FIXTURE)
    return CompactedBlock(
        chains=parse_chains(CORPUS.compact_markdown(FIXTURE)),
        latest_chunk_id=chunk_session(session)[-1].id,
        rendered_for=agent,
    )


def rendered_path(agent: Agent, directory: Path) -> Path:
    """Render the fixture for one agent into a file under `directory`."""
    session = CORPUS.session(FIXTURE)
    return write_transcript(
        block_of(agent),
        session=session,
        chunks=chunk_session(session),
        path=directory / f"{agent.value}.jsonl",
    )


def reimported(agent: Agent, directory: Path) -> Session:
    """Render the fixture for one agent, write it, and read it back."""
    return import_transcript(agent, rendered_path(agent, directory)).session


def transcript_of(agent: Agent, source: SessionSource, directory: Path) -> Session:
    """Return a transcript of one agent that should read back under `source`."""
    if source is SessionSource.INTENT_COMPACTION:
        return reimported(agent, directory)
    if source is SessionSource.NATIVE_COMPACTION:
        return import_transcript(agent, NATIVE.path(agent, "native_compaction")).session
    name, suffix = RAW_FIXTURE[agent]
    return import_transcript(agent, NATIVE.path(agent, name, suffix)).session


def summary_of(session: Session) -> Message:
    """Return the message the session names as its summary."""
    return next(one for one in session.messages if one.id == session.native_summary_id)


def history(source: SessionSource) -> Session:
    """Build a compacted session that has a history in front of its summary.

    The shape a rendered file does not have: the renderer writes the summary
    first, so the only transcripts with messages before a summary are the ones an
    agent compacted in place. Used to assert that the cut does not depend on
    which of the two compactions wrote the summary.
    """
    stamp = datetime(2026, 3, 4, 9, 0, tzinfo=UTC)
    return Session(
        id="sess-cut",
        agent=Agent.CLAUDE_CODE,
        source=source,
        native_summary_id=SUMMARY_MESSAGE_ID,
        created_at=stamp,
        messages=[
            Message(id="u01", role=Role.USER, content="add a retry", timestamp=stamp),
            Message(id="a01", role=Role.ASSISTANT, content="done", timestamp=stamp),
            Message(
                id=SUMMARY_MESSAGE_ID,
                role=Role.SYSTEM,
                content="## [C1] Intent: add a retry",
                timestamp=stamp,
                is_system_injected=True,
            ),
            Message(id="u02", role=Role.USER, content="now the timeout", timestamp=stamp),
        ],
    )


# --- the three-way distinction ----------------------------------------------


def test_the_source_table_is_complete() -> None:
    """Guard the parametrization: a half-filled table would skip, not fail."""
    assert SOURCE_CASES
    assert {agent for agent, _ in SOURCE_CASES} == set(Agent)
    assert {source for _, source in SOURCE_CASES} == set(SessionSource)


@pytest.mark.parametrize(("agent", "source"), SOURCE_CASES)
def test_each_provenance_reads_back_as_itself(
    agent: Agent, source: SessionSource, tmp_path: Path
) -> None:
    """Three transcripts per agent, three provenances, told apart by the file alone.

    No filename, path or timestamp is consulted: the two fixtures are read
    through their agent's importer and the third is written by the renderer, and
    all three are imported the same way.
    """
    session = transcript_of(agent, source, tmp_path)

    assert session.agent is agent
    assert session.source is source


@pytest.mark.parametrize("agent", AGENTS)
def test_the_rendered_file_names_the_summary_it_wrote(agent: Agent, tmp_path: Path) -> None:
    """`intent-compaction` plus the identifier — both halves, or the cut is lost."""
    session = reimported(agent, tmp_path)

    assert session.source is SessionSource.INTENT_COMPACTION
    assert session.native_summary_id == SUMMARY_MESSAGE_ID
    summary = summary_of(session)
    assert summary.role is Role.SYSTEM
    assert summary.is_system_injected


# --- the marker itself -------------------------------------------------------


def test_the_marker_decides_which_compaction_wrote_a_summary() -> None:
    """The one rule, in the one place both the builder and a reader can see."""
    assert summary_provenance(f"{INTENT_SUMMARY_MARKER}\n## [C1] Intent: x") is (
        SessionSource.INTENT_COMPACTION
    )
    assert summary_provenance("Session continued from a summary.") is (
        SessionSource.NATIVE_COMPACTION
    )


def test_the_builder_sets_the_provenance_the_marker_names() -> None:
    """One decision point: the importers hand the text over and set nothing themselves."""
    builder = SessionBuilder(Agent.CLAUDE_CODE)
    builder.add_native_summary(
        message_id="c01",
        content=f"{INTENT_SUMMARY_MARKER}\n## [C1] Intent: x",
        timestamp=datetime(2026, 3, 4, 9, 0, tzinfo=UTC),
    )

    session = builder.build(session_id="sess-1").session

    assert session.source is SessionSource.INTENT_COMPACTION
    assert session.native_summary_id == "c01"


@pytest.mark.parametrize("agent", AGENTS)
def test_the_rendered_file_carries_the_marker_once(agent: Agent, tmp_path: Path) -> None:
    """In the summary's own text, and nowhere else in the file."""
    text = rendered_path(agent, tmp_path).read_text(encoding="utf-8")

    assert text.count(INTENT_SUMMARY_MARKER) == 1
    assert INTENT_SUMMARY_MARKER in summary_of(reimported(agent, tmp_path)).content


@pytest.mark.parametrize("agent", AGENTS)
def test_the_block_comes_back_once_the_marker_is_stripped(agent: Agent, tmp_path: Path) -> None:
    """The marker is a prefix line, not a rewrite: the block's bytes are untouched.

    `parse_chains` refuses the summary's text as it stands — a loud refusal, as
    it is for Gemini CLI's snapshot tag — and the chains come back from the text
    with the marker line removed.

    The refusal is asserted twice on purpose. For Gemini CLI the first one is
    satisfied by the `<state_snapshot>` wrapper alone, so it would hold with no
    marker in the file at all; the second strips the wrapper and leaves the
    marker, which is the only thing that can still be refused. Without it this
    parameter pinned nothing of this ticket.
    """
    content = summary_of(reimported(agent, tmp_path)).content

    with pytest.raises(StrayContentError):
        parse_chains(content)

    unwrapped = content.removeprefix(f"{SNAPSHOT_OPEN}\n").removesuffix(SNAPSHOT_CLOSE)
    with pytest.raises(StrayContentError):
        parse_chains(unwrapped)

    stripped = (
        content.removeprefix(f"{SNAPSHOT_OPEN}\n")
        .removesuffix(SNAPSHOT_CLOSE)
        .removeprefix(f"{INTENT_SUMMARY_MARKER}\n")
    )
    session = CORPUS.session(FIXTURE)
    assert (
        stripped
        == assemble(block_of(agent), session=session, chunks=chunk_session(session)).summary
    )
    assert parse_chains(stripped) == block_of(agent).chains


# --- what the measurement counts --------------------------------------------


@pytest.mark.parametrize("agent", AGENTS)
def test_the_summary_this_tool_wrote_is_context(agent: Agent, tmp_path: Path) -> None:
    """The whole rendered file is context, the summary block included.

    The load-bearing half of naming the summary: `visible_context` hides an
    injected `system` message as a harness notice unless it is the one
    `native_summary_id` names, and this summary is an injected `system` message.
    An unnamed one would leave the measured "after" side without the block the
    study's condition consists of.
    """
    session = reimported(agent, tmp_path)

    context = visible_context(session)

    assert context.hidden == ()
    assert context.message_ids[0] == SUMMARY_MESSAGE_ID
    assert list(context.message_ids) == [message.id for message in session.messages]
    assert context.session.source is SessionSource.INTENT_COMPACTION
    assert context.session.native_summary_id == SUMMARY_MESSAGE_ID


def test_the_cut_does_not_depend_on_which_compaction_wrote_the_summary() -> None:
    """The messages before the summary are not context, for either provenance."""
    native = visible_context(history(SessionSource.NATIVE_COMPACTION))
    intent = visible_context(history(SessionSource.INTENT_COMPACTION))

    assert list(intent.message_ids) == [SUMMARY_MESSAGE_ID, "u02"]
    assert list(intent.hidden) == [
        HiddenMessage("u01", HIDDEN_REPLACED_BY_COMPACTION),
        HiddenMessage("a01", HIDDEN_REPLACED_BY_COMPACTION),
    ]
    assert intent.message_ids == native.message_ids
    assert intent.hidden == native.hidden
