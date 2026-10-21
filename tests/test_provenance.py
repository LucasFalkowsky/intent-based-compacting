"""Provenance normalisation: the context the agent would see, for all three agents."""

from datetime import UTC, datetime

import pytest

from idc.chunk import reduce_session
from idc.importers import (
    HIDDEN_HARNESS_NOTICE,
    HIDDEN_REPLACED_BY_COMPACTION,
    NOTICE_MARKERS,
    HiddenMessage,
    ImportedSession,
    TranscriptImportError,
    VisibleContext,
    import_transcript,
    visible_context,
)
from idc.schema import (
    Agent,
    Message,
    Role,
    SchemaValidationError,
    Session,
    SessionSource,
    ToolCall,
    ToolCategory,
    ToolResult,
    dumps,
    loads,
)
from native_transcripts import NATIVE

RAW_FIXTURES = {
    Agent.CLAUDE_CODE: ("golden_session", ".jsonl"),
    Agent.OPENCODE: ("golden_session", ".jsonl"),
    Agent.GEMINI_CLI: ("golden_session", ".json"),
}
"""One raw, full-coverage transcript per agent — the fixture's name and suffix."""

COMPACTED_FIXTURES = {
    Agent.CLAUDE_CODE: ("native_compaction", ".jsonl"),
    Agent.OPENCODE: ("native_compaction", ".jsonl"),
    Agent.GEMINI_CLI: ("native_compaction", ".jsonl"),
}
"""The natively compacted transcript of each agent."""

RAW_VISIBLE = {
    Agent.CLAUDE_CODE: ["u01", "a01", "t01", "a02", "t02#0", "t02#1", "u02", "a03", "t03", "a04"],
    Agent.OPENCODE: [
        "msg_01u",
        "msg_02a",
        "msg_02a#3",
        "msg_03a",
        "msg_03a#1",
        "msg_03a#2",
        "msg_04u",
        "msg_05a",
        "msg_05a#0",
        "msg_06a",
    ],
    # info01 and err01 are the harness's own notices and are not context.
    Agent.GEMINI_CLI: [
        "u01",
        "g01#0",
        "g01#1",
        "g02#0",
        "g02#1",
        "g02#2",
        "u02",
        "g03#0",
        "g03#1",
        "g04",
    ],
}
"""The visible message identifiers of each agent's raw transcript."""

RAW_HIDDEN = {
    Agent.CLAUDE_CODE: [],
    Agent.OPENCODE: [],
    Agent.GEMINI_CLI: [
        HiddenMessage("info01", HIDDEN_HARNESS_NOTICE),
        HiddenMessage("err01", HIDDEN_HARNESS_NOTICE),
    ],
}
"""What each raw transcript hides, in transcript order."""

COMPACTED_VISIBLE = {
    Agent.CLAUDE_CODE: ["c40", "u41", "a40", "t40"],
    Agent.OPENCODE: ["msg_22c", "msg_23u"],
    Agent.GEMINI_CLI: ["c40", "c41", "u71"],
}
"""The visible identifiers after the agent compacted the session itself."""

COMPACTED_HIDDEN = {
    Agent.CLAUDE_CODE: [HiddenMessage("u40", HIDDEN_REPLACED_BY_COMPACTION)],
    Agent.OPENCODE: [HiddenMessage("msg_21u", HIDDEN_REPLACED_BY_COMPACTION)],
    # Gemini CLI's own `$set.messages` already dropped the replaced history.
    Agent.GEMINI_CLI: [],
}
"""What the compaction replaced, in transcript order."""

SUMMARY_IDS = {
    Agent.CLAUDE_CODE: "c40",
    Agent.OPENCODE: "msg_22c",
    Agent.GEMINI_CLI: "c40",
}
"""The native summary of each compacted fixture."""


def imported(agent: Agent, fixtures: dict[Agent, tuple[str, str]]) -> ImportedSession:
    name, suffix = fixtures[agent]
    return import_transcript(agent, NATIVE.path(agent, name, suffix))


def message(message_id: str, role: Role = Role.USER, **extra) -> Message:
    return Message(id=message_id, role=role, **extra)


def session(
    *messages: Message,
    source: SessionSource = SessionSource.RAW,
    agent: Agent = Agent.CLAUDE_CODE,
    native_summary_id: str | None = None,
) -> Session:
    return Session(
        id="sess-1",
        agent=agent,
        source=source,
        native_summary_id=native_summary_id,
        created_at=datetime(2026, 9, 10, 9, tzinfo=UTC),
        messages=list(messages),
    )


# --- the same visible set for all three agents -------------------------------


@pytest.mark.parametrize("agent", list(Agent))
def test_a_raw_session_is_visible_apart_from_the_harness_notices(agent):
    context = visible_context(imported(agent, RAW_FIXTURES))

    assert list(context.message_ids) == RAW_VISIBLE[agent]
    assert list(context.hidden) == RAW_HIDDEN[agent]


@pytest.mark.parametrize("agent", list(Agent))
def test_a_natively_compacted_session_is_visible_from_its_summary_on(agent):
    context = visible_context(imported(agent, COMPACTED_FIXTURES))

    assert list(context.message_ids) == COMPACTED_VISIBLE[agent]
    assert list(context.hidden) == COMPACTED_HIDDEN[agent]


@pytest.mark.parametrize("agent", list(Agent))
def test_the_native_summary_is_the_first_visible_message_of_every_agent(agent):
    context = visible_context(imported(agent, COMPACTED_FIXTURES))

    first = context.messages[0]

    assert first.id == SUMMARY_IDS[agent]
    assert (first.role, first.is_system_injected) == (Role.SYSTEM, True)


@pytest.mark.parametrize("agent", list(Agent))
def test_every_message_is_either_visible_or_hidden_exactly_once(agent):
    for fixtures in (RAW_FIXTURES, COMPACTED_FIXTURES):
        result = imported(agent, fixtures)
        context = visible_context(result)

        accounted = list(context.message_ids) + list(context.hidden_ids)

        assert sorted(accounted) == sorted(m.id for m in result.session.messages)


@pytest.mark.parametrize("agent", list(Agent))
def test_the_visible_session_keeps_the_sessions_own_provenance(agent):
    result = imported(agent, COMPACTED_FIXTURES)

    visible = visible_context(result).session

    assert visible.source is SessionSource.NATIVE_COMPACTION
    assert (visible.id, visible.agent, visible.model) == (
        result.session.id,
        result.session.agent,
        result.session.model,
    )
    assert visible.created_at == result.session.created_at


@pytest.mark.parametrize("agent", list(Agent))
def test_the_visible_session_is_a_valid_canonical_document(agent):
    context = visible_context(imported(agent, RAW_FIXTURES))

    assert loads(Session, dumps(context.session)) == context.session


# --- what the importers mark ------------------------------------------------


@pytest.mark.parametrize("agent", list(Agent))
def test_the_importer_names_the_native_summary_it_imported(agent):
    result = imported(agent, COMPACTED_FIXTURES)

    assert result.native_summary_id == SUMMARY_IDS[agent]
    assert result.session.native_summary_id == SUMMARY_IDS[agent]


@pytest.mark.parametrize("agent", list(Agent))
def test_a_raw_transcript_names_no_native_summary(agent):
    result = imported(agent, RAW_FIXTURES)

    assert result.native_summary_id is None
    assert result.session.native_summary_id is None


# --- harness text in the user role ------------------------------------------


def test_an_injected_user_message_stays_visible():
    # `<system-reminder>`, `<session_context>` and OpenCode's `synthetic` text are
    # harness-written, but the agent is sent them, so they are context.
    for agent, message_id in (
        (Agent.CLAUDE_CODE, "u02"),
        (Agent.OPENCODE, "msg_04u"),
    ):
        context = visible_context(imported(agent, RAW_FIXTURES))

        assert message_id in context.message_ids

    gemini = visible_context(
        import_transcript(Agent.GEMINI_CLI, NATIVE.path("gemini-cli", "history_parts"))
    )

    assert "env01" in gemini.message_ids


def test_a_harness_written_notice_is_marked_injected_and_stays_visible():
    result = import_transcript(Agent.CLAUDE_CODE, NATIVE.path("claude-code", "interrupted_turn"))

    context = visible_context(result)
    notice = context.messages[-1]

    assert result.session.messages[-1].is_system_injected is False
    assert (notice.id, notice.role, notice.is_system_injected) == ("u21", Role.USER, True)
    assert list(context.normalised_ids) == ["u21"]
    assert list(context.message_ids) == ["u20", "a20", "u21"]


def test_a_user_turn_that_only_resembles_a_notice_is_left_alone():
    marker = NOTICE_MARKERS[Agent.CLAUDE_CODE][0]
    typed = message("u1", content=f"Explain why you print {marker!r}")

    context = visible_context(session(typed))

    assert context.normalised_ids == ()
    assert context.messages[0].is_system_injected is False


def test_a_user_turn_that_starts_with_a_notice_and_keeps_typing_is_left_alone():
    # The notice is matched as a whole text, not as a prefix: a user who pastes it
    # and keeps typing has opened a genuine intent, and flagging it would let T3.2
    # reduce that intent away.
    typed = message("u1", content="[Request interrupted by user] was printed by my script, explain")

    context = visible_context(session(typed))

    assert context.normalised_ids == ()
    assert context.messages[0].is_system_injected is False


def test_a_notice_marker_is_the_whole_text_the_transcript_records():
    result = import_transcript(Agent.CLAUDE_CODE, NATIVE.path("claude-code", "interrupted_turn"))

    notice = result.session.messages[-1]

    assert notice.content in NOTICE_MARKERS[Agent.CLAUDE_CODE]


def test_flagging_a_notice_changes_nothing_but_the_flag():
    # Driven by the model's own field list rather than by a hand-written one, so
    # that normalisation cannot quietly reset a field to its default.
    notice = Message(
        id="u1",
        role=Role.USER,
        content=NOTICE_MARKERS[Agent.CLAUDE_CODE][0],
        timestamp=datetime(2026, 9, 12, 8, 0, 7, tzinfo=UTC),
        is_command=True,
        tool_calls=[
            ToolCall(id="t1", name="Edit", category=ToolCategory.WRITE, args={"p": "a.py"})
        ],
        tool_result=ToolResult(call_id="t0", output="done", exit_code=0),
    )

    normalised = visible_context(session(notice)).messages[0]

    differing = {
        name for name in Message.model_fields if getattr(normalised, name) != getattr(notice, name)
    }

    assert differing == {"is_system_injected"}
    assert normalised.is_system_injected is True
    assert notice.is_system_injected is False


def test_every_agent_has_an_entry_in_the_notice_markers():
    assert set(NOTICE_MARKERS) == set(Agent)


def test_every_notice_marker_entry_is_a_tuple_of_whole_texts():
    # `_is_harness_notice` asks `message.content in NOTICE_MARKERS[agent]`, and that
    # `in` is an equality test only while the entry is a tuple of texts. A single
    # dropped comma turns a one-element entry into a bare `str`, and the same `in`
    # becomes substring matching — so the container's type is part of the contract.
    assert all(isinstance(markers, tuple) for markers in NOTICE_MARKERS.values())
    assert all(isinstance(marker, str) for markers in NOTICE_MARKERS.values() for marker in markers)


@pytest.mark.parametrize("agent", list(Agent))
def test_no_fragment_of_a_notice_marker_is_matched_on_its_own(agent):
    # The behaviour the tuple protects, pinned without naming the container: only the
    # whole notice text is a notice. A single character of a marker is a user turn a
    # human typed — flagging it would hand T3.2 an intent boundary to reduce away.
    for marker in NOTICE_MARKERS[agent]:
        fragments = {marker[0], marker[-1], marker[1:], marker[:-1], marker[: len(marker) // 2]}
        for fragment in sorted(fragments - {marker}):
            context = visible_context(session(message("u1", content=fragment), agent=agent))

            assert context.normalised_ids == (), fragment
            assert context.messages[0].is_system_injected is False, fragment


# --- a session the agent compacted more than once ---------------------------


def test_a_session_compacted_twice_is_visible_from_the_second_summary():
    result = import_transcript(Agent.CLAUDE_CODE, NATIVE.path("claude-code", "compacted_twice"))

    context = visible_context(result)

    assert result.native_summary_id == "c51"
    assert list(context.message_ids) == ["c51", "u52"]
    assert list(context.hidden) == [
        HiddenMessage("u50", HIDDEN_REPLACED_BY_COMPACTION),
        HiddenMessage("c50", HIDDEN_REPLACED_BY_COMPACTION),
        HiddenMessage("u51", HIDDEN_REPLACED_BY_COMPACTION),
    ]


# --- a session that was not imported in this process ------------------------


def test_a_raw_session_can_be_normalised_on_its_own():
    context = visible_context(session(message("u1", content="go")))

    assert list(context.message_ids) == ["u1"]
    assert isinstance(context, VisibleContext)


@pytest.mark.parametrize("source", [SessionSource.RAW, SessionSource.INTENT_COMPACTION])
def test_a_session_the_agent_never_compacted_is_normalised_as_it_is(source):
    # Only `native-compaction` needs the summary identifier; the other two
    # provenances are normalised on their own, which is what the module promises.
    plain = session(message("u1", content="go"), source=source)

    context = visible_context(plain)

    assert list(context.message_ids) == ["u1"]
    assert context.hidden == ()
    assert context.session.source is source


@pytest.mark.parametrize("source", [SessionSource.RAW, SessionSource.INTENT_COMPACTION])
def test_an_imported_session_without_a_native_summary_is_normalised_as_it_is(source):
    result = ImportedSession(session=session(message("u1", content="go"), source=source))

    context = visible_context(result)

    assert list(context.message_ids) == ["u1"]
    assert context.hidden == ()
    assert context.session.source is source


def test_a_compacted_session_without_its_summary_is_refused():
    plain = session(message("u1", content="go"), source=SessionSource.NATIVE_COMPACTION)

    with pytest.raises(TranscriptImportError, match="native summary"):
        visible_context(plain)


def test_a_summary_identifier_that_is_not_in_the_session_is_refused():
    # Schema-valid on purpose — referential completeness is not enforced (#3) — and
    # refused here, because the cut-off point cannot be located.
    truncated = session(
        message("u1", content="go"),
        source=SessionSource.NATIVE_COMPACTION,
        native_summary_id="c9",
    )

    with pytest.raises(TranscriptImportError, match="c9"):
        visible_context(truncated)

    with pytest.raises(TranscriptImportError, match="c9"):
        visible_context(ImportedSession(session=truncated))


# --- provenance and the summary identifier say the same thing ----------------


def test_an_imported_compacted_session_that_names_no_summary_cannot_be_built():
    # The gap this closes: such a session used to normalise silently, with every
    # message the agent had already dropped counted as context.
    compacted = session(message("u1", content="go"), source=SessionSource.NATIVE_COMPACTION)

    with pytest.raises(TranscriptImportError, match="native summary"):
        ImportedSession(session=compacted)


def test_a_session_that_names_a_summary_must_be_compacted_natively():
    # The other half of the pair is now a canonical invariant: both fields live on
    # `Session`, so the contradiction is refused by the schema and `ImportedSession`
    # can no longer be handed one — the identifier is read off the session it names.
    summary = message("c1", role=Role.SYSTEM, content="summary", is_system_injected=True)

    for source in (SessionSource.RAW, SessionSource.INTENT_COMPACTION):
        with pytest.raises(SchemaValidationError, match="native-compaction"):
            session(summary, source=source, native_summary_id="c1")


def test_the_imported_session_reads_its_summary_identifier_off_the_session():
    # One place holds the fact, so the two cannot drift apart: `ImportedSession`
    # no longer carries a second copy that a caller could set independently.
    result = import_transcript(Agent.CLAUDE_CODE, NATIVE.path("claude-code", "native_compaction"))

    assert result.native_summary_id == result.session.native_summary_id == "c40"
    with pytest.raises(TypeError):
        ImportedSession(session=result.session, native_summary_id="c40")


def test_the_importers_build_sessions_that_satisfy_the_invariant():
    for agent in Agent:
        assert imported(agent, RAW_FIXTURES).native_summary_id is None
        assert imported(agent, COMPACTED_FIXTURES).native_summary_id is not None


# --- the summary survives a file, so two CLI invocations can see it ----------


@pytest.mark.parametrize("agent", list(Agent))
def test_a_natively_compacted_session_round_trips_with_its_summary_identified(agent):
    # The gap this ticket closes: T10.1 runs one command per stage with files in
    # between, so what the file does not carry is lost between two invocations.
    original = imported(agent, COMPACTED_FIXTURES).session

    reloaded = loads(Session, dumps(original))

    assert reloaded.native_summary_id == SUMMARY_IDS[agent]
    assert reloaded == original


@pytest.mark.parametrize("agent", list(Agent))
def test_a_reloaded_document_normalises_exactly_like_the_imported_session(agent):
    result = imported(agent, COMPACTED_FIXTURES)

    from_import = visible_context(result)
    from_file = visible_context(loads(Session, dumps(result.session)))

    assert from_file.message_ids == from_import.message_ids
    assert from_file.hidden == from_import.hidden
    assert from_file.normalised_ids == from_import.normalised_ids
    assert from_file == from_import


@pytest.mark.parametrize("agent", list(Agent))
def test_a_reloaded_raw_document_normalises_exactly_like_the_imported_session(agent):
    result = imported(agent, RAW_FIXTURES)

    from_file = visible_context(loads(Session, dumps(result.session)))

    assert from_file == visible_context(result)


@pytest.mark.parametrize("agent", list(Agent))
def test_the_visible_session_identifies_the_summary_it_starts_at(agent):
    # The visible context is itself a canonical document a later stage reads from a
    # file, so it has to stay self-describing — otherwise normalising it a second
    # time would be the refusal this ticket exists to remove.
    visible = visible_context(imported(agent, COMPACTED_FIXTURES)).session

    assert visible.native_summary_id == SUMMARY_IDS[agent]
    renormalised = visible_context(loads(Session, dumps(visible)))

    assert list(renormalised.message_ids) == COMPACTED_VISIBLE[agent]
    assert renormalised.hidden == ()


@pytest.mark.parametrize("agent", list(Agent))
def test_a_reloaded_raw_session_names_no_summary(agent):
    assert loads(Session, dumps(imported(agent, RAW_FIXTURES).session)).native_summary_id is None


def test_a_session_compacted_twice_round_trips_naming_the_second_summary():
    original = import_transcript(
        Agent.CLAUDE_CODE, NATIVE.path("claude-code", "compacted_twice")
    ).session

    reloaded = loads(Session, dumps(original))

    assert reloaded.native_summary_id == "c51"
    assert list(visible_context(reloaded).message_ids) == ["c51", "u52"]
    assert list(visible_context(reloaded).hidden) == [
        HiddenMessage("u50", HIDDEN_REPLACED_BY_COMPACTION),
        HiddenMessage("c50", HIDDEN_REPLACED_BY_COMPACTION),
        HiddenMessage("u51", HIDDEN_REPLACED_BY_COMPACTION),
    ]


# --- …and survives the stages between those two invocations ------------------


@pytest.mark.parametrize("agent", list(Agent))
def test_a_reduced_natively_compacted_session_normalises_like_the_imported_one(agent):
    # The identifier is only useful if the stages between the import and the
    # normalisation carry it. The chunker's reduction rebuilds the session, so
    # it is the first stage that can lose it — and a reduced transcript that no
    # longer says where the agent's context starts is refused, which is the
    # refusal this ticket exists to remove.
    result = imported(agent, COMPACTED_FIXTURES)

    reduced = reduce_session(result.session).session

    assert reduced.native_summary_id == SUMMARY_IDS[agent]
    context = visible_context(reduced)
    assert list(context.message_ids) == COMPACTED_VISIBLE[agent]
    assert list(context.hidden) == COMPACTED_HIDDEN[agent]
    assert context.messages[0].id == SUMMARY_IDS[agent]


@pytest.mark.parametrize("agent", list(Agent))
def test_every_message_of_a_reduced_session_is_either_visible_or_hidden_exactly_once(agent):
    reduced = reduce_session(imported(agent, COMPACTED_FIXTURES).session).session

    context = visible_context(reduced)

    accounted = list(context.message_ids) + list(context.hidden_ids)
    assert sorted(accounted) == sorted(message.id for message in reduced.messages)


def test_a_session_compacted_twice_still_names_its_second_summary_after_reduction():
    # The shape the preamble rule cannot protect by position: the second summary
    # block sits after a real user turn, so reduction would have dropped it and
    # left the identifier naming a message the document no longer holds.
    session = import_transcript(
        Agent.CLAUDE_CODE, NATIVE.path("claude-code", "compacted_twice")
    ).session

    reduced = reduce_session(session).session

    assert reduced.native_summary_id == "c51"
    assert "c51" in [message.id for message in reduced.messages]
    assert list(visible_context(reduced).message_ids) == ["c51", "u52"]


# --- the result is stable ----------------------------------------------------


@pytest.mark.parametrize("agent", list(Agent))
def test_normalising_the_same_session_twice_yields_the_same_result(agent):
    result = imported(agent, COMPACTED_FIXTURES)

    assert visible_context(result) == visible_context(result)


def test_a_hidden_message_reads_as_one_diagnostic_line():
    context = visible_context(imported(Agent.CLAUDE_CODE, COMPACTED_FIXTURES))

    assert [str(hidden) for hidden in context.hidden] == ["replaced-by-compaction: u40"]


def test_the_hidden_reasons_are_the_two_the_module_declares():
    gemini = visible_context(imported(Agent.GEMINI_CLI, RAW_FIXTURES))
    claude = visible_context(imported(Agent.CLAUDE_CODE, COMPACTED_FIXTURES))

    reasons = {hidden.reason for hidden in gemini.hidden + claude.hidden}

    assert reasons == {HIDDEN_HARNESS_NOTICE, HIDDEN_REPLACED_BY_COMPACTION}
