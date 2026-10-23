"""Provenance normalisation: the context the agent would see, for every agent.

`visible_context` answers one question — *which messages of this transcript is
the agent still being sent?* — and answers it the same way for all three agents,
which is what makes the study's conditions comparable. Without it, measurement
would count three different things: a Claude Code or OpenCode transcript keeps
the history its own compaction replaced, Gemini CLI's does not, and each agent
records a different amount of harness chatter in its transcript.

```python
from idc.importers import import_transcript, visible_context

context = visible_context(import_transcript("claude-code", "session.jsonl"))
context.message_ids                    # ('c40', 'u41', 'a40', 't40')
[h.reason for h in context.hidden]     # ['replaced-by-compaction']
context.session                        # a canonical Session holding only those
```

A canonical `Session` read back from a file works just as well as the
`ImportedSession` the importer returned: everything this function needs is in
the document.

**Two reasons hide a message, and nothing else does.**

- `replaced-by-compaction` — it sits before the summary a compaction wrote,
  whichever compaction that was: the agent's own in a `native-compaction`
  transcript, this tool's in an `intent-compaction` one. The message the session
  names in `native_summary_id` is where its context starts, and the history
  before it is exactly what was thrown away; counting it would make the
  compaction look like it saved nothing — the over-count both conditions of the
  study are measured against. Gemini CLI's importer
  already drops it, because the agent's own `$set.messages` does (see
  `idc.importers.gemini_cli`); for Claude Code and OpenCode the records stay in
  the file, and this is where they stop being context.
- `harness-notice` — it is an **injected `system`** message that is not that
  summary: Claude Code's `system` records and Gemini CLI's `info` and `error`
  notices. Those are the harness talking to the user — a model switch, a failed
  edit — recorded for the transcript, not sent to the model.

**Injected text in the *user* role stays visible.** A Claude Code
`<system-reminder>`, Gemini CLI's `<session_context>` and OpenCode's `synthetic`
text are all written by the harness rather than typed, but the model is sent
them, so they are context and they are counted. `is_system_injected` is about
*who wrote* a message, not about whether the agent sees it — only the role tells
those two apart, which is why the rule above keys on both.

**What normalisation changes.** Exactly one thing, and only on the way out: a
harness-written notice in the user role is returned with
`is_system_injected = True`. Claude Code records `[Request interrupted by user
for tool use]` as a plain `user` message, and importing it faithfully (T2.2
does) leaves T3.1 reading it as a user turn, i.e. as an intent boundary no human
opened. Flagging it here is what #10 decided: the flag then means the same thing
for all three agents — *the harness wrote this* — and T3.2 reduces it away. The
message stays visible, because the model is sent it. The identifiers this
happened to are in `VisibleContext.normalised_ids`; the imported session itself
is left exactly as the importer produced it.

Known limitations, both deliberate:

- **Only notices with an observed spelling are flagged, and only when the text
  matches whole.** `NOTICE_MARKERS` holds one tuple per agent and is empty for
  Gemini CLI and OpenCode: Gemini CLI's loop-recovery notice (the pointer in
  `idc.importers.gemini_cli`) has no spelling corroborated anywhere in this
  repository, and OpenCode marks harness text with `synthetic` already, which its
  importer reads. Inventing a marker would flag real user turns away — and so
  would matching a corroborated one as a *prefix*: a user who pastes the notice
  and keeps typing has opened a genuine intent, which T3.2 would then reduce
  away. The comparison is equality for that reason, which also means a notice
  spelling this repository has not seen stays unflagged.
- **A session whose summary cannot be located is refused, not guessed at.**
  Where the summary sits is `Session.native_summary_id`, a
  canonical field since T2.6, so a document written by one CLI command and read
  by the next answers the question itself — an `ImportedSession` and the file it
  was dumped to normalise identically, which is what both compacted conditions
  need in order to cross two invocations. Two cases are refused, and both
  are documents the canonical schema accepts on purpose:

  - the provenance says `native-compaction` but `native_summary_id` is `null`.
    An `intent-compaction` document without the identifier is accepted instead:
    T1.4 left the field permitted and never required under that provenance, and
    `idc.render` names the summary it wrote anyway;
  - the identifier names no message of the session, whichever compaction wrote
    it, e.g. because the transcript was truncated at the front past its own
    summary.

  Either way the cut-off point is unknown, and no content marker could recover
  it: the text of a summary the *agent* wrote is arbitrary (OpenCode's
  especially), and `INTENT_SUMMARY_MARKER` says who compacted a session, not
  which message the summary is. Guessing would silently count history the
  compaction had already dropped — the exact failure this tool is built to
  measure — so `visible_context` raises instead. Every other session is accepted
  as it is.

`toolUseResult.interrupted` is **not** read here either, and that is the second
pointer #10 was asked about: a Claude Code tool run the user cancelled still
arrives in the agent's context as an ordinary result, so visibility is not what
is wrong with it. Representing "this run was cancelled" needs a canonical field
the schema does not have, so it belongs to a schema ticket and to T5/T6, which
decide what may become the accepted solution's evidence — not here.
"""

from dataclasses import dataclass

from idc.importers.base import ImportedSession, TranscriptImportError
from idc.schema import Agent, Message, Role, Session, SessionSource

__all__ = [
    "HIDDEN_HARNESS_NOTICE",
    "HIDDEN_REPLACED_BY_COMPACTION",
    "NOTICE_MARKERS",
    "HiddenMessage",
    "VisibleContext",
    "visible_context",
]

HIDDEN_REPLACED_BY_COMPACTION = "replaced-by-compaction"
"""The message sits before the summary the session names, so it was replaced."""

HIDDEN_HARNESS_NOTICE = "harness-notice"
"""The message is an injected `system` notice, recorded but never sent."""

NOTICE_MARKERS: dict[Agent, tuple[str, ...]] = {
    # The one spelling this repository corroborates, carried verbatim by
    # `tests/native/claude-code/interrupted_turn.jsonl`: the harness records the
    # interruption as a `user` message whose whole text is the notice.
    Agent.CLAUDE_CODE: ("[Request interrupted by user for tool use]",),
    # Gemini CLI injects a loop-recovery notice as a user turn, but no spelling of
    # it is corroborated in this repository, so none is matched here.
    Agent.GEMINI_CLI: (),
    # OpenCode marks harness-written text with `synthetic`, which its importer
    # reads into `is_system_injected` already.
    Agent.OPENCODE: (),
}
"""The complete texts that identify a harness-written notice in the user role.

One entry per agent, so that a missing agent is a `KeyError` rather than a
silent "this agent has no notices". A text is only listed when a transcript or a
fixture in this repository shows it, and it is matched **whole**: a guessed
marker, or a prefix test, would mark a real user turn as harness-written and
T3.2 would reduce the user's own intent away. A notice spelling this repository
has never seen is therefore not flagged — under-flagging keeps a harness turn in
the context, over-flagging deletes an intent boundary.
"""


@dataclass(frozen=True, slots=True)
class HiddenMessage:
    """One message that is in the transcript but not in the agent's context.

    Attributes:
        message_id: `Message.id` of the message that is not counted.
        reason: Why it is not — `HIDDEN_REPLACED_BY_COMPACTION` or
            `HIDDEN_HARNESS_NOTICE`.
    """

    message_id: str
    reason: str

    def __str__(self) -> str:
        """Return the exclusion as one diagnostic line."""
        return f"{self.reason}: {self.message_id}"


@dataclass(frozen=True, slots=True)
class VisibleContext:
    """The part of a transcript the agent would see, plus what was left out.

    Attributes:
        session: The imported session restricted to the visible messages. Its
            identifier, agent, model, provenance and `created_at` are the
            original's — this is the same session, measured as the agent sees
            it, not a different one — so it dumps and validates like any
            canonical document.
        hidden: One entry per message that is not context, in transcript order.
        normalised_ids: Identifiers of the visible messages this normalisation
            marked `is_system_injected`, in transcript order.
    """

    session: Session
    hidden: tuple[HiddenMessage, ...] = ()
    normalised_ids: tuple[str, ...] = ()

    @property
    def messages(self) -> tuple[Message, ...]:
        """Return the visible messages, in transcript order."""
        return tuple(self.session.messages)

    @property
    def message_ids(self) -> tuple[str, ...]:
        """Return the identifiers of the visible messages, in transcript order."""
        return tuple(message.id for message in self.session.messages)

    @property
    def hidden_ids(self) -> tuple[str, ...]:
        """Return the identifiers of the hidden messages, in transcript order."""
        return tuple(hidden.message_id for hidden in self.hidden)


def visible_context(session: ImportedSession | Session) -> VisibleContext:
    """Return the messages of a transcript the agent is still being sent.

    The one place that decides what counts as context, so that measurement
    counts the same thing whichever agent recorded the transcript. See this
    module's docstring for the two reasons a message is left out and for the one
    flag normalisation sets.

    Args:
        session: An `ImportedSession`, or a canonical `Session` on its own —
            including one loaded back from a file, which carries its own
            `native_summary_id`.

    Returns:
        The visible messages as a canonical session, carrying the provenance and
        the summary identifier of the original, together with what was left out
        and what was flagged.

    Raises:
        TranscriptImportError: If a compacted session does not locate its
            summary. Two cases, and they are not symmetric: a
            `native-compaction` session whose `native_summary_id` is `null` is
            refused, and so is *any* session — either compaction — whose
            identifier names no message of it. An `intent-compaction` session
            that names no summary is **not** refused, because T1.4 left the
            identifier permitted and never required there; it normalises as it
            stands, and `idc.render` always names the summary it wrote. Both
            refusals are about counting history a compaction dropped, which is
            the error this function exists to prevent.
        SchemaValidationError: If the visible messages do not satisfy the
            canonical schema.
    """
    imported, summary_id = _unpack(session)
    cut = _summary_index(imported, summary_id)
    visible: list[Message] = []
    hidden: list[HiddenMessage] = []
    normalised: list[str] = []
    for index, message in enumerate(imported.messages):
        reason = _hidden_reason(message, index=index, cut=cut, summary_id=summary_id)
        if reason is not None:
            hidden.append(HiddenMessage(message.id, reason))
        elif _is_harness_notice(imported.agent, message):
            visible.append(_as_injected(message))
            normalised.append(message.id)
        else:
            visible.append(message)
    return VisibleContext(
        session=Session(
            id=imported.id,
            agent=imported.agent,
            model=imported.model,
            source=imported.source,
            native_summary_id=summary_id,
            created_at=imported.created_at,
            messages=visible,
        ),
        hidden=tuple(hidden),
        normalised_ids=tuple(normalised),
    )


def _unpack(session: ImportedSession | Session) -> tuple[Session, str | None]:
    """Return the canonical session and the identifier of its native summary.

    Both arguments are read the same way: the identifier is a field of `Session`,
    so an `ImportedSession` and the document it was dumped to answer identically
    and a stage does not have to have done the import itself.

    Args:
        session: What the caller passed — an `ImportedSession` or a `Session`.

    Returns:
        The session, and the summary's identifier or `None` when the session
        names none. `None` therefore covers two unlike cases: a transcript no
        compaction touched, and an `intent-compaction` one that names no block —
        which the Raises clause below says passes. Only `native-compaction` must
        name a summary, so only there does `None` mean the document is unusable.

    Raises:
        TranscriptImportError: If a `native-compaction` session names no
            summary. The canonical schema accepts that document on purpose (see
            `idc.schema.models`); it is this function that cannot proceed,
            because nothing in it says where the agent's context starts. An
            `intent-compaction` session without the identifier passes here: the
            schema permits it and the field is not required there, so it has no
            cut and is normalised as it stands.
    """
    unpacked = session.session if isinstance(session, ImportedSession) else session
    if unpacked.source is SessionSource.NATIVE_COMPACTION and unpacked.native_summary_id is None:
        raise TranscriptImportError(
            f"session {unpacked.id!r} was compacted by {unpacked.agent.value} itself, "
            "but names no native summary; the document has to record the message the "
            "agent continued from in native_summary_id, or the history it dropped "
            "would be counted as context"
        )
    return unpacked, unpacked.native_summary_id


def _summary_index(session: Session, summary_id: str | None) -> int | None:
    """Return the position of the native summary in the session's messages.

    Args:
        session: The session being normalised.
        summary_id: Identifier the importer recorded for the summary, or `None`.

    Returns:
        The index of that message, or `None` when there is no summary.

    Raises:
        TranscriptImportError: If the identifier names no message of the
            session, which would leave the cut-off point unknown.
    """
    if summary_id is None:
        return None
    for index, message in enumerate(session.messages):
        if message.id == summary_id:
            return index
    raise TranscriptImportError(
        f"session {session.id!r} names {summary_id!r} as its native summary, "
        "but no message of it carries that identifier"
    )


def _hidden_reason(
    message: Message, *, index: int, cut: int | None, summary_id: str | None
) -> str | None:
    """Return why a message is not context, or `None` when it is.

    Args:
        message: The message being judged.
        index: Its position in the session.
        cut: Position of the native summary, or `None` when there is none.
        summary_id: Identifier of the native summary, or `None`.

    Returns:
        `HIDDEN_REPLACED_BY_COMPACTION` for a message that sits before the
        summary, whichever compaction wrote it — the agent's own in a
        `native-compaction` transcript, this tool's in an `intent-compaction`
        one, as the module docstring says. `HIDDEN_HARNESS_NOTICE` for an
        injected `system` notice. `None` for everything else — including the
        summary itself, which is the one injected `system` message the agent
        does see, **when the session names it**. An unnamed summary is hidden as
        a harness notice like any other injected message, which is why a
        transcript this tool compacted names its own block.
    """
    if cut is not None and index < cut:
        return HIDDEN_REPLACED_BY_COMPACTION
    if message.id == summary_id:
        return None
    if message.role is Role.SYSTEM and message.is_system_injected:
        return HIDDEN_HARNESS_NOTICE
    return None


def _is_harness_notice(agent: Agent, message: Message) -> bool:
    """Return whether a user message is a notice the harness wrote.

    Args:
        agent: The agent whose markers apply.
        message: The message being judged.

    Returns:
        Whether the message is in the user role, is not flagged as injected
        already, and its content is exactly one of the agent's notice texts. The
        comparison is equality, not a prefix test: the harness writes the notice
        as the message's whole text, so anything around it is text a human typed.

    Raises:
        KeyError: If the agent has no entry in `NOTICE_MARKERS`, which would
            mean an agent was added without deciding what its notices look like.
    """
    if message.role is not Role.USER or message.is_system_injected:
        return False
    return message.content in NOTICE_MARKERS[agent]


def _as_injected(message: Message) -> Message:
    """Return a copy of a message marked as written by the harness.

    Built through the canonical constructor rather than copied field by field in
    place: the imported session must stay exactly as the importer produced it,
    and the copy has to satisfy the schema like any other message. The fields
    come from `Message.model_fields` rather than from a list written out here,
    so a field the canonical schema gains later is carried over instead of being
    silently reset to its default — `extra="forbid"` catches a field too many,
    nothing would catch a field too few.

    Args:
        message: The message to flag.

    Returns:
        The same message with `is_system_injected` set.

    Raises:
        SchemaValidationError: If the copy does not satisfy the canonical schema.
    """
    fields = {name: getattr(message, name) for name in Message.model_fields}
    return Message(**{**fields, "is_system_injected": True})
