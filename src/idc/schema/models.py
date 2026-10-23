"""The canonical types — the contract between all pipeline stages.

The seven types below cover every state a transcript passes through: the
imported session (`Session` with its `Message`, `ToolCall` and `ToolResult`),
the chunked session (`Chunk`), the classified session (`Chain`) and the
compacted output (`CompactedBlock`). Agent-specific shapes exist only inside
`idc.importers` and `idc.render`; everything in between sees these types.

Four conventions hold throughout and are enforced, not documented only:

- **No unknown fields.** Every model forbids extras, so a field an importer
  invents is an error instead of silent data that no later stage reads.
- **Timestamps are timezone-aware.** Agents record instants in different
  offsets; a naive timestamp cannot be ordered against them, so importers must
  attach the offset (UTC when the agent records none).
- **Identifiers are unique where they are referenced.** `ToolResult.call_id`,
  `Chunk`'s message range and `Chain.chunk_ids` are references; the containers
  reject duplicate message and tool-call identifiers so that resolving a
  reference can never be ambiguous.
- **Only one error type leaves the package.** Construction and assignment are
  translated into `SchemaValidationError` exactly like the load path, so an
  importer that builds a `Message` from a malformed transcript cannot surface a
  pydantic traceback at the CLI.

What is *not* enforced: that a reference resolves inside the document that
carries it — neither that every `ToolResult.call_id` has a matching `ToolCall`
in the same session, nor that `Session.native_summary_id` names one of its
messages. Transcripts can be truncated at the front, and what the cut removed is
still data the measurement needs; the stage that has to resolve a reference is
the one that refuses when it cannot.
"""

from collections.abc import Iterable
from typing import Annotated, Any, Self

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, ValidationError, model_validator
from pydantic_core import InitErrorDetails, PydanticCustomError

from idc.schema.enums import Agent, ChunkOrigin, Role, SessionSource, ToolCategory
from idc.schema.errors import as_schema_error

Identifier = Annotated[str, Field(min_length=1)]
"""A non-empty identifier taken from the transcript or assigned by a stage."""

CHAIN_ID_PATTERN = r"^C[1-9][0-9]*$"
"""`C<n>` with `n` counted from 1 — the stable chain ID the summary blocks use."""


_ModelMetaclass: type = type(BaseModel)
"""Pydantic's own model metaclass, read off `BaseModel`.

Taken this way rather than imported from `pydantic._internal`, which is private
and may move between pydantic releases.
"""


class _CanonicalMeta(_ModelMetaclass):
    """Metaclass that reports a failed construction as a `SchemaValidationError`.

    The translation sits on the metaclass, not on `__init__`, and that placement
    is the whole point: pydantic calls a model's `__init__` again for every
    nested model it builds, so wrapping `__init__` would turn the one located
    error of `Session(messages=[{...}])` into an error about `messages.0` with a
    second error nested inside its text — and it would do the same on the load
    path, which must keep reporting exactly as it does today. A metaclass
    `__call__` runs only for an explicit `Model(...)` from Python: pydantic
    builds nested models and `model_validate` without going through it.
    """

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:  # noqa: N805 - metaclass method
        """Construct an instance, reporting invalid input as a schema error."""
        try:
            return super().__call__(*args, **kwargs)
        except ValidationError as exc:
            raise as_schema_error(exc, cls) from None


class CanonicalModel(BaseModel, metaclass=_CanonicalMeta):
    """Base class carrying the configuration every canonical type shares.

    Unknown fields are rejected, assignments are validated like constructor
    arguments (a stage cannot mutate an object into an invalid one), and enum
    members are kept as enum members in Python while serializing to their wire
    strings.

    Invalid input is reported as a `SchemaValidationError` whichever way it
    arrives — `load`/`loads`/`validate_document`, construction, or assignment —
    so no stage and no CLI user ever sees a pydantic traceback.
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        use_enum_values=False,
    )

    def __setattr__(self, name: str, value: Any) -> None:
        """Assign a field, reporting an invalid value as a schema error."""
        try:
            super().__setattr__(name, value)
        except ValidationError as exc:
            raise as_schema_error(exc, type(self)) from None


def _field_error(model: type[BaseModel], field: str, value: Any, message: str) -> ValidationError:
    """Build a pydantic error located at one field instead of at the document.

    A `model_validator` sees the whole object, so the error it raises carries no
    field location and would be reported as `<document>`. An invariant that is
    *about* one field raises this instead, so that construction, assignment and
    the load path all name that field — one dotted field path per problem.

    Args:
        model: The canonical type the invariant belongs to.
        field: Name of the field the problem is reported against.
        value: The offending value, echoed as the error's input.
        message: The complete message; it is used verbatim, so it must not
            contain `{}` placeholders.

    Returns:
        The pydantic error to raise from the validator.
    """
    return ValidationError.from_exception_data(
        model.__name__,
        [
            InitErrorDetails(
                type=PydanticCustomError("value_error", message),
                loc=(field,),
                input=value,
            )
        ],
    )


def _reject_duplicates(values: Iterable[str], what: str) -> None:
    """Raise if a value occurs twice, naming every repeated one.

    Args:
        values: The identifiers to check, in document order.
        what: What is repeated, used to build the message `duplicate <what>: …`.

    Raises:
        ValueError: If any value occurs more than once.
    """
    seen: set[str] = set()
    duplicated: list[str] = []
    for value in values:
        if value in seen and value not in duplicated:
            duplicated.append(value)
        seen.add(value)
    if duplicated:
        raise ValueError(f"duplicate {what}: {', '.join(duplicated)}")


class ToolCall(CanonicalModel):
    """A call the agent made to one of its tools.

    The arguments stay as the agent sent them: an arbitrary JSON object, never
    reformatted, so that the resolver can read a file path or an edit out of
    them and the renderer can reproduce them verbatim.
    """

    id: Identifier = Field(description="Identifier the agent gave the call; unique per session.")
    name: Identifier = Field(description="Tool name as the agent reports it, e.g. `Edit`.")
    category: ToolCategory = Field(description="What the call does to the workspace.")
    args: dict[str, Any] = Field(
        default_factory=dict,
        description="The call's arguments as a JSON object, unmodified.",
    )


class ToolResult(CanonicalModel):
    """The result of a tool call, as the agent received it back.

    Results are never summarized by this tool: downstream either keeps the
    output verbatim (it belongs to the solution) or drops it (it belongs to the
    path).
    """

    call_id: Identifier = Field(description="`ToolCall.id` this result answers.")
    output: str = Field(default="", description="Output verbatim, including stderr if merged.")
    exit_code: int | None = Field(
        default=None,
        description="Process exit status for `exec` calls; `null` when the tool reports none.",
    )
    is_error: bool = Field(
        default=False,
        description="Whether the agent saw this result as a failure.",
    )


class Message(CanonicalModel):
    """One message of a canonical transcript.

    `is_system_injected` and `is_command` are the two properties the chunker
    needs to reduce noise: an injected reminder and the echo of a slash command
    are not user turns, even though the agent stores them in the user role.
    """

    id: Identifier = Field(description="Identifier of the message; unique per session.")
    role: Role = Field(description="Who authored the message.")
    content: str = Field(
        default="",
        description="Text content; empty for a message that only carries tool calls.",
    )
    timestamp: AwareDatetime | None = Field(
        default=None,
        description="When the message was recorded; `null` when the agent records none.",
    )
    is_system_injected: bool = Field(
        default=False,
        description="Whether the harness injected this message rather than the user or agent.",
    )
    is_command: bool = Field(
        default=False,
        description="Whether this message is a slash command or its echo.",
    )
    tool_calls: list[ToolCall] = Field(
        default_factory=list,
        description="Tool calls this message issued, in the order the agent made them.",
    )
    tool_result: ToolResult | None = Field(
        default=None,
        description="The tool result this message reports; required for role `tool`.",
    )

    @model_validator(mode="after")
    def _check_tool_fields(self) -> Self:
        """Reject a tool message without a result and repeated tool-call IDs."""
        if self.role is Role.TOOL and self.tool_result is None:
            raise ValueError("a message with role 'tool' must carry a tool_result")
        _reject_duplicates((call.id for call in self.tool_calls), "tool-call id in one message")
        return self


class Session(CanonicalModel):
    """A whole transcript in canonical form, with its provenance.

    `source` is what makes the conditions of the study comparable: the same type
    holds a raw transcript, one the agent compacted natively, and one this tool
    compacted, so measurement can take a before and an after without knowing
    which agent produced either.

    For either compaction condition the provenance alone is not enough: where
    the summary sits is what separates the context that is still being sent from
    the history it replaced, so `native_summary_id` records it and the document
    stays readable after a `dump`/`load` — one CLI command per stage, with files
    in between, is what the pipeline is. The two fields answer different
    questions, which is why they are both here: `source` says **who** compacted
    the transcript, `native_summary_id` **which message its context now starts
    at**. A `native-compaction` session names the agent's own summary and an
    `intent-compaction` one names the summary this tool wrote, and the stages
    read both the same way: `idc.importers.visible_context` hides what precedes
    the named message, and `idc.chunk.reduction` keeps the message it names.

    Three things about that identifier are deliberately **not** enforced:

    - That it names a message of this session. It is a reference like
      `ToolResult.call_id`, and referential completeness is not enforced for
      those either: a transcript truncated at the front loses the summary record
      before it loses anything else, and a document that still says *which*
      message the context starts at is more useful than one that dropped the
      fact with it. `idc.importers.provenance` refuses such a session, because
      it cannot locate the cut — the stage that needs the reference resolves it.
    - That a `native-compaction` session names one at all. That is an obligation
      on whoever produced the document, not a property of the document: the
      importers are held to it by `ImportedSession` (see `idc.importers.base`),
      and the one computation that cannot proceed without it refuses. Making it
      a schema rule would mean a stage that rebuilds a session without carrying
      the field turns a valid transcript into an unloadable one, and it would be
      the single place this schema stops accepting incomplete data.
    - That an `intent-compaction` session names one. It **should**, though, and
      an earlier version of this paragraph was wrong about why. The identifier
      does two things, not one: it moves the cut, and it **exempts the named
      message from the harness-notice rule** —
      `idc.importers.provenance._hidden_reason` tests `message.id == summary_id`
      *before* the injected-`system` branch. This tool writes its summary as an
      injected `system` message, so leaving the field `null` does not merely
      fail to hide earlier history: it hides **the summary block itself**, as a
      harness notice. Measured on
      `tests/fixtures/already_compacted_block.json`: with the field `null`,
      `visible_context` returns 11 messages holding 103 words of content; with
      `m01` named, 12 messages holding 348 words. The 245 words in between are
      the compacted block — the "after" side of the condition under test, which
      would otherwise be counted as if the agent were never sent it. The fixture
      names its summary for that reason.

    One combination is refused, and only one: `raw` plus an identifier (see
    `_check_native_summary_agrees_with_the_provenance`).
    """

    id: Identifier = Field(description="Session identifier, taken from the source transcript.")
    agent: Agent = Field(description="The agent the transcript comes from.")
    model: str | None = Field(
        default=None,
        description="Model identifier the agent ran on; `null` when the transcript omits it.",
    )
    source: SessionSource = Field(
        default=SessionSource.RAW,
        description="Whether and how this transcript was compacted before.",
    )
    native_summary_id: Identifier | None = Field(
        default=None,
        description=(
            "`Message.id` of the summary a compaction wrote, and the message the "
            "context now starts at — the agent's own summary under "
            "`native-compaction`, this tool's under `intent-compaction`; `null` "
            "when `source` is `raw`, and `null` is allowed under either "
            "compaction. A transcript compacted twice names the last summary."
        ),
    )
    created_at: AwareDatetime = Field(description="When the session started.")
    messages: list[Message] = Field(
        default_factory=list,
        description="The messages in transcript order; index positions are what chunks range over.",
    )

    @model_validator(mode="after")
    def _check_identifiers_are_unique(self) -> Self:
        """Reject repeated message or tool-call IDs, which references rely on."""
        _reject_duplicates((message.id for message in self.messages), "message id")
        _reject_duplicates(
            (call.id for message in self.messages for call in message.tool_calls),
            "tool-call id across messages",
        )
        return self

    @model_validator(mode="after")
    def _check_native_summary_agrees_with_the_provenance(self) -> Self:
        """Reject a summary identifier on a transcript nothing compacted.

        The identifier marks where the context now starts, which is a fact only
        a compaction creates: under `raw` nothing replaced anything, so an
        identifier there is not incomplete data but two fields contradicting
        each other, and no stage could decide which to believe. Both compaction
        provenances accept one, because both describe a transcript whose context
        starts at a summary — the agent's own, or this tool's.

        Neither converse is rejected: a session of either compaction provenance
        may name no summary (see the class docstring).
        """
        if self.native_summary_id is not None and self.source is SessionSource.RAW:
            raise _field_error(
                type(self),
                "native_summary_id",
                self.native_summary_id,
                f"native_summary_id names {self.native_summary_id!r} as the message "
                f"the context starts at, but source is {self.source.value!r}: a "
                "transcript nothing compacted has no summary its context starts at",
            )
        return self


class Chunk(CanonicalModel):
    """One user input plus everything the agent produced in response.

    The chunk does not copy its messages: it is a half-open range over
    `Session.messages`, `start_index` included and `end_index` excluded, so that
    a chunked session carries no duplicated content and the chunker stays
    verifiably lossless.
    """

    id: Identifier = Field(description="Identifier of the chunk, assigned by the chunker.")
    start_index: int = Field(ge=0, description="Index of the chunk's first message, included.")
    end_index: int = Field(ge=1, description="Index after the chunk's last message, excluded.")
    is_latest: bool = Field(
        default=False,
        description="Whether this is the last chunk, which is rendered verbatim, not compacted.",
    )
    origin: ChunkOrigin = Field(
        default=ChunkOrigin.USER,
        description="Whether the chunk starts at a user message or holds subagent work.",
    )

    @model_validator(mode="after")
    def _check_range(self) -> Self:
        """Reject an empty or reversed message range, reported against `end_index`."""
        if self.end_index <= self.start_index:
            raise _field_error(
                type(self),
                "end_index",
                self.end_index,
                f"end_index ({self.end_index}) must be greater than "
                f"start_index ({self.start_index}); a chunk holds at least one message",
            )
        return self

    @property
    def message_count(self) -> int:
        """Return how many messages the chunk spans."""
        return self.end_index - self.start_index


class Chain(CanonicalModel):
    """A sequence of chunks that pursue the same user intent.

    A chain keeps its ID for the lifetime of a transcript: the compactor writes
    it into the summary block, the parser reads it back, and the classifier
    offers the parsed chain as a candidate again — which is how a chain can be
    extended on a later run instead of being summarized a second time.
    """

    id: str = Field(
        pattern=CHAIN_ID_PATTERN,
        description="Stable chain ID of the form `C<n>`, numbered from 1.",
    )
    label: Identifier = Field(description="Short natural-language name of the intent.")
    chunk_ids: list[Identifier] = Field(
        min_length=1,
        description="`Chunk.id`s of this chain, in transcript order.",
    )
    compacted: bool = Field(
        default=False,
        description="Whether `compacted_text` holds this chain's summary block.",
    )
    compacted_text: str | None = Field(
        default=None,
        description="The chain's Markdown summary block; set exactly when `compacted` is true.",
    )

    @model_validator(mode="after")
    def _check_chunks_and_compaction(self) -> Self:
        """Reject repeated chunk IDs and a compaction flag that contradicts the text."""
        _reject_duplicates(self.chunk_ids, "id in chunk_ids")
        if self.compacted and not self.compacted_text:
            raise ValueError("a compacted chain must carry a non-empty compacted_text")
        if not self.compacted and self.compacted_text is not None:
            raise ValueError("compacted_text is only allowed when compacted is true")
        return self


class CompactedBlock(CanonicalModel):
    """The summary of a transcript: its chains plus the chunk left verbatim.

    This is what the renderer turns into a target agent's native summary tags
    and what the parser reads back on the next run. The chains are held in
    chronological order; the latest chunk is referenced, not compacted, because
    it is appended to the output unchanged.
    """

    chains: list[Chain] = Field(
        default_factory=list,
        description="The chains of this transcript, in chronological order.",
    )
    latest_chunk_id: Identifier = Field(
        description="`Chunk.id` of the chunk that is appended verbatim instead of compacted."
    )
    rendered_for: Agent = Field(description="The agent whose native format this block targets.")

    @model_validator(mode="after")
    def _check_chain_ids_are_unique(self) -> Self:
        """Reject two chains with the same ID in one block."""
        _reject_duplicates((chain.id for chain in self.chains), "chain id")
        return self


CANONICAL_MODELS: tuple[type[CanonicalModel], ...] = (
    ToolCall,
    ToolResult,
    Message,
    Session,
    Chunk,
    Chain,
    CompactedBlock,
)
"""Every canonical type, ordered from the innermost to the outermost.

This tuple is the single list the JSON Schema document is generated from, so a
new type is part of the committed schema as soon as it is added here.
"""
