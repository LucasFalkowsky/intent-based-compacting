"""The canonical types — the contract between all pipeline stages.

The seven types below cover every state a transcript passes through: the
imported session (`Session` with its `Message`, `ToolCall` and `ToolResult`),
the chunked session (`Chunk`), the classified session (`Chain`) and the
compacted output (`CompactedBlock`). Agent-specific shapes exist only inside
`idc.importers` and `idc.render`; everything in between sees these types.

Three conventions hold throughout and are enforced, not documented only:

- **No unknown fields.** Every model forbids extras, so a field an importer
  invents is an error instead of silent data that no later stage reads.
- **Timestamps are timezone-aware.** Agents record instants in different
  offsets; a naive timestamp cannot be ordered against them, so importers must
  attach the offset (UTC when the agent records none).
- **Identifiers are unique where they are referenced.** `ToolResult.call_id`,
  `Chunk`'s message range and `Chain.chunk_ids` are references; the containers
  reject duplicate message and tool-call identifiers so that resolving a
  reference can never be ambiguous.

What is *not* enforced: that every `ToolResult.call_id` has a matching
`ToolCall` in the same session. Transcripts can be truncated at the front, and
a result whose call was cut off is still data the measurement needs.
"""

from typing import Annotated, Any, Self

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

from idc.schema.enums import Agent, ChunkOrigin, Role, SessionSource, ToolCategory

Identifier = Annotated[str, Field(min_length=1)]
"""A non-empty identifier taken from the transcript or assigned by a stage."""

CHAIN_ID_PATTERN = r"^C[1-9][0-9]*$"
"""`C<n>` with `n` counted from 1 — the stable chain ID the summary blocks use."""


class CanonicalModel(BaseModel):
    """Base class carrying the configuration every canonical type shares.

    Unknown fields are rejected, assignments are validated like constructor
    arguments (a stage cannot mutate an object into an invalid one), and enum
    members are kept as enum members in Python while serializing to their wire
    strings.
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        use_enum_values=False,
    )


def _duplicates(values: list[str]) -> list[str]:
    """Return the values that occur more than once, in order of first repeat."""
    seen: set[str] = set()
    duplicated: list[str] = []
    for value in values:
        if value in seen and value not in duplicated:
            duplicated.append(value)
        seen.add(value)
    return duplicated


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
        duplicated = _duplicates([call.id for call in self.tool_calls])
        if duplicated:
            raise ValueError(f"duplicate tool-call id in one message: {', '.join(duplicated)}")
        return self


class Session(CanonicalModel):
    """A whole transcript in canonical form, with its provenance.

    `source` is what makes the conditions of the study comparable: the same type
    holds a raw transcript, one the agent compacted natively, and one this tool
    compacted, so measurement can take a before and an after without knowing
    which agent produced either.
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
    created_at: AwareDatetime = Field(description="When the session started.")
    messages: list[Message] = Field(
        default_factory=list,
        description="The messages in transcript order; index positions are what chunks range over.",
    )

    @model_validator(mode="after")
    def _check_identifiers_are_unique(self) -> Self:
        """Reject repeated message or tool-call IDs, which references rely on."""
        duplicated = _duplicates([message.id for message in self.messages])
        if duplicated:
            raise ValueError(f"duplicate message id: {', '.join(duplicated)}")
        duplicated = _duplicates(
            [call.id for message in self.messages for call in message.tool_calls]
        )
        if duplicated:
            raise ValueError(f"duplicate tool-call id across messages: {', '.join(duplicated)}")
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
        """Reject an empty or reversed message range."""
        if self.end_index <= self.start_index:
            raise ValueError(
                f"end_index ({self.end_index}) must be greater than "
                f"start_index ({self.start_index}); a chunk holds at least one message"
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
        duplicated = _duplicates(list(self.chunk_ids))
        if duplicated:
            raise ValueError(f"duplicate id in chunk_ids: {', '.join(duplicated)}")
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
        duplicated = _duplicates([chain.id for chain in self.chains])
        if duplicated:
            raise ValueError(f"duplicate chain id: {', '.join(duplicated)}")
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
