"""The tool calls of one chain: where each sits, and what it did to which file.

Two things are read here, and both are read from the transcript alone:

- **Where a call sits.** A `Chain` holds chunk *ids* and a `Chunk` holds message
  *indices* (#3), so the calls of a chain are two dereferences away from the
  messages that carry them. `chain_calls` does them once and returns a
  `LocatedCall` per call — the call, the chunk and the message it came from, its
  position in the transcript, and the result the agent received back. Every
  later stage of the resolver walks that list instead of the session.
- **What a call did.** `file_of` answers which file a call's arguments name and
  `write_kind` whether a write replaced the file whole or patched a part of it.
  Both are argument-shape questions, which is the one place this stage has to
  know something about the agents' spellings — see `FILE_PATH_KEYS` and
  `WriteKind` for why that knowledge sits here and not in the tool registry.

The order is the transcript's. `chain_chunks` resolves a chain's ids against the
chunk list and sorts them by position, so "the last write" means the last one in
the transcript even if a chain lists its chunks in another order; `Chain.chunk_ids`
is documented to be in transcript order, and this does not depend on it.

Nothing here reads a clock, a model or the filesystem, and nothing summarizes:
a `LocatedCall` points at the canonical `ToolCall` and `ToolResult` objects of
the session, so a result that is kept is kept verbatim by construction.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

from idc.resolve.errors import ForeignChunkError, UnknownChunkError
from idc.schema import Chain, Chunk, Session, ToolCall, ToolCategory, ToolResult

__all__ = [
    "CONTENT_KEYS",
    "FILE_PATH_KEYS",
    "REPLACEMENT_KEYS",
    "LocatedCall",
    "WriteKind",
    "chain_calls",
    "chain_chunks",
    "content_of",
    "file_of",
    "write_kind",
]

FILE_PATH_KEYS = ("file_path", "filePath", "absolute_path")
"""Argument keys the three agents use for the file a call works on.

`ToolCall.args` keeps a call's arguments exactly as the agent sent them (#3), so
the same operation arrives under three spellings: `file_path` (Claude Code, and
Gemini CLI's `write_file`/`replace`), `filePath` (OpenCode) and `absolute_path`
(Gemini CLI's `read_file`).

Only **observed** spellings are listed, the evidence rule
`tool_registry.toml` follows: a key no agent sends would pin behaviour that was
never seen, while a missing one makes a call resolve to *no file* — which is
reported (`LocatedCall.path` is `None`) and keeps the write out of a file's
solution, rather than attributing it to the wrong file.

This knowledge belongs in the registry's data file next to the tool categories,
which is where a new spelling should be addable without touching Python. It is
written here because the registry is `idc.importers`' and extending it is not
this ticket's scope; the decision is recorded on #21.
"""

CONTENT_KEYS = ("content",)
"""Argument keys that carry the **whole** new content of a file.

All three agents spell it `content` — Claude Code's `Write`, Gemini CLI's
`write_file`, OpenCode's `write`. A one-element tuple for the same reason
`COMMAND_KEYS` is one in the fixture corpus: it is the spelling that was
observed, not a place for prospective ones.
"""

REPLACEMENT_KEYS = ("old_string", "new_string", "oldString", "newString")
"""Argument keys that carry a *part* of a file being replaced.

The agents send them in pairs — `old_string`/`new_string` (Claude Code's `Edit`,
Gemini CLI's `replace`) and `oldString`/`newString` (OpenCode's `edit`) — but
**one** of them is enough to know what the call does: a call that names the text
it replaces patches the file, so the file's earlier content is still part of
what the file now holds, which is exactly why a patch does not supersede the
write before it. Asking for the whole pair would make a call that carries one
replacement key *and* a content argument look like a whole-file write, which is
the one reading of such a call that loses content.
"""


class WriteKind(StrEnum):
    """How a write-category call changed the file it names.

    Attributes:
        FULL: It wrote the whole file, so everything written before it is gone.
        EDIT: It replaced a part of the file, so the write before it still shows.
    """

    FULL = "full"
    EDIT = "edit"


def file_of(call: ToolCall) -> str | None:
    """Return the file a call works on, or `None` if its arguments name none.

    Args:
        call: Any tool call, of any category.

    Returns:
        The path exactly as the agent wrote it — not normalized, not resolved:
        two spellings of one file are two files here, because the resolver has
        no filesystem and `/repo/src/a.py` and `src/a.py` cannot be compared
        without one. The first key of `FILE_PATH_KEYS` that holds a path wins;
        `None` means no argument named a file, which covers three cases: no
        such key, a key spelled in a way this build has never seen, and a key
        whose value holds no word. The last one is read as *no* path rather
        than as the file `""`, which would group into an entry naming a file
        nothing names — a `SolutionFile` refuses that, and a transcript must
        not be able to trigger a refusal outside `idc.resolve.errors`.
    """
    for key in FILE_PATH_KEYS:
        value = call.args.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def _replaces_a_part(call: ToolCall) -> bool:
    """Return whether the call's arguments name a part of the file to replace."""
    return any(isinstance(call.args.get(key), str) for key in REPLACEMENT_KEYS)


def content_of(call: ToolCall) -> str | None:
    """Return the whole file content a call wrote, or `None` if it wrote none.

    Args:
        call: Any tool call, of any category.

    Returns:
        The text under the first key of `CONTENT_KEYS` that holds a string, for
        a call that is a `WriteKind.FULL` write; `None` for an edit, for a
        non-write call, and for a full write whose content argument is missing.
    """
    if write_kind(call) is not WriteKind.FULL:
        return None
    for key in CONTENT_KEYS:
        value = call.args.get(key)
        if isinstance(value, str):
            return value
    return None


def write_kind(call: ToolCall) -> WriteKind | None:
    """Return how a call changed its file, read off its arguments.

    The canonical schema has one `write` category for writing, editing and
    patching (#3), and the tool registry declares that category and not more —
    so the distinction the last-write-per-file rule needs is taken from the
    arguments the agent sent: a call that carries the whole content wrote the
    file, a call that names the text it replaces patched it.

    Args:
        call: Any tool call, of any category.

    Returns:
        `None` for a call that is not of category `write`, `WriteKind.FULL` for
        one that carries the whole content and no replacement key, and
        `WriteKind.EDIT` for everything else — a replacement key, a shape this
        build does not know (Claude Code's `NotebookEdit`, a patch tool), or a
        call carrying both.

        `EDIT` is the default on purpose: reading an edit as a full write would
        drop the write it patched from the solution and lose content, while
        reading a full write as an edit only keeps a superseded write that the
        summary does not need. The lossless direction is the one to fall into.
    """
    if call.category is not ToolCategory.WRITE:
        return None
    if _replaces_a_part(call):
        return WriteKind.EDIT
    if any(isinstance(call.args.get(key), str) for key in CONTENT_KEYS):
        return WriteKind.FULL
    return WriteKind.EDIT


@dataclass(frozen=True, slots=True)
class LocatedCall:
    """One tool call of a chain, with where it sits and what came back.

    Attributes:
        call: The canonical call itself, arguments verbatim.
        chunk_id: `Chunk.id` of the chunk the call sits in.
        message_id: `Message.id` of the message that issued it.
        message_index: Index of that message in `Session.messages` — the
            position everything in this stage is ordered by.
        result: The `ToolResult` the agent received back, or `None` when the
            transcript holds none (it can be truncated before the answer
            arrived). The object itself, never a copy or a summary of it.
    """

    call: ToolCall
    chunk_id: str
    message_id: str
    message_index: int
    result: ToolResult | None

    @property
    def id(self) -> str:
        """`ToolCall.id` — what the path and the solution are expressed in."""
        return self.call.id

    @property
    def name(self) -> str:
        """The tool name as the agent reports it, e.g. `Edit`."""
        return self.call.name

    @property
    def category(self) -> ToolCategory:
        """What the call does to the workspace, as the importer resolved it."""
        return self.call.category

    @property
    def path(self) -> str | None:
        """The file the call's arguments name, or `None` — see `file_of`."""
        return file_of(self.call)

    @property
    def kind(self) -> WriteKind | None:
        """How the call changed its file, or `None` — see `write_kind`."""
        return write_kind(self.call)

    @property
    def failed(self) -> bool:
        """Whether the agent saw this call fail.

        Read off the result's `is_error` flag, which is what the agent itself
        acted on. A call whose result the transcript does not hold counts as
        not failed: nothing says it did, and treating a truncated transcript as
        a chain of failures would empty its solution.
        """
        return self.result is not None and self.result.is_error


def chain_chunks(chunks: Sequence[Chunk], chain: Chain) -> tuple[Chunk, ...]:
    """Resolve a chain's chunk ids against a chunk list, in transcript order.

    Args:
        chunks: The chunks of the transcript, as the chunker produced them.
        chain: The chain whose chunks are wanted.

    Returns:
        The chain's chunks, ordered by `start_index`. The objects of `chunks`
        themselves, not copies.

    Raises:
        UnknownChunkError: If the chain names a chunk the list does not hold.
    """
    by_id: dict[str, Chunk] = {}
    for chunk in chunks:
        by_id.setdefault(chunk.id, chunk)
    resolved: list[Chunk] = []
    for chunk_id in chain.chunk_ids:
        chunk = by_id.get(chunk_id)
        if chunk is None:
            raise UnknownChunkError(chunk_id, chain.id, tuple(by_id))
        resolved.append(chunk)
    return tuple(sorted(resolved, key=lambda chunk: chunk.start_index))


def chain_calls(session: Session, chunks: Sequence[Chunk], chain: Chain) -> tuple[LocatedCall, ...]:
    """Return every tool call of a chain, located and in transcript order.

    Args:
        session: The transcript the chunks range over.
        chunks: The chunks of that transcript.
        chain: The chain to walk.

    Returns:
        One `LocatedCall` per call the chain's chunks carry, in transcript
        order and with the result that answers it attached. The result is
        looked up across the **whole** session, not only inside the chain: a
        call sits in the chunk that made it, while its answer is wherever the
        agent recorded it.

    Raises:
        UnknownChunkError: If the chain names a chunk the list does not hold.
        ForeignChunkError: If a chunk's range reaches past the session.
    """
    results = {
        message.tool_result.call_id: message.tool_result
        for message in session.messages
        if message.tool_result is not None
    }
    located: list[LocatedCall] = []
    for chunk in chain_chunks(chunks, chain):
        if chunk.end_index > len(session.messages):
            raise ForeignChunkError(chunk.id, session.id, chunk.end_index, len(session.messages))
        for index in range(chunk.start_index, chunk.end_index):
            message = session.messages[index]
            located += [
                LocatedCall(
                    call=call,
                    chunk_id=chunk.id,
                    message_id=message.id,
                    message_index=index,
                    result=results.get(call.id),
                )
                for call in message.tool_calls
            ]
    return tuple(located)
