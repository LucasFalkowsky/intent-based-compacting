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
  Which argument names the file is the tool registry's knowledge, declared
  next to the category that says the tool writes (T5.5); which *shape* a write
  call has is read here — see `CONTENT_KEYS` for why that one stays.

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

from idc.importers import tool_registry
from idc.resolve.errors import ForeignChunkError, UnknownChunkError
from idc.schema import Chain, Chunk, Session, ToolCall, ToolCategory, ToolResult

__all__ = [
    "COMMAND_KEYS",
    "CONTENT_KEYS",
    "FILE_PATH_KEYS",
    "REPLACEMENT_KEYS",
    "LocatedCall",
    "WriteKind",
    "chain_calls",
    "chain_chunks",
    "command_of",
    "content_of",
    "file_of",
    "write_kind",
]

FILE_PATH_KEYS = tool_registry().file_path_keys
"""Argument keys the three agents use for the file a call works on.

`ToolCall.args` keeps a call's arguments exactly as the agent sent them (#3), so
the same operation arrives under four spellings: `file_path` (Claude Code, and
Gemini CLI's `write_file`/`replace`), `notebook_path` (Claude Code's
`NotebookEdit`), `absolute_path` (Gemini CLI's `read_file`) and `filePath`
(OpenCode).

This stage holds no list of its own: the keys are
`tool_registry.toml`'s `path_keys` declarations, collected across the agents
(T5.5, #75). The registry is where "this tool writes" and "this is its path
argument" are one line, so a declared write tool whose spelling is missing is
rejected on load instead of resolving to *no file* — which is what
`notebook_path` did while the two lived apart, keeping every notebook out of
every solution. A spelling is added to the data file and only when the agent was
observed to send it, the evidence rule that file states.

A key this build has never seen still makes a call resolve to no file, which is
reported (`LocatedCall.path` is `None`) and keeps the write out of a file's
solution rather than attributing it to the wrong file. Reading the whole union
rather than the declaring tool's own keys is deliberate: a `ToolCall` names its
tool but not the session, so the agent is not available here.
"""

COMMAND_KEYS = tool_registry().command_keys
"""Argument keys an `exec` call uses for the command line it runs.

The registry's `command_keys` declarations, collected across the agents, read
the way `FILE_PATH_KEYS` is read and moved there with it (T5.5, #75): an `exec`
tool declared without its key is rejected on load, rather than running commands
that no verification detection ever sees. All three agents spell it `command` —
Claude Code's `Bash`, Gemini CLI's `run_shell_command` and OpenCode's `bash` —
so the union is one element, and it stays one until a transcript shows
otherwise.

A spelling this build has never seen makes `command_of` return `None`, which
keeps the call out of the verification detection and reports it as an ordinary
`exec` call instead of guessing at a command line.

The same list exists once more, in `tests/fixture_corpus.py`, so that the stage
is not its own oracle; `tests/test_resolve_verification.py` pins that copy
against this one.
"""

CONTENT_KEYS = ("content",)
"""Argument keys that carry the **whole** new content of a file.

All three agents spell it `content` — Claude Code's `Write`, Gemini CLI's
`write_file`, OpenCode's `write`. A one-element tuple for the same reason
`COMMAND_KEYS` is one: it is the spelling that was observed, not a place for
prospective ones.

This one stayed here when the file and command keys moved into the registry
(T5.5, #75), and `REPLACEMENT_KEYS` with it. The two that moved answer *where*
a declared write or exec call names its subject, so a tool missing from them is
a gap the registry can refuse: every `write` tool has a file, every `exec` tool
a command. These two answer *which of two shapes* a write call has, and the
shapes are not per tool but per call — `write_kind` reads both lists on every
write call and falls back to `EDIT` for a shape it does not know, which is the
lossless reading. There is no declaration a registry could require of a write
tool here, and **four** of the seven write tools are not corpus-recorded with a
content argument: `NotebookEdit` and `write` have no recorded call at all, and
`replace` and `edit` are recorded only with replacement keys. Declaring a
content key per tool would therefore be the invented data the evidence rule
refuses — for four of seven it would rest on nothing.
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


def _argument_of(call: ToolCall, keys: Sequence[str]) -> str | None:
    """Return the first of `keys` the call answers with a word, or `None`.

    Args:
        call: Any tool call, of any category.
        keys: The argument keys to try, in order.

    Returns:
        The value verbatim, for the first key that holds a string with a word
        in it. `None` covers three cases, and folding them together is what
        both callers document: no such key, a key spelled in a way this build
        has never seen, and a key whose value holds no word. The last one is
        why the value is tested with `strip()` and returned unstripped — a
        blank argument names nothing, while a path that *does* name something
        is the agent's own spelling and is never rewritten here.
    """
    for key in keys:
        value = call.args.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


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
    return _argument_of(call, FILE_PATH_KEYS)


def command_of(call: ToolCall) -> str | None:
    """Return the command line a call runs, or `None` if its arguments name none.

    Args:
        call: Any tool call, of any category. The category is not consulted —
            `verification_runs` selects the `exec` calls before it asks for
            their command, which keeps this a pure argument-shape question like
            `file_of`.

    Returns:
        The command line exactly as the agent sent it, prefixes and quoting
        included: it is the string the tool registry's own prefix stripping and
        token anchoring are defined over, and the string a summary quotes. The
        first key of `COMMAND_KEYS` that holds a non-blank string wins; `None`
        means no argument named a command, which covers a missing key, a key
        spelled in a way this build has never seen, and a value that holds no
        word — the same three cases `file_of` folds into `None`, and for the
        same reason: a blank command line is no command, and matching it
        against the patterns could only ever produce a run nothing ran.
    """
    return _argument_of(call, COMMAND_KEYS)


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

    No production code calls this yet — T5.4 (#24), the edit replay, is the
    stage that would, and it is `optional`. It is public anyway because
    `CONTENT_KEYS` is: exporting the keys without the one lookup that applies
    them is an invitation to re-implement that lookup per caller, which is the
    duplication the two copies of `FILE_PATH_KEYS` already cost this stage. It
    is also what makes `write_kind`'s `FULL` checkable from outside — the test
    that asserts a full write is exactly a write carrying whole content.
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
        build does not know (a patch tool), or a call carrying both.

        `EDIT` is the default on purpose: reading an edit as a full write would
        drop the write it patched from the solution and lose content, while
        reading a full write as an edit only keeps a superseded write that the
        summary does not need. Of those two the lossless one is the direction to
        fall into.

        A call reaches the default by naming no shape this build knows, and
        it is read as an edit, so the write before it stays in the solution.
        Every tool `tool_registry.toml` declares `category = "write"` can get
        that far: the registry requires each of them to declare the argument
        key under which it names its file, so none of them resolves to no file
        the way `NotebookEdit` did before T5.5 (#75).
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
