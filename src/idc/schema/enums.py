"""Closed value sets of the canonical schema.

Every enum is a `StrEnum`, so a member *is* its wire string: it compares equal
to the string a transcript carries, and it serializes to that string without a
custom encoder. The sets are deliberately closed — an agent or a tool category
the study does not cover must fail validation loudly instead of travelling
through the pipeline as an unknown string.
"""

from enum import StrEnum


class Agent(StrEnum):
    """A coding agent whose transcripts the tool reads and writes.

    The three members are the agents of the study. They name both the source of
    an imported transcript (`Session.agent`) and the target of a rendered one
    (`CompactedBlock.rendered_for`).
    """

    CLAUDE_CODE = "claude-code"
    GEMINI_CLI = "gemini-cli"
    OPENCODE = "opencode"


class SessionSource(StrEnum):
    """Provenance of a canonical session — which condition of the study it is.

    Attributes:
        RAW: Imported from an agent transcript that was never compacted.
        NATIVE_COMPACTION: Imported from a transcript the agent compacted with
            its own built-in summarization (the baseline condition).
        INTENT_COMPACTION: Written by this tool (the condition under test).
    """

    RAW = "raw"
    NATIVE_COMPACTION = "native-compaction"
    INTENT_COMPACTION = "intent-compaction"


class Role(StrEnum):
    """Author of a canonical message.

    `TOOL` is the result of a tool call, reported back to the agent; which role
    the agent itself uses on the wire for that message is the importer's
    problem, not the schema's.
    """

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


class ToolCategory(StrEnum):
    """What a tool call does to the workspace.

    The category is what makes the exploration rate and the solution/path split
    computable without interpreting tool names downstream.

    Attributes:
        READ: Reads or searches without changing anything (read, grep, glob).
        WRITE: Changes a file (write, edit, patch).
        EXEC: Runs a command whose effect is not visible in the transcript.
        OTHER: Anything the tool registry does not recognize.
    """

    READ = "read"
    WRITE = "write"
    EXEC = "exec"
    OTHER = "other"


class ChunkOrigin(StrEnum):
    """Where a chunk's leading input came from.

    Attributes:
        USER: The chunk starts at a user message of the main transcript.
        SUBAGENT: The chunk holds the work of a subagent, extracted into a chunk
            of its own and placed after the chunk that spawned it.
    """

    USER = "user"
    SUBAGENT = "subagent"
