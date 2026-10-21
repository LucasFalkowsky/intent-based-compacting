"""The failures of the compaction stage, as one catchable family.

Two kinds of input reach this stage from outside the transcript, and both are
refused here rather than carried: the **instruction data file**, which is read
from disk, and a **model's response**, which is text nothing has checked. Every
refusal is a `CompactError`, so a caller (and T10's CLI) catches the whole stage
with one `except`, and each subclass carries the values its message is built
from — the shape `idc.classify.errors` and `idc.resolve.errors` established.

The line between the two families those modules draw is drawn here the same
way, and by **where the data comes from**:

- Everything the stage is *handed* is in the family. The instruction file is
  package data, but it is still a file: a broken one is a diagnostic for the
  CLI user, not a traceback, which is the reason
  `idc.importers.tool_registry.ToolRegistryError` exists for its own data file.
  A model's response is the clearer case — it is the one input to this stage
  that no schema and no importer has validated.
- A value object a caller built wrongly is **not**. `Narrative` raises a plain
  `ValueError` for a label that holds no word or a body line that is a header,
  the way `idc.classify.adapters.Decision` does for a bad target: a bad literal
  in a caller's own construction is a programming error.

The two sides meet in `NarrativeContractError`, and deliberately overlap rather
than split: a response that cannot be read is both a bad value and a failure of
the stage, so it is a `ValueError` **and** a `CompactError` — exactly as
`AdapterContractError` is for the classifier's adapters. A caller never has to
catch two families for one event, and a caller that reads it as a value problem
is not made wrong.

`InstructionError` is a `ValueError` for the same reason, which is also what
`ToolRegistryError` is.
"""

__all__ = [
    "CarriedChainError",
    "CompactError",
    "InstructionError",
    "NarrativeContractError",
    "NarrativeScriptExhaustedError",
]


class CompactError(Exception):
    """Base class of every failure the compaction stage raises."""


class InstructionError(CompactError, ValueError):
    """The narrative instruction could not be read, or is not one this build reads.

    The instruction is a versioned data file shipped inside the package
    (`idc.compact.instruction.INSTRUCTION_PATH`), so that a run can record
    *which* instruction produced it. Everything that can go wrong with it is
    therefore input: a file that is not there, that is not UTF-8, that is not
    TOML, that declares another `version`, or whose `instruction` holds no word.
    The message names the file, so a packaging mistake is readable from the
    error alone.
    """


class NarrativeContractError(CompactError, ValueError):
    """A narrative client answered with something that is not a narrative.

    The response has to carry the intent line and the two narrative headers,
    each exactly once and in the template's order, and nothing else that looks
    like a heading — see `idc.compact.narrative.as_narrative` for the whole
    rule. Everything else is this error, because the alternative is worse than a
    failure: a block whose `Solution` section a model wrote, or whose second
    `## [C…]` heading invents a chain the classifier never opened, is a summary
    that reads as the tool's own and is not.
    """


class NarrativeScriptExhaustedError(CompactError):
    """The scripted narrative client was asked for one response more than it holds.

    A test-authoring mistake, and a loud one on purpose: a fake that repeated
    its last response would render a block that says the wrong thing about a
    chain, and the test would pass for a reason it never stated. The sibling of
    `idc.classify.errors.ScriptExhaustedError`, named apart from it so that a
    test importing both reads which stage ran out.

    Attributes:
        chain_id: The chain a narrative was wanted for.
        script_length: How many responses the script held.
    """

    def __init__(self, chain_id: str, script_length: int) -> None:
        """Build the error from the chain asked about and the script's length.

        Args:
            chain_id: The chain a narrative was wanted for.
            script_length: How many responses the script held.
        """
        self.chain_id = chain_id
        self.script_length = script_length
        super().__init__(
            f"the scripted narrative client ran out of responses at chain {chain_id!r}: "
            f"its script holds {script_length} response(s)"
        )


class CarriedChainError(CompactError, ValueError):
    """A chain with no part in the transcript at hand was asked to be resolved.

    `idc.compact.incremental.transcript_chain` reduces a chain to the chunks
    this transcript holds, and a chain that has none is carried over from an
    earlier run: its block is copied, and there is nothing in it to resolve.
    Only a caller that asked the wrong question of the wrong chain reaches this
    — `is_carried_over` is the question to ask first.

    `CompactError` **and** `ValueError`, exactly as `NarrativeContractError`
    is: the mistake is in input the stage was handed, and the stage's refusals
    are one family a caller can catch. It was a bare `ValueError` once, which a
    caller could not tell from a pydantic or a built-in failure and which would
    have reached a CLI as a raw traceback.
    """
