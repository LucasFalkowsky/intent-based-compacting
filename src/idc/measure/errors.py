"""The failures of the measurement stage, as one catchable family.

Measurement is handed three kinds of input from outside its own code: a
**transcript** — whose `model` field says which tokenizer the numbers have to be
produced with, and whose tool calls are what the exploration rate is a rate of —
a pair of **token counts** that a previous stage produced, and the **files** a
run manifest records: the transcript that went in, the one that came out, and a
manifest document handed back to be read. Everything that can go wrong with any
of them is a `MeasureError`, so a caller (and
T10's CLI) catches the whole stage with one `except`, and each subclass carries
the values its message is built from — the shape `idc.classify.errors`,
`idc.resolve.errors`, `idc.compact.errors` and `idc.parse.errors` established.

The line between the family and a plain `ValueError` is drawn here the way
`idc.parse.errors` draws it, by **where the data comes from**:

- Everything the stage is *handed* is in the family. A model name it does not
  know and a family with no tokenizer behind it are both diagnostics for the
  CLI user — "this transcript cannot be measured, and here is what to do" — not
  tracebacks.
- A value only a caller's own code can produce is **not**. A negative token
  count is one case: nothing in this stage counts below zero, so a negative
  `before` or `after` is a programming error and raises a plain `ValueError`
  that is deliberately *not* a `MeasureError`. A manifest a caller builds or
  mutates wrongly, and a run window whose finish precedes its start, are the
  others; `idc.measure.manifest` translates pydantic's own error into a plain
  `ValueError` on that path and into `ManifestError` on the load path, so the
  two sides of the line stay apart while no pydantic traceback escapes either.

`EmptyBaselineError` and `NoToolCallsError` are both, and deliberately overlap
rather than split —
exactly as `idc.compact.errors.NarrativeContractError` and
`idc.classify.errors.AdapterContractError` do. A `before` of zero is at once a
property of the transcript the stage was handed (a visible context that holds
no word) and a value no caller may divide by, so a caller never has to catch two
families for one event and a caller that reads it as a value problem is not made
wrong. A visible context that holds no tool call is the same event one metric
further on, and is reported the same way.

What is **not** here: the refusal of `idc.importers.visible_context`. A session
the agent compacted itself but whose summary cannot be located raises
`TranscriptImportError`, and `idc.measure.tokens` lets it through untouched.
Wrapping it would turn "the history the agent dropped cannot be located" into a
vaguer measurement failure, and that refusal exists precisely so measurement
cannot count that history.
"""

from collections.abc import Sequence

__all__ = [
    "EmptyBaselineError",
    "ManifestError",
    "MeasureError",
    "NoTokenizerError",
    "NoToolCallsError",
    "UnknownModelError",
]


class MeasureError(Exception):
    """Base class of every failure the measurement stage raises."""


class UnknownModelError(MeasureError):
    """A transcript's model belongs to no model family this build knows.

    Refused rather than measured with a default tokenizer, which is the
    decision of T9.1 (#37) and the reason this error exists: a silent fallback
    would produce two numbers that look comparable and were counted with
    different tokenizers, and nothing in the output would say so. The message
    names the families that are known, so the repair is readable from the error
    alone.

    A transcript that records **no** model (`Session.model` is `None`) raises
    this too: there is nothing to pick a tokenizer from, and guessing one from
    the agent would be the same silent fallback by another route.

    Attributes:
        model: The model name as the document recorded it, or `None` when it
            recorded none.
        known: The families this build knows, in declaration order.
    """

    def __init__(self, model: str | None, known: Sequence[str]) -> None:
        """Build the error from the model that was not placed and the known families.

        Args:
            model: The model name the document recorded, or `None`.
            known: The model families this build knows.
        """
        self.model = model
        self.known = tuple(known)
        families = ", ".join(repr(family) for family in self.known) or "no family at all"
        recorded = (
            "the transcript records no model"
            if model is None
            else f"the model {model!r} belongs to no model family this build knows"
        )
        super().__init__(
            f"{recorded}; a tokenizer cannot be picked for it, and measuring it with a "
            f"default one would make two runs comparable that were not counted the same "
            f"way. Known families: {families}"
        )


class NoTokenizerError(MeasureError):
    """A model family is known, but no tokenizer was registered for it.

    The counterpart of `UnknownModelError`: the transcript's model was placed
    in a family and the registry has nothing behind that family. This build
    ships no tokenizer at all — every real one needs a downloaded vocabulary or
    a network call, and determinism forbids both inside the tool's test suite —
    so a caller registers the tokenizer it wants the run counted with.

    Attributes:
        family: Value of the model family that has no tokenizer.
        model: The model name that resolved to it, as the document recorded it.
            Typed `str | None` because `Session.model` is, and never `None` in
            practice: a session that records no model is refused by
            `UnknownModelError` before a tokenizer is looked for.
        registered: The families that do have one, in registration order.
    """

    def __init__(self, family: str, model: str | None, registered: Sequence[str]) -> None:
        """Build the error from the family, the model and what is registered.

        Args:
            family: Value of the model family that has no tokenizer.
            model: The model name that resolved to it.
            registered: The families that do have a tokenizer.
        """
        self.family = family
        self.model = model
        self.registered = tuple(registered)
        have = ", ".join(repr(name) for name in self.registered) or "no family"
        super().__init__(
            f"no tokenizer is registered for the model family {family!r}, which the model "
            f"{model!r} belongs to; register one for it before measuring. Registered: {have}"
        )


class EmptyBaselineError(MeasureError, ValueError):
    """A compression rate was asked for against a baseline of zero tokens.

    `(before - after) / before` has no value at `before == 0`, and the number
    this tool reports may have no undefined region — so the zero baseline is
    refused where it is handed over instead of being turned into a `0.0`, a
    `nan` or an `inf` that would travel into a results table unnoticed. It
    happens when the "before" side holds no visible message, or none that
    carries a single word.

    `MeasureError` **and** `ValueError`; see this module's docstring for why the
    two sides overlap here.

    Attributes:
        after: The "after" count that was handed in with it, so the message can
            tell "nothing in, nothing out" from "nothing in, something out".
    """

    def __init__(self, after: int) -> None:
        """Build the error from the "after" count the zero baseline came with.

        Args:
            after: The "after" token count.
        """
        self.after = after
        super().__init__(
            f"the baseline counts no token, so no compression rate is defined for it "
            f"(after = {after}); measure a context that holds at least one word"
        )


class NoToolCallsError(MeasureError, ValueError):
    """An exploration rate was asked for from a session that made no tool call.

    `read_calls / total_calls` has no value at `total_calls == 0`, and the
    refusal is the one `EmptyBaselineError` makes for the same shape of
    question: a `0.0` would read as "this session explored nothing" for a
    session that used no tool at all, and a `nan` would travel into a results
    table without a line of output saying so.

    The refusal is raised by `ToolStatistics.exploration_rate` and **not** by
    the constructor, because the other metrics of that object are defined at
    zero calls — no file was edited, no line was added, the histogram is empty —
    and refusing to build it would discard three good numbers with the one that
    is undefined.

    `MeasureError` **and** `ValueError`; see this module's docstring for why the
    two sides overlap here.

    Attributes:
        session_id: `Session.id` of the transcript that made no tool call, so
            the message names the transcript to look at.
    """

    def __init__(self, session_id: str) -> None:
        """Build the error from the session whose rate was asked for.

        Args:
            session_id: `Session.id` of that session.
        """
        self.session_id = session_id
        super().__init__(
            f"the visible context of {session_id!r} holds no tool call, so no exploration "
            f"rate is defined for it; its other statistics are reported as they stand"
        )


class ManifestError(MeasureError):
    """A run manifest could not be read, or a file it records could not be hashed.

    The manifest is the one artifact of this stage that is written to disk and
    read back, so it is handed two things from outside its own code: the
    **files** it records the digests of, and a **document** a later run or a
    reader hands back. Both are in the family for the reason this module's
    docstring gives — "this file is not there" and "this manifest declares a
    version I cannot read" are diagnostics for the CLI user.

    What is deliberately **not** here: a manifest a caller's own code builds
    wrongly. `idc.measure.manifest` translates pydantic's `ValidationError`
    into a plain `ValueError` for construction and assignment, and into this
    error only on the load path, which is the same line
    `EmptyBaselineError`'s neighbours are drawn along. Either way pydantic's
    own error never escapes the module.

    Attributes:
        source: The file the document or the digest came from, or `None` when a
            manifest was read from text with no source named.
        errors: One `(location, message)` pair per problem, with the location
            given as a dotted field path such as `input_file.sha256`. Empty for
            a failure that is not about a field — an unreadable file, or text
            that is not JSON.
    """

    def __init__(
        self,
        reason: str,
        *,
        source: str | None = None,
        errors: Sequence[tuple[str, str]] = (),
    ) -> None:
        """Build the error and render its message.

        Args:
            reason: Summary line, e.g. `2 problems` or `not valid JSON`.
            source: Where the document or the file came from, named in the
                message when given.
            errors: The located problems, listed one per line after the summary.
        """
        self.source = source
        self.errors = tuple(errors)
        origin = f" from {source}" if source else ""
        lines = [f"invalid run manifest{origin}: {reason}"]
        lines += [f"  {location}: {message}" for location, message in self.errors]
        super().__init__("\n".join(lines))
