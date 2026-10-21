"""The failures of the measurement stage, as one catchable family.

Measurement is handed two kinds of input from outside its own code: a
**transcript**, whose `model` field says which tokenizer the numbers have to be
produced with, and a pair of **token counts** that a previous stage produced.
Everything that can go wrong with either is a `MeasureError`, so a caller (and
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
  count is the one case: nothing in this stage counts below zero, so a negative
  `before` or `after` is a programming error and raises a plain `ValueError`
  that is deliberately *not* a `MeasureError`.

`EmptyBaselineError` is both, and deliberately overlaps rather than splits —
exactly as `idc.compact.errors.NarrativeContractError` and
`idc.classify.errors.AdapterContractError` do. A `before` of zero is at once a
property of the transcript the stage was handed (a visible context that holds
no word) and a value no caller may divide by, so a caller never has to catch two
families for one event and a caller that reads it as a value problem is not made
wrong.

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
    "MeasureError",
    "NoTokenizerError",
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
        model: The model name that resolved to it.
        registered: The families that do have one, in registration order.
    """

    def __init__(self, family: str, model: str, registered: Sequence[str]) -> None:
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
