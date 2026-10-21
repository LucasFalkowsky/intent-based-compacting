"""The compression rate: the one number the study reports, pinned.

```python
from idc.measure import CompressionRate

rate = CompressionRate.between(before, after)   # two ContextCounts
rate.saved_fraction                              # 0.62 — what the study reports
```

**The formula.** With `before` the tokens of the visible context as it stands
and `after` the tokens of the compacted one, both counted by the one definition
in `idc.measure.tokens`:

```
saved_tokens   = before - after
saved_fraction = (before - after) / before
percent_saved  = saved_fraction * 100
ratio          = after / before
```

`saved_fraction` is the reported figure, and it is a **saved fraction**, not a
ratio and not a percentage: `0.62` means 62% of the tokens are gone. The other
three are derived from the same two counts and offered so that nothing
downstream recomputes them with a different rounding. `ratio` is the
`after / before` the literature usually calls the compression ratio, kept under
its own name so the two can never be mistaken for one another.

**The edge cases, each decided rather than left to the arithmetic:**

- `before == 0` is **refused**, as `EmptyBaselineError`. The expression has no
  value there, and the alternatives are all worse than a refusal: `0.0` would
  read as "compaction saved nothing" for a transcript that was never
  measurable, and a `nan` or an `inf` would travel into a results table without
  a single line of output saying so. A visible context that counts no token is
  a transcript to look at, not a number to report.
- `after > before` gives a **negative** saved fraction and is never clamped.
  Intent-driven compaction can make a context bigger — a transcript of one
  short turn compacts to a block with four section headers — and that is a
  finding of the study, not an error to swallow at the last step. `grew` says
  so without a reader having to compare a float against zero.
- `after == before` is exactly `0.0`, and `after == 0` is exactly `1.0`: the
  two ends of the normal range, pinned because the obvious variant
  `1 - before / after` divides by the wrong side and has no value at all for
  the second of them.
- A negative count raises a plain `ValueError` and deliberately **not** a
  `MeasureError`: nothing in this stage counts below zero, so only a caller's
  own construction can produce one. See `idc.measure.errors`.

The counts are integers and the rate is derived from them on access, so a
`CompressionRate` is a value: two of them built from the same counts are equal,
and nothing about it depends on when it was built.
"""

from dataclasses import dataclass
from typing import Self

from idc.measure.errors import EmptyBaselineError
from idc.measure.tokens import ContextCount

__all__ = ["CompressionRate"]


@dataclass(frozen=True, slots=True)
class CompressionRate:
    """How much of a context a compaction removed.

    Attributes:
        before: Tokens of the visible context before compaction. Has to be
            positive — see `EmptyBaselineError`.
        after: Tokens of the visible context after it. May exceed `before`.
    """

    before: int
    after: int

    def __post_init__(self) -> None:
        """Reject a negative count and a baseline of zero.

        Raises:
            ValueError: If either count is negative. A plain `ValueError` and
                not a `MeasureError`: a count below zero can only come from a
                caller's own construction.
            EmptyBaselineError: If `before` is zero, which leaves the rate
                undefined. Both a `MeasureError` and a `ValueError`.
        """
        if self.before < 0 or self.after < 0:
            raise ValueError(
                f"a token count cannot be negative; got before={self.before}, after={self.after}"
            )
        if self.before == 0:
            raise EmptyBaselineError(self.after)

    def __str__(self) -> str:
        """Return the rate as one diagnostic line."""
        return f"{self.before} -> {self.after} tokens, {self.percent_saved:.1f}% saved"

    @property
    def saved_tokens(self) -> int:
        """Return `before - after`; negative when the context grew."""
        return self.before - self.after

    @property
    def saved_fraction(self) -> float:
        """Return `(before - after) / before` — the figure the study reports.

        At most `1.0` (everything removed), `0.0` when nothing changed, and
        negative when the compacted context is the larger one. Never clamped.
        """
        return self.saved_tokens / self.before

    @property
    def percent_saved(self) -> float:
        """Return the saved fraction in percent, unrounded."""
        return self.saved_fraction * 100

    @property
    def ratio(self) -> float:
        """Return `after / before` — the compression ratio, not the saving."""
        return self.after / self.before

    @property
    def grew(self) -> bool:
        """Return whether the compacted context holds more tokens than the original."""
        return self.after > self.before

    @classmethod
    def between(cls, before: ContextCount, after: ContextCount) -> Self:
        """Return the rate between two counted contexts.

        Args:
            before: The count of the context as it stands.
            after: The count of the compacted context.

        Returns:
            The rate over their totals.

        Raises:
            EmptyBaselineError: If `before` counts no token.
        """
        return cls(before=before.tokens, after=after.tokens)
