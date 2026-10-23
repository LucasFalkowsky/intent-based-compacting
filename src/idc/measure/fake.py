"""The fakes of the measurement stage: a tokenizer and a clock, both scripted.

Measurement computes everything it reports from the transcript — except two
things it cannot: how many tokens a piece of text is, and when the run
happened. Both are injected seams, and both have their fake here:
`FakeTokenizer` for `idc.measure.tokenizers.Tokenizer` and `FakeClock` for
`idc.measure.clock.Clock`. Neither is a test helper on the side; they are the
only implementations the suite uses, which is what makes the suite
deterministic.

Every measurement test in this repository counts with this tokenizer. That is
not a convenience, it is the condition under which the suite can exist at all:
a real tokenizer needs a downloaded vocabulary or a network call, and
`CLAUDE.md` forbids both — no network, no unpinned data, nothing that makes a
test's verdict depend on what is cached on the machine running it. The fake is
therefore part of the contract rather than a test helper, exactly as
`idc.classify.fake.FakeAdapter` is.

```python
from idc.measure import FakeTokenizer, count_context

tokenizer = FakeTokenizer()
counted = count_context(session, tokenizer)
tokenizer.calls          # every text it was handed, in order
```

Two properties make it usable as an oracle:

- **A token is a run of non-whitespace.** `FakeTokenizer().count(text)` is
  `len(text.split())`, so an expectation can be written down by counting the
  words of a fixture by hand, and a test that disagrees with the implementation
  says which of the two is wrong. It is *not* a model of any real tokenizer's
  numbers, and nothing may read it as one — a real tokenizer splits words into
  several tokens and punctuation into its own.
- **It records what it was asked, in order.** Like the fake adapter's call
  record, that is what lets a test pin *which text* was counted rather than
  only the total: a counter that silently dropped tool output would still
  produce a plausible number, and only the record shows it.

It holds no script: its answer is fixed by the rule above, which is the one
thing T9.1 (#37) requires of it. A test that needs a specific number arranges
the text, not the tokenizer.

`FakeClock` is the other half, and it *does* hold a script, because there is no
rule that could fix what time it is:

```python
from idc.measure import FakeClock, RunTimer

clock = FakeClock([START, END])
timer = RunTimer(clock)
timer.start()
window = timer.finish()  # exactly START .. END
clock.calls              # every instant it answered, in order
```

It plays its instants in order and refuses to answer past the end of its
script, so a test's window is what the test wrote down and never a consequence
of how often the code under test asked the clock.
"""

from collections.abc import Iterable
from datetime import datetime

__all__ = ["FakeClock", "FakeTokenizer"]


class FakeTokenizer:
    """A `Tokenizer` that counts one token per whitespace-separated word.

    Stateless as far as its answers go — the same text always counts the same —
    and stateful only in the record of what it was asked.
    """

    def __init__(self) -> None:
        """Build the tokenizer with an empty call record."""
        self._calls: list[str] = []

    def __repr__(self) -> str:
        """Return a representation naming how much it has counted so far."""
        return f"{type(self).__name__}(calls={len(self._calls)}, total={self.total})"

    @property
    def calls(self) -> tuple[str, ...]:
        """Return every text the tokenizer was handed, in order."""
        return tuple(self._calls)

    @property
    def total(self) -> int:
        """Return the sum of every count the tokenizer has returned."""
        return sum(len(text.split()) for text in self._calls)

    def count(self, text: str) -> int:
        """Return how many whitespace-separated words `text` holds.

        Args:
            text: The text to count. Whitespace-only text counts zero, and the
                record keeps it either way.

        Returns:
            `len(text.split())` — the number of runs of non-whitespace.
        """
        self._calls.append(text)
        return len(text.split())


class FakeClock:
    """A `Clock` that plays scripted instants and records what it answered.

    The counterpart of `FakeTokenizer` for the one value measurement cannot
    compute from a transcript: `CLAUDE.md` forbids a clock in a test, so a test
    that needs a run's two instants writes them down. Built like
    `idc.compact.fake.FakeNarrativeClient` — a script consumed eagerly and a
    call record — so a test can pin which instants a `RunTimer` took and in
    what order, not only that a manifest ended up with two.

    Args:
        instants: The instants to play, in order. Consumed eagerly, so an
            iterator works and a later change to a mutable list cannot change
            what the clock will answer.

    Raises:
        ValueError: If an entry is not an aware `datetime`. A naive instant
            cannot be compared with one from another machine and is refused by
            `RunWindow` anyway, so a script holding one is a test-authoring
            mistake, caught where the script was written. A plain `ValueError`
            and not a `MeasureError`: see `idc.measure.errors` for the line.
    """

    def __init__(self, instants: Iterable[datetime]) -> None:
        """Build the clock from its script, checking that every entry is aware."""
        entries = tuple(instants)
        for position, entry in enumerate(entries):
            if not isinstance(entry, datetime) or entry.utcoffset() is None:
                raise ValueError(
                    f"a scripted instant is an aware datetime; entry {position} is {entry!r}"
                )
        self._script = entries
        self._calls: list[datetime] = []

    def __repr__(self) -> str:
        """Return a representation naming how long the script is and what is left."""
        return (
            f"{type(self).__name__}(script={len(self._script)} instant(s), "
            f"remaining={self.remaining})"
        )

    @property
    def script(self) -> tuple[datetime, ...]:
        """Return the script, in order."""
        return self._script

    @property
    def calls(self) -> tuple[datetime, ...]:
        """Return every instant the clock answered, in order."""
        return tuple(self._calls)

    @property
    def remaining(self) -> int:
        """Return how many scripted instants have not been played yet."""
        return len(self._script) - len(self._calls)

    def now(self) -> datetime:
        """Return the next scripted instant, recording that it was asked for.

        Returns:
            The next instant of the script.

        Raises:
            ValueError: If the script holds no further instant. A clock that
                guessed one would make a test's window depend on how often the
                code under test happened to ask.
        """
        if not self.remaining:
            raise ValueError(
                f"the clock script holds {len(self._script)} instant(s) and all of them have "
                f"been played; script one more"
            )
        instant = self._script[len(self._calls)]
        self._calls.append(instant)
        return instant
