"""The fake tokenizer: one token per word, so no test calls a real one.

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
"""

__all__ = ["FakeTokenizer"]


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
