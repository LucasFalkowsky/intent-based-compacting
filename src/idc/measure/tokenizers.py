"""The tokenizer seam: one method, and which model family gets which tokenizer.

A tokenizer is anything that can say how many tokens a piece of text is, and
nothing else. That is the whole contract:

```python
from idc.measure import FakeTokenizer, ModelFamily, TokenizerRegistry, count_context

registry = TokenizerRegistry({ModelFamily.CLAUDE: FakeTokenizer()})
counted = count_context(session, registry.tokenizer_for(session))
```

It follows `idc.classify.adapters` deliberately: a `Protocol` with a single
method, so a real tokenizer inherits nothing and the fake
(`idc.measure.fake.FakeTokenizer`) is not a special case, and the tokenizer is
**injected** into the counting functions rather than looked up by them.

Three properties of this shape are the point of it:

- **A tokenizer sees text, and only text.** It is handed the strings
  `idc.measure.tokens.message_parts` resolved out of a message — never a
  `Message`, a `Session` or a model handle. A backend therefore cannot decide
  differently for a tool result than for a user turn, which is what keeps two
  runs comparable, and the fake can produce everything a real one gets.
- **The registry picks, it does not guess.** `model_family` places a model name
  in a family by its first dash-separated segment; a name it cannot place is
  `UnknownModelError`, and a family with nothing registered behind it is
  `NoTokenizerError`. Neither falls back to a default — see those errors for
  why.
- **There is no module-level registry.** A `TokenizerRegistry` is an object a
  caller builds and passes, so a run's tokenizer set is explicit and two runs
  cannot differ by which module happened to register something first.

**This build ships no real tokenizer.** Every real one needs a downloaded
vocabulary or a network call, and `CLAUDE.md` forbids both inside the test
suite, so a fresh `TokenizerRegistry()` is empty and the only tokenizer in the
package is the fake. Wiring a real one in is registering it against the family
it belongs to; nothing else in this package changes.
"""

from collections.abc import Mapping
from enum import StrEnum
from typing import Final, Protocol, runtime_checkable

from idc.measure.errors import NoTokenizerError, UnknownModelError
from idc.schema import Session

__all__ = [
    "FAMILY_SEPARATOR",
    "ModelFamily",
    "Tokenizer",
    "TokenizerRegistry",
    "model_family",
]

FAMILY_SEPARATOR: Final = "-"
"""What separates a model family from the rest of a model name.

`claude-opus-5` and `claude-sonnet-4-5` are both of the `claude` family,
`gemini-2.5-pro` is of the `gemini` family. The rule is the whole of it: a
model name's family is its first segment, lowercased, with surrounding
whitespace stripped. A name that carries a provider in front of it
(`anthropic/claude-opus-5`) therefore does **not** resolve — it is refused
rather than guessed at, which is the decision `UnknownModelError` records.
"""


class ModelFamily(StrEnum):
    """The model families this build can place a transcript's model in.

    Exactly the families the three supported agents' models fall into, and no
    speculative ones: a family that nothing can produce a transcript for would
    be a tokenizer slot no run could fill, and an unknown model is refused
    loudly instead of being absorbed by a catch-all member.
    """

    CLAUDE = "claude"
    GEMINI = "gemini"


def model_family(model: str | None) -> ModelFamily:
    """Return the model family a transcript's model belongs to.

    Args:
        model: The model name a canonical `Session` recorded, or `None` when it
            recorded none.

    Returns:
        The family: the model name's first `FAMILY_SEPARATOR`-separated
        segment, lowercased and stripped of surrounding whitespace.

    Raises:
        UnknownModelError: If the name is `None`, holds no word, or its first
            segment is not a member of `ModelFamily`. Never a fallback: see
            that error for why a default tokenizer would be worse than a
            refusal.
    """
    known = tuple(family.value for family in ModelFamily)
    if model is None or not model.strip():
        raise UnknownModelError(model, known)
    segment = model.strip().lower().split(FAMILY_SEPARATOR, 1)[0]
    try:
        return ModelFamily(segment)
    except ValueError as unknown:
        raise UnknownModelError(model, known) from unknown


@runtime_checkable
class Tokenizer(Protocol):
    """The one method a tokenizer has to provide.

    A protocol rather than a base class, for the reason
    `idc.classify.adapters.IntentAdapter` is one: a tokenizer is anything with a
    `count` method of this shape. It is `runtime_checkable` so that a test can
    assert the shape — but only the *presence* of `count` is checkable at
    runtime, not its signature, so nothing in this package uses `isinstance` as
    a guard and nothing here promises that it does.
    """

    def count(self, text: str) -> int:
        """Return how many tokens `text` is.

        Args:
            text: One piece of a message's text, as
                `idc.measure.tokens.message_parts` resolved it. Never empty —
                the counter does not hand an empty part over at all.

        Returns:
            The number of tokens, zero or more.
        """
        ...


class TokenizerRegistry:
    """Which tokenizer a model family is counted with.

    An object a caller builds and passes rather than a module-level table, so
    that the tokenizers a run used are part of the call and not of the import
    order. It is empty by default because this build ships no real tokenizer;
    see the module docstring.

    Args:
        tokenizers: The tokenizer per family to start with. Copied, so a later
            change to the caller's mapping cannot change what the registry
            hands out.
    """

    def __init__(self, tokenizers: Mapping[ModelFamily, Tokenizer] | None = None) -> None:
        """Build the registry from an optional mapping of family to tokenizer."""
        self._tokenizers: dict[ModelFamily, Tokenizer] = dict(tokenizers or {})

    def __repr__(self) -> str:
        """Return a representation naming the families that have a tokenizer."""
        families = ", ".join(family.value for family in self.families) or "empty"
        return f"{type(self).__name__}({families})"

    @property
    def families(self) -> tuple[ModelFamily, ...]:
        """Return the families that have a tokenizer, in registration order."""
        return tuple(self._tokenizers)

    def register(self, family: ModelFamily, tokenizer: Tokenizer) -> None:
        """Register `tokenizer` for `family`, replacing one already registered.

        Args:
            family: The model family to count with this tokenizer.
            tokenizer: The tokenizer itself.
        """
        self._tokenizers[family] = tokenizer

    def tokenizer_for_model(self, model: str | None) -> Tokenizer:
        """Return the tokenizer a model is to be counted with.

        Args:
            model: The model name a canonical `Session` recorded, or `None`.

        Returns:
            The tokenizer registered for that model's family.

        Raises:
            UnknownModelError: If the name belongs to no known family.
            NoTokenizerError: If it does, and nothing is registered for it.
        """
        family = model_family(model)
        tokenizer = self._tokenizers.get(family)
        if tokenizer is None:
            raise NoTokenizerError(
                family.value,
                str(model),
                tuple(known.value for known in self.families),
            )
        return tokenizer

    def tokenizer_for(self, session: Session) -> Tokenizer:
        """Return the tokenizer a session is to be counted with.

        Args:
            session: The canonical session whose `model` decides the family.

        Returns:
            The tokenizer registered for that model's family.

        Raises:
            UnknownModelError: If the session's model belongs to no known
                family, `None` included.
            NoTokenizerError: If it does, and nothing is registered for it.
        """
        return self.tokenizer_for_model(session.model)
