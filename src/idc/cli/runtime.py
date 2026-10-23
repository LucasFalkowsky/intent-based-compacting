"""The four dependencies a command cannot compute, and the seam they arrive through.

Everything this tool reports is computed from the transcript — that is the
determinism rule in `CLAUDE.md` — with exactly four exceptions, and every one of
them is already a one-method `Protocol` in the stage that needs it:

| What | Protocol | Real implementation in this build |
|---|---|---|
| the classifier's decisions | `idc.classify.IntentAdapter` | **none** |
| the three narrative sections | `idc.compact.NarrativeClient` | **none** |
| a token count | `idc.measure.Tokenizer` | **none** |
| the run's two instants | `idc.measure.Clock` | `SystemClock` |

The first two are models, and `idc.classify.LlmAdapter` and `S1MAdapter` take
their backend (`ModelClient`, `ContinuationScorer`) as a constructor argument
precisely so that the suite can run without one; this build ships no
implementation of either, and none of a `Tokenizer` — every real one needs a
downloaded vocabulary (see `idc.measure.NoTokenizerError`). So the seam is not a
convenience for tests that could be bypassed in production: for three of the
four rows there is nothing to bypass it with, and a command that tried would
have to invent a backend.

`Runtime` is therefore how all four reach a command, and `idc.cli.main` passes
it straight through:

```python
from idc.cli import Runtime, main
from idc.classify import NEW, FakeAdapter
from idc.compact import FakeNarrativeClient
from idc.measure import FakeClock, FakeTokenizer, ModelFamily, TokenizerRegistry

main(
    ["run", "--agent", "claude-code", "--in", "session.jsonl", "--out", "compacted.jsonl"],
    runtime=Runtime(
        adapter=FakeAdapter([NEW]),
        narrative_client=FakeNarrativeClient(["## Intent: …"]),
        tokenizers=TokenizerRegistry({ModelFamily.CLAUDE: FakeTokenizer()}),
        clock=FakeClock([started, finished]),
    ),
)
```

A seam that is declared and never wired fails *silently*, with the suite green,
so the absence of each dependency is a `BackendUnavailableError` naming it
rather than a default that quietly does something else. The one exception is the
clock, which really does have a production implementation and defaults to it;
`tests/test_cli_run.py` pins that every one of the four is reached by injecting
all four and reading what they recorded.
"""

from dataclasses import dataclass

from idc.classify import IntentAdapter
from idc.cli.errors import BackendUnavailableError
from idc.compact import NarrativeClient
from idc.measure import Clock, SystemClock, TokenizerRegistry

__all__ = ["Runtime"]


@dataclass(frozen=True, slots=True)
class Runtime:
    """The dependencies a command runs with, as a caller supplies them.

    Attributes:
        adapter: The classifier adapter that decides the intent chains. No
            default: see the module docstring.
        narrative_client: The backend that writes a chain's two narrative
            sections. No default, for the same reason.
        tokenizers: The tokenizer per model family the token counts are
            produced with. An empty registry when omitted, so that the refusal
            comes from `idc.measure` and names the model family that has
            nothing behind it, rather than from here.
        clock: Where a run manifest's two instants come from. `SystemClock`
            when omitted — the one of the four that has a real implementation.
    """

    adapter: IntentAdapter | None = None
    narrative_client: NarrativeClient | None = None
    tokenizers: TokenizerRegistry | None = None
    clock: Clock | None = None

    def require_adapter(self, requested: str) -> IntentAdapter:
        """Return the classifier adapter, refusing to invent one.

        Args:
            requested: The adapter the configuration asked for, named in the
                error so that the message is about the run and not about this
                class.

        Returns:
            The injected adapter.

        Raises:
            BackendUnavailableError: If none was injected.
        """
        if self.adapter is None:
            raise BackendUnavailableError(
                f"the {requested!r} classifier adapter",
                "idc.classify.LlmAdapter and S1MAdapter each take their model backend as an "
                "injected dependency and this build ships neither, so the adapter object is "
                "passed in: idc.cli.Runtime(adapter=...)",
            )
        return self.adapter

    def require_narrative_client(self) -> NarrativeClient:
        """Return the narrative backend, refusing to invent one.

        Returns:
            The injected client.

        Raises:
            BackendUnavailableError: If none was injected.
        """
        if self.narrative_client is None:
            raise BackendUnavailableError(
                "a narrative backend",
                "the three narrative sections of a chain's block are the only model call the "
                "compaction stage makes, and this build ships no client for it: "
                "idc.cli.Runtime(narrative_client=...)",
            )
        return self.narrative_client

    def tokenizer_registry(self) -> TokenizerRegistry:
        """Return the tokenizer registry, empty when none was injected.

        Returns:
            The injected registry, or an empty one. Empty rather than refused
            here, because `idc.measure` already refuses it precisely: its
            `NoTokenizerError` names the model family the transcript's model
            was placed in, which this class does not know.
        """
        return self.tokenizers if self.tokenizers is not None else TokenizerRegistry()

    def run_clock(self) -> Clock:
        """Return the clock a run's manifest takes its two instants from.

        Returns:
            The injected clock, or `SystemClock`.
        """
        return self.clock if self.clock is not None else SystemClock()
