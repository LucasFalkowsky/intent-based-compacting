"""The fake adapter: a scripted decision list, so no test calls a model.

Every test of everything downstream of the classifier runs on this adapter —
the resolver's, the compactor's, the renderer's and above all the end-to-end
test (#42), which is network-free because of it. That makes the fake part of the
contract rather than a test helper: a test writes down the decisions it means,
and gets a result that is deterministic and inspectable.

```python
from idc.classify import NEW, FakeAdapter, classify

adapter = FakeAdapter([NEW, NEW, "C1"])      # A, B, back to A
chains = classify(session, chunks, adapter)
adapter.calls                                 # what it was asked, and answered
```

Three properties make it usable as an oracle:

- **It plays its script in order, and nothing else.** One decision per `decide`
  call, in the order written. When the script runs out it raises
  `ScriptExhaustedError` instead of repeating its last answer or defaulting to
  `NEW` — a fake that guessed would make a test pass for a reason the test never
  stated.
- **It is normalized when it is built.** The script may be written in any of the
  forms the interface allows (`NEW`, `"C1"`, `Decision.extend("C1", 0.8)`), and
  an entry that is none of them is `AdapterContractError` at construction, where
  the script was written, not at the call that would have used it.
- **It is not smarter than a real backend.** It does *not* check its chain ids
  against the candidates it was given: a model can name a chain that is not on
  offer, and the pipeline's own guard (`UnknownChainError`) is what has to catch
  that. A fake that refused here could not produce the case at all.

It records every call it answered, so a test can assert on what the classifier
actually offered — which is how the growing candidate list is pinned.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from idc.classify.adapters import ChainCandidate, Decision, DecisionLike, as_decision
from idc.classify.errors import ScriptExhaustedError
from idc.schema import Chunk


@dataclass(frozen=True, slots=True)
class FakeCall:
    """One question the fake adapter was asked, and the answer it gave.

    Attributes:
        chunk_id: `Chunk.id` of the chunk it was asked about.
        candidates: The candidates it was shown, in the order they were offered.
        decision: The decision it returned.
    """

    chunk_id: str
    candidates: tuple[ChainCandidate, ...]
    decision: Decision

    @property
    def candidate_ids(self) -> tuple[str, ...]:
        """Return the ids of the candidates that were on offer."""
        return tuple(candidate.id for candidate in self.candidates)


class FakeAdapter:
    """An `IntentAdapter` that returns a scripted decision per chunk.

    Args:
        script: The decisions to play, in order, each a `Decision`, a chain id
            or `NEW`. Consumed eagerly, so an iterator works and a later change
            to a mutable list cannot change what the adapter will answer.

    Raises:
        AdapterContractError: If an entry is not one of the allowed forms.
        ValueError: If an entry is a string that is not a chain id.
    """

    def __init__(self, script: Iterable[DecisionLike]) -> None:
        """Build the adapter from its script, normalizing every entry."""
        self._script = tuple(as_decision(entry) for entry in script)
        self._calls: list[FakeCall] = []

    def __repr__(self) -> str:
        """Return a representation naming the script and how much is left."""
        return f"{type(self).__name__}(script={list(self._script)}, remaining={self.remaining})"

    @property
    def script(self) -> tuple[Decision, ...]:
        """Return the normalized script, in order."""
        return self._script

    @property
    def calls(self) -> tuple[FakeCall, ...]:
        """Return every call the adapter answered, in order."""
        return tuple(self._calls)

    @property
    def remaining(self) -> int:
        """Return how many scripted decisions have not been played yet."""
        return len(self._script) - len(self._calls)

    def decide(self, chunk: Chunk, candidates: Sequence[ChainCandidate]) -> Decision:
        """Return the next scripted decision, recording what was asked.

        Args:
            chunk: The chunk to classify; only its id is read, for the record.
            candidates: The open chains, kept in the record unexamined.

        Returns:
            The next decision of the script.

        Raises:
            ScriptExhaustedError: If the script holds no further decision.
        """
        if not self.remaining:
            raise ScriptExhaustedError(chunk.id, len(self._script))
        decision = self._script[len(self._calls)]
        self._calls.append(
            FakeCall(chunk_id=chunk.id, candidates=tuple(candidates), decision=decision)
        )
        return decision
