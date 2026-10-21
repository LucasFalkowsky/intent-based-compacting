r"""The fake narrative client: scripted responses, so no test calls a model.

Every test of this stage and of everything downstream of it runs on this
client — which is what keeps the suite network-free, and what makes the
end-to-end test (#42) possible at all. That makes the fake part of the contract
rather than a test helper, exactly as `idc.classify.fake.FakeAdapter` is: a test
writes down the response it means and gets a deterministic block back.

```python
from idc.compact import FakeNarrativeClient, compact_chain

client = FakeNarrativeClient(["## Intent: Stamp timestamps in UTC\\n…"])
block = compact_chain(chain, resolution, client)
client.calls                      # what it was asked, and what it answered
```

Three properties make it usable as an oracle, and they are the three
`FakeAdapter` has:

- **It plays its script in order, and nothing else.** One response per `write`
  call. When the script runs out it raises `NarrativeScriptExhaustedError`
  instead of repeating its last answer — a fake that repeated would write one
  chain's narrative into another chain's block and the test would still pass.
- **It records every call.** `calls` is what pins the ticket's hardest number:
  one model call per chain, read off the record rather than trusted.
- **It is not smarter than a real backend.** Its script is checked for being
  *text* and nothing else: a malformed response is exactly what the stage has to
  refuse, so a fake that validated its own script could not produce the case.
  `as_narrative` is the only validator, and it runs where a real answer would
  arrive.
"""

from collections.abc import Iterable
from dataclasses import dataclass

from idc.compact.errors import NarrativeScriptExhaustedError
from idc.compact.narrative import NarrativeRequest

__all__ = [
    "FakeNarrativeCall",
    "FakeNarrativeClient",
]


@dataclass(frozen=True, slots=True)
class FakeNarrativeCall:
    """One request the fake client answered, and the text it answered with.

    Attributes:
        request: The request it was handed, unexamined — which is how a test
            asserts what the stage actually offered a backend.
        response: The scripted text it returned.
    """

    request: NarrativeRequest
    response: str

    @property
    def chain_id(self) -> str:
        """Return the id of the chain the request was about."""
        return self.request.chain_id


class FakeNarrativeClient:
    """A `NarrativeClient` that returns a scripted response per chain.

    Args:
        script: The responses to play, in order. Consumed eagerly, so an
            iterator works and a later change to a mutable list cannot change
            what the client will answer.

    Raises:
        ValueError: If an entry is not a string. A response is text — that much
            is the client's own contract and a test-authoring mistake, caught
            where the script was written. Whether the text *is* a narrative is
            not checked: refusing a malformed one is the stage's job, and a
            script that could not hold one could not test it.
    """

    def __init__(self, script: Iterable[str]) -> None:
        """Build the client from its script, checking that every entry is text."""
        entries = tuple(script)
        for position, entry in enumerate(entries):
            if not isinstance(entry, str):
                raise ValueError(
                    f"a scripted narrative response is text; entry {position} is {entry!r} "
                    f"of type {type(entry).__name__}"
                )
        self._script = entries
        self._calls: list[FakeNarrativeCall] = []

    def __repr__(self) -> str:
        """Return a representation naming how long the script is and what is left."""
        return (
            f"{type(self).__name__}(script={len(self._script)} response(s), "
            f"remaining={self.remaining})"
        )

    @property
    def script(self) -> tuple[str, ...]:
        """Return the script, in order."""
        return self._script

    @property
    def calls(self) -> tuple[FakeNarrativeCall, ...]:
        """Return every call the client answered, in order."""
        return tuple(self._calls)

    @property
    def remaining(self) -> int:
        """Return how many scripted responses have not been played yet."""
        return len(self._script) - len(self._calls)

    def write(self, request: NarrativeRequest) -> str:
        """Return the next scripted response, recording what was asked.

        Args:
            request: The chain's facts and the instruction; kept in the record
                unexamined — the fake answers from its script and not from the
                request, which is what makes it an oracle rather than a second
                compactor.

        Returns:
            The next response of the script.

        Raises:
            NarrativeScriptExhaustedError: If the script holds no further
                response. The chain it ran out at is named.
        """
        if not self.remaining:
            raise NarrativeScriptExhaustedError(request.chain_id, len(self._script))
        response = self._script[len(self._calls)]
        self._calls.append(FakeNarrativeCall(request=request, response=response))
        return response
