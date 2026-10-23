"""Where a run manifest's two instants come from, and why they are injected.

A manifest records when a run started and when it finished, and those are the
only two values in the whole tool that cannot be computed from a transcript.
`CLAUDE.md` forbids a clock in a test, so the clock is a seam rather than a
call: `Clock` is a `Protocol` with one method, exactly as
`idc.classify.adapters.IntentAdapter`, `idc.compact.narrative.NarrativeClient`
and `idc.measure.tokenizers.Tokenizer` are, `SystemClock` is the one real
implementation, and `idc.measure.fake.FakeClock` plays a scripted list of
instants. Nothing in this package calls `datetime.now` outside `SystemClock`.

```python
from idc.measure import RunTimer, SystemClock

timer = RunTimer(SystemClock())
timer.start()
...                      # the run
window = timer.finish()  # what build_manifest records
```

Two properties are the point of the shape:

- **The two instants are taken once each, in order.** `RunTimer` asks the clock
  at `start` and at `finish` and holds nothing else, so a manifest cannot
  acquire a timestamp as a side effect of being built, and a test can pin
  *which* instants were taken by reading the fake's call record.
- **An instant without a time zone is refused.** A manifest is study material
  that will be read next to other runs' manifests, and a naive timestamp cannot
  be compared with one from another machine. `RunWindow` therefore refuses a
  naive instant and a finish that precedes its start, both as a plain
  `ValueError`: both can only come from a caller's own code, which is the line
  `idc.measure.errors` draws.
"""

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Protocol, runtime_checkable

__all__ = ["Clock", "RunTimer", "RunWindow", "SystemClock"]


@runtime_checkable
class Clock(Protocol):
    """The one method a clock has to provide.

    A protocol rather than a base class, for the reason
    `idc.measure.tokenizers.Tokenizer` is one: a clock is anything with a `now`
    method of this shape. It is `runtime_checkable`, which checks the
    *presence* of `now` and not its signature, so nothing here uses
    `isinstance` as a guard — an instant that is not an aware `datetime` is
    refused by `RunWindow` instead.
    """

    def now(self) -> datetime:
        """Return the current instant.

        Returns:
            An aware `datetime`. A naive one is refused by every consumer in
            this package, so an implementation that cannot name its zone has
            nothing to offer here.
        """
        ...


class SystemClock:
    """The clock a real run reads: the machine's own, in UTC.

    The only place in the package that calls `datetime.now`. It exists because
    a tool that cannot obtain an instant cannot write a manifest at all; no
    test asserts a value it returned, and the suite's manifests are built from
    `idc.measure.fake.FakeClock`'s scripted instants.
    """

    def __repr__(self) -> str:
        """Return a representation naming the class."""
        return f"{type(self).__name__}()"

    def now(self) -> datetime:
        """Return the current instant in UTC.

        Returns:
            `datetime.now(UTC)` — aware, and in UTC rather than in the
            machine's local zone, so that two runs' manifests are comparable
            without knowing where each ran.
        """
        return datetime.now(UTC)


def _require_aware(field: str, instant: datetime) -> None:
    """Raise unless `instant` is an aware `datetime`.

    Args:
        field: Name of the field being checked, for the message.
        instant: The value to check.

    Raises:
        ValueError: If the value is not a `datetime`, or carries no time zone.
    """
    if not isinstance(instant, datetime):
        raise ValueError(
            f"{field} must be an aware instant, not {instant!r} of type {type(instant).__name__}"
        )
    if instant.utcoffset() is None:
        raise ValueError(
            f"{field} must be an aware instant; {instant.isoformat()} carries no time zone"
        )


@dataclass(frozen=True, slots=True)
class RunWindow:
    """When a run started and when it finished.

    Attributes:
        started_at: The instant the run began; aware, normally UTC.
        finished_at: The instant it ended; aware, and not before `started_at`.
            Equal to it for a run that took less than the clock's resolution,
            which is accepted rather than refused.
    """

    started_at: datetime
    finished_at: datetime

    def __post_init__(self) -> None:
        """Reject a naive instant and a finish that precedes the start.

        Raises:
            ValueError: If either instant is naive or not a `datetime`, or if
                `finished_at` is before `started_at`. A plain `ValueError` and
                not a `MeasureError`: a window is built by the caller's own
                code, never read out of a document.
        """
        _require_aware("started_at", self.started_at)
        _require_aware("finished_at", self.finished_at)
        if self.finished_at < self.started_at:
            raise ValueError(
                f"a run cannot finish before it started; finished_at "
                f"{self.finished_at.isoformat()} is before started_at "
                f"{self.started_at.isoformat()}"
            )

    def __str__(self) -> str:
        """Return the window as one diagnostic line."""
        return f"{self.started_at.isoformat()} .. {self.finished_at.isoformat()} ({self.duration})"

    @property
    def duration(self) -> timedelta:
        """Return how long the run took; never negative."""
        return self.finished_at - self.started_at


class RunTimer:
    """Takes a run's two instants from an injected clock, once each.

    Args:
        clock: The clock to ask. `SystemClock` in a real run,
            `idc.measure.fake.FakeClock` in the suite.
    """

    def __init__(self, clock: Clock) -> None:
        """Build the timer around a clock, with no instant taken yet."""
        self._clock = clock
        self._started_at: datetime | None = None

    def __repr__(self) -> str:
        """Return a representation naming the clock and whether the run started."""
        state = "not started" if self._started_at is None else self._started_at.isoformat()
        return f"{type(self).__name__}({self._clock!r}, {state})"

    @property
    def started_at(self) -> datetime | None:
        """Return the instant `start` took, or `None` before it was called."""
        return self._started_at

    def start(self) -> datetime:
        """Take the run's first instant from the clock.

        Returns:
            The instant the clock answered.

        Raises:
            ValueError: If the timer was already started. One timer records one
                run, so a second `start` would silently discard the first
                instant.
        """
        if self._started_at is not None:
            raise ValueError(
                f"this run was already started at {self._started_at.isoformat()}; "
                f"use a new RunTimer for a new run"
            )
        self._started_at = self._clock.now()
        return self._started_at

    def finish(self) -> RunWindow:
        """Take the run's second instant and return the window.

        Returns:
            The window between the two instants the clock answered.

        Raises:
            ValueError: If `start` was never called, or if the clock answered a
                naive instant or one before the start.
        """
        if self._started_at is None:
            raise ValueError("this run was never started; call start() before finish()")
        return RunWindow(started_at=self._started_at, finished_at=self._clock.now())
