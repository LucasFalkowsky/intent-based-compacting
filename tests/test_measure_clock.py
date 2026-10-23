"""The injected clock and the run window a manifest records (T9.3, #39).

`CLAUDE.md` forbids a clock in the test suite, so the two instants a manifest
carries are not read inside the stage: `Clock` is a one-method protocol, the
same seam `IntentAdapter`, `NarrativeClient` and `Tokenizer` are, and
`FakeClock` plays a scripted list of instants and records what it was asked.
Every assertion here is over scripted instants; the one test that calls
`SystemClock` asserts only its *shape* — that the value is an aware datetime in
UTC — and never a value, so it cannot depend on when the suite runs.
"""

from datetime import UTC, datetime, timedelta

import pytest

from idc.measure import Clock, FakeClock, MeasureError, RunTimer, RunWindow, SystemClock

START = datetime(2026, 10, 21, 9, 0, 0, tzinfo=UTC)
"""First scripted instant; a fixed value, never now()."""

END = datetime(2026, 10, 21, 9, 2, 30, tzinfo=UTC)
"""Second scripted instant, 150 seconds after `START`."""

NAIVE = datetime(2026, 10, 21, 9, 0, 0)
"""The same wall time without a zone — refused everywhere a clock is read."""


def test_both_clocks_satisfy_the_protocol():
    assert isinstance(SystemClock(), Clock)
    assert isinstance(FakeClock([START]), Clock)


def test_system_clock_answers_an_aware_utc_instant():
    answered = SystemClock().now()

    assert isinstance(answered, datetime)
    assert answered.utcoffset() == timedelta(0)


def test_fake_clock_plays_its_script_in_order_and_records_the_calls():
    clock = FakeClock([START, END])

    assert clock.now() == START
    assert clock.now() == END
    assert clock.calls == (START, END)
    assert clock.remaining == 0


def test_fake_clock_refuses_to_answer_past_its_script():
    clock = FakeClock([START])
    clock.now()

    with pytest.raises(ValueError, match="script") as raised:
        clock.now()

    assert not isinstance(raised.value, MeasureError)


@pytest.mark.parametrize("entry", [NAIVE, "2026-10-21T09:00:00+00:00", None])
def test_fake_clock_refuses_a_script_entry_that_is_not_an_aware_instant(entry):
    with pytest.raises(ValueError, match="instant"):
        FakeClock([START, entry])


def test_run_window_carries_the_two_instants_and_their_duration():
    window = RunWindow(started_at=START, finished_at=END)

    assert (window.started_at, window.finished_at) == (START, END)
    assert window.duration == timedelta(seconds=150)
    assert "150" in str(window) or "0:02:30" in str(window)


def test_run_window_accepts_two_equal_instants():
    window = RunWindow(started_at=START, finished_at=START)

    assert window.duration == timedelta(0)


def test_run_window_refuses_a_finish_before_its_start():
    with pytest.raises(ValueError, match="before"):
        RunWindow(started_at=END, finished_at=START)


@pytest.mark.parametrize("field", ["started_at", "finished_at"])
def test_run_window_refuses_a_naive_instant(field):
    instants = {"started_at": START, "finished_at": END} | {field: NAIVE}

    with pytest.raises(ValueError, match="time zone"):
        RunWindow(**instants)


def test_run_timer_takes_both_instants_from_the_clock_in_order():
    clock = FakeClock([START, END])
    timer = RunTimer(clock)

    timer.start()
    window = timer.finish()

    assert window == RunWindow(started_at=START, finished_at=END)
    assert clock.calls == (START, END)


def test_run_timer_refuses_to_finish_a_run_it_never_started():
    timer = RunTimer(FakeClock([START, END]))

    with pytest.raises(ValueError, match="started"):
        timer.finish()


def test_run_timer_refuses_a_second_start():
    timer = RunTimer(FakeClock([START, END]))
    timer.start()

    with pytest.raises(ValueError, match="already"):
        timer.start()
