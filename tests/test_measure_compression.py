"""The compression rate: the number the study reports, and its edge cases."""

import pytest

from idc.measure import (
    CompressionRate,
    EmptyBaselineError,
    FakeTokenizer,
    MeasureError,
    count_context,
)


@pytest.mark.parametrize(
    ("before", "after", "saved_tokens", "saved_fraction", "ratio"),
    [
        (100, 25, 75, 0.75, 0.25),  # three quarters of the context gone
        (200, 100, 100, 0.5, 0.5),
        (4, 3, 1, 0.25, 0.75),
        (10, 10, 0, 0.0, 1.0),  # nothing compacted: no saving, and not an error
        (10, 0, 10, 1.0, 0.0),  # everything removed: the upper bound, exactly 1
        (100, 150, -50, -0.5, 1.5),  # it grew — negative, never clamped to 0
        (1, 3, -2, -2.0, 3.0),
    ],
)
def test_the_rate_is_before_minus_after_over_before(
    before, after, saved_tokens, saved_fraction, ratio
):
    rate = CompressionRate(before=before, after=after)

    assert rate.saved_tokens == saved_tokens
    assert rate.saved_fraction == pytest.approx(saved_fraction)
    assert rate.ratio == pytest.approx(ratio)
    assert rate.percent_saved == pytest.approx(saved_fraction * 100)


def test_a_compaction_that_made_the_context_bigger_is_reported_as_such():
    # The case a clamp would hide: the fraction is negative and `grew` says so,
    # because a method that enlarges the context is a result, not a bug to swallow.
    rate = CompressionRate(before=80, after=120)

    assert rate.saved_fraction < 0
    assert rate.grew is True
    assert CompressionRate(before=120, after=80).grew is False
    assert CompressionRate(before=80, after=80).grew is False


@pytest.mark.parametrize("after", [0, 1, 99])
def test_a_zero_baseline_is_refused_rather_than_given_a_value(after):
    # `(0 - after) / 0` has no value, and the study's number may have no
    # undefined region: a context that counts no token is refused where it is
    # handed over, not divided by.
    with pytest.raises(EmptyBaselineError) as refusal:
        CompressionRate(before=0, after=after)

    assert refusal.value.after == after
    assert "no token" in str(refusal.value)


def test_the_zero_baseline_refusal_is_a_measurement_error_and_a_value_error():
    # Both, the way `idc.compact.errors.NarrativeContractError` is: the counts
    # come from a transcript this stage was handed, and zero is at the same time
    # a value no caller may divide by.
    with pytest.raises(MeasureError):
        CompressionRate(before=0, after=1)
    with pytest.raises(ValueError, match="no token"):
        CompressionRate(before=0, after=1)


@pytest.mark.parametrize(("before", "after"), [(-1, 10), (10, -1), (-1, -1)])
def test_a_negative_count_is_a_plain_value_error(before, after):
    # The other side of the line: nothing in this stage produces a negative
    # count, so only a caller's own construction can, and that is a programming
    # error rather than a refusal of the measurement.
    with pytest.raises(ValueError, match="negative") as refusal:
        CompressionRate(before=before, after=after)

    assert not isinstance(refusal.value, MeasureError)


def test_a_rate_is_a_value_and_compares_by_its_counts():
    assert CompressionRate(before=10, after=5) == CompressionRate(before=10, after=5)
    assert CompressionRate(before=10, after=5) != CompressionRate(before=10, after=6)


def test_between_reads_the_totals_of_two_context_counts(corpus):
    before = count_context(corpus.session("single_intent"), FakeTokenizer())
    after = count_context(corpus.session("system_injected_message"), FakeTokenizer())

    rate = CompressionRate.between(before, after)

    assert (rate.before, rate.after) == (197, 156)
    assert rate.saved_tokens == 41
    assert rate.saved_fraction == pytest.approx(41 / 197)


def test_the_same_counts_give_the_same_rate_twice(corpus):
    before = count_context(corpus.session("single_intent"), FakeTokenizer())
    after = count_context(corpus.session("system_injected_message"), FakeTokenizer())

    assert CompressionRate.between(before, after) == CompressionRate.between(before, after)


def test_a_rate_says_its_counts_and_its_saving_when_printed():
    assert str(CompressionRate(before=100, after=25)) == "100 -> 25 tokens, 75.0% saved"
    assert str(CompressionRate(before=100, after=150)) == "100 -> 150 tokens, -50.0% saved"
