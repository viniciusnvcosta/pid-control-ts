# ABOUTME: Behavioral tests for causal benchmark selection and raw-prefix preparation.
# ABOUTME: Protects holdout purging, observed truth, interpolation boundaries, and seed isolation.
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from hcp.forecast_types import PreparedHistory


def _history(n_weeks: int = 60) -> PreparedHistory:
    dates = pd.date_range("2020-01-04", periods=n_weeks, freq="7D").to_numpy()
    values = np.arange(n_weeks, dtype=float)
    return PreparedHistory(
        "us",
        dates,
        values,
        values,
        np.ones(n_weeks, bool),
        dates - np.timedelta64(26, "D"),
    )


def test_qr_responses_are_purged_by_forecast_date() -> None:
    from hcp.forecast_validation import prepare_prefix, qr_split

    history = _history()
    split = qr_split(history.dates, history.observed, history.forecast_dates)
    np.testing.assert_array_equal(split.validation_targets, history.dates[48:60])
    np.testing.assert_array_equal(
        split.validation_cutoffs, history.forecast_dates[48:60]
    )
    training = prepare_prefix(history, split.training_cutoff)
    # Target 48 is forecast on date 44+2 days: responses 45..47 are not available.
    np.testing.assert_array_equal(training.values, np.arange(45.0))
    assert split.training_cutoff == history.forecast_dates[48]
    assert not split.validation_targets.flags.writeable
    assert not split.validation_cutoffs.flags.writeable


def test_neural_holdout_scores_nine_fourth_steps() -> None:
    from hcp.forecast_validation import neural_split, prepare_prefix

    history = _history()
    observed = np.vstack([history.observed, history.observed])
    split = neural_split(history.dates, observed, history.forecast_dates)
    np.testing.assert_array_equal(split.validation_targets, history.dates[51:60])
    np.testing.assert_array_equal(
        split.validation_cutoffs, history.forecast_dates[51:60]
    )
    training = prepare_prefix(history, split.training_cutoff)
    np.testing.assert_array_equal(training.values, np.arange(48.0))
    assert split.training_cutoff == history.forecast_dates[51]


def test_gap_interpolation_cannot_cross_split() -> None:
    from hcp.forecast_validation import InsufficientHistoryError, prepare_prefix

    history = _history()
    raw = history.raw_values.copy()
    raw[44] = np.nan
    history = replace(history, raw_values=raw, observed=np.isfinite(raw))
    # Outer values contain 44, interpolated using 45: not available at this cutoff.
    with pytest.raises(InsufficientHistoryError, match="trailing"):
        prepare_prefix(history, history.forecast_dates[48])


def test_prefix_drops_leading_gaps_and_reinterpolates_raw_observed_values() -> None:
    from hcp.forecast_validation import prepare_prefix

    history = _history()
    raw = history.raw_values.copy()
    raw[:2] = np.nan
    raw[10] = 9999  # Finite but unobserved is not genuine truth.
    observed = np.isfinite(raw)
    observed[10] = False
    history = replace(
        history, values=np.full(60, -100.0), raw_values=raw, observed=observed
    )
    prefix = prepare_prefix(history, history.forecast_dates[48])
    np.testing.assert_array_equal(prefix.values, np.arange(2.0, 45.0))
    assert prefix.raw_values[8] == 9999
    assert not prefix.observed[8]
    assert not prefix.values.flags.writeable
    changed = raw.copy()
    changed[45:] += 100000
    shifted = prepare_prefix(
        replace(history, raw_values=changed), history.forecast_dates[48]
    )
    np.testing.assert_array_equal(prefix.values, shifted.values)


@pytest.mark.parametrize("split_name", ["qr_split", "neural_split"])
def test_unobserved_holdout_is_ineligible_without_searching_older_windows(
    split_name: str,
) -> None:
    from hcp import forecast_validation

    history = _history()
    observed = history.observed.copy()
    observed[49] = False
    assert (
        getattr(forecast_validation, split_name)(
            history.dates, observed, history.forecast_dates
        )
        is None
    )


def test_qr_requires_twelve_observed_purged_training_responses() -> None:
    from hcp.forecast_validation import qr_split

    # With 42 weeks the first holdout is30, cutoff permits responses 15..26 =12.
    history = _history(42)
    assert qr_split(history.dates, history.observed, history.forecast_dates) is not None
    observed = history.observed.copy()
    observed[25] = False  # Eleven responses remain; prefix end26 stays observed.
    assert qr_split(history.dates, observed, history.forecast_dates) is None
    shorter = _history(41)
    assert qr_split(shorter.dates, shorter.observed, shorter.forecast_dates) is None


@pytest.mark.parametrize("split_name", ["qr_split", "neural_split"])
def test_inner_training_trailing_gap_is_ineligible(split_name: str) -> None:
    from hcp import forecast_validation

    history = _history()
    observed = history.observed.copy()
    observed[44 if split_name == "qr_split" else 47] = False
    assert (
        getattr(forecast_validation, split_name)(
            history.dates, observed, history.forecast_dates
        )
        is None
    )


@pytest.mark.parametrize("problem", ["irregular", "lag", "shape", "nat", "missing"])
def test_invalid_split_calendar_or_metadata_raises(problem: str) -> None:
    from hcp.forecast_validation import neural_split, qr_split

    history = _history()
    dates, origins, observed = (
        history.dates.copy(),
        history.forecast_dates.copy(),
        history.observed.copy(),
    )
    if problem == "irregular":
        dates[10] += np.timedelta64(1, "D")
    elif problem == "lag":
        origins += np.timedelta64(1, "D")
    elif problem == "shape":
        observed = observed[:-1]
    elif problem == "nat":
        dates[0] = np.datetime64("NaT")
    else:
        origins = None
    for split_function in (qr_split, neural_split):
        with pytest.raises(ValueError):
            split_function(dates, observed, origins)


@pytest.mark.parametrize("field", ["raw_values", "observed", "forecast_dates"])
def test_v2_prefix_rejects_missing_raw_metadata(field: str) -> None:
    from hcp.forecast_validation import prepare_prefix

    history = _history()
    with pytest.raises(ValueError, match="metadata"):
        prepare_prefix(replace(history, **{field: None}), history.forecast_dates[48])


def test_neural_seed_streams_are_independent_and_repeatable() -> None:
    from hcp.forecast_validation import neural_seed_streams

    streams = neural_seed_streams(
        np.random.SeedSequence(20261021, spawn_key=(0, 6, 2)), 3
    )
    assert list(streams) == ["training", "validation", "refit", "prediction"]
    assert [len(value) for value in streams.values()] == [3, 3, 1, 1]
    keys = [seed.spawn_key for group in streams.values() for seed in group]
    assert keys == [(0, 6, 2, i) for i in range(8)]
    for seed in streams["validation"]:
        first_epoch = np.random.default_rng(seed).normal(size=10)
        next_epoch = np.random.default_rng(seed).normal(size=10)
        np.testing.assert_array_equal(first_epoch, next_epoch)
    with pytest.raises(ValueError):
        neural_seed_streams(np.random.SeedSequence(1), 0)


def test_neural_requires_seventeen_available_inner_training_weeks() -> None:
    from hcp.forecast_validation import neural_split

    history = _history()
    observed = np.vstack([history.observed, history.observed]).copy()
    observed[1, :31] = False  # Preceding prefix 31..47 contains 17 weeks.
    assert neural_split(history.dates, observed, history.forecast_dates) is not None
    observed[1, 31] = False  # Only 16 weeks: insufficient shifted training sequence.
    assert neural_split(history.dates, observed, history.forecast_dates) is None
