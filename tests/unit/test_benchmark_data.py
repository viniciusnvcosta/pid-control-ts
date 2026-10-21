# ABOUTME: Unit tests for canonical benchmark truth and aligned COVIDhub reference inputs.
# ABOUTME: Covers duplicate truth, weekly grids, immutability, forecast LOCF, and alignment errors.
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from hcp.benchmark_data import ActualPanel, canonical_actuals, reference_bundle
from hcp.data import FORECASTER, build_bundle
from tests.helpers import benchmark_rows


def _frame() -> pd.DataFrame:
    return pd.DataFrame(benchmark_rows())


def test_canonical_actuals_collapses_identical_duplicates_root_first() -> None:
    panel = canonical_actuals(_frame())

    assert panel.nodes == ("us", "aa")
    np.testing.assert_array_equal(
        panel.dates, pd.date_range("2022-01-01", periods=3, freq="7D").to_numpy()
    )
    np.testing.assert_array_equal(panel.y[0], [30.0, 33.0, 36.0])
    np.testing.assert_array_equal(panel.y[1], [10.0, 11.0, 12.0])
    assert panel.observed.all()


def test_canonical_actuals_marks_unobserved_only_when_all_duplicates_nan() -> None:
    frame = _frame()
    mask = (frame["geo_value"] == "aa") & (
        frame["target_end_date"] == pd.Timestamp("2022-01-15")
    )
    frame.loc[mask, "actual"] = np.nan

    panel = canonical_actuals(frame)

    assert np.isnan(panel.y[1, 2])
    assert not panel.observed[1, 2]
    assert panel.observed[:, :2].all()


def test_canonical_actuals_returns_read_only_arrays() -> None:
    panel = canonical_actuals(_frame())

    for array in (panel.dates, panel.y, panel.observed):
        assert not array.flags.writeable
        with pytest.raises(ValueError):
            array.flat[0] = array.flat[0]


def test_canonical_actuals_rejects_any_finite_disagreement() -> None:
    frame = _frame()
    key = (frame["geo_value"] == "aa") & (
        frame["target_end_date"] == pd.Timestamp("2022-01-01")
    )
    frame.loc[key, "actual"] = [10.0, 12.0]
    nan_row = frame.loc[key].iloc[[0]].copy()
    nan_row["actual"] = np.nan
    frame = pd.concat([frame, nan_row], ignore_index=True)

    with pytest.raises(ValueError, match=r"aa.*2022-01-01"):
        canonical_actuals(frame)


def test_canonical_actuals_rejects_nonweekly_grid() -> None:
    frame = _frame()
    frame = frame[frame["target_end_date"] != pd.Timestamp("2022-01-08")]

    with pytest.raises(ValueError, match="seven-day|weekly"):
        canonical_actuals(frame)


def test_reference_bundle_uses_submitted_quantiles_and_canonical_truth() -> None:
    frame = _frame()
    actuals = canonical_actuals(frame)
    submitted = build_bundle(frame, FORECASTER)

    bundle = reference_bundle(frame, actuals, FORECASTER)

    np.testing.assert_array_equal(bundle.q, submitted.q)
    np.testing.assert_array_equal(bundle.imputed, submitted.imputed)
    np.testing.assert_array_equal(bundle.y, actuals.y)
    np.testing.assert_array_equal(bundle.dates, submitted.dates)
    np.testing.assert_array_equal(bundle.forecast_dates, submitted.forecast_dates)
    assert bundle.imputed[1, 1]
    assert bundle.y[1, 1] == 11.0
    for array in (
        bundle.dates,
        bundle.forecast_dates,
        bundle.y,
        bundle.q,
        bundle.imputed,
    ):
        assert not array.flags.writeable


def test_reference_bundle_rejects_mismatched_nodes() -> None:
    frame = _frame()
    actuals = canonical_actuals(frame)
    mismatched = replace(actuals, nodes=("us", "bb"))

    with pytest.raises(ValueError, match="nodes"):
        reference_bundle(frame, mismatched, FORECASTER)


def test_reference_bundle_rejects_unavailable_date() -> None:
    frame = _frame()
    actuals = canonical_actuals(frame)
    shortened = ActualPanel(
        dates=actuals.dates[:-1],
        nodes=actuals.nodes,
        y=actuals.y[:, :-1],
        observed=actuals.observed[:, :-1],
    )

    with pytest.raises(ValueError, match="date"):
        reference_bundle(frame, shortened, FORECASTER)
