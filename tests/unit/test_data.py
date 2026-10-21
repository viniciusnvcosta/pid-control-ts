# ABOUTME: Unit tests for the data stage on the synthetic hierarchy CSV.
# ABOUTME: Covers node order, filtering, SHA-256 check, week grid, trimming, LOCF and truth handling.
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hcp.data import (
    FORECASTER,
    TERRITORIES,
    build_bundle,
    build_bundles,
    file_sha256,
    load_hierarchy,
    read_forecasts,
)
from tests.helpers import (
    ALT_FORECASTER,
    DUP_FORECASTER,
    GAP_LEAF,
    GAP_WEEK,
    LATE_WEEKS,
    N_WEEKS,
    SHORT_FORECASTER,
    SHORT_WEEKS,
    START,
    TRUTHLESS_US_WEEKS,
    synthetic_rows,
)

GAP_INDEX = GAP_WEEK - LATE_WEEKS


def test_nodes_are_root_then_sorted_leaves(hier_bundle) -> None:
    assert hier_bundle.nodes == ("us", "aa", "bb", "cc")


def test_grid_drops_truthless_us_weeks_and_trims_late_leaf(hier_bundle) -> None:
    n_weeks = N_WEEKS - TRUTHLESS_US_WEEKS - LATE_WEEKS
    assert hier_bundle.y.shape == (4, n_weeks)
    assert hier_bundle.q.shape == (4, n_weeks, 3)
    assert pd.Timestamp(hier_bundle.dates[0]) == START + pd.Timedelta(weeks=LATE_WEEKS)


def test_keeps_only_ensemble_at_ahead_4(hier_bundle) -> None:
    assert (hier_bundle.q > 0).all()  # decoy rows carry -1 and -2


def test_quantile_levels_are_on_last_axis(hier_bundle) -> None:
    q = hier_bundle.q
    np.testing.assert_allclose(q[..., 0], 0.4 * q[..., 1])
    np.testing.assert_allclose(q[..., 2], 1.6 * q[..., 1])


def test_forecast_dates_are_26_days_before_target(hier_bundle) -> None:
    lag = (hier_bundle.dates - hier_bundle.forecast_dates).astype("timedelta64[D]")
    assert (lag.astype(int) == 26).all()


def test_missing_leaf_week_is_carried_forward_and_flagged(hier_bundle) -> None:
    row = hier_bundle.nodes.index(GAP_LEAF)
    np.testing.assert_array_equal(
        hier_bundle.q[row, GAP_INDEX], hier_bundle.q[row, GAP_INDEX - 1]
    )
    assert hier_bundle.imputed[row, GAP_INDEX]
    assert hier_bundle.imputed.sum() == 1


def test_truth_is_never_imputed(hier_bundle) -> None:
    row = hier_bundle.nodes.index(GAP_LEAF)
    assert np.isnan(hier_bundle.y[row, GAP_INDEX])
    assert np.isfinite(np.delete(hier_bundle.y, GAP_INDEX, axis=1)).all()


def test_us_truth_is_sum_of_leaves(hier_bundle) -> None:
    y = np.delete(hier_bundle.y, GAP_INDEX, axis=1)
    np.testing.assert_allclose(y[0], y[1:].sum(axis=0))


def test_arrays_are_read_only(hier_bundle) -> None:
    for name in ("dates", "forecast_dates", "y", "q", "imputed"):
        assert not getattr(hier_bundle, name).flags.writeable, name


def test_sha256_mismatch_raises(hier_csv: Path) -> None:
    with pytest.raises(ValueError, match="sha256"):
        load_hierarchy(hier_csv, expected_sha256="0" * 64)


def test_sha256_match_loads(hier_csv: Path) -> None:
    bundle = load_hierarchy(hier_csv, expected_sha256=file_sha256(hier_csv))
    assert bundle.nodes[0] == "us"


def test_missing_root_raises(tmp_path: Path) -> None:
    path = tmp_path / "leaves.csv"
    rows = [row for row in synthetic_rows() if row["geo_value"] != "us"]
    pd.DataFrame(rows).to_csv(path)
    with pytest.raises(ValueError, match="root"):
        load_hierarchy(path)


def test_duplicate_rows_raise(tmp_path: Path) -> None:
    path = tmp_path / "dup.csv"
    rows = synthetic_rows()
    pd.DataFrame(rows + rows[:1]).to_csv(path)
    with pytest.raises(ValueError, match="duplicate"):
        load_hierarchy(path)


def _assert_same_bundle(a, b) -> None:
    assert a.nodes == b.nodes
    for name in ("dates", "forecast_dates", "y", "q", "imputed"):
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name))


def test_load_hierarchy_equals_build_bundle_of_read_forecasts(hier_csv: Path) -> None:
    _assert_same_bundle(
        load_hierarchy(hier_csv), build_bundle(read_forecasts(hier_csv))
    )


def test_read_forecasts_keeps_only_ahead_4(hier_csv: Path) -> None:
    assert set(read_forecasts(hier_csv)["ahead"]) == {4}


def test_read_forecasts_checks_sha256(hier_csv: Path) -> None:
    with pytest.raises(ValueError, match="sha256"):
        read_forecasts(hier_csv, "0" * 64)


def test_build_bundle_selects_forecaster(panel_frame) -> None:
    base = build_bundle(panel_frame)
    alt = build_bundle(panel_frame, ALT_FORECASTER)
    np.testing.assert_array_equal(alt.dates, base.dates)
    np.testing.assert_allclose(alt.q, 1.1 * base.q)
    np.testing.assert_array_equal(alt.y, base.y)


def test_build_bundle_unknown_forecaster_raises(panel_frame) -> None:
    with pytest.raises(ValueError, match="no rows for forecaster 'nope'"):
        build_bundle(panel_frame, "nope")


def test_build_bundles_maps_every_forecaster_and_none_on_failure(panel_frame) -> None:
    bundles = build_bundles(panel_frame)
    assert list(bundles) == sorted(panel_frame["forecaster"].unique())
    assert (
        bundles[SHORT_FORECASTER] is None
        or len(bundles[SHORT_FORECASTER].dates) <= SHORT_WEEKS
    )
    _assert_same_bundle(bundles[DUP_FORECASTER], bundles[FORECASTER])


def test_territories_constant() -> None:
    assert TERRITORIES == ("as", "gu", "mp", "pr", "vi")
