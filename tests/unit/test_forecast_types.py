# ABOUTME: Tests immutable native forecast payloads and legacy history compatibility.
# ABOUTME: Guards node/horizon validation, defensive storage, and fourth-step extraction.

import numpy as np
import pytest


def test_native_quantiles_own_their_storage() -> None:
    from hcp.forecast_types import NativePrediction

    source = np.array([[1.0, 2.0, 3.0]])
    prediction = NativePrediction(
        nodes=("us",), ahead=4, kind="quantiles", quantiles=source
    )
    source[0, 0] = 99.0
    np.testing.assert_array_equal(prediction.quantiles, [[1.0, 2.0, 3.0]])
    assert not prediction.quantiles.flags.writeable
    with pytest.raises(ValueError):
        prediction.quantiles[0, 0] = 0.0


@pytest.mark.parametrize(
    "overrides",
    [
        {"kind": "point"},
        {"ahead": 1},
        {"ahead": 4.0},
        {"nodes": ()},
        {"nodes": ("us", "us"), "quantiles": np.zeros((2, 3))},
        {"nodes": ("us", "aa")},
        {"quantiles": None},
        {"samples": np.zeros((1, 4, 5))},
        {"levels": (0.2, 0.5, 0.8)},
        {"levels": (0.9, 0.5, 0.1)},
        {"quantiles": np.zeros((1, 1, 3))},
        {"quantiles": [[1, np.nan, 3]]},
        {"quantiles": [[1, 2, np.inf]]},
    ],
)
def test_rejects_invalid_native_quantile_contract(overrides: dict) -> None:
    from hcp.forecast_types import NativePrediction

    arguments = {
        "nodes": ("us",),
        "ahead": 4,
        "kind": "quantiles",
        "quantiles": [[1, 2, 3]],
    }
    arguments.update(overrides)
    with pytest.raises(ValueError):
        NativePrediction(**arguments)


@pytest.mark.parametrize(
    "samples",
    [
        np.zeros((1, 3, 5)),
        np.zeros((2, 4, 5)),
        np.zeros((1, 4, 0)),
        np.full((1, 4, 5), np.nan),
        np.zeros((1, 4, 1, 5)),
    ],
)
def test_rejects_invalid_native_sample_contract(samples: np.ndarray) -> None:
    from hcp.forecast_types import NativePrediction

    with pytest.raises(ValueError):
        NativePrediction(nodes=("us",), ahead=4, kind="samples", samples=samples)


def test_sample_payload_owns_storage_and_extracts_fourth_step() -> None:
    from hcp.forecast_types import NativePrediction, prediction_quantiles

    source = np.array(
        [[[100, 100, 100], [200, 200, 200], [300, 300, 300], [-10, 0, 10]]], dtype=float
    )
    prediction = NativePrediction(
        nodes=("us",), ahead=4, kind="samples", samples=source
    )
    source[:] = 999
    assert not prediction.samples.flags.writeable
    result = prediction_quantiles(prediction)
    np.testing.assert_allclose(result, [[-8, 0, 8]])
    assert not result.flags.writeable


def test_direct_quantile_extraction_preserves_values() -> None:
    from hcp.forecast_types import NativePrediction, prediction_quantiles

    prediction = NativePrediction(
        nodes=("us", "aa"),
        ahead=4,
        kind="quantiles",
        quantiles=[[-3, 1, 8], [2, 4, 6]],
    )
    np.testing.assert_array_equal(
        prediction_quantiles(prediction), [[-3, 1, 8], [2, 4, 6]]
    )


def test_native_diagnostics_cannot_alias_mutable_input() -> None:
    from hcp.forecast_types import NativePrediction

    values = np.array([1.0])
    diagnostics = {"scale": values, "nested": {"counts": [2]}}
    prediction = NativePrediction(
        nodes=["us"],
        ahead=4,
        kind="quantiles",
        quantiles=[[1, 2, 3]],
        levels=[0.1, 0.5, 0.9],
        diagnostics=diagnostics,
    )
    values[0] = 10
    diagnostics["nested"]["counts"][0] = 20
    assert prediction.diagnostics["scale"][0] == 1.0
    assert prediction.diagnostics["nested"]["counts"] == (2,)
    assert not prediction.diagnostics["scale"].flags.writeable
    with pytest.raises(TypeError):
        prediction.diagnostics["new"] = 1


def test_history_keeps_legacy_construction_and_protects_optional_raw_metadata() -> None:
    from dataclasses import FrozenInstanceError

    from hcp.forecast import ForecastSettings as OrchestrationSettings
    from hcp.forecast import PreparedHistory as OrchestrationHistory
    from hcp.forecast_models import ForecastSettings as LegacySettings
    from hcp.forecast_models import PreparedHistory as LegacyHistory
    from hcp.forecast_types import ForecastSettings, PreparedHistory

    assert OrchestrationHistory is LegacyHistory is PreparedHistory
    assert OrchestrationSettings is LegacySettings is ForecastSettings
    dates = np.array(["2020-01-04", "2020-01-11"], dtype="datetime64[ns]")
    values = np.array([1.0, 2.0])
    raw_values = np.array([1.0, np.nan])
    observed = np.array([True, False])
    forecast_dates = dates - np.timedelta64(26, "D")
    history = PreparedHistory("us", dates, values)
    assert history.raw_values is history.observed is history.forecast_dates is None
    enriched = PreparedHistory(
        "us",
        dates,
        values,
        raw_values=raw_values,
        observed=observed,
        forecast_dates=forecast_dates,
    )
    dates[:] = np.datetime64("2030-01-01")
    values[:] = 99
    raw_values[:] = 99
    observed[:] = True
    forecast_dates[:] = np.datetime64("2030-01-01")
    np.testing.assert_array_equal(history.values, [1.0, 2.0])
    np.testing.assert_array_equal(enriched.raw_values, [1.0, np.nan])
    np.testing.assert_array_equal(enriched.observed, [True, False])
    assert enriched.dates[0] == np.datetime64("2020-01-04")
    assert enriched.forecast_dates[0] == np.datetime64("2019-12-09")
    for name in ("dates", "values", "raw_values", "observed", "forecast_dates"):
        assert not getattr(enriched, name).flags.writeable
    with pytest.raises(FrozenInstanceError):
        history.node = "aa"


def test_v2_settings_can_drive_legacy_adapter_without_mutable_settings() -> None:
    from dataclasses import FrozenInstanceError

    from hcp.forecast_models import arima_samples
    from hcp.forecast_types import PreparedHistory, V2ModelSettings

    settings = V2ModelSettings(n_samples=7)
    dates = np.arange("2020-01-04", "2021-03-27", 7, dtype="datetime64[D]")
    values = 20 + np.random.default_rng(np.random.SeedSequence(1)).normal(
        size=len(dates)
    )
    result = arima_samples(
        (PreparedHistory("us", dates, values),), settings, np.random.SeedSequence(11)
    )
    assert result.shape == (1, 4, 7)
    assert np.isfinite(result).all()
    with pytest.raises(FrozenInstanceError):
        settings.neural_max_epochs = 2
