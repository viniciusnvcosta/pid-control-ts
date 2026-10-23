# ABOUTME: Unit tests for causal rolling-origin benchmark forecast generation.
# ABOUTME: Covers history preparation, seed/model ordering, quantiles, hard failures, and immutable outputs.
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from hcp.benchmark_data import ActualPanel
from hcp.forecast import (
    FORECASTERS,
    ForecastConfig,
    generate_model_bundles,
)
from hcp.forecast_models import ForecastSettings


def _panel(n_weeks: int = 70, *, gaps: bool = False) -> ActualPanel:
    dates = pd.date_range("2020-01-04", periods=n_weeks, freq="7D").to_numpy()
    index = np.arange(n_weeks, dtype=float)
    y = np.vstack([100.0 + index, 10.0 + index])
    if gaps:
        y[1, 0] = np.nan
        y[1, 10] = np.nan
    observed = np.isfinite(y)
    return ActualPanel(dates=dates, nodes=("us", "aa"), y=y, observed=observed)


def _config(*models: str, seed: int = 17) -> ForecastConfig:
    return ForecastConfig(
        models=models or ("literal",),
        root_seed=seed,
        settings=replace(ForecastSettings(), n_samples=7),
    )


def _grid(panel: ActualPanel, start: int, stop: int) -> tuple[np.ndarray, np.ndarray]:
    dates = panel.dates[start:stop]
    return dates, dates - np.timedelta64(26, "D")


def _literal_adapter(histories, settings, seed):
    samples = np.empty((len(histories), settings.ahead, settings.n_samples))
    for node, history in enumerate(histories):
        for horizon in range(settings.ahead):
            samples[node, horizon] = (
                history.values[-1] + 100.0 * horizon + np.arange(settings.n_samples)
            )
    return samples


def test_generation_is_causal_interpolates_prefix_and_uses_horizon_four(
    monkeypatch,
) -> None:
    panel = _panel(gaps=True)
    dates, forecast_dates = _grid(panel, 64, 65)
    seen = []

    def recording_adapter(histories, settings, seed):
        seen.append(histories)
        return _literal_adapter(histories, settings, seed)

    monkeypatch.setitem(FORECASTERS, "literal", recording_adapter)

    result = generate_model_bundles(panel, dates, forecast_dates, _config())

    histories = seen[0]
    assert all(history.dates[-1] <= forecast_dates[0] for history in histories)
    assert all(
        history.dates[-1] == dates[0] - np.timedelta64(28, "D") for history in histories
    )
    assert histories[1].dates[0] == panel.dates[1]
    assert histories[1].values[9] == 20.0
    expected = histories[0].values[-1] + np.array([300.6, 303.0, 305.4])
    np.testing.assert_allclose(result.bundles["literal"].q[0, 0], expected)
    record = result.diagnostics["models"]["literal"]["interpolation"][1]
    assert record["node"] == "aa"
    assert record["raw_observations"] == 59
    assert record["interpolated_observations"] == 1


def test_changing_post_origin_actuals_leaves_forecast_identical(monkeypatch) -> None:
    panel = _panel()
    dates, forecast_dates = _grid(panel, 64, 65)
    monkeypatch.setitem(FORECASTERS, "literal", _literal_adapter)

    original = generate_model_bundles(panel, dates, forecast_dates, _config())
    changed_y = panel.y.copy()
    changed_y[:, 61:] += 10000.0
    changed = replace(panel, y=changed_y, observed=np.isfinite(changed_y))
    rerun = generate_model_bundles(changed, dates, forecast_dates, _config())

    np.testing.assert_array_equal(
        original.bundles["literal"].q, rerun.bundles["literal"].q
    )


def test_configuration_order_controls_models_and_origin_seeds(monkeypatch) -> None:
    panel = _panel()
    dates, forecast_dates = _grid(panel, 64, 66)
    calls = []

    def adapter(name):
        def run(histories, settings, seed):
            calls.append((name, int(seed.generate_state(1)[0])))
            return _literal_adapter(histories, settings, seed)

        return run

    monkeypatch.setattr(
        "hcp.forecast.FORECASTERS",
        {"alpha": adapter("alpha"), "beta": adapter("beta")},
    )
    cfg = _config("beta", "alpha")

    result = generate_model_bundles(panel, dates, forecast_dates, cfg)

    root = np.random.SeedSequence(cfg.root_seed, spawn_key=cfg.seed_spawn_key)
    expected = []
    for name, model_seed in zip(cfg.models, root.spawn(2), strict=True):
        expected.extend(
            (name, int(origin.generate_state(1)[0])) for origin in model_seed.spawn(2)
        )
    assert calls == expected
    assert tuple(result.bundles) == ("beta", "alpha")


def test_preeligibility_forecasts_are_nan_but_later_origins_are_finite(
    monkeypatch,
) -> None:
    panel = _panel()
    dates, forecast_dates = _grid(panel, 58, 65)
    monkeypatch.setitem(FORECASTERS, "literal", _literal_adapter)

    result = generate_model_bundles(panel, dates, forecast_dates, _config())

    q = result.bundles["literal"].q
    assert np.isnan(q[:, :5]).all()
    assert np.isfinite(q[:, 5:]).all()


def test_zero_eligible_origins_raises(monkeypatch) -> None:
    panel = _panel()
    dates, forecast_dates = _grid(panel, 50, 55)
    monkeypatch.setitem(FORECASTERS, "literal", _literal_adapter)

    with pytest.raises(ValueError, match="eligible origin"):
        generate_model_bundles(panel, dates, forecast_dates, _config())


def test_unknown_model_raises() -> None:
    panel = _panel()
    dates, forecast_dates = _grid(panel, 64, 65)
    with pytest.raises(ValueError, match="unknown model"):
        generate_model_bundles(panel, dates, forecast_dates, _config("unknown"))


def test_nonweekly_grid_raises(monkeypatch) -> None:
    panel = _panel()
    dates, forecast_dates = _grid(panel, 64, 66)
    dates = dates.copy()
    dates[1] += np.timedelta64(1, "D")
    monkeypatch.setitem(FORECASTERS, "literal", _literal_adapter)
    with pytest.raises(ValueError, match="weekly"):
        generate_model_bundles(panel, dates, forecast_dates, _config())


def test_non_horizon_four_origin_target_lag_raises(monkeypatch) -> None:
    panel = _panel()
    dates, forecast_dates = _grid(panel, 64, 65)
    forecast_dates = forecast_dates + np.timedelta64(7, "D")
    monkeypatch.setitem(FORECASTERS, "literal", _literal_adapter)

    with pytest.raises(ValueError, match="26-day|horizon-four"):
        generate_model_bundles(panel, dates, forecast_dates, _config())


@pytest.mark.parametrize("mismatch", ["shape", "date"])
def test_node_or_date_mismatch_raises(monkeypatch, mismatch: str) -> None:
    panel = _panel()
    dates, forecast_dates = _grid(panel, 64, 65)
    if mismatch == "shape":
        panel = replace(panel, y=panel.y[:1], observed=panel.observed[:1])
    else:
        dates = dates + np.timedelta64(700, "D")
        forecast_dates = forecast_dates + np.timedelta64(700, "D")
    monkeypatch.setitem(FORECASTERS, "literal", _literal_adapter)
    with pytest.raises(ValueError, match="node|date|shape"):
        generate_model_bundles(panel, dates, forecast_dates, _config())


def test_invalid_adapter_shape_raises(monkeypatch) -> None:
    panel = _panel()
    dates, forecast_dates = _grid(panel, 64, 65)
    monkeypatch.setitem(
        FORECASTERS, "literal", lambda histories, settings, seed: np.zeros((1, 4, 7))
    )
    with pytest.raises(ValueError, match="shape"):
        generate_model_bundles(panel, dates, forecast_dates, _config())


def test_invalid_quantile_order_raises() -> None:
    with pytest.raises(ValueError, match="quantile"):
        ForecastConfig(
            quantiles=(0.5, 0.1, 0.9),
            settings=replace(ForecastSettings(), n_samples=7),
        )


@pytest.mark.parametrize("failure", ["nonfinite", "exception"])
def test_nonfinite_or_failed_adapter_aborts_generation(
    monkeypatch, failure: str
) -> None:
    panel = _panel()
    dates, forecast_dates = _grid(panel, 64, 65)

    def bad_adapter(histories, settings, seed):
        if failure == "exception":
            raise RuntimeError("fit failed")
        return np.full((len(histories), settings.ahead, settings.n_samples), np.nan)

    monkeypatch.setitem(FORECASTERS, "literal", bad_adapter)
    with pytest.raises((RuntimeError, ValueError), match="fit failed|non-finite"):
        generate_model_bundles(panel, dates, forecast_dates, _config())


def test_outputs_are_read_only_with_exact_long_columns(monkeypatch) -> None:
    panel = _panel()
    dates, forecast_dates = _grid(panel, 64, 65)
    monkeypatch.setitem(FORECASTERS, "literal", _literal_adapter)

    result = generate_model_bundles(panel, dates, forecast_dates, _config())

    bundle = result.bundles["literal"]
    for array in (
        bundle.dates,
        bundle.forecast_dates,
        bundle.y,
        bundle.q,
        bundle.imputed,
    ):
        assert not array.flags.writeable
    assert list(result.base_forecasts) == [
        "model",
        "node",
        "week",
        "forecast_date",
        "q_lo",
        "q_med",
        "q_hi",
    ]


def test_legacy_results_gain_independent_schema_bearing_diagnostic_tables() -> None:
    from hcp.forecast import ForecastResult
    from hcp.forecast_types import DiagnosticsTables

    diagnostics = {"legacy": "preserved"}
    first = ForecastResult(
        bundles={}, base_forecasts=pd.DataFrame(), diagnostics=diagnostics
    )
    second = ForecastResult(bundles={}, base_forecasts=pd.DataFrame(), diagnostics={})
    assert first.diagnostics is diagnostics
    assert isinstance(first.tables, DiagnosticsTables)
    for name in ("forecast_diagnostics", "tuning", "quantile_repairs", "warnings"):
        table = getattr(first.tables, name)
        assert table.empty
        assert len(table.columns) >= 3
        table.loc[0, table.columns[0]] = "arima"
        assert getattr(second.tables, name).empty
    source = pd.DataFrame(
        {"model": ["qr_l1"], "node": ["us"], "forecast_date": ["2020-01-04"]}
    )
    tables = DiagnosticsTables(tuning=source)
    source.loc[0, "model"] = "changed"
    assert tables.tuning.loc[0, "model"] == "qr_l1"


def test_ineligible_origins_reserve_seed_slots(monkeypatch) -> None:
    from hcp.forecast_validation import neural_seed_streams

    panel = _panel()
    dates, forecast_dates = _grid(panel, 62, 66)

    def adapter(histories, settings, seed):
        streams = neural_seed_streams(seed, 1)
        value = np.random.default_rng(streams["prediction"][0]).normal()
        return np.full((len(histories), settings.ahead, settings.n_samples), value)

    monkeypatch.setitem(FORECASTERS, "literal", adapter)
    original = generate_model_bundles(panel, dates, forecast_dates, _config())
    raw = panel.y.copy()
    raw[:, 0] = np.nan  # Origin 63 becomes ineligible; origin 64 retains its stream.
    raw[:, 63:] += 100000  # Future values are invisible to all these fits.
    changed = replace(panel, y=raw, observed=np.isfinite(raw))
    rerun = generate_model_bundles(changed, dates, forecast_dates, _config())
    assert np.isnan(original.bundles["literal"].q[:, 0]).all()
    assert np.isfinite(original.bundles["literal"].q[:, 1]).all()
    assert np.isnan(rerun.bundles["literal"].q[:, :2]).all()
    np.testing.assert_array_equal(
        original.bundles["literal"].q[:, 2:], rerun.bundles["literal"].q[:, 2:]
    )


def test_outer_history_preserves_raw_metadata_before_interpolation(monkeypatch) -> None:
    panel = _panel(gaps=True)
    dates, forecast_dates = _grid(panel, 64, 65)

    def adapter(histories, settings, seed):
        history = histories[1]
        assert history.raw_values is not None
        assert np.isnan(history.raw_values[9])
        assert history.values[9] == 20.0
        assert not history.observed[9]
        np.testing.assert_array_equal(
            history.forecast_dates, history.dates - np.timedelta64(26, "D")
        )
        return _literal_adapter(histories, settings, seed)

    monkeypatch.setitem(FORECASTERS, "literal", adapter)
    generate_model_bundles(panel, dates, forecast_dates, _config())
