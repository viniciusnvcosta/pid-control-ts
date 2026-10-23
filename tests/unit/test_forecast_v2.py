# ABOUTME: Exercises v2 native adapter dispatch and diagnostics without model fitting.
# ABOUTME: Protects the unchanged first three forecast seed streams and warning evidence.
import warnings

import numpy as np

from hcp.forecast import FORECASTERS, ForecastConfig, generate_model_bundles
from hcp.forecast_types import ForecastSettings, NativePrediction, V2ModelSettings
from tests.unit.test_forecast import _grid, _panel


def test_v2_adapter_retains_native_quantiles_and_warning_rows(monkeypatch) -> None:
    panel = _panel()
    dates, forecast_dates = _grid(panel, 64, 65)
    seen = {1: [], 2: []}

    def adapter(histories, settings, seed, execution):
        version = 2 if isinstance(settings, V2ModelSettings) else 1
        seen[version].append(seed.generate_state(4).tolist())
        for _ in range(2):
            warnings.warn("sample warning", UserWarning)
        if version == 2:
            return NativePrediction(
                tuple(h.node for h in histories),
                4,
                "quantiles",
                quantiles=np.ones((len(histories), 3)),
            )
        return np.ones((len(histories), 4, settings.n_samples))

    models = (
        "arima",
        "sarima",
        "deepar",
        "prophet",
        "transformer",
        "qr_l1",
        "deepar_enhanced",
    )
    for model in models:
        monkeypatch.setitem(FORECASTERS, model, adapter)
    settings = V2ModelSettings()
    cfg = ForecastConfig(
        models=models,
        root_seed=3,
        protocol_version=2,
        settings=settings,
        v2_settings=settings,
    )
    result = generate_model_bundles(panel, dates, forecast_dates, cfg)
    v2_streams = seen[2][:3]
    generate_model_bundles(
        panel,
        dates,
        forecast_dates,
        ForecastConfig(models=models[:3], root_seed=3, settings=ForecastSettings()),
    )
    assert seen[1] == v2_streams
    assert len(result.tables.warnings) == 7
    assert (result.tables.warnings["count"] == 2).all()
    assert {
        "category",
        "message",
        "source",
        "model",
        "node",
        "origin",
        "candidate",
        "phase",
        "count",
    } <= set(result.tables.warnings.columns)
    assert result.tables.forecast_diagnostics.empty
    assert set(result.base_forecasts.query("model == 'qr_l1'")["source_kind"]) == {
        "native_quantiles"
    }
    assert set(result.base_forecasts.query("model == 'arima'")["source_kind"]) == {
        "sample_derived"
    }


def test_v2_ineligible_origin_reserves_seed_before_v1_matched_origin(monkeypatch):
    panel = _panel()
    y = panel.y.copy()
    y[:, :2] = np.nan
    panel = panel.__class__(
        dates=panel.dates, nodes=panel.nodes, y=y, observed=np.isfinite(y)
    )
    dates, forecast_dates = _grid(panel, 64, 66)
    models = (
        "arima",
        "sarima",
        "deepar",
        "prophet",
        "transformer",
        "qr_l1",
        "deepar_enhanced",
    )
    calls = {1: [], 2: []}

    def adapter_for(model):
        def adapter(histories, settings, seed, execution):
            version = 2 if isinstance(settings, V2ModelSettings) else 1
            calls[version].append((model, seed.generate_state(4).tolist()))
            samples = np.zeros((len(histories), 4, settings.n_samples))
            if version == 2:
                return NativePrediction(
                    tuple(history.node for history in histories),
                    4,
                    "samples",
                    samples=samples,
                )
            return samples

        return adapter

    for model in models:
        monkeypatch.setitem(FORECASTERS, model, adapter_for(model))
    settings = V2ModelSettings()
    v2 = ForecastConfig(
        models=models,
        root_seed=3,
        protocol_version=2,
        settings=settings,
        v2_settings=settings,
    )
    v2_result = generate_model_bundles(panel, dates, forecast_dates, v2)
    generate_model_bundles(
        panel,
        dates,
        forecast_dates,
        ForecastConfig(models=models[:3], root_seed=3, settings=ForecastSettings()),
    )

    assert np.isnan(v2_result.bundles["arima"].q[:, 0]).all()
    assert np.isfinite(v2_result.bundles["arima"].q[:, 1]).all()
    assert [model for model, _ in calls[2]] == list(models)
    assert calls[1] == calls[2][:3]


def test_registry_classical_ignores_cuda_execution(monkeypatch):
    from hcp import forecast
    from hcp.benchmark_batch import NeuralExecution

    execution = NeuralExecution(
        "cuda:0", "NVIDIA GeForce RTX 4080", ":4096:8", False, "32-true"
    )
    for name, target in (
        ("arima", "arima_samples"),
        ("sarima", "sarima_samples"),
        ("prophet", "prophet_prediction"),
        ("qr_l1", "qr_prediction"),
    ):
        seen = []

        def classical(histories, settings, seed, seen=seen):
            seen.append(seed.spawn_key)
            return "cpu result"

        monkeypatch.setattr(forecast, target, classical)
        assert (
            forecast.FORECASTERS[name](
                (), ForecastSettings(), np.random.SeedSequence(1), execution
            )
            == "cpu result"
        )
        assert seen == [()]


def test_registered_execution_reaches_every_neural_adapter(monkeypatch):
    from hcp.benchmark_batch import NeuralExecution

    execution = NeuralExecution(
        "cuda:0", "NVIDIA GeForce RTX 4080", ":4096:8", False, "32-true"
    )
    panel = _panel()
    dates, origins = _grid(panel, 64, 65)
    seen = []

    def neural(histories, settings, seed, received):
        seen.append(received)
        return np.ones((len(histories), 4, settings.n_samples))

    models = (
        "arima",
        "sarima",
        "deepar",
        "prophet",
        "transformer",
        "qr_l1",
        "deepar_enhanced",
    )
    for name in models:
        monkeypatch.setitem(FORECASTERS, name, neural)
    generate_model_bundles(
        panel,
        dates,
        origins,
        ForecastConfig(
            models=models,
            settings=(settings := V2ModelSettings()),
            v2_settings=settings,
            protocol_version=2,
            execution=execution,
        ),
    )
    assert seen == [execution] * 7
