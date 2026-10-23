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

    def adapter(histories, settings, seed):
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
