# ABOUTME: Characterizes the upstream core model-score forecast harness without changing it.
# ABOUTME: Pins the AR(3) constructor, historical-forecast arguments, tail alignment, and cache reuse.
import importlib
import sys
from types import ModuleType

import numpy as np
import pandas as pd


def test_generate_forecasts_ar_contract_and_cache_reuse(tmp_path, monkeypatch) -> None:
    prophet_module = ModuleType("darts.models.forecasting.prophet_model")
    prophet_module.Prophet = object
    monkeypatch.setitem(
        sys.modules, "darts.models.forecasting.prophet_model", prophet_module
    )
    transformer_module = ModuleType("darts.models.forecasting.transformer_model")
    transformer_module.TransformerModel = object
    monkeypatch.setitem(
        sys.modules, "darts.models.forecasting.transformer_model", transformer_module
    )
    monkeypatch.delitem(sys.modules, "core.model_scores", raising=False)
    model_scores = importlib.import_module("core.model_scores")
    calls: dict[str, object] = {}

    class FakeForecast:
        def values(self) -> np.ndarray:
            return np.array([[11.0], [12.0], [13.0]])

    class FakeModel:
        def historical_forecasts(self, series, **kwargs):
            calls["series"] = series
            calls["historical"] = kwargs
            return FakeForecast()

    def fake_arima(*, p: int, d: int, q: int) -> FakeModel:
        calls["order"] = (p, d, q)
        return FakeModel()

    class FakeTimeSeries:
        @staticmethod
        def from_dataframe(frame, *, time_col: str, value_cols: str):
            calls["frame"] = frame.copy()
            calls["columns"] = (time_col, value_cols)
            return "series"

    monkeypatch.setattr(model_scores, "ARIMA", fake_arima)
    monkeypatch.setattr(model_scores, "TimeSeries", FakeTimeSeries)
    data = pd.DataFrame(
        {"y": [1.0, np.nan, 3.0, 4.0, 5.0]},
        index=pd.date_range("2020-01-01", periods=5, freq="7D"),
    )
    cache = tmp_path / "ar-cache.npz"

    result = model_scores.generate_forecasts(data, "ar", cache, True, False, 2, 4)

    assert calls["order"] == (3, 0, 0)
    assert calls["historical"] == {
        "forecast_horizon": 2,
        "retrain": 2,
        "verbose": True,
    }
    np.testing.assert_array_equal(result, [0.0, 0.0, 11.0, 12.0, 13.0])
    assert calls["columns"] == ("index", "y")

    monkeypatch.setattr(
        model_scores,
        "ARIMA",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("cache not reused")),
    )
    cached = model_scores.generate_forecasts(data, "ar", cache, False, False, 2, 4)
    np.testing.assert_array_equal(cached, result)
