# ABOUTME: Behavioral contracts for seeded local Prophet and classical native adapters.
# ABOUTME: Uses boundary fakes to expose settings, weekly chronology, failures, and RNG preservation.
"""Local adapter contracts without invoking slow statistical backends."""

from importlib import import_module

import numpy as np
import pandas as pd
import pytest

from hcp.forecast_types import PreparedHistory, V2ModelSettings

pytestmark = [
    pytest.mark.filterwarnings(
        "ignore:Maximum Likelihood optimization failed to converge"
    ),
    pytest.mark.filterwarnings(
        "ignore:Non-stationary starting seasonal autoregressive"
    ),
    pytest.mark.filterwarnings(
        "ignore:Non-stationary starting autoregressive parameters"
    ),
]


def _histories() -> tuple[PreparedHistory, ...]:
    dates = pd.date_range("2020-01-04", periods=64, freq="7D").to_numpy()
    return tuple(
        PreparedHistory(node, dates, np.linspace(-20.0, -10.0, 64) * scale)
        for node, scale in (("bb", 2), ("aa", 1))
    )


def _fake_prophet(monkeypatch, *, fail: str | None = None):
    import darts.models

    calls = []

    class Backend:
        def fit(self, *, seed):
            return seed

    class FakeProphet:
        def __init__(self, **kwargs):
            self._model_builder = lambda **kwargs: Backend()
            self.kwargs = kwargs
            calls.append(self)

        def fit(self, series):
            self.backend_seed = self._model_builder().fit()
            self.series = series
            self.fit_draw = np.random.randint(0, 100000)
            if fail == "fit":
                raise ArithmeticError("optimizer failed")
            return self

        def predict(self, ahead, *, num_samples, random_state):
            from darts import TimeSeries

            self.prediction = (ahead, num_samples, random_state)
            if fail == "predict":
                raise ArithmeticError("sampler failed")
            values = np.full((ahead, 1, num_samples), self.series.last_value())
            values[:, 0, 0] -= 5
            dates = pd.date_range(self.series.end_time(), periods=ahead + 1, freq="7D")[
                1:
            ]
            return TimeSeries.from_times_and_values(dates, values)

    monkeypatch.setattr(darts.models, "Prophet", FakeProphet)
    return calls


def test_prophet_uses_registered_settings_node_seeds_and_signed_native_paths(
    monkeypatch,
):
    # Wrong defaults, node streams, time frequency or clipping must change this output.
    calls = _fake_prophet(monkeypatch)
    adapter = import_module("hcp.forecast_classical")
    result = adapter.prophet_prediction(
        _histories(), V2ModelSettings(), np.random.SeedSequence(127)
    )

    assert result.nodes == ("bb", "aa")
    assert result.kind == "samples"
    assert result.ahead == 4
    assert result.samples.shape == (2, 4, 500)
    assert not result.samples.flags.writeable
    np.testing.assert_array_equal(result.samples[:, :, 0], [[-25] * 4, [-15] * 4])
    np.testing.assert_array_equal(
        result.samples[:, :, 1:],
        np.broadcast_to(np.array([-20, -10])[:, None, None], (2, 4, 499)),
    )
    seeds = [2567351042, 3478827648]
    for call, history, seed in zip(calls, _histories(), seeds, strict=True):
        assert call.kwargs == {
            "growth": "linear",
            "yearly_seasonality": False,
            "weekly_seasonality": False,
            "daily_seasonality": False,
            "n_changepoints": 25,
            "changepoint_range": 0.8,
            "changepoint_prior_scale": 0.05,
            "mcmc_samples": 0,
            "random_state": seed,
        }
        np.testing.assert_array_equal(call.series.time_index.to_numpy(), history.dates)
        np.testing.assert_array_equal(call.series.univariate_values(), history.values)
        assert call.prediction == (4, 500, seed)
        assert call.backend_seed == seed
    assert result.diagnostics["node_seeds"] == tuple(seeds)


def test_prophet_seeds_fit_and_restores_callers_numpy_state(monkeypatch):
    calls = _fake_prophet(monkeypatch)
    adapter = import_module("hcp.forecast_classical")
    before = np.random.get_state()
    first = adapter.prophet_prediction(
        _histories(), V2ModelSettings(), np.random.SeedSequence(127)
    )
    after = np.random.get_state()
    second = adapter.prophet_prediction(
        _histories(), V2ModelSettings(), np.random.SeedSequence(127)
    )
    np.testing.assert_array_equal(first.samples, second.samples)
    assert [c.fit_draw for c in calls[:2]] == [c.fit_draw for c in calls[2:]]
    assert calls[0].fit_draw != calls[1].fit_draw
    assert before[0] == after[0]
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]


@pytest.mark.parametrize("stage", ["fit", "predict"])
def test_prophet_failure_retains_node_origin_and_cause(monkeypatch, stage):
    _fake_prophet(monkeypatch, fail=stage)
    adapter = import_module("hcp.forecast_classical")
    before = np.random.get_state()
    with pytest.raises(RuntimeError, match=r"prophet.*bb.*2021-03-20") as exc:
        adapter.prophet_prediction(
            _histories(), V2ModelSettings(), np.random.SeedSequence(127)
        )
    assert isinstance(exc.value.__cause__, ArithmeticError)
    after = np.random.get_state()
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]


@pytest.mark.parametrize("name", ["arima", "sarima"])
def test_classical_native_wrapper_preserves_legacy_stream_and_paths(name):
    adapter = import_module("hcp.forecast_classical")
    legacy = import_module("hcp.forecast_models")
    settings = V2ModelSettings(n_samples=7)
    original = getattr(legacy, f"{name}_samples")(
        _histories(), settings, np.random.SeedSequence(127)
    )
    prediction = getattr(adapter, f"{name}_prediction")(
        _histories(), settings, np.random.SeedSequence(127)
    )
    np.testing.assert_array_equal(prediction.samples, original)
    assert prediction.nodes == ("bb", "aa")
    assert prediction.kind == "samples"


def test_prophet_refuses_backend_without_seedable_fit_boundary(monkeypatch):
    import darts.models

    monkeypatch.setattr(darts.models, "Prophet", lambda **kwargs: object())
    adapter = import_module("hcp.forecast_classical")
    with pytest.raises(RuntimeError) as exc:
        adapter.prophet_prediction(
            _histories(), V2ModelSettings(), np.random.SeedSequence(127)
        )
    assert "seedable" in str(exc.value.__cause__)
