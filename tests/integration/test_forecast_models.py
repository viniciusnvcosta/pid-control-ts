# ABOUTME: Integration tests for real Darts probabilistic benchmark model adapters.
# ABOUTME: Checks shape, finiteness, seed determinism, stochasticity, and horizon-four sensitivity.
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from hcp import forecast_neural
from hcp.forecast_models import (
    ForecastSettings,
    PreparedHistory,
    arima_samples,
    deepar_samples,
    sarima_samples,
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.filterwarnings(
        "ignore:Maximum Likelihood optimization failed to converge"
    ),
    pytest.mark.filterwarnings(
        "ignore:Non-stationary starting seasonal autoregressive"
    ),
    pytest.mark.filterwarnings("ignore:Can't initialize NVML"),
    pytest.mark.filterwarnings(
        "ignore:`isinstance\\(treespec, LeafSpec\\)` is deprecated"
    ),
    pytest.mark.filterwarnings("ignore:Total length of `list` across ranks is zero"),
    pytest.mark.filterwarnings("ignore:'pin_memory' argument is set as true"),
]


def _history(last_shift: float = 0.0) -> PreparedHistory:
    dates = pd.date_range("2020-01-04", periods=64, freq="7D").to_numpy()
    index = np.arange(64, dtype=float)
    values = 40.0 + 0.15 * index + 5.0 * np.sin(2.0 * np.pi * index / 13.0)
    values[-4:] += last_shift
    return PreparedHistory(node="aa", dates=dates, values=values)


@pytest.mark.parametrize("adapter", [arima_samples, sarima_samples])
def test_classical_adapter_is_seeded_probabilistic_and_uses_recent_history(
    adapter,
) -> None:
    settings = replace(ForecastSettings(), n_samples=7)

    first = adapter((_history(),), settings, np.random.SeedSequence(123))
    repeated = adapter((_history(),), settings, np.random.SeedSequence(123))
    changed_seed = adapter((_history(),), settings, np.random.SeedSequence(124))
    changed_tail = adapter(
        (_history(last_shift=20.0),), settings, np.random.SeedSequence(123)
    )

    assert first.shape == (1, 4, 7)
    assert np.isfinite(first).all()
    np.testing.assert_array_equal(first, repeated)
    assert not np.array_equal(first, changed_seed)
    assert not np.array_equal(first[:, -1, :], changed_tail[:, -1, :])


def test_sarima_accepts_fixed_annual_seasonal_order_with_64_weeks() -> None:
    settings = replace(ForecastSettings(), n_samples=7)

    samples = sarima_samples((_history(),), settings, np.random.SeedSequence(9))

    assert settings.sarima_seasonal_order == (1, 0, 0, 52)
    assert samples.shape == (1, 4, 7)


def test_deepar_is_repeatable_preserves_order_and_restores_scale() -> None:
    settings = replace(ForecastSettings(), n_samples=7)
    small = _history()
    large = PreparedHistory(
        node="large",
        dates=small.dates.copy(),
        values=small.values * 10.0,
    )

    first = deepar_samples((large, small), settings, np.random.SeedSequence(321))
    repeated = deepar_samples((large, small), settings, np.random.SeedSequence(321))

    assert first.shape == (2, 4, 7)
    assert np.isfinite(first).all()
    np.testing.assert_array_equal(first, repeated)
    assert np.mean(np.abs(first[0])) > 5.0 * np.mean(np.abs(first[1]))


def test_deepar_uses_cpu_deterministic_trainer_without_writes(
    tmp_path, monkeypatch
) -> None:
    settings = replace(ForecastSettings(), n_samples=7)
    constructor: dict[str, object] = {}
    real_rnn_model = forecast_neural.RNNModel

    def recording_rnn_model(*args, **kwargs):
        constructor.update(kwargs)
        return real_rnn_model(*args, **kwargs)

    monkeypatch.setattr(forecast_neural, "RNNModel", recording_rnn_model)
    monkeypatch.chdir(tmp_path)

    result = deepar_samples((_history(),), settings, np.random.SeedSequence(7))

    trainer = constructor["pl_trainer_kwargs"]
    assert trainer == {
        "accelerator": "cpu",
        "devices": 1,
        "deterministic": True,
        "logger": False,
        "enable_checkpointing": False,
        "enable_progress_bar": False,
        "enable_model_summary": False,
    }
    assert constructor["save_checkpoints"] is False
    assert constructor["log_tensorboard"] is False
    assert result.shape == (1, 4, 7)
    assert list(tmp_path.iterdir()) == []
