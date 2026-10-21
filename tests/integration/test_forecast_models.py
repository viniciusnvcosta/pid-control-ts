# ABOUTME: Integration tests for real Darts probabilistic benchmark model adapters.
# ABOUTME: Checks shape, finiteness, seed determinism, stochasticity, and horizon-four sensitivity.
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from hcp.forecast_models import (
    ForecastSettings,
    PreparedHistory,
    arima_samples,
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
