# ABOUTME: Real Prophet integration on small synthetic weekly histories.
# ABOUTME: Checks native units, negative paths, seeded MAP uncertainty and recent-prefix sensitivity.
"""Seed reproducibility and scale contracts against the real Darts Prophet."""

from importlib import import_module

import numpy as np
import pandas as pd
import pytest

from hcp.forecast_types import PreparedHistory, V2ModelSettings

pytestmark = pytest.mark.integration


def _history(shift: float = 0.0) -> PreparedHistory:
    dates = pd.date_range("2020-01-04", periods=64, freq="7D").to_numpy()
    index = np.arange(64, dtype=float)
    values = -400.0 + 0.4 * index + 2 * np.sin(index)
    values[-4:] += shift
    return PreparedHistory("negative", dates, values)


def test_real_prophet_has_seeded_original_unit_predictive_paths():
    adapter = import_module("hcp.forecast_classical")
    settings = V2ModelSettings()
    first = adapter.prophet_prediction(
        (_history(),), settings, np.random.SeedSequence(81)
    )
    repeat = adapter.prophet_prediction(
        (_history(),), settings, np.random.SeedSequence(81)
    )
    reseeded = adapter.prophet_prediction(
        (_history(),), settings, np.random.SeedSequence(82)
    )
    changed = adapter.prophet_prediction(
        (_history(40),), settings, np.random.SeedSequence(81)
    )

    assert first.samples.shape == (1, 4, 500)
    assert np.isfinite(first.samples).all()
    assert np.mean(first.samples) < -300
    assert np.std(first.samples[0, -1]) > 0
    np.testing.assert_array_equal(first.samples, repeat.samples)
    assert not np.array_equal(first.samples, reseeded.samples)
    assert not np.array_equal(first.samples[:, -1], changed.samples[:, -1])


def test_real_prophet_passes_child_seed_to_map_fit(monkeypatch):
    import prophet

    observed_seeds = []
    original_fit = prophet.Prophet.fit

    def recording_fit(self, *args, **kwargs):
        observed_seeds.append(kwargs.get("seed"))
        return original_fit(self, *args, **kwargs)

    monkeypatch.setattr(prophet.Prophet, "fit", recording_fit)
    adapter = import_module("hcp.forecast_classical")
    result = adapter.prophet_prediction(
        (_history(),), V2ModelSettings(n_samples=7), np.random.SeedSequence(127)
    )
    assert observed_seeds == [2567351042]
    assert result.diagnostics["node_seeds"] == (2567351042,)
