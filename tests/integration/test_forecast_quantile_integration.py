# ABOUTME: Exercises local QR with real HiGHS fits on synthetic weekly observations.
# ABOUTME: Checks reproducibility, causal outer slicing, units, and independent node order.
from dataclasses import replace

import numpy as np
import pandas as pd

from hcp.forecast_types import PreparedHistory, V2ModelSettings
from hcp.forecast_validation import prepare_prefix


def test_real_solver_repeats_and_ignores_post_cutoff_truth() -> None:
    from hcp.forecast_quantile import qr_prediction

    dates = pd.date_range("2020-01-04", periods=70, freq="7D").to_numpy()
    raw = -100 + np.arange(70, dtype=float)
    history = PreparedHistory(
        "us", dates, raw, raw, np.ones(70, bool), dates - np.timedelta64(26, "D")
    )
    cutoff = history.forecast_dates[64]
    first = prepare_prefix(history, cutoff)
    changed_raw = raw.copy()
    changed_raw[61:] += 1000000
    changed = prepare_prefix(replace(history, raw_values=changed_raw), cutoff)
    settings = V2ModelSettings()
    prediction = qr_prediction((first,), settings, np.random.SeedSequence(123))
    repeated = qr_prediction((changed,), settings, np.random.SeedSequence(123))
    np.testing.assert_array_equal(prediction.quantiles, repeated.quantiles)
    assert np.all(prediction.quantiles < 0)
    assert np.all(np.diff(prediction.quantiles, axis=-1) >= 0)
    assert prediction.samples is None
    assert prediction.quantiles.shape == (1, 3)
    assert not prediction.quantiles.flags.writeable
    # Intercept and extrapolation are restored into original death-count units.
    np.testing.assert_allclose(prediction.quantiles, [[-36, -36, -36]], atol=0.1)


def test_real_solver_keeps_nodes_local_and_in_declared_order() -> None:
    from hcp.forecast_quantile import qr_prediction

    dates = pd.date_range("2020-01-04", periods=60, freq="7D").to_numpy()
    histories = tuple(
        PreparedHistory(
            node,
            dates,
            np.full(60, value),
            np.full(60, value),
            np.ones(60, bool),
            dates - np.timedelta64(26, "D"),
        )
        for node, value in (("ca", -20.0), ("us", -5.0))
    )
    prediction = qr_prediction(
        histories, V2ModelSettings(), np.random.SeedSequence(123)
    )
    assert prediction.nodes == ("ca", "us")
    np.testing.assert_array_equal(prediction.quantiles, [[-20, -20, -20], [-5, -5, -5]])
