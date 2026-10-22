# ABOUTME: Exercises selected enhanced DeepAR with real Darts on synthetic 57-node data.
# ABOUTME: Checks repeatability, causal outer preparation, native scales, and warning evidence.
"""Reduced development budgets never alter the production defaults."""

import json
from dataclasses import replace

import numpy as np
import pytest

from hcp.forecast_deepar import enhanced_deepar_prediction
from hcp.forecast_types import PreparedHistory, V2ModelSettings
from hcp.forecast_validation import prepare_prefix

pytestmark = pytest.mark.integration


def _available_histories(future_shift=0.0):
    dates = np.datetime64("2020-01-04") + np.arange(44) * np.timedelta64(7, "D")
    result = []
    for node in range(57):
        values = -np.arange(44, dtype=float) * (node + 1)
        values[40:] += future_shift
        raw = PreparedHistory(
            str(node),
            dates,
            values,
            values,
            np.ones(44, bool),
            dates - np.timedelta64(26, "D"),
        )
        result.append(prepare_prefix(raw, dates[39] + np.timedelta64(2, "D")))
    return tuple(result)


def test_real_57_node_paths_repeat_and_ignore_unavailable_future(record_property):
    settings = replace(
        V2ModelSettings(), neural_max_epochs=1, neural_patience=1, n_samples=8
    )
    predictions = [
        enhanced_deepar_prediction(
            _available_histories(shift), settings, np.random.SeedSequence(71)
        )
        for shift in (0.0, 0.0, 1000000.0)
    ]
    first = predictions[0]
    assert first.nodes == tuple(str(i) for i in range(57))
    assert first.samples.shape == (57, 4, 8)
    assert np.isfinite(first.samples).all() and not first.samples.flags.writeable
    assert np.std(first.samples[0, -1]) > 0
    assert np.any(first.samples < 0)  # native Gaussian paths are never clipped
    for prediction in predictions:
        np.testing.assert_array_equal(first.samples, prediction.samples)
        tuning = prediction.diagnostics["tuning"]
        assert [row["width"] for row in tuning] == [40, 80, 120]
        for row in tuning:
            # Inner 28 weeks: mean 0..27 = 13.5; full 40 weeks: mean = 19.5.
            np.testing.assert_array_equal(row["scales"], 1.0 + 13.5 * np.arange(1, 58))
            assert row["best_epoch"] == row["epochs_run"] == 1
            assert np.isfinite(row["best_nll"])
        fit = prediction.diagnostics["final_fit"]
        np.testing.assert_array_equal(fit["scales"], 1.0 + 19.5 * np.arange(1, 58))
        assert fit["epochs"] == 1
        for covariance in fit["covariates"]:
            assert covariance["mean"][0] == 19.5
        rows = prediction.diagnostics["warnings"]
        assert rows and all(row["filename"] and row["lineno"] > 0 for row in rows)
        assert all(
            row["node"] is None and row["model"] == "deepar_enhanced" for row in rows
        )
        assert sum(row["count"] for row in rows) == sum(
            prediction.diagnostics["warning_counts"].values()
        )
        assert {row["phase"] for row in rows} >= {"training", "refit"}
    record_property(
        "warning_evidence",
        json.dumps(
            [
                [dict(row) for row in prediction.diagnostics["warnings"]]
                for prediction in predictions
            ],
            default=str,
        ),
    )
