# ABOUTME: Hand-computed v2 metrics for empty sets and per-node coverage error.
# ABOUTME: Protects signed legacy spans and the unchanged quantile-pair score.
import numpy as np
import pytest

from hcp import panel_metrics
from hcp.evaluate import evaluate_batch, interval_score


def test_empty_set_has_zero_physical_width_and_signed_legacy_span() -> None:
    y, lo, hi = np.array([[1.0]]), np.array([[2.0]]), np.array([[0.0]])
    actual = panel_metrics.v2_node_metrics(
        y, lo, hi, np.array([2.0]), alpha=0.2, window=1
    )
    expected = {
        "marginal_coverage": 0.0,
        "marginal_coverage_error": 0.8,
        "longest_miss_run": 1.0,
        "mean_width": -2.0,
        "median_width": -2.0,
        "scaled_mean_width": -1.0,
        "mean_set_width": 0.0,
        "scaled_mean_set_width": 0.0,
        "empty_set_rate": 1.0,
        "interval_score": 18.0,
        "scaled_interval_score": 9.0,
    }
    for name, value in expected.items():
        np.testing.assert_allclose(actual[name], [value])
    legacy = evaluate_batch(y, lo, hi, alpha=0.2, window=1)
    for name, value in legacy.items():
        np.testing.assert_array_equal(actual[name], value)


def test_absolute_coverage_error_precedes_leaf_aggregation() -> None:
    y = np.ones((2, 5))
    lo = np.zeros((2, 5))
    hi = np.array([[2.0, 2.0, 2.0, 0.0, 0.0], [2.0] * 5])
    actual = panel_metrics.v2_node_metrics(y, lo, hi, np.ones(2), alpha=0.2, window=2)
    np.testing.assert_allclose(actual["marginal_coverage"], [0.6, 1.0])
    np.testing.assert_allclose(actual["marginal_coverage_error"], [0.2, 0.2])
    assert actual["marginal_coverage_error"].mean() == pytest.approx(0.2)


def test_set_width_averages_clipped_weekly_spans_and_counts_only_strict_empties() -> (
    None
):
    y = np.array([[1.0, 0.0, 1.0]])
    lo = np.array([[2.0, 0.0, 0.0]])
    hi = np.array([[0.0, 0.0, 4.0]])
    actual = panel_metrics.v2_node_metrics(
        y, lo, hi, np.array([2.0]), alpha=0.2, window=1
    )
    np.testing.assert_allclose(actual["mean_set_width"], [4 / 3])
    np.testing.assert_allclose(actual["scaled_mean_set_width"], [2 / 3])
    np.testing.assert_allclose(actual["empty_set_rate"], [1 / 3])
    np.testing.assert_array_equal(lo, [[2.0, 0.0, 0.0]])
    np.testing.assert_array_equal(hi, [[0.0, 0.0, 4.0]])


def test_score_retains_two_pinball_loss_value_for_crossed_bounds() -> None:
    np.testing.assert_array_equal(
        interval_score(np.array([1.0]), np.array([2.0]), np.array([0.0]), 0.2), [18.0]
    )
