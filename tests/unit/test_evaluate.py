# ABOUTME: Unit tests for temporal-coverage metrics against hand-computed values.
# ABOUTME: Covers inclusive coverage, rolling windows, longest miss run, interval score and the metric dict.
import numpy as np
import pytest

from hcp.evaluate import (
    METRIC_NAMES,
    coverage_indicator,
    evaluate,
    interval_score,
    longest_miss_run,
    rolling_coverage,
)

Y = np.array([5.0, 0.0, 12.0, 5.0])
LO = np.array([0.0, 1.0, 0.0, 0.0])
HI = np.array([10.0, 10.0, 10.0, 10.0])


def test_coverage_is_inclusive() -> None:
    hit = coverage_indicator(
        np.array([1.0, 2, 3, 4]), np.array([1.0, 0, 4, 0]), np.array([2.0, 1, 5, 4])
    )
    np.testing.assert_array_equal(hit, [True, False, False, True])


def test_rolling_coverage_valid_windows() -> None:
    hit = np.array([True, False, True, True])
    np.testing.assert_allclose(rolling_coverage(hit, 2), [0.5, 0.5, 1.0])


@pytest.mark.parametrize("window", [0, 5])
def test_rolling_window_out_of_range_raises(window: int) -> None:
    with pytest.raises(ValueError, match="window"):
        rolling_coverage(np.ones(4, dtype=bool), window)


def test_longest_miss_run() -> None:
    hit = np.array([1, 0, 0, 1, 0, 0, 0, 1], dtype=bool)
    assert longest_miss_run(hit) == 3
    assert longest_miss_run(np.ones(5, dtype=bool)) == 0


def test_interval_score_hand_values() -> None:
    np.testing.assert_allclose(
        interval_score(Y[:3], LO[:3], HI[:3], 0.2), [10.0, 19.0, 30.0]
    )


def test_evaluate_hand_values() -> None:
    metrics = evaluate(Y, LO, HI, alpha=0.2, window=2)
    assert tuple(metrics) == METRIC_NAMES
    assert metrics["coverage_deviation"] == pytest.approx((0.3 + 0.8 + 0.3) / 3)
    assert metrics["longest_miss_run"] == 2.0
    assert metrics["marginal_coverage"] == 0.5
    assert metrics["mean_width"] == 9.75
    assert metrics["median_width"] == 10.0
    assert metrics["interval_score"] == pytest.approx(17.25)
