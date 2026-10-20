# ABOUTME: Unit tests for temporal-coverage metrics against hand-computed values.
# ABOUTME: Covers inclusive coverage, rolling windows, longest miss run, interval score and the metric dict.
import numpy as np
import pytest

from hcp.evaluate import (
    METRIC_NAMES,
    coverage_deviation_floor,
    coverage_indicator,
    evaluate,
    evaluate_batch,
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


def _random_arms(shape: tuple[int, ...], seed: int = 0):
    rng = np.random.default_rng(seed)
    y = rng.normal(size=shape)
    lo = y - rng.normal(0.5, 1.0, size=shape)
    hi = y + rng.normal(0.5, 1.0, size=shape)
    return y, lo, hi


def test_evaluate_batch_matches_evaluate_row_by_row() -> None:
    y, lo, hi = _random_arms((7, 40))
    batch = evaluate_batch(y, lo, hi, alpha=0.2, window=10)
    assert list(batch) == list(METRIC_NAMES)
    for row in range(7):
        single = evaluate(y[row], lo[row], hi[row], alpha=0.2, window=10)
        for name in METRIC_NAMES:
            np.testing.assert_allclose(
                batch[name][row], single[name], rtol=1e-12, atol=1e-12
            )


def test_evaluate_batch_keeps_leading_axes() -> None:
    y, lo, hi = _random_arms((3, 4, 30), seed=1)
    batch = evaluate_batch(y, lo, hi, alpha=0.2, window=5)
    assert all(value.shape == (3, 4) for value in batch.values())


def test_evaluate_batch_longest_miss_run_hand_values() -> None:
    hits = np.array(
        [[1, 0, 0, 1, 0], [0, 0, 0, 0, 0], [1, 1, 1, 1, 1], [0, 1, 0, 0, 0]]
    )
    y = np.zeros(hits.shape)
    lo = np.where(hits == 1, -1.0, 1.0)
    hi = np.ones(hits.shape)
    batch = evaluate_batch(y, lo, hi, alpha=0.2, window=2)
    np.testing.assert_array_equal(batch["longest_miss_run"], [2.0, 5.0, 0.0, 3.0])


@pytest.mark.parametrize("window", [0, 6])
def test_evaluate_batch_window_out_of_range_raises(window: int) -> None:
    y, lo, hi = _random_arms((2, 5))
    with pytest.raises(ValueError, match="window"):
        evaluate_batch(y, lo, hi, alpha=0.2, window=window)


def test_coverage_deviation_floor_matches_binomial_expectation() -> None:
    floor = coverage_deviation_floor(
        112, alpha=0.2, window=10, n_sim=4000, rng=np.random.default_rng(0)
    )
    assert floor.shape == (4000,)
    # E|Bin(10, 0.8)/10 - 0.8| = 0.0966368 exactly.
    assert abs(floor.mean() - 0.0966368) < 0.003


def test_coverage_deviation_floor_is_reproducible() -> None:
    first = coverage_deviation_floor(
        50, alpha=0.2, window=5, n_sim=10, rng=np.random.default_rng(3)
    )
    again = coverage_deviation_floor(
        50, alpha=0.2, window=5, n_sim=10, rng=np.random.default_rng(3)
    )
    np.testing.assert_array_equal(first, again)
