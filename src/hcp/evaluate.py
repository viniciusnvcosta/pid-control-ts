# ABOUTME: Temporal-coverage metrics for one conformalized series: rolling deviation, miss runs, width, score.
# ABOUTME: The primary endpoint is coverage_deviation, the mean |rolling coverage - (1 - alpha)| (D10).
"""Evaluation stage (decisions D10, D11)."""

import numpy as np

METRIC_NAMES = (
    "coverage_deviation",
    "longest_miss_run",
    "marginal_coverage",
    "mean_width",
    "median_width",
    "interval_score",
)


def coverage_indicator(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """True where ``lo <= y <= hi``."""
    return (lo <= y) & (y <= hi)


def rolling_coverage(hit: np.ndarray, window: int) -> np.ndarray:
    """Coverage over each full trailing window, length ``len(hit) - window + 1``.

    Raises:
        ValueError: If ``window`` is not in ``[1, len(hit)]``.
    """
    if not 1 <= window <= len(hit):
        raise ValueError(f"window must be in [1, {len(hit)}], got {window}")
    return np.convolve(hit.astype(float), np.ones(window) / window, mode="valid")


def longest_miss_run(hit: np.ndarray) -> int:
    """Length of the longest run of consecutive misses."""
    longest = current = 0
    for covered in hit:
        current = 0 if covered else current + 1
        longest = max(longest, current)
    return longest


def interval_score(
    y: np.ndarray, lo: np.ndarray, hi: np.ndarray, alpha: float
) -> np.ndarray:
    """Gneiting-Raftery interval score of the central ``1 - alpha`` interval."""
    below = (lo - y) * (y < lo)
    above = (y - hi) * (y > hi)
    return (hi - lo) + 2 / alpha * below + 2 / alpha * above


def evaluate(
    y: np.ndarray, lo: np.ndarray, hi: np.ndarray, *, alpha: float, window: int
) -> dict[str, float]:
    """All metrics in ``METRIC_NAMES`` order."""
    hit = coverage_indicator(y, lo, hi)
    width = hi - lo
    return {
        "coverage_deviation": float(
            np.mean(np.abs(rolling_coverage(hit, window) - (1 - alpha)))
        ),
        "longest_miss_run": float(longest_miss_run(hit)),
        "marginal_coverage": float(hit.mean()),
        "mean_width": float(np.mean(width)),
        "median_width": float(np.median(width)),
        "interval_score": float(np.mean(interval_score(y, lo, hi, alpha))),
    }
