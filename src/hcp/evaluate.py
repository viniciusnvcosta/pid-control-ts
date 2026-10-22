# ABOUTME: Temporal-coverage metrics for one conformalized series: rolling deviation, miss runs, width, score.
# ABOUTME: The primary endpoint is coverage_deviation, the mean |rolling coverage - (1 - alpha)| (D10); add batched evaluation and noise floor (D19).
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
    """Central interval score, extended to crossed bounds as a quantile-pair score.

    Retains the two-pinball-loss value for empty sets without sorting their
    bounds. Signed spans remain in the formula, not physical set widths.
    """
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


def _coverage_deviation(hit: np.ndarray, alpha: float, window: int) -> np.ndarray:
    """Mean ``|rolling coverage - (1 - alpha)|`` over the last axis of ``hit``."""
    n = hit.shape[-1]
    if not 1 <= window <= n:
        raise ValueError(f"window must be in [1, {n}], got {window}")
    totals = np.cumsum(hit, axis=-1, dtype=float)
    padded = np.concatenate([np.zeros((*hit.shape[:-1], 1)), totals], axis=-1)
    rolling = (padded[..., window:] - padded[..., :-window]) / window
    return np.mean(np.abs(rolling - (1 - alpha)), axis=-1)


def evaluate_batch(
    y: np.ndarray, lo: np.ndarray, hi: np.ndarray, *, alpha: float, window: int
) -> dict[str, np.ndarray]:
    """``evaluate`` over the last (week) axis of ``[..., n]`` arrays.

    Args:
        y: Truth ``[..., n]``.
        lo: Lower bounds, broadcastable to ``y``.
        hi: Upper bounds, broadcastable to ``y``.
        alpha: Miscoverage level; the target coverage is ``1 - alpha``.
        window: Rolling-coverage window in weeks.

    Returns:
        Every metric in ``METRIC_NAMES`` order, each shaped like the leading axes.

    Raises:
        ValueError: If ``window`` is not in ``[1, n]``.
    """
    y, lo, hi = np.broadcast_arrays(y, lo, hi)
    hit = coverage_indicator(y, lo, hi)
    positions = np.arange(hit.shape[-1])
    last_hit = np.maximum.accumulate(np.where(hit, positions, -1), axis=-1)
    width = hi - lo
    return {
        "coverage_deviation": _coverage_deviation(hit, alpha, window),
        "longest_miss_run": np.where(hit, 0, positions - last_hit)
        .max(axis=-1)
        .astype(float),
        "marginal_coverage": hit.mean(axis=-1),
        "mean_width": width.mean(axis=-1),
        "median_width": np.median(width, axis=-1),
        "interval_score": interval_score(y, lo, hi, alpha).mean(axis=-1),
    }


def coverage_deviation_floor(
    n_weeks: int, *, alpha: float, window: int, n_sim: int, rng: np.random.Generator
) -> np.ndarray:
    """Coverage deviation of ``n_sim`` series covered i.i.d. with probability ``1 - alpha``.

    This is what a calibrated interval with independent misses still scores on the
    primary endpoint (D19): its noise floor.

    Returns:
        Simulated coverage deviations ``[n_sim]``.
    """
    hits = rng.random((n_sim, n_weeks)) < 1 - alpha
    return _coverage_deviation(hits, alpha, window)
