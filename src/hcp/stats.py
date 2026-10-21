# ABOUTME: Paired moving-block bootstrap over weeks, percentile intervals, shift p-values and Holm adjustment.
# ABOUTME: Resample indices are drawn once per block length and shared by every arm (D12).
"""Statistics stage (decision D12)."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Contrast:
    """Treatment minus control for one metric."""

    estimate: float
    low: float
    high: float
    p_value: float


def draw_blocks(
    n: int, block: int, n_boot: int, rng: np.random.Generator
) -> np.ndarray:
    """Moving-block bootstrap indices ``[n_boot, n]`` built from contiguous blocks.

    Raises:
        ValueError: If ``block`` is not in ``[1, n]``.
    """
    if not 1 <= block <= n:
        raise ValueError(f"block must be in [1, {n}], got {block}")
    n_blocks = -(-n // block)
    starts = rng.integers(0, n - block + 1, size=(n_boot, n_blocks))
    return (starts[:, :, None] + np.arange(block)).reshape(n_boot, -1)[:, :n]


def summarize_diff(estimate: float, diffs: np.ndarray, level: float = 0.95) -> Contrast:
    """Percentile interval and two-sided shift p-value for a bootstrapped difference.

    The p-value centers the bootstrap differences at zero and counts how often
    ``|centered| >= |estimate|``, with the +1 correction.
    """
    tail = (1 - level) / 2
    low, high = np.quantile(diffs, [tail, 1 - tail])
    centered = diffs - diffs.mean()
    p_value = (np.sum(np.abs(centered) >= abs(estimate)) + 1) / (len(diffs) + 1)
    return Contrast(float(estimate), float(low), float(high), float(p_value))


def holm(p_values: Sequence[float] | np.ndarray) -> np.ndarray:
    """Holm step-down adjusted p-values, in the input order."""
    p = np.asarray(p_values, dtype=float)
    adjusted = np.empty_like(p)
    running = 0.0
    for rank, index in enumerate(np.argsort(p)):
        running = max(running, (len(p) - rank) * p[index])
        adjusted[index] = min(running, 1.0)
    return adjusted
