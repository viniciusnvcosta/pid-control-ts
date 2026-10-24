# ABOUTME: Prepares causal age/calendar features and explicit shifted recurrent windows.
# ABOUTME: Samples nodes by fit scale then windows uniformly with a dedicated generator.
"""In-memory data boundary for the enhanced Darts recurrent candidate."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from hcp.forecast_types import PreparedHistory
from hcp.forecast_validation import prepare_prefix


def seed_integer(seed: np.random.SeedSequence) -> int:
    """Derive a reproducible torch-compatible integer from an explicit child."""
    return int(seed.generate_state(1, dtype=np.uint32)[0])


def _features(dates: np.ndarray, first_date: np.datetime64) -> np.ndarray:
    age = (dates - first_date) / np.timedelta64(7, "D")
    week = pd.DatetimeIndex(dates).isocalendar().week.to_numpy(dtype=float)
    return np.column_stack((age, week))


@dataclass(frozen=True)
class CovariateTransform:
    """Owned training-only normalization for age and ISO numeric week-of-year."""

    first_date: np.datetime64
    mean: np.ndarray
    scale: np.ndarray

    def __post_init__(self) -> None:
        for name in ("mean", "scale"):
            array = np.array(getattr(self, name), copy=True)
            array.setflags(write=False)
            object.__setattr__(self, name, array)

    @classmethod
    def fit(cls, prefix: PreparedHistory) -> "CovariateTransform":
        """Fit on an independently prepared training prefix only."""
        values = _features(prefix.dates, prefix.dates[0])
        scale = values.std(axis=0)
        return cls(
            prefix.dates[0], values.mean(axis=0), np.where(scale == 0, 1.0, scale)
        )

    def transform(self, dates: np.ndarray) -> np.ndarray:
        """Extend known calendar features using fixed training statistics."""
        return (_features(dates, self.first_date) - self.mean) / self.scale


def enhanced_covariates(
    histories: tuple[PreparedHistory, ...],
    training_cutoff: np.datetime64,
    prediction_dates: np.ndarray,
) -> tuple[np.ndarray, ...]:
    """Return standardized age/week arrays over each prefix plus prediction dates.

    Args:
        histories: Raw outer histories in fixed node order.
        training_cutoff: Actual forecast-date boundary for fitting statistics.
        prediction_dates: Known future calendar dates to append after each prefix.

    Returns:
        Per-node arrays with columns age and numeric ISO week, in date order.
    """
    result = []
    for history in histories:
        prefix = prepare_prefix(history, training_cutoff)
        dates = np.union1d(prefix.dates, prediction_dates)
        result.append(CovariateTransform.fit(prefix).transform(dates))
    return tuple(result)


def weighted_window_sampler(
    window_counts: np.ndarray, scales: np.ndarray, seed: np.random.SeedSequence
) -> WeightedRandomSampler:
    """Draw total-window epoch size with replacement, with node mass proportional to nu.

    Args:
        window_counts: Positive counts, contiguous nodes in dataset order.
        scales: Positive fit-only node scales nu.
        seed: Dedicated sampler child, consumed continuously across epochs.

    Returns:
        A sampler whose per-window weight is nu divided by that node's count.
    """
    counts, nu = np.asarray(window_counts), np.asarray(scales, dtype=float)
    if (
        counts.ndim != 1
        or not len(counts)
        or counts.shape != nu.shape
        or not np.issubdtype(counts.dtype, np.integer)
        or np.any(counts <= 0)
        or not np.isfinite(nu).all()
        or np.any(nu <= 0)
    ):
        raise ValueError(
            "sampler requires positive integer counts and finite positive scales"
        )
    weights = np.repeat(nu / counts, counts)
    generator = torch.Generator().manual_seed(seed_integer(seed))
    return WeightedRandomSampler(
        weights, int(counts.sum()), replacement=True, generator=generator
    )


class EnhancedWindowDataset(Dataset):
    """All valid 16-step shift-one windows, without equalizing node lengths."""

    def __init__(
        self,
        prefixes: tuple[PreparedHistory, ...],
        scales: np.ndarray,
        transforms: tuple[CovariateTransform, ...],
        training_length: int = 16,
    ):
        self.training_length = training_length
        self.window_counts = np.array(
            [len(h.values) - training_length for h in prefixes]
        )
        if np.any(self.window_counts <= 0):
            raise ValueError(
                "each node needs at least training_length + 1 observations"
            )
        self.ends = np.cumsum(self.window_counts)
        self.targets = tuple(
            torch.tensor(h.values / s, dtype=torch.float32)[:, None]
            for h, s in zip(prefixes, scales, strict=True)
        )
        self.covariates = tuple(
            torch.tensor(t.transform(h.dates), dtype=torch.float32)
            for h, t in zip(prefixes, transforms, strict=True)
        )

    def __len__(self) -> int:
        return int(self.ends[-1])

    def __getitem__(self, index: int) -> tuple:
        node = int(np.searchsorted(self.ends, index, side="right"))
        start = index - (int(self.ends[node - 1]) if node else 0)
        stop = start + self.training_length
        # Darts training batch: past target, past cov, historic future cov,
        # shifted future cov, static node ID, sample weight, shifted target.
        return (
            self.targets[node][start:stop],
            None,
            None,
            self.covariates[node][start + 1 : stop + 1],
            torch.tensor([[node]], dtype=torch.float32),
            None,
            self.targets[node][start + 1 : stop + 1],
        )


def collate_windows(samples: list[tuple]) -> tuple:
    """Stack Darts' optional-tensor seven-field training batch."""
    return tuple(
        None if values[0] is None else torch.stack(values)
        for values in zip(*samples, strict=True)
    )
