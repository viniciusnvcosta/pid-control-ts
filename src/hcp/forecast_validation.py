# ABOUTME: Builds purged inner holdouts and independently prepared causal raw prefixes.
# ABOUTME: Separates structural ineligibility from invalid metadata and allocates explicit neural streams.
"""In-memory causal validation shared by the v2 forecast adapters."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from hcp.forecast_types import PreparedHistory


class InsufficientHistoryError(ValueError):
    """A valid calendar has too little usable observed history for a prefix."""


@dataclass(frozen=True)
class TemporalSplit:
    """Selection boundaries expressed as dates, never row-number cutoffs.

    Attributes:
        training_cutoff: First scored validation forecast date.
        validation_targets: Observed target dates to score in chronological order.
        validation_cutoffs: Each target's registered forecast date.
    """

    training_cutoff: np.datetime64
    validation_targets: np.ndarray
    validation_cutoffs: np.ndarray

    def __post_init__(self) -> None:
        for name in ("validation_targets", "validation_cutoffs"):
            array = np.array(getattr(self, name), dtype="datetime64[ns]", copy=True)
            array.setflags(write=False)
            object.__setattr__(self, name, array)


def _calendar(
    dates: np.ndarray, observed: np.ndarray, forecast_dates: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if dates is None or observed is None or forecast_dates is None:
        raise ValueError("raw history metadata is required for v2 validation")
    targets = np.asarray(dates, dtype="datetime64[ns]")
    cutoffs = np.asarray(forecast_dates, dtype="datetime64[ns]")
    mask = np.asarray(observed)
    if targets.ndim != 1 or cutoffs.shape != targets.shape:
        raise ValueError("date metadata must have matching one-dimensional shapes")
    if np.isnat(targets).any() or np.isnat(cutoffs).any():
        raise ValueError("date metadata cannot contain NaT")
    if len(targets) > 1 and not np.all(np.diff(targets) == np.timedelta64(7, "D")):
        raise ValueError("validation dates must form a strict weekly grid")
    if np.any(targets - cutoffs != np.timedelta64(26, "D")):
        raise ValueError("validation forecast dates require the registered 26-day lag")
    if mask.ndim not in (1, 2) or mask.shape[-1] != len(targets) or mask.dtype != bool:
        raise ValueError("observed metadata must be a boolean date-aligned mask")
    if mask.ndim == 2 and mask.shape[0] == 0:
        raise ValueError("observed metadata must include at least one node")
    return targets, mask, cutoffs


def _holdout(
    dates: np.ndarray, observed: np.ndarray, cutoffs: np.ndarray, *, neural: bool
) -> TemporalSplit | None:
    if len(dates) < 16:
        return None
    first = len(dates) - 12
    masks = np.atleast_2d(observed)
    # Do not move the holdout backwards to rescue an incomplete latest window.
    if not masks[:, first:].all():
        return None
    scored = first + 3 if neural else first
    training_end = int(np.searchsorted(dates, cutoffs[scored], side="right")) - 1
    if training_end < 0 or not masks[:, training_end].all():
        return None
    if dates[training_end] != dates[scored] - np.timedelta64(28, "D"):
        raise ValueError("inner training prefix violates the horizon-four boundary")
    # A missing endpoint cannot be interpolated with the next, unavailable week.
    if not masks[:, scored - 4 : len(dates) - 4].all():
        return None
    for mask in masks:
        available = np.flatnonzero(mask[: training_end + 1])
        if not len(available):
            return None
        start = int(available[0])
        if neural:
            # The enhanced RNN needs a 16-week sequence shifted by one week.
            if training_end - start + 1 < 17:
                return None
        else:
            # Twelve lags end at response-4; only response truth is scored.
            responses = np.arange(start + 15, training_end + 1)
            eligible = mask[responses] & mask[responses - 4]
            if np.count_nonzero(eligible) < 12:
                return None
    return TemporalSplit(cutoffs[scored], dates[scored:], cutoffs[scored:])


def qr_split(
    dates: np.ndarray, observed: np.ndarray, forecast_dates: np.ndarray
) -> TemporalSplit | None:
    """Reserve twelve observed targets and purge unavailable training responses.

    Args:
        dates: Strict weekly dates of one node's available outer history.
        observed: One-dimensional original truth availability, never imputed.
        forecast_dates: Target-aligned forecast cutoffs with the 26-day lag.

    Returns:
        The split, or None for an incomplete holdout/prefix or fewer than twelve
        eligible training pairs known at the first validation forecast date.

    Raises:
        ValueError: If calendar or metadata are malformed or missing.
    """
    targets, mask, cutoffs = _calendar(dates, observed, forecast_dates)
    if mask.ndim != 1:
        raise ValueError("QR observed metadata must describe one node")
    return _holdout(targets, mask, cutoffs, neural=False)


def neural_split(
    dates: np.ndarray, observed: np.ndarray, forecast_dates: np.ndarray
) -> TemporalSplit | None:
    """Score nine fourth steps inside the final twelve fully observed weeks.

    Args:
        dates: Strict weekly dates shared by the outer node histories.
        observed: Original availability with shape ``[T]`` or ``[nodes, T]``.
        forecast_dates: Target-aligned forecast cutoffs with the 26-day lag.

    Returns:
        A split with the preceding training prefix, or None if any node has
        incomplete holdout/context or fewer than seventeen inner training weeks.

    Raises:
        ValueError: If calendar or metadata are malformed or missing.
    """
    targets, mask, cutoffs = _calendar(dates, observed, forecast_dates)
    return _holdout(targets, mask, cutoffs, neural=True)


def prepare_prefix(history: PreparedHistory, cutoff: np.datetime64) -> PreparedHistory:
    """Slice original observations before interpolation at an inner forecast date.

    Args:
        history: Outer history retaining raw values, observed mask, and cutoffs.
        cutoff: Registered forecast date defining the available inner prefix.

    Returns:
        Protected history after dropping leading gaps and filling internal gaps.
        Raw metadata remains available for subsequent independent preparation.

    Raises:
        InsufficientHistoryError: If no usable prefix exists or its end is missing.
        ValueError: If metadata or the requested calendar boundary are invalid.
    """
    if history.raw_values is None:
        raise ValueError("raw history metadata is required for v2 validation")
    dates, observed, origins = _calendar(
        history.dates, history.observed, history.forecast_dates
    )
    raw = np.asarray(history.raw_values, dtype=float)
    if (
        observed.ndim != 1
        or raw.shape != dates.shape
        or history.values.shape != dates.shape
    ):
        raise ValueError("raw history metadata must match the one-node date shape")
    if np.any(observed & ~np.isfinite(raw)):
        raise ValueError("observed metadata requires finite original truth")
    boundary = np.asarray(cutoff, dtype="datetime64[ns]")
    if boundary.ndim != 0 or np.isnat(boundary):
        raise ValueError("prefix cutoff must be a finite scalar date")
    if len(dates) and (boundary + np.timedelta64(26, "D") - dates[0]) % np.timedelta64(
        7, "D"
    ):
        raise ValueError(
            "prefix cutoff must align with the registered forecast calendar"
        )
    stop = int(np.searchsorted(dates, boundary, side="right"))
    available = np.flatnonzero(observed[:stop])
    if not len(available):
        raise InsufficientHistoryError("prefix has no observed history")
    if dates[stop - 1] != boundary - np.timedelta64(2, "D"):
        raise InsufficientHistoryError(
            "prefix does not reach the horizon-four boundary"
        )
    if not observed[stop - 1]:
        raise InsufficientHistoryError(
            "prefix has a trailing gap with no available endpoint"
        )
    start = int(available[0])
    values = np.where(observed[start:stop], raw[start:stop], np.nan)
    filled = pd.Series(values).interpolate(limit_area="inside").to_numpy()
    return PreparedHistory(
        history.node,
        dates[start:stop],
        filled,
        raw[start:stop],
        observed[start:stop],
        origins[start:stop],
    )


def neural_seed_streams(
    seed: np.random.SeedSequence, n_candidates: int
) -> dict[str, tuple[np.random.SeedSequence, ...]]:
    """Spawn distinct streams in training, validation, refit, prediction order.

    Args:
        seed: Reserved model/origin child from the complete outer target grid.
        n_candidates: Positive number of architectures in fixed candidate order.

    Returns:
        Candidate training/validation tuples and single final refit/prediction
        tuples. Recreate validation generators from their child at each epoch.

    Raises:
        ValueError: If the candidate count is not a positive integer.
    """
    if (
        isinstance(n_candidates, bool)
        or not isinstance(n_candidates, (int, np.integer))
        or n_candidates < 1
    ):
        raise ValueError("n_candidates must be a positive integer")
    children = seed.spawn(2 * n_candidates + 2)
    return {
        "training": tuple(children[:n_candidates]),
        "validation": tuple(children[n_candidates : 2 * n_candidates]),
        "refit": (children[-2],),
        "prediction": (children[-1],),
    }
