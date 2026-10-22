# ABOUTME: Fits node-local direct horizon-four L1 quantiles with purged alpha selection.
# ABOUTME: Rebuilds causal raw prefixes, retains negative forecasts, and records native repairs.
"""Deterministic HiGHS quantile regression without pseudo-sample expansion."""

import warnings
from dataclasses import dataclass

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import QuantileRegressor

from hcp.forecast_types import NativePrediction, PreparedHistory, V2ModelSettings
from hcp.forecast_validation import (
    InsufficientHistoryError,
    TemporalSplit,
    prepare_prefix,
    qr_split,
)
from hcp.reconcile import crossing_mask


@dataclass(frozen=True)
class _NodeData:
    outer: PreparedHistory
    split: TemporalSplit
    training: PreparedHistory
    features: np.ndarray
    responses: np.ndarray
    response_dates: np.ndarray
    validation_features: np.ndarray
    validation_truth: np.ndarray
    refit_features: np.ndarray
    refit_responses: np.ndarray


def direct_examples(
    history: PreparedHistory, cutoff: np.datetime64
) -> tuple[np.ndarray, np.ndarray]:
    """Build twelve descending lags ending four weeks before observed responses.

    Args:
        history: Weekly raw truth, availability mask, and registered forecast dates.
        cutoff: Forecast date by which every returned response must be available.

    Returns:
        Features ``[example, 12]`` and original observed responses ``[example]``.
        Every feature row uses its independently prepared causal raw prefix.

    Raises:
        InsufficientHistoryError: If the requested prefix is structurally unavailable.
        ValueError: If raw metadata or the calendar are malformed.
    """
    features, responses, _ = _examples(history, cutoff)
    return features, responses


def _examples(
    history: PreparedHistory, cutoff: np.datetime64
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    prefix = prepare_prefix(history, cutoff)
    features, responses, response_dates = [], [], []
    for index in np.flatnonzero(prefix.observed):
        if index < 15:
            continue
        try:
            context = prepare_prefix(history, prefix.forecast_dates[index])
        except InsufficientHistoryError:
            continue
        if len(context.values) >= 12:
            features.append(context.values[-12:][::-1])
            responses.append(prefix.raw_values[index])
            response_dates.append(prefix.dates[index])
    return (
        np.asarray(features, dtype=float).reshape(-1, 12),
        np.asarray(responses, dtype=float),
        np.asarray(response_dates, dtype="datetime64[ns]"),
    )


def _node_data(history: PreparedHistory, settings: V2ModelSettings) -> _NodeData:
    split = qr_split(history.dates, history.observed, history.forecast_dates)
    outer_cutoff = (
        history.dates[-1] + np.timedelta64(2, "D")
        if len(history.dates)
        else np.datetime64("1970-01-05")
    )
    # Raw metadata must be valid even when the split is structurally unavailable.
    outer = prepare_prefix(history, outer_cutoff)
    if split is None:
        raise InsufficientHistoryError(f"{history.node}: unavailable QR holdout")
    training = prepare_prefix(history, split.training_cutoff)
    features, responses, response_dates = _examples(history, split.training_cutoff)
    refit_features, refit_responses = direct_examples(
        history, outer.dates[-1] + np.timedelta64(2, "D")
    )
    if min(len(responses), len(refit_responses)) < settings.qr_min_train_examples:
        raise InsufficientHistoryError(f"{history.node}: insufficient QR examples")
    validation_features = []
    for cutoff in split.validation_cutoffs:
        prefix = prepare_prefix(history, cutoff)
        if len(prefix.values) < 12:
            raise InsufficientHistoryError(f"{history.node}: short validation context")
        validation_features.append(prefix.values[-12:][::-1])
    indices = np.searchsorted(history.dates, split.validation_targets)
    return _NodeData(
        outer,
        split,
        training,
        features,
        responses,
        response_dates,
        np.asarray(validation_features),
        history.raw_values[indices],
        refit_features,
        refit_responses,
    )


def _scale(history: PreparedHistory) -> float:
    return max(float(np.mean(np.abs(history.values))), 1.0)


def _fit_predict(
    features: np.ndarray,
    responses: np.ndarray,
    prediction_features: np.ndarray,
    scale: float,
    alpha: float,
    settings: V2ModelSettings,
) -> np.ndarray:
    predictions = []
    for level in settings.levels:
        model = QuantileRegressor(
            quantile=level,
            alpha=alpha,
            solver=settings.qr_solver,
            fit_intercept=settings.qr_fit_intercept,
        )
        # sklearn can return finite coefficients despite an unsuccessful linprog.
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            model.fit(features / scale, responses / scale)
        predicted = np.asarray(model.predict(prediction_features / scale), dtype=float)
        if (
            predicted.shape != (len(prediction_features),)
            or not np.isfinite(predicted).all()
        ):
            raise ValueError("QR solver returned invalid or nonfinite predictions")
        predictions.append(predicted * scale)
    output = np.column_stack(predictions)
    if not np.isfinite(output).all():
        raise ValueError("QR restored predictions must be finite")
    return output


def _validate_settings(settings: V2ModelSettings) -> None:
    if (settings.ahead, settings.qr_input_length, settings.qr_holdout_targets) != (
        4,
        12,
        12,
    ):
        raise ValueError(
            "QR requires horizon four, twelve lags and twelve holdout targets"
        )
    if tuple(settings.levels) != (0.1, 0.5, 0.9):
        raise ValueError("QR requires levels=(0.1, 0.5, 0.9)")
    alphas = np.asarray(settings.qr_alphas, dtype=float)
    if (
        alphas.ndim != 1
        or not len(alphas)
        or not np.isfinite(alphas).all()
        or np.any(alphas <= 0)
        or np.any(np.diff(alphas) <= 0)
    ):
        raise ValueError("QR alphas must be finite, positive and strictly increasing")


def qr_prediction(
    histories: tuple[PreparedHistory, ...],
    settings: V2ModelSettings,
    seed: np.random.SeedSequence,
) -> NativePrediction:
    """Select one L1 alpha per node, refit, and return repaired direct quantiles.

    Args:
        histories: Ordered outer histories retaining original raw truth metadata.
        settings: QR fit and joint three-level holdout settings.
        seed: Reserved outer-origin seed; deterministic HiGHS consumes no randomness.

    Returns:
        Native horizon-four quantiles with per-node tuning and repair diagnostics.

    Raises:
        InsufficientHistoryError: Before fitting, if any node lacks an eligible split.
        ValueError: If metadata, settings, solver predictions or losses are invalid.
        Exception: Eligible solver failures propagate without dropping candidates.
    """
    _validate_settings(settings)
    nodes = tuple(history.node for history in histories)
    if not nodes or len(set(nodes)) != len(nodes):
        raise ValueError("QR histories must have nonempty unique nodes")
    # Preflight every node before the first candidate, for structural eligibility.
    data = tuple(_node_data(history, settings) for history in histories)
    final, tuning, repairs, adjustments = [], [], [], []
    levels = np.asarray(settings.levels)
    for node in data:
        scale = _scale(node.training)
        losses = {}
        for alpha in settings.qr_alphas:
            predicted = _fit_predict(
                node.features,
                node.responses,
                node.validation_features,
                scale,
                alpha,
                settings,
            )
            residual = node.validation_truth[:, None] - predicted
            loss = float(
                np.mean(np.maximum(levels * residual, (levels - 1) * residual))
            )
            if not np.isfinite(loss):
                raise ValueError("QR selection loss must be finite")
            losses[alpha] = loss
        selected = min(losses, key=lambda alpha: (losses[alpha], -alpha))
        refit_scale = _scale(node.outer)
        before = _fit_predict(
            node.refit_features,
            node.refit_responses,
            node.outer.values[-12:][::-1][None, :],
            refit_scale,
            selected,
            settings,
        )[0]
        after = np.sort(before)
        final.append(after)
        origin = node.outer.dates[-1] + np.timedelta64(2, "D")
        tuning.append(
            {
                "model": "qr_l1",
                "node": node.outer.node,
                "forecast_date": origin,
                "chosen_alpha": selected,
                "candidate_losses": losses,
                "training_scale": scale,
                "training_start": node.training.dates[0],
                "training_end": node.training.dates[-1],
                "training_response_start": node.response_dates[0],
                "training_response_end": node.response_dates[-1],
                "training_cutoff": node.split.training_cutoff,
                "training_examples": len(node.responses),
                "validation_targets": len(node.validation_truth),
                "validation_start": node.split.validation_targets[0],
                "validation_end": node.split.validation_targets[-1],
                "refit_scale": refit_scale,
                "refit_start": node.outer.dates[0],
                "refit_end": node.outer.dates[-1],
                "refit_examples": len(node.refit_responses),
            }
        )
        if np.any(before != after):
            adjustment = np.abs(after - before)
            adjustments.append(adjustment)
            repairs.append(
                {
                    "model": "qr_l1",
                    "node": node.outer.node,
                    "forecast_date": origin,
                    "stage": "base",
                    "levels": settings.levels,
                    "before": before,
                    "after": after,
                    "crossing": bool(crossing_mask(before)),
                    "max_absolute_adjustment": float(np.max(adjustment)),
                    "mean_absolute_adjustment": float(np.mean(adjustment)),
                }
            )
    diagnostics = {
        "tuning": tuning,
        "quantile_repairs": repairs,
        "crossing_count": sum(row["crossing"] for row in repairs),
        "root_crossing_count": sum(
            row["crossing"] for row in repairs if row["node"] == "us"
        ),
        "adjusted_cell_count": len(repairs),
        "root_adjusted_cell_count": sum(row["node"] == "us" for row in repairs),
        # Summaries average over all three levels of the adjusted node/origin cells.
        "max_absolute_adjustment": float(np.max(adjustments)) if repairs else 0.0,
        "mean_absolute_adjustment": float(np.mean(adjustments)) if repairs else 0.0,
    }
    return NativePrediction(
        nodes,
        settings.ahead,
        "quantiles",
        quantiles=np.asarray(final),
        levels=settings.levels,
        diagnostics=diagnostics,
    )
