# ABOUTME: Per-node and per-level metric layer for the panel stage: node scaling,
# ABOUTME: node/level metric aggregation, and the long-format frame builders.
"""Panel metric layer (decisions D23, D25).

Scales each node by ``max(mean |y|, 1)`` over its evaluation window (D23), aggregates
per-node metrics into node- and level-level values -- ``leaf_mean`` averaging over
scored leaves outside ``TERRITORIES`` (D25) and ``us`` taking the root row directly --
and builds the long-format intervals, metrics and level-metrics frames consumed by
``hcp.panel``.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from hcp.evaluate import METRIC_NAMES, evaluate_batch

LEVELS = ("leaf_mean", "us")
PANEL_METRICS = (
    "coverage_deviation",
    "longest_miss_run",
    "marginal_coverage",
    "scaled_mean_width",
    "scaled_interval_score",
)


@dataclass(frozen=True)
class ForecasterArms:
    """One forecaster's reconciled, conformalized arms on its own evaluation window.

    Attributes:
        weeks: Target dates of the evaluation window ``[n]``.
        nodes: Scored node names, ``"us"`` first (D25).
        y: Truth ``[n_scored, n]``.
        q: Reconciler -> reconciled quantiles ``[n_scored, n, 3]``.
        intervals: ``(reconciler, controller, lr) -> (lo, hi)``, each ``[n_scored, n]``.
        endpoint: ``[n_scored]``; True for scored leaves outside ``TERRITORIES`` (D25).
        diagnostics: Per-forecaster diagnostics, mirroring ``run.ExperimentResult``.
    """

    weeks: np.ndarray
    nodes: tuple[str, ...]
    y: np.ndarray
    q: dict[str, np.ndarray]
    intervals: dict[tuple[str, str, float], tuple[np.ndarray, np.ndarray]]
    endpoint: np.ndarray
    diagnostics: dict


def node_scale(y: np.ndarray) -> np.ndarray:
    """Per-node scale ``max(mean |y|, 1)`` over the week axis (D23)."""
    return np.maximum(np.abs(y).mean(axis=-1), 1.0)


def node_metrics(
    y: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    scale: np.ndarray,
    *,
    alpha: float,
    window: int,
) -> dict[str, np.ndarray]:
    """Per-node metrics, with ``mean_width``/``interval_score`` additionally scaled (D23).

    Args:
        y: Truth, nodes on axis 0.
        lo: Lower interval bounds, same shape as ``y``.
        hi: Upper interval bounds, same shape as ``y``.
        scale: Per-node scale from ``node_scale``.
        alpha: Target miscoverage rate.
        window: Rolling-coverage window in weeks.

    Returns:
        Metric name -> per-node value(s), including ``scaled_mean_width`` and
        ``scaled_interval_score``.
    """
    values = evaluate_batch(y, lo, hi, alpha=alpha, window=window)
    values["scaled_mean_width"] = values["mean_width"] / scale
    values["scaled_interval_score"] = values["interval_score"] / scale
    return values


def v2_node_metrics(
    y: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    scale: np.ndarray,
    *,
    alpha: float,
    window: int,
) -> dict[str, np.ndarray]:
    """Add v2 coverage-error and physical-width metrics to legacy node metrics.

    Args:
        y: Truth, with weeks on the last axis.
        lo: Final lower bounds, broadcastable to ``y``; retained even if crossed.
        hi: Final upper bounds, broadcastable to ``y``.
        scale: Fixed scoring scales, broadcastable to per-node metric values.
        alpha: Target miscoverage rate.
        window: Rolling-coverage window in weeks.

    Returns:
        Legacy metrics plus absolute marginal coverage error computed before
        leaf aggregation, mean nonnegative set width and its scaled value,
        and the fraction of weeks with strictly crossed (empty) final bounds.
        Legacy widths remain signed spans and scores remain quantile-pair loss.
    """
    y, lo, hi = np.broadcast_arrays(y, lo, hi)
    values = evaluate_batch(y, lo, hi, alpha=alpha, window=window)
    scale = np.asarray(scale, dtype=float)
    scale_shape = (len(scale),) + (1,) * (values["interval_score"].ndim - 1)
    broadcast_scale = scale.reshape(scale_shape)
    values["scaled_mean_width"] = values["mean_width"] / broadcast_scale
    values["scaled_interval_score"] = values["interval_score"] / broadcast_scale
    values["marginal_coverage_error"] = np.abs(
        values["marginal_coverage"] - (1 - alpha)
    )
    values["mean_set_width"] = np.maximum(hi - lo, 0.0).mean(axis=-1)
    values["scaled_mean_set_width"] = values["mean_set_width"] / broadcast_scale
    values["empty_set_rate"] = (lo > hi).mean(axis=-1)
    return values


def level_values(
    values: dict[str, np.ndarray], endpoint: np.ndarray
) -> dict[str, dict[str, np.ndarray]]:
    """Aggregate per-node metrics into level-level values (D25).

    Args:
        values: Metric name -> value(s) with nodes on axis 0.
        endpoint: ``[n_scored]``; True for scored leaves outside ``TERRITORIES``.

    Returns:
        ``level -> metric -> value(s)``; ``values[metric]`` has nodes on axis 0.
    """
    return {
        "leaf_mean": {m: values[m][endpoint].mean(axis=0) for m in PANEL_METRICS},
        "us": {m: values[m][0] for m in PANEL_METRICS},
    }


def intervals_frame(forecaster: str, arms: ForecasterArms) -> pd.DataFrame:
    """Long-format node x week x arm intervals for one forecaster."""
    n_nodes, n_weeks = len(arms.nodes), len(arms.weeks)
    nodes_col = np.repeat(arms.nodes, n_weeks)
    weeks_col = np.tile(arms.weeks, n_nodes)
    frames = []
    for (reconciler, controller, lr), (lo, hi) in arms.intervals.items():
        q = arms.q[reconciler]
        frames.append(
            pd.DataFrame(
                {
                    "forecaster": forecaster,
                    "node": nodes_col,
                    "week": weeks_col,
                    "reconciler": reconciler,
                    "controller": controller,
                    "lr": lr,
                    "y": arms.y.reshape(-1),
                    "q_lo": q[:, :, 0].reshape(-1),
                    "q_med": q[:, :, 1].reshape(-1),
                    "q_hi": q[:, :, 2].reshape(-1),
                    "lo": lo.reshape(-1),
                    "hi": hi.reshape(-1),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def metrics_frame(
    forecaster: str, arms: ForecasterArms, *, alpha: float, window: int
) -> pd.DataFrame:
    """Per-node metrics for every arm of one forecaster.

    Args:
        forecaster: Forecaster name, stamped into the ``forecaster`` column.
        arms: The forecaster's reconciled, conformalized arms.
        alpha: Target miscoverage rate.
        window: Rolling-coverage window in weeks.
    """
    scale = node_scale(arms.y)
    columns = (*METRIC_NAMES, "scaled_mean_width", "scaled_interval_score")
    rows = []
    for (reconciler, controller, lr), (lo, hi) in arms.intervals.items():
        values = node_metrics(arms.y, lo, hi, scale, alpha=alpha, window=window)
        for i, node in enumerate(arms.nodes):
            rows.append(
                {
                    "forecaster": forecaster,
                    "node": node,
                    "reconciler": reconciler,
                    "controller": controller,
                    "lr": lr,
                    **{m: float(values[m][i]) for m in columns},
                }
            )
    return pd.DataFrame(rows)


def level_metrics_frame(
    forecaster: str, arms: ForecasterArms, *, alpha: float, window: int
) -> pd.DataFrame:
    """Per-level metrics for every arm of one forecaster.

    Args:
        forecaster: Forecaster name, stamped into the ``forecaster`` column.
        arms: The forecaster's reconciled, conformalized arms.
        alpha: Target miscoverage rate.
        window: Rolling-coverage window in weeks.
    """
    scale = node_scale(arms.y)
    rows = []
    for (reconciler, controller, lr), (lo, hi) in arms.intervals.items():
        levels = level_values(
            node_metrics(arms.y, lo, hi, scale, alpha=alpha, window=window),
            arms.endpoint,
        )
        for level in LEVELS:
            rows.append(
                {
                    "forecaster": forecaster,
                    "level": level,
                    "reconciler": reconciler,
                    "controller": controller,
                    "lr": lr,
                    **{m: float(levels[level][m]) for m in PANEL_METRICS},
                }
            )
    return pd.DataFrame(rows)
