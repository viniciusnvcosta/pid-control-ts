# ABOUTME: Validates forecast grids and prepares causal node histories and bundles.
# ABOUTME: Contains numeric in-memory helpers shared by the forecast orchestrator.
"""Causal history preparation and forecast bundle helpers."""

import numpy as np
import pandas as pd

from hcp.benchmark_data import ActualPanel
from hcp.data import HierarchyBundle
from hcp.forecast_models import PreparedHistory


def _weekly(name: str, values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype="datetime64[ns]")
    if array.ndim != 1 or len(array) == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional date grid")
    if len(array) > 1 and not np.all(np.diff(array) == np.timedelta64(7, "D")):
        raise ValueError(f"{name} must be a strict weekly grid")
    return array


def _validate_inputs(
    actuals: ActualPanel, dates: np.ndarray, forecast_dates: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    actual_dates = _weekly("canonical actual dates", actuals.dates)
    targets = _weekly("target dates", dates)
    origins = _weekly("forecast dates", forecast_dates)
    if origins.shape != targets.shape:
        raise ValueError("forecast-date and target-date grids must have the same shape")
    if np.any(targets - origins != np.timedelta64(26, "D")):
        raise ValueError("benchmark forecasts require the registered 26-day lag")
    expected = (len(actuals.nodes), len(actual_dates))
    if actuals.y.shape != expected or actuals.observed.shape != expected:
        raise ValueError(f"canonical node/date shape must be {expected}")
    target_index = pd.Index(actual_dates).get_indexer(pd.Index(targets))
    if np.any(target_index < 0):
        raise ValueError("target date is unavailable in canonical actuals")
    return targets, origins, target_index


def _prepare_origin(
    actuals: ActualPanel,
    forecast_date: np.datetime64,
    min_train_weeks: int,
) -> tuple[tuple[PreparedHistory, ...], list[dict]] | None:
    prefix = actuals.dates <= forecast_date
    histories = []
    records = []
    for row, node in enumerate(actuals.nodes):
        dates = np.asarray(actuals.dates[prefix], dtype="datetime64[ns]")
        values = np.asarray(actuals.y[row, prefix], dtype=float)
        finite = np.flatnonzero(np.isfinite(values))
        if not len(finite):
            return None
        start = int(finite[0])
        dates = dates[start:].copy()
        values = values[start:].copy()
        raw = int(np.isfinite(values).sum())
        missing = ~np.isfinite(values)
        filled = pd.Series(values).interpolate(limit_area="inside").to_numpy()
        interpolated = int(np.sum(missing & np.isfinite(filled)))
        if len(filled) < min_train_weeks or not np.isfinite(filled).all():
            return None
        dates.setflags(write=False)
        filled.setflags(write=False)
        histories.append(
            PreparedHistory(
                node=node,
                dates=dates,
                values=filled,
                raw_values=values,
                observed=actuals.observed[row, prefix][start:],
                forecast_dates=dates - np.timedelta64(26, "D"),
            )
        )
        records.append(
            {
                "node": node,
                "training_start": str(pd.Timestamp(dates[0]).date()),
                "training_end": str(pd.Timestamp(dates[-1]).date()),
                "raw_observations": raw,
                "interpolated_observations": interpolated,
            }
        )
    return tuple(histories), records


def _bundle(
    actuals: ActualPanel,
    dates: np.ndarray,
    forecast_dates: np.ndarray,
    target_index: np.ndarray,
    quantiles: np.ndarray,
) -> HierarchyBundle:
    arrays = {
        "dates": dates.copy(),
        "forecast_dates": forecast_dates.copy(),
        "y": actuals.y[:, target_index].copy(),
        "q": quantiles,
        "imputed": np.zeros(quantiles.shape[:2], dtype=bool),
    }
    for array in arrays.values():
        array.setflags(write=False)
    return HierarchyBundle(nodes=actuals.nodes, **arrays)


def _long_forecasts(
    model: str, bundle: HierarchyBundle, source_kind: str | None = None
) -> pd.DataFrame:
    rows = []
    for week, forecast_date, q_week in zip(
        bundle.dates, bundle.forecast_dates, bundle.q.transpose(1, 0, 2), strict=True
    ):
        for node, values in zip(bundle.nodes, q_week, strict=True):
            row = {
                "model": model,
                "node": node,
                "week": week,
                "forecast_date": forecast_date,
                "q_lo": values[0],
                "q_med": values[1],
                "q_hi": values[2],
            }
            if source_kind is not None:
                row["source_kind"] = source_kind
            rows.append(row)
    return pd.DataFrame(rows)
