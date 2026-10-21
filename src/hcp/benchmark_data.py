# ABOUTME: Canonicalizes benchmark truth across forecast submissions and aligns reference forecasts.
# ABOUTME: Enforces exact duplicate agreement, weekly grids, and immutable numeric input bundles.
"""Benchmark data boundaries for canonical truth and external reference forecasts."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from hcp.data import ROOT, HierarchyBundle, build_bundle


@dataclass(frozen=True)
class ActualPanel:
    """Canonical weekly truth for the complete hierarchy.

    Attributes:
        dates: Sorted weekly target dates with shape ``[T]``.
        nodes: Root followed by alphabetically sorted leaves.
        y: Canonical actual values with shape ``[n_nodes, T]``.
        observed: Whether at least one finite actual was reported for each cell.
    """

    dates: np.ndarray
    nodes: tuple[str, ...]
    y: np.ndarray
    observed: np.ndarray


def canonical_actuals(frame: pd.DataFrame) -> ActualPanel:
    """Collapse reported actuals into one exact, immutable hierarchy panel.

    Args:
        frame: Forecast rows containing node, target date, and actual columns.

    Returns:
        Root-first canonical truth on a strict seven-day grid.

    Raises:
        ValueError: If the root is absent, dates are not weekly, or finite duplicate
            actuals disagree for a node and target week.
    """
    nodes_present = set(frame["geo_value"])
    if ROOT not in nodes_present:
        raise ValueError(f"root series {ROOT!r} is missing")
    nodes = (ROOT, *sorted(nodes_present - {ROOT}))
    dates = pd.DatetimeIndex(
        pd.to_datetime(frame["target_end_date"]).unique()
    ).sort_values()
    if len(dates) > 1 and not np.all(
        np.diff(dates.to_numpy()) == np.timedelta64(7, "D")
    ):
        raise ValueError(
            "canonical actual dates must form a strict seven-day weekly grid"
        )

    y = np.full((len(nodes), len(dates)), np.nan, dtype=float)
    observed = np.zeros_like(y, dtype=bool)
    node_index = {node: index for index, node in enumerate(nodes)}
    date_index = {date: index for index, date in enumerate(dates)}
    grouped = frame.assign(
        target_end_date=pd.to_datetime(frame["target_end_date"])
    ).groupby(["geo_value", "target_end_date"], sort=False, dropna=False)
    for (node, date), rows in grouped:
        finite = rows.loc[np.isfinite(rows["actual"]), "actual"].to_numpy(dtype=float)
        unique = np.unique(finite)
        if len(unique) > 1:
            raise ValueError(
                f"finite actuals disagree for ({node}, {pd.Timestamp(date).date()})"
            )
        if len(unique) == 1:
            row = node_index[str(node)]
            column = date_index[pd.Timestamp(date)]
            y[row, column] = unique[0]
            observed[row, column] = True

    date_values = dates.to_numpy()
    for array in (date_values, y, observed):
        array.setflags(write=False)
    return ActualPanel(dates=date_values, nodes=nodes, y=y, observed=observed)


def reference_bundle(
    frame: pd.DataFrame, actuals: ActualPanel, forecaster: str
) -> HierarchyBundle:
    """Build a submitted-forecast bundle aligned to canonical truth.

    Args:
        frame: Forecast rows containing the selected forecaster.
        actuals: Canonical truth across all forecast submissions.
        forecaster: Submitted forecast name to use for base quantiles.

    Returns:
        The submitted forecast grid with its truth replaced by canonical values.

    Raises:
        ValueError: If the canonical and submitted node sets differ or a submitted
            target date is absent from the canonical panel.
    """
    submitted = build_bundle(frame, forecaster)
    if submitted.nodes != actuals.nodes:
        raise ValueError(
            f"reference nodes {submitted.nodes!r} do not match canonical nodes "
            f"{actuals.nodes!r}"
        )
    canonical_dates = {
        pd.Timestamp(date): index for index, date in enumerate(actuals.dates)
    }
    missing = [
        pd.Timestamp(date)
        for date in submitted.dates
        if pd.Timestamp(date) not in canonical_dates
    ]
    if missing:
        raise ValueError(
            f"reference target date is unavailable in canonical truth: {missing[0]}"
        )
    columns = [canonical_dates[pd.Timestamp(date)] for date in submitted.dates]
    arrays = {
        "dates": submitted.dates.copy(),
        "forecast_dates": submitted.forecast_dates.copy(),
        "y": actuals.y[:, columns].copy(),
        "q": submitted.q.copy(),
        "imputed": submitted.imputed.copy(),
    }
    for array in arrays.values():
        array.setflags(write=False)
    return HierarchyBundle(nodes=submitted.nodes, **arrays)
