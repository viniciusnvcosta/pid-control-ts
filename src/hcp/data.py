# ABOUTME: Loads COVIDhub 4-week-ahead death forecasts into a US -> 56-leaf hierarchy bundle.
# ABOUTME: Verifies the input SHA-256, carries missing leaf quantiles forward causally, never imputes truth.
"""Data stage (decisions D1, D5, D6).

Rows are the root ``us`` followed by the leaves in alphabetical order. The week grid is
the set of target weeks where the US forecast and truth are complete. Leading weeks in
which some leaf has never been forecast are trimmed, and later leaf gaps are filled by
last observation carried forward (LOCF) and flagged in ``imputed``.
"""

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = "us"
FORECASTER = "COVIDhub-4_week_ensemble"
AHEAD = 4
QUANTILE_LEVELS = (0.1, 0.5, 0.9)
QUANTILE_COLUMNS = tuple(f"forecast_{level}" for level in QUANTILE_LEVELS)
_COLUMNS = [
    "ahead",
    "geo_value",
    "forecaster",
    "forecast_date",
    "target_end_date",
    *QUANTILE_COLUMNS,
    "actual",
]


@dataclass(frozen=True)
class HierarchyBundle:
    """Aligned truth and base quantiles for the root and its leaves.

    Attributes:
        dates: Target end dates ``[T]``.
        forecast_dates: Earliest forecast date among the nodes for each target ``[T]``.
        nodes: ``ROOT`` then the leaves, sorted.
        y: Reported truth ``[n_nodes, T]``; NaN where not reported.
        q: Base quantiles ``[n_nodes, T, 3]`` at ``QUANTILE_LEVELS`` after LOCF.
        imputed: ``[n_nodes, T]``; True where ``q`` was carried forward.
    """

    dates: np.ndarray
    forecast_dates: np.ndarray
    nodes: tuple[str, ...]
    y: np.ndarray
    q: np.ndarray
    imputed: np.ndarray


def file_sha256(path: Path) -> str:
    """Return the hex SHA-256 of a file, read in 1 MiB chunks."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_hierarchy(path: Path, expected_sha256: str | None = None) -> HierarchyBundle:
    """Load the ensemble's 4-week-ahead forecasts for ``us`` and every leaf.

    Args:
        path: The deaths.csv file (paper's LFS input schema).
        expected_sha256: If given, the file's SHA-256 must match.

    Raises:
        ValueError: On a SHA-256 mismatch, duplicate rows, a missing root, or no usable week.
    """
    path = Path(path)
    if expected_sha256 is not None:
        digest = file_sha256(path)
        if digest != expected_sha256:
            raise ValueError(f"{path} has sha256 {digest}, expected {expected_sha256}")
    frame = pd.read_csv(
        path, usecols=_COLUMNS, parse_dates=["forecast_date", "target_end_date"]
    )
    frame = frame[(frame["ahead"] == AHEAD) & (frame["forecaster"] == FORECASTER)]
    if frame.duplicated(["geo_value", "target_end_date"]).any():
        raise ValueError("duplicate (geo_value, target_end_date) rows")
    if ROOT not in set(frame["geo_value"]):
        raise ValueError(f"root series {ROOT!r} is missing")
    nodes = (ROOT, *sorted(set(frame["geo_value"]) - {ROOT}))
    root = frame[frame["geo_value"] == ROOT].dropna(
        subset=[*QUANTILE_COLUMNS, "actual"]
    )
    dates = pd.DatetimeIndex(sorted(root["target_end_date"]))

    def wide(column: str) -> np.ndarray:
        table = frame.pivot(index="target_end_date", columns="geo_value", values=column)
        return table.reindex(index=dates, columns=list(nodes)).to_numpy(dtype=float).T

    raw_q = np.stack([wide(column) for column in QUANTILE_COLUMNS], axis=-1)
    filled = np.stack(
        [
            pd.DataFrame(raw_q[:, :, k].T).ffill().to_numpy().T
            for k in range(raw_q.shape[-1])
        ],
        axis=-1,
    )
    complete = np.isfinite(filled).all(axis=(0, 2))
    if not complete.any():
        raise ValueError("no week has forecasts for every node")
    start = int(np.argmax(complete))
    imputed = np.isnan(raw_q).any(axis=-1) & np.isfinite(filled).all(axis=-1)
    forecast_dates = (
        frame.groupby("target_end_date")["forecast_date"].min().reindex(dates)
    )
    arrays = {
        "dates": dates.to_numpy()[start:],
        "forecast_dates": forecast_dates.to_numpy()[start:],
        "y": wide("actual")[:, start:],
        "q": filled[:, start:],
        "imputed": imputed[:, start:],
    }
    for array in arrays.values():
        array.setflags(write=False)
    return HierarchyBundle(nodes=nodes, **arrays)
