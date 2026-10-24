# ABOUTME: Normalizes forecast event diagnostics and deduplicates warning occurrences.
# ABOUTME: Preserves warning provenance and stable empty schemas without writing files.
"""Forecast diagnostic table helpers."""

import json
from collections.abc import Mapping

import numpy as np
import pandas as pd


def _diagnostic_frame(name: str, rows: list[dict], empty_columns) -> pd.DataFrame:
    """Build an owned event table, retaining adapter-specific evidence fields."""
    if name == "warnings":
        deduplicated = {}
        for row in rows:
            normalized = {
                "category": row.get("category"),
                "message": row.get("message"),
                "source": row.get("source")
                or f"{row.get('filename', '')}:{row.get('lineno', '')}",
                "model": row.get("model"),
                "node": row.get("node"),
                "origin": row.get("origin", row.get("forecast_date")),
                "candidate": row.get("candidate"),
                "phase": row.get("phase"),
                "count": int(row.get("count", 1)),
            }
            key = tuple(
                normalized[column] for column in empty_columns if column != "count"
            )
            if key in deduplicated:
                deduplicated[key]["count"] += normalized["count"]
            else:
                deduplicated[key] = normalized
        rows = list(deduplicated.values())
    if not rows:
        return pd.DataFrame(columns=empty_columns)
    columns = list(
        dict.fromkeys([*empty_columns, *(key for row in rows for key in row)])
    )
    return pd.DataFrame(
        [
            {
                key: json.dumps(
                    _json_diagnostic(value), sort_keys=True, allow_nan=False
                )
                if isinstance(value, Mapping)
                else _json_diagnostic(value)
                for key, value in row.items()
            }
            for row in rows
        ],
        columns=columns,
    )


def _json_diagnostic(value: object) -> object:
    """Convert frozen adapter evidence to lossless JSON-shaped native values."""
    if isinstance(value, Mapping):
        return {str(key): _json_diagnostic(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return _json_diagnostic(value.tolist())
    if isinstance(value, (list, tuple)):
        return [_json_diagnostic(item) for item in value]
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def adapter_diagnostic_rows(
    diagnostics: Mapping, model: str, origin: pd.Timestamp
) -> dict[str, list[dict]]:
    """Normalize adapter evidence into canonical event and candidate records.

    Scalar candidate scores remain numeric; every QR alpha gets its own row.
    Arbitrary nested provenance is canonical JSON in event ``value`` cells,
    preserving stream addresses and separate inner/full-prefix preprocessing.

    Args:
        diagnostics: Immutable numeric evidence returned by a native adapter.
        model: Canonical registry key assigned to these records.
        origin: Outer forecast issuance date used for absent row identifiers.

    Returns:
        Records for the four canonical diagnostic tables, ready for Arrow-safe
        conversion with ``_diagnostic_frame``.
    """
    names = ("forecast_diagnostics", "tuning", "quantile_repairs", "warnings")
    rows = {name: [] for name in names}
    tuning = list(diagnostics.get("tuning", ()))
    best = (
        min(range(len(tuning)), key=lambda i: tuning[i].get("best_nll", np.inf))
        if tuning
        else None
    )
    for name in names:
        for index, original in enumerate(diagnostics.get(name, ())):
            row = {
                "model": model,
                "origin": origin,
                "forecast_date": origin,
                **original,
            }
            if name == "tuning" and "candidate_losses" in row:
                losses = row.pop("candidate_losses")
                for alpha, score in losses.items():
                    rows[name].append(
                        {
                            **row,
                            "candidate": str(alpha),
                            "selected": alpha == row["chosen_alpha"],
                            "score": score,
                        }
                    )
            elif name == "tuning":
                rows[name].append(
                    {
                        **row,
                        "candidate": str(row.get("width", row.get("candidate", index))),
                        "selected": index == best,
                        "score": row.get("best_nll", row.get("score")),
                    }
                )
            else:
                if name == "quantile_repairs":
                    row.setdefault("count", 1)
                rows[name].append(row)
    for name, value in diagnostics.items():
        if name in names or name == "warning_counts":
            continue
        rows["forecast_diagnostics"].append(
            {
                "model": model,
                "node": None,
                "forecast_date": origin,
                "origin": origin,
                "phase": "full_refit" if name == "final_fit" else "adapter",
                "name": name,
                "value": json.dumps(
                    _json_diagnostic(value),
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ),
            }
        )
    rows["forecast_diagnostics"].append(
        {
            "model": model,
            "node": None,
            "forecast_date": origin,
            "origin": origin,
            "phase": "outer_fit",
            "name": "adapter_completed",
            "value": "true",
        }
    )
    return rows


def diagnostic_records(frame: pd.DataFrame) -> list[dict]:
    """Return structured manifest records, using null for absent schema fields.

    Args:
        frame: Canonical diagnostic table, possibly combining adapter schemas.

    Returns:
        Native JSON-shaped records retaining numeric arrays as lists.
    """
    return [
        _json_diagnostic(row)
        for row in frame.astype(object).where(frame.notna(), None).to_dict("records")
    ]
