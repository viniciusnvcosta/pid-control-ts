# ABOUTME: Normalizes forecast event diagnostics and deduplicates warning occurrences.
# ABOUTME: Preserves warning provenance and stable empty schemas without writing files.
"""Forecast diagnostic table helpers."""

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
    return pd.DataFrame(rows, columns=columns)
