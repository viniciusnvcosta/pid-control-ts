# ABOUTME: Test helpers: a synthetic forecast CSV in the real deaths.csv schema, a TOML writer, real-data paths.
# ABOUTME: The synthetic hierarchy is us + aa/bb/cc with a late leaf, a missing leaf week and truthless US weeks.
from pathlib import Path

import numpy as np
import pandas as pd

from hcp.data import FORECASTER, QUANTILE_COLUMNS, QUANTILE_LEVELS

ROOT_DIR = Path(__file__).resolve().parents[1]
REAL_DATA = ROOT_DIR / "data" / "deaths.csv"
REAL_SHA256 = "cb4dccfd21d6c08b1247aa9a7c6a8d93f7c373871526cfcb4860df884b54ac06"
PICKLES = ROOT_DIR / "tests" / "datasets" / "covid-ts-proc" / "statewide"
PICKLE_STATES = ("ak", "ca", "fl", "ga", "ks", "ny", "tx")

START = pd.Timestamp("2021-01-02")
N_WEEKS = 30
LEAVES = ("aa", "bb", "cc")
LATE_LEAF, LATE_WEEKS = "bb", 2  # no bb rows in weeks 0-1, so both weeks are trimmed
GAP_LEAF, GAP_WEEK = "cc", 10  # no cc row in week 10, so LOCF fills it
TRUTHLESS_US_WEEKS = 2  # last two weeks lack US truth, so they leave the grid


def synthetic_rows(seed: int = 0) -> list[dict]:
    """Rows in the deaths.csv schema, plus decoy forecaster and ahead=1 rows."""
    rng = np.random.default_rng(seed)
    rows = []
    for week in range(N_WEEKS):
        target = START + pd.Timedelta(weeks=week)
        leaf_truth = {
            leaf: float(rng.poisson(50 * (i + 1))) for i, leaf in enumerate(LEAVES)
        }
        truth = {"us": sum(leaf_truth.values()), **leaf_truth}
        for geo, value in truth.items():
            if geo == LATE_LEAF and week < LATE_WEEKS:
                continue
            if geo == GAP_LEAF and week == GAP_WEEK:
                continue
            median = value * float(rng.uniform(0.8, 1.2))
            truthless = geo == "us" and week >= N_WEEKS - TRUTHLESS_US_WEEKS
            row = {
                "ahead": 4,
                "geo_value": geo,
                "forecaster": FORECASTER,
                "forecast_date": (target - pd.Timedelta(days=26)).date(),
                "target_end_date": target.date(),
                "forecast_0.25": median,
                "actual": np.nan if truthless else value,
            }
            for level, column in zip(QUANTILE_LEVELS, QUANTILE_COLUMNS, strict=True):
                row[column] = median * (1 + 1.5 * (level - 0.5))
            rows.append(row)
            rows.append(
                {
                    **row,
                    "forecaster": "COVIDhub-baseline",
                    **dict.fromkeys(QUANTILE_COLUMNS, -1.0),
                }
            )
            rows.append({**row, "ahead": 1, **dict.fromkeys(QUANTILE_COLUMNS, -2.0)})
    return rows


def _toml_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return repr(value)
    if isinstance(value, list | tuple):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    return '"' + str(value) + '"'


def write_config(path: Path, values: dict) -> Path:
    """Write a flat TOML file from ``values``."""
    lines = [f"{key} = {_toml_value(value)}" for key, value in values.items()]
    path.write_text("\n".join(lines) + "\n")
    return path
