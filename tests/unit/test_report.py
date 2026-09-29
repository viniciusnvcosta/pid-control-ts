# ABOUTME: Unit tests for the report stage: every expected PDF is written and non-empty.
# ABOUTME: Uses a small synthetic intervals/contrasts table in the run stage's column schema.
from pathlib import Path

import numpy as np
import pandas as pd

from hcp.report import render_figures


def _intervals() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    weeks = pd.date_range("2021-01-02", periods=30, freq="7D")
    frames = []
    for reconciler in ("none", "bottom_up", "mint_shrink"):
        for controller, lr in (("raw", 0.0), ("pi", 0.1)):
            y = rng.normal(size=30)
            frames.append(
                pd.DataFrame(
                    {
                        "week": weeks,
                        "reconciler": reconciler,
                        "controller": controller,
                        "lr": lr,
                        "y": y,
                        "q_lo": y - 1,
                        "q_med": y,
                        "q_hi": y + 1,
                        "lo": y - rng.uniform(0, 2, 30),
                        "hi": y + rng.uniform(0, 2, 30),
                    }
                )
            )
    return pd.concat(frames, ignore_index=True)


def _contrasts() -> pd.DataFrame:
    rows = [
        {
            "block": 8,
            "controller": controller,
            "lr": lr,
            "treatment": treatment,
            "control": "none",
            "metric": "coverage_deviation",
            "estimate": -0.01,
            "low": -0.05,
            "high": 0.03,
            "p_value": 0.4,
            "p_holm": 0.8,
            "primary": False,
        }
        for controller, lr in (("raw", 0.0), ("pi", 0.1))
        for treatment in ("bottom_up", "mint_shrink")
    ]
    return pd.DataFrame(rows)


def test_render_figures_writes_every_pdf(tmp_path: Path) -> None:
    paths = render_figures(
        _intervals(),
        _contrasts(),
        tmp_path / "figures",
        alpha=0.2,
        window=5,
        lr=0.1,
        block=8,
    )
    names = sorted(path.name for path in paths)
    assert names == [
        "contrasts_coverage_deviation.pdf",
        "intervals_pi.pdf",
        "intervals_raw.pdf",
        "rolling_coverage_pi.pdf",
        "rolling_coverage_raw.pdf",
    ]
    assert all(path.stat().st_size > 0 for path in paths)
