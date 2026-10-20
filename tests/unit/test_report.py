# ABOUTME: Unit tests for the report stage: every expected PDF is written and non-empty.
# ABOUTME: Covers Phase 1 run figures and the Phase 2 leaves x forecasters panel figures.
from pathlib import Path

import numpy as np
import pandas as pd

from hcp.report import (
    leaf_contrast_figure,
    panel_forest_figure,
    render_figures,
    render_panel_figures,
)


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


def _panel_metrics() -> pd.DataFrame:
    nodes = ("us", "aa", "bb", "pr")
    reconcilers = ("none", "bottom_up", "mint_shrink", "mint_median")
    rows = [
        {
            "forecaster": forecaster,
            "node": node,
            "reconciler": reconciler,
            "controller": controller,
            "lr": lr,
            "coverage_deviation": 0.1 if reconciler == "none" else 0.05,
        }
        for forecaster in ("fc_primary", "fc_other")
        for node in nodes
        for reconciler in reconcilers
        for controller, lr in (("raw", 0.0), ("pi", 0.1))
    ]
    return pd.DataFrame(rows)


def _panel_contrasts() -> pd.DataFrame:
    rows = [
        {
            "block": 8,
            "forecaster": forecaster,
            "level": level,
            "controller": controller,
            "lr": lr,
            "treatment": treatment,
            "control": "none",
            "metric": "coverage_deviation",
            "estimate": -0.02,
            "low": -0.05,
            "high": 0.01,
            "p_value": 0.3,
            "primary": False,
            "p_holm": 0.6,
        }
        for forecaster in ("fc_primary", "fc_other")
        for level in ("leaf_mean", "us")
        for controller, lr in (("raw", 0.0), ("pi", 0.1))
        for treatment in ("mint_shrink", "bottom_up", "mint_median")
    ]
    return pd.DataFrame(rows)


def test_render_panel_figures_writes_every_pdf(tmp_path: Path) -> None:
    paths = render_panel_figures(
        _panel_metrics(),
        _panel_contrasts(),
        tmp_path / "panel_figures",
        lr=0.1,
        block=8,
        primary_forecaster="fc_primary",
    )
    names = sorted(path.name for path in paths)
    assert names == sorted(
        [
            "panel_forest_leaf_mean_raw.pdf",
            "panel_forest_leaf_mean_pi.pdf",
            "panel_forest_us_raw.pdf",
            "panel_forest_us_pi.pdf",
            "leaf_contrasts_raw.pdf",
            "leaf_contrasts_pi.pdf",
        ]
    )
    assert all(path.stat().st_size > 0 for path in paths)


def test_leaf_contrast_figure_excludes_us_and_territories() -> None:
    figure = leaf_contrast_figure(
        _panel_metrics(), forecaster="fc_primary", controller="pi", lr=0.1
    )
    axes = figure.axes[0]
    labels = {label.get_text() for label in axes.get_yticklabels()}
    assert labels == {"aa", "bb"}


def test_panel_forest_figure_bolds_primary_forecaster() -> None:
    figure = panel_forest_figure(
        _panel_contrasts(),
        level="leaf_mean",
        controller="pi",
        lr=0.1,
        block=8,
        primary_forecaster="fc_primary",
    )
    axes = figure.axes[0]
    bold = {
        label.get_text()
        for label in axes.get_yticklabels()
        if label.get_fontweight() in ("bold", 700)
    }
    assert bold == {"fc_primary"}
