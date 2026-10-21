# ABOUTME: Unit tests for pure in-memory benchmark figure construction.
# ABOUTME: Covers figure inventory, labels, clipped asymmetric errors, and the no-I/O contract.
from pathlib import Path

import pandas as pd

from hcp.benchmark import BenchmarkResult
from hcp.benchmark_report import benchmark_figures

MODELS = ("arima", "sarima", "deepar", "covidhub")


def _result() -> BenchmarkResult:
    hierarchy = pd.DataFrame(
        [
            {
                "block": 8,
                "model": model,
                "level": level,
                "controller": controller,
                "lr": 0.0 if controller == "raw" else 0.1,
                "treatment": treatment,
                "control": "none",
                "metric": "coverage_deviation",
                "estimate": 0.1 if model == "arima" else -0.1,
                "low": 0.2 if model == "arima" else -0.3,
                "high": 0.3 if model == "arima" else -0.2,
                "p_value": 0.2,
                "primary": False,
                "p_holm": 0.8,
            }
            for model in MODELS
            for level in ("leaf_mean", "us")
            for controller in ("raw", "pi")
            for treatment in ("bottom_up", "mint_shrink", "mint_median")
        ]
    )
    level_metrics = pd.DataFrame(
        [
            {
                "model": model,
                "level": level,
                "reconciler": reconciler,
                "controller": "pi",
                "lr": 0.1,
                "coverage_deviation": 0.02 + 0.01 * model_index,
                "scaled_interval_score": 1.0
                + 0.1 * model_index
                + (0.05 if reconciler == "mint_shrink" else 0.0),
            }
            for model_index, model in enumerate(MODELS)
            for level in ("leaf_mean", "us")
            for reconciler in ("none", "mint_shrink")
        ]
    )
    empty = pd.DataFrame()
    return BenchmarkResult(
        base_forecasts=empty,
        intervals=empty,
        metrics=empty,
        level_metrics=level_metrics,
        hierarchy_contrasts=hierarchy,
        model_contrasts=empty,
        diagnostics={"model_order": list(MODELS)},
    )


def _all_labels(figure) -> set[str]:
    labels = set()
    for axes in figure.axes:
        labels.update(text.get_text() for text in axes.texts)
        labels.update(text.get_text() for text in axes.get_xticklabels())
        labels.update(text.get_text() for text in axes.get_yticklabels())
        if axes.get_legend() is not None:
            labels.update(text.get_text() for text in axes.get_legend().get_texts())
    return labels


def test_benchmark_figures_are_complete_labeled_and_do_no_io(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)

    figures = benchmark_figures(_result(), block=8, lr=0.1)

    assert set(figures) == {
        "forest-leaf-mean-pi",
        "forest-us-pi",
        "coverage-score-tradeoff",
        "hierarchy-effect",
    }
    assert all(figure.axes for figure in figures.values())
    labels = set().union(*(_all_labels(figure) for figure in figures.values()))
    assert "DeepAR-style (Darts LSTM)" in labels
    assert "COVIDhub (external reference)" in labels
    assert {"none", "mint_shrink", "bottom_up", "mint_median"} <= labels
    assert list(tmp_path.iterdir()) == []


def test_benchmark_figures_clip_ci_errors_that_exclude_estimates() -> None:
    figures = benchmark_figures(_result(), block=8, lr=0.1)

    for figure in figures.values():
        figure.canvas.draw()
