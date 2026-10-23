# ABOUTME: Contract tests for four-arm v2 benchmark figures.
# ABOUTME: Checks visible arms, empty sets, inferential labels, and the in-memory contract.
from pathlib import Path

import pandas as pd
import pytest
from matplotlib import pyplot as plt

from hcp.benchmark import BenchmarkResult
from hcp.benchmark_report import benchmark_figures


@pytest.fixture
def result():
    levels = []
    nodes = []
    for model in ("arima", "covidhub"):
        for level in ("leaf_mean", "us"):
            for reconciler in ("none", "mint_median", "bottom_up"):
                for controller in ("raw", "pi_norm", "pi"):
                    levels.append(
                        {
                            "model": model,
                            "level": level,
                            "reconciler": reconciler,
                            "controller": controller,
                            "lr": 0.1,
                            "marginal_coverage_error": 0.1,
                            "scaled_interval_score": 2.0,
                            "mean_set_width": 4.0,
                            "scaled_mean_set_width": 0.4,
                            "empty_set_rate": 0.25,
                            "mean_width": -1.0,
                        }
                    )
        for node, scale in (("US", 1000.0), ("CA", 12.0)):
            for reconciler in ("none", "mint_median", "bottom_up"):
                for controller in ("pi", "pi_norm"):
                    nodes.append(
                        {
                            "model": model,
                            "node": node,
                            "scale": scale,
                            "reconciler": reconciler,
                            "controller": controller,
                            "empty_set_rate": 0.25,
                            "mean_set_width": 4.0,
                            "mean_width": -1.0,
                            "scaled_mean_width": -0.1,
                        }
                    )
    contrasts = pd.DataFrame(
        [
            {
                "block": 8,
                "model": "arima",
                "metric": metric,
                "contrast": contrast,
                "estimate": 0.1,
                "low": -0.1,
                "high": 0.2,
                "p_value": 0.001,
                "p_holm": 0.12,
                "family_size": 96,
            }
            for metric in ("marginal_coverage_error", "scaled_interval_score")
            for contrast in (
                "pi_effect",
                "reconciliation_effect",
                "combined_effect",
                "reconciliation_given_pi",
                "pi_given_reconciliation",
                "interaction",
            )
        ]
    )
    empty = pd.DataFrame()
    return BenchmarkResult(
        empty,
        empty,
        pd.DataFrame(nodes),
        pd.DataFrame(levels),
        contrasts,
        empty,
        {
            "protocol_version": 2,
            "model_order": ["arima", "covidhub"],
            "pi_scales": {"US": 1000.0, "CA": 12.0},
        },
    )


def _labels(fig):
    labels = [fig._suptitle.get_text()] if fig._suptitle is not None else []
    for axes in fig.axes:
        labels.extend([axes.get_title(), axes.get_xlabel(), axes.get_ylabel()])
        labels.extend(text.get_text() for text in axes.texts)
        legend = axes.get_legend()
        if legend is not None:
            labels.extend(text.get_text() for text in legend.get_texts())
    return " ".join(labels)


def test_v2_report_exposes_all_arms_and_diagnostic_semantics(
    result, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    figures = benchmark_figures(result, block=8, lr=0.1)
    assert set(figures) == {
        "model-quality",
        "factorial-effects",
        "coverage-score-tradeoff",
        "pi-scale-diagnostics",
        "us-descriptive",
    }
    all_labels = " ".join(map(_labels, figures.values())).lower()
    for arm in ("a", "b", "c", "d", "raw", "normalized pi"):
        assert arm in all_labels
    assert "empty-set rate" in all_labels
    assert "physical" in all_labels and "signed span" in all_labels
    assert "pointwise" in all_labels and "holm" in all_labels
    assert "descriptive" in _labels(figures["us-descriptive"]).lower()
    assert "ordinary width" not in all_labels
    width_plot = figures["coverage-score-tradeoff"].axes[0]
    assert all(
        (collection.get_offsets()[:, 1] >= 0).all()
        for collection in width_plot.collections
    )
    assert list(Path(tmp_path).iterdir()) == []
    for figure in figures.values():
        plt.close(figure)


def test_v2_factorial_uses_holm_values_and_does_not_clip_intervals(result):
    figures = benchmark_figures(result, block=8, lr=0.1)
    forest = figures["factorial-effects"].axes[0]
    assert any("0.12" in text.get_text() for text in forest.texts)
    segments = [
        segment
        for collection in forest.collections
        if hasattr(collection, "get_segments")
        for segment in collection.get_segments()
    ]
    assert any(
        abs(segment[0, 0] + 0.1) < 1e-8 and abs(segment[-1, 0] - 0.2) < 1e-8
        for segment in segments
    )
    for figure in figures.values():
        plt.close(figure)
