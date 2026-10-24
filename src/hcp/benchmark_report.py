# ABOUTME: Pure in-memory figures for probabilistic model hierarchy benchmark results.
# ABOUTME: Builds forests, trade-offs, and hierarchy effects; artifact writing remains in run.py.
"""Benchmark-specific matplotlib figures without file I/O."""

import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from hcp.benchmark import BenchmarkResult
from hcp.benchmark_report_v2 import v2_figures
from hcp.report import RECONCILER_COLORS

MODEL_LABELS = {
    "arima": "ARIMA",
    "sarima": "SARIMA",
    "deepar": "DeepAR-style (Darts LSTM)",
    "covidhub": "COVIDhub (external reference)",
}
TREATMENTS = ("bottom_up", "mint_shrink", "mint_median")


def _figure(width: float = 8.0, height: float = 4.0) -> Figure:
    figure = Figure(figsize=(width, height))
    FigureCanvasAgg(figure)
    return figure


def _model_label(model: str) -> str:
    return MODEL_LABELS.get(model, model)


def _clipped_error(record: pd.Series) -> list[list[float]]:
    estimate = float(record["estimate"])
    return [
        [max(estimate - float(record["low"]), 0.0)],
        [max(float(record["high"]) - estimate, 0.0)],
    ]


def _forest(result: BenchmarkResult, *, level: str, block: int, lr: float) -> Figure:
    rows = result.hierarchy_contrasts[
        (result.hierarchy_contrasts["block"] == block)
        & (result.hierarchy_contrasts["level"] == level)
        & (result.hierarchy_contrasts["controller"] == "pi")
        & (result.hierarchy_contrasts["lr"] == lr)
        & (result.hierarchy_contrasts["metric"] == "coverage_deviation")
    ]
    order = result.diagnostics["model_order"]
    figure = _figure(8.0, 1.0 + 0.32 * max(len(order) * len(TREATMENTS), 1))
    axes = figure.add_subplot()
    labels = []
    row = 0
    for model in order:
        for treatment in TREATMENTS:
            point = rows[(rows["model"] == model) & (rows["treatment"] == treatment)]
            if point.empty:
                continue
            record = point.iloc[0]
            axes.errorbar(
                [record["estimate"]],
                [row],
                xerr=_clipped_error(record),
                fmt="o",
                color=RECONCILER_COLORS[treatment],
                label=treatment,
                capsize=3,
            )
            labels.append(f"{_model_label(model)} · {treatment}")
            row += 1
    handles, legend_labels = axes.get_legend_handles_labels()
    unique = dict(zip(legend_labels, handles, strict=True))
    axes.legend(unique.values(), unique.keys(), frameon=False)
    axes.set_yticks(range(len(labels)), labels)
    axes.invert_yaxis()
    axes.axvline(0.0, color="#6b7280", linestyle="--", linewidth=1)
    axes.set_xlabel(
        f"Δ coverage deviation vs none (95% block-bootstrap CI, block={block})"
    )
    axes.set_title(f"Hierarchy effects · {level} · PI")
    axes.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    return figure


def _tradeoff(result: BenchmarkResult, *, lr: float) -> Figure:
    rows = result.level_metrics[
        (result.level_metrics["controller"] == "pi")
        & (result.level_metrics["lr"] == lr)
        & (result.level_metrics["reconciler"].isin(("none", "mint_shrink")))
    ]
    order = result.diagnostics["model_order"]
    figure = _figure(10.0, 4.0)
    axes_by_level = figure.subplots(1, 2)
    for axes, level in zip(axes_by_level, ("leaf_mean", "us"), strict=True):
        level_rows = rows[rows["level"] == level]
        for reconciler in ("none", "mint_shrink"):
            points = level_rows[level_rows["reconciler"] == reconciler].set_index(
                "model"
            )
            points = points.reindex(order).dropna(how="all")
            axes.scatter(
                points["coverage_deviation"],
                points["scaled_interval_score"],
                color=RECONCILER_COLORS[reconciler],
                label=reconciler,
            )
            for model, record in points.iterrows():
                axes.annotate(
                    _model_label(model),
                    (record["coverage_deviation"], record["scaled_interval_score"]),
                    xytext=(4, 3),
                    textcoords="offset points",
                    fontsize=7,
                )
        axes.set_title(level)
        axes.set_xlabel("Coverage deviation")
        axes.set_ylabel("Scaled interval score")
        axes.spines[["top", "right"]].set_visible(False)
        axes.legend(frameon=False)
    figure.suptitle("Coverage–score trade-off · PI")
    figure.tight_layout()
    return figure


def _hierarchy_effect(result: BenchmarkResult, *, block: int, lr: float) -> Figure:
    rows = result.hierarchy_contrasts[
        (result.hierarchy_contrasts["block"] == block)
        & (result.hierarchy_contrasts["level"] == "leaf_mean")
        & (result.hierarchy_contrasts["treatment"] == "mint_shrink")
        & (result.hierarchy_contrasts["metric"] == "coverage_deviation")
    ]
    order = result.diagnostics["model_order"]
    figure = _figure(8.0, 1.4 + 0.55 * len(order))
    axes = figure.add_subplot()
    positions = np.arange(len(order), dtype=float)
    for offset, (controller, arm_lr) in zip(
        (-0.12, 0.12), (("raw", 0.0), ("pi", lr)), strict=True
    ):
        selected = rows[
            (rows["controller"] == controller) & (rows["lr"] == arm_lr)
        ].set_index("model")
        selected = selected.reindex(order)
        left = np.maximum(selected["estimate"] - selected["low"], 0.0)
        right = np.maximum(selected["high"] - selected["estimate"], 0.0)
        axes.errorbar(
            selected["estimate"],
            positions + offset,
            xerr=[left, right],
            fmt="o",
            label=controller,
            capsize=3,
        )
    axes.set_yticks(positions, [_model_label(model) for model in order])
    axes.invert_yaxis()
    axes.axvline(0.0, color="#6b7280", linestyle="--", linewidth=1)
    axes.set_xlabel(f"mint_shrink − none coverage deviation (block={block})")
    axes.set_title("Leaf-mean hierarchy effect under raw and PI")
    axes.legend(frameon=False)
    axes.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    return figure


def benchmark_figures(
    result: BenchmarkResult, *, block: int, lr: float
) -> dict[str, Figure]:
    """Build the four registered benchmark figures without writing files.

    Args:
        result: Completed in-memory benchmark evaluation.
        block: Bootstrap block length selected for forest plots.
        lr: PI learning rate selected for reporting.

    Returns:
        Stable artifact stem to in-memory matplotlib figure.
    """
    if result.diagnostics.get("protocol_version", 1) == 2:
        return v2_figures(result, block=block)
    return {
        "forest-leaf-mean-pi": _forest(result, level="leaf_mean", block=block, lr=lr),
        "forest-us-pi": _forest(result, level="us", block=block, lr=lr),
        "coverage-score-tradeoff": _tradeoff(result, lr=lr),
        "hierarchy-effect": _hierarchy_effect(result, block=block, lr=lr),
    }
