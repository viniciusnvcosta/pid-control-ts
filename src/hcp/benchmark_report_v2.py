# ABOUTME: In-memory diagnostic figures for the four-arm benchmark protocol.
# ABOUTME: Makes empty sets, fixed PI scales, and exploratory inference visible.
"""V2 benchmark figures; artifact writing is owned by ``hcp.run``."""

import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from hcp.benchmark import BenchmarkResult

ARMS = {
    "A": ("none", "raw"),
    "B": ("none", "pi_norm"),
    "C": ("mint_median", "raw"),
    "D": ("mint_median", "pi_norm"),
}
CONTRASTS = (
    "pi_effect",
    "reconciliation_effect",
    "combined_effect",
    "reconciliation_given_pi",
    "pi_given_reconciliation",
    "interaction",
)


def _figure(width: float = 9, height: float = 4) -> Figure:
    """Create a canvas-backed figure without opening a GUI or writing files."""
    fig = Figure(figsize=(width, height))
    FigureCanvasAgg(fig)
    return fig


def _arm_data(result: BenchmarkResult, level: str) -> pd.DataFrame:
    data = result.level_metrics
    if data.empty:
        return data
    return data[(data["level"] == level) & data["controller"].isin(("raw", "pi_norm"))]


def _quality(result: BenchmarkResult) -> Figure:
    rows = _arm_data(result, "leaf_mean")
    fig = _figure(11, 5)
    axes = fig.subplots(1, 2)
    metrics = (
        ("marginal_coverage_error", "Absolute marginal coverage error"),
        ("scaled_interval_score", "Scaled interval score"),
    )
    models = result.diagnostics["model_order"]
    for ax, (metric, label) in zip(axes, metrics, strict=True):
        for arm, (reconciler, controller) in ARMS.items():
            part = rows[
                (rows["reconciler"] == reconciler) & (rows["controller"] == controller)
            ]
            part = part.set_index("model").reindex(models)
            ax.plot(
                np.arange(len(models)),
                part[metric],
                marker="o",
                linestyle="none",
                label=arm,
            )
        ax.set_xticks(np.arange(len(models)), models, rotation=30, ha="right")
        ax.set_ylabel(label)
        ax.set_title(f"{label} · four principal arms")
        ax.legend(title="Arm", frameon=False)
    fig.tight_layout()
    return fig


def _factorial(result: BenchmarkResult, block: int) -> Figure:
    rows = result.hierarchy_contrasts
    rows = rows[(rows["block"] == block) & rows["contrast"].isin(CONTRASTS)]
    rows = rows[
        rows["metric"].isin(("marginal_coverage_error", "scaled_interval_score"))
    ]
    fig = _figure(12, max(5, 0.35 * len(rows)))
    axes = fig.add_subplot()
    ys = np.arange(len(rows))
    axes.hlines(ys, rows["low"], rows["high"], color="#586174", linewidth=1.2)
    axes.plot(rows["estimate"], ys, "o", color="#111827", markersize=4)
    axes.set_yticks(
        ys,
        [
            f"{m} · {c} · {metric}"
            for m, c, metric in zip(
                rows["model"], rows["contrast"], rows["metric"], strict=True
            )
        ],
    )
    for y, (_, record) in enumerate(rows.iterrows()):
        axes.text(
            record["high"],
            y,
            f"  pointwise 95% CI; Holm p={record['p_holm']:.3g}",
            va="center",
            fontsize=7,
        )
    axes.invert_yaxis()
    axes.axvline(0, color="#666", linestyle="--", linewidth=1)
    axes.set_xlabel(f"Factorial contrast (block={block}); intervals are pointwise")
    axes.set_title("Four-arm factorial effects · Holm-adjusted significance")
    fig.tight_layout()
    return fig


def _tradeoff(result: BenchmarkResult) -> Figure:
    rows = _arm_data(result, "leaf_mean")
    fig = _figure(11, 5)
    axes = fig.subplots(1, 2)
    for ax, y, label in zip(
        axes,
        ("scaled_mean_set_width", "scaled_interval_score"),
        ("Scaled physical set width (nonnegative)", "Scaled interval score"),
        strict=True,
    ):
        for arm, (reconciler, controller) in ARMS.items():
            part = rows[
                (rows["reconciler"] == reconciler) & (rows["controller"] == controller)
            ]
            ax.scatter(part["marginal_coverage_error"], part[y], label=arm)
            for _, point in part.iterrows():
                ax.annotate(
                    f"{point['model']} · {arm}; empty-set rate={point['empty_set_rate']:.1%}",
                    (point["marginal_coverage_error"], point[y]),
                    fontsize=6,
                    xytext=(3, 2),
                    textcoords="offset points",
                )
        ax.set_xlabel("Absolute marginal coverage error")
        ax.set_ylabel(label)
        ax.legend(title="Arm", frameon=False)
    fig.suptitle("Coverage and score trade-off · empty-set rates shown")
    fig.tight_layout()
    return fig


def _pi_scale(result: BenchmarkResult) -> Figure:
    rows = result.metrics
    rows = rows[rows["controller"].isin(("pi", "pi_norm"))]
    scales = result.diagnostics.get("pi_scales", {})
    rows = rows.copy()
    rows["scale"] = rows["node"].map(scales)
    fig = _figure(10, 5)
    ax = fig.add_subplot()
    for controller, controller_label in (
        ("pi", "Original PI (KI=1000)"),
        ("pi_norm", "Normalized PI (KI=1)"),
    ):
        part = rows[rows["controller"] == controller]
        for reconciler, reconciler_rows in part.groupby("reconciler"):
            ax.scatter(
                reconciler_rows["scale"],
                reconciler_rows["scaled_mean_width"],
                label=f"{controller_label} · {reconciler}",
                alpha=0.65,
            )
    ax.set_xscale("log")
    ax.set_xlabel("Fixed pre-evaluation node scale")
    ax.set_ylabel("Legacy scaled signed span (empty sets retained)")
    ax.set_title("Original and normalized PI across node scales; descriptive")
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def _us(result: BenchmarkResult) -> Figure:
    rows = result.level_metrics
    rows = rows[
        (rows["level"] == "us") & rows["controller"].isin(("raw", "pi_norm", "pi"))
    ]
    rows = rows[rows["reconciler"].isin(("none", "mint_median", "bottom_up"))]
    fig = _figure(10, 5)
    axes = fig.subplots(1, 2)
    for ax, metric, label in zip(
        axes,
        ("scaled_interval_score", "empty_set_rate"),
        ("Scaled interval score", "Empty-set rate"),
        strict=True,
    ):
        for (controller, reconciler), part in rows.groupby(
            ["controller", "reconciler"]
        ):
            ax.scatter(
                part["model"], part[metric], label=f"{controller} · {reconciler}"
            )
        ax.set_title(label)
        ax.tick_params(axis="x", rotation=30)
        ax.legend(frameon=False)
    fig.suptitle("US root · descriptive only; no inferential claims")
    fig.tight_layout()
    return fig


def v2_figures(result: BenchmarkResult, *, block: int = 8) -> dict[str, Figure]:
    """Build the five registered v2 figures without file I/O.

    Args:
        result: Completed in-memory v2 benchmark result.
        block: Bootstrap block length shown in the factorial forest.

    Returns:
        Figures keyed by stable artifact stems.
    """
    return {
        "model-quality": _quality(result),
        "factorial-effects": _factorial(result, block),
        "coverage-score-tradeoff": _tradeoff(result),
        "pi-scale-diagnostics": _pi_scale(result),
        "us-descriptive": _us(result),
    }
