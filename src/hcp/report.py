# ABOUTME: PDF figures for a run (Phase 1) and the leaves x forecasters panel (Phase 2):
# ABOUTME: rolling coverage, intervals, contrast forests, and per-forecaster/per-leaf contrasts.
"""Report stage. Called only by ``run.write_artifacts`` and ``panel`` reporting scripts."""

from pathlib import Path

import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from hcp.data import TERRITORIES
from hcp.evaluate import coverage_indicator, rolling_coverage

RAW_LR = 0.0
RECONCILER_COLORS = {
    "none": "#6b7280",
    "bottom_up": "#2563eb",
    "mint_shrink": "#d97706",
    "mint_median": "#059669",
}


def _arms(
    intervals: pd.DataFrame, controller: str, lr: float
) -> dict[str, pd.DataFrame]:
    selected = intervals[
        (intervals["controller"] == controller) & (intervals["lr"] == lr)
    ]
    return {name: group for name, group in selected.groupby("reconciler", sort=False)}


def _figure(width: float = 8.0, height: float = 3.5) -> Figure:
    figure = Figure(figsize=(width, height))
    FigureCanvasAgg(figure)
    return figure


def rolling_coverage_figure(
    intervals: pd.DataFrame, *, controller: str, lr: float, alpha: float, window: int
) -> Figure:
    """Rolling coverage of the US interval for each reconciler."""
    figure = _figure()
    axes = figure.add_subplot()
    for reconciler, group in _arms(intervals, controller, lr).items():
        hit = coverage_indicator(
            group["y"].to_numpy(), group["lo"].to_numpy(), group["hi"].to_numpy()
        )
        axes.plot(
            group["week"].to_numpy()[window - 1 :],
            rolling_coverage(hit, window),
            label=reconciler,
            color=RECONCILER_COLORS.get(reconciler),
            linewidth=1.5,
        )
    axes.axhline(1 - alpha, color="black", linestyle="--", linewidth=1)
    axes.set_ylim(0, 1.05)
    axes.set_ylabel(f"Rolling coverage (w={window})")
    axes.set_title(f"US 4-week-ahead deaths · {controller}, lr={lr}")
    axes.spines[["top", "right"]].set_visible(False)
    axes.legend(frameon=False)
    figure.tight_layout()
    return figure


def intervals_figure(intervals: pd.DataFrame, *, controller: str, lr: float) -> Figure:
    """US truth with each reconciler's conformalized interval as a band."""
    figure = _figure()
    axes = figure.add_subplot()
    truth = None
    for reconciler, group in _arms(intervals, controller, lr).items():
        weeks = group["week"].to_numpy()
        axes.fill_between(
            weeks,
            group["lo"].to_numpy(),
            group["hi"].to_numpy(),
            color=RECONCILER_COLORS.get(reconciler),
            alpha=0.25,
            label=reconciler,
            linewidth=0,
        )
        truth = (weeks, group["y"].to_numpy())
    if truth is not None:
        axes.plot(
            *truth,
            color="black",
            marker="o",
            markersize=2,
            linewidth=0.8,
            label="truth",
        )
    axes.set_ylabel("Weekly deaths")
    axes.set_title(f"US 80% intervals · {controller}, lr={lr}")
    axes.spines[["top", "right"]].set_visible(False)
    axes.legend(frameon=False)
    figure.tight_layout()
    return figure


def contrasts_figure(contrasts: pd.DataFrame, *, block: int) -> Figure:
    """Forest plot of coverage-deviation contrasts (treatment - control) at ``block``."""
    rows = contrasts[
        (contrasts["block"] == block) & (contrasts["metric"] == "coverage_deviation")
    ].reset_index(drop=True)
    figure = _figure(6.0, 0.5 + 0.4 * max(len(rows), 1))
    axes = figure.add_subplot()
    positions = range(len(rows))
    axes.errorbar(
        rows["estimate"],
        list(positions),
        xerr=[rows["estimate"] - rows["low"], rows["high"] - rows["estimate"]],
        fmt="o",
        color="black",
        capsize=3,
    )
    axes.set_yticks(
        list(positions),
        [f"{r.controller}: {r.treatment} − {r.control}" for r in rows.itertuples()],
    )
    axes.axvline(0, color="#6b7280", linestyle="--", linewidth=1)
    axes.set_xlabel(f"Δ coverage deviation (95% block-bootstrap CI, block={block})")
    axes.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    return figure


def render_figures(
    intervals: pd.DataFrame,
    contrasts: pd.DataFrame,
    out_dir: Path,
    *,
    alpha: float,
    window: int,
    lr: float,
    block: int,
) -> list[Path]:
    """Write every figure as PDF into ``out_dir`` and return their paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    figures: dict[str, Figure] = {}
    for controller in intervals["controller"].unique():
        arm_lr = RAW_LR if controller == "raw" else lr
        figures[f"rolling_coverage_{controller}.pdf"] = rolling_coverage_figure(
            intervals, controller=controller, lr=arm_lr, alpha=alpha, window=window
        )
        figures[f"intervals_{controller}.pdf"] = intervals_figure(
            intervals, controller=controller, lr=arm_lr
        )
    figures["contrasts_coverage_deviation.pdf"] = contrasts_figure(
        contrasts, block=block
    )
    paths = []
    for name, figure in figures.items():
        path = out_dir / name
        figure.savefig(path)
        paths.append(path)
    return paths


def panel_forest_figure(
    contrasts: pd.DataFrame,
    *,
    level: str,
    controller: str,
    lr: float,
    block: int,
    primary_forecaster: str,
) -> Figure:
    """Forest plot of coverage-deviation contrasts, one row group per forecaster.

    Each forecaster's group holds one point and 95% CI per treatment, colored by
    ``RECONCILER_COLORS``; the primary forecaster's tick label is bold.
    """
    rows = contrasts[
        (contrasts["level"] == level)
        & (contrasts["controller"] == controller)
        & (contrasts["lr"] == lr)
        & (contrasts["block"] == block)
        & (contrasts["metric"] == "coverage_deviation")
    ]
    forecasters = list(pd.unique(rows["forecaster"]))
    treatments = [name for name in RECONCILER_COLORS if name != "none"]
    figure = _figure(7.0, 0.8 + 0.4 * max(len(forecasters) * len(treatments), 1))
    axes = figure.add_subplot()
    row, centers = 0, []
    for forecaster in forecasters:
        start = row
        group = rows[rows["forecaster"] == forecaster]
        for treatment in treatments:
            point = group[group["treatment"] == treatment]
            if point.empty:
                continue
            record = point.iloc[0]
            axes.errorbar(
                [record["estimate"]],
                [row],
                xerr=[
                    [record["estimate"] - record["low"]],
                    [record["high"] - record["estimate"]],
                ],
                fmt="o",
                color=RECONCILER_COLORS[treatment],
                label=treatment,
                capsize=3,
            )
            row += 1
        centers.append((start + row - 1) / 2)
        row += 1
    handles, labels = axes.get_legend_handles_labels()
    unique = dict(zip(labels, handles, strict=True))
    axes.legend(unique.values(), unique.keys(), frameon=False)
    axes.set_yticks(centers, forecasters)
    for tick, forecaster in zip(axes.get_yticklabels(), forecasters, strict=True):
        if forecaster == primary_forecaster:
            tick.set_fontweight("bold")
    axes.invert_yaxis()
    axes.axvline(0, color="#6b7280", linestyle="--", linewidth=1)
    axes.set_xlabel(
        f"Δ coverage deviation vs none (95% block-bootstrap CI, block={block})"
    )
    axes.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    return figure


def leaf_contrast_figure(
    metrics: pd.DataFrame,
    *,
    forecaster: str,
    controller: str,
    lr: float,
    treatment: str = "mint_shrink",
    control: str = "none",
    metric: str = "coverage_deviation",
) -> Figure:
    """Per-node ``metric[treatment] - metric[control]`` for one forecaster and arm.

    Excludes ``us`` and ``TERRITORIES``; sorted ascending as a horizontal dot plot.
    """
    rows = metrics[
        (metrics["forecaster"] == forecaster)
        & (metrics["controller"] == controller)
        & (metrics["lr"] == lr)
        & (metrics["node"] != "us")
        & (~metrics["node"].isin(TERRITORIES))
    ]
    treated = rows[rows["reconciler"] == treatment].set_index("node")[metric]
    control_values = rows[rows["reconciler"] == control].set_index("node")[metric]
    diff = (treated - control_values).sort_values()
    figure = _figure(6.0, 0.8 + 0.3 * max(len(diff), 1))
    axes = figure.add_subplot()
    positions = range(len(diff))
    axes.plot(
        diff.to_numpy(),
        list(positions),
        "o",
        color=RECONCILER_COLORS.get(treatment, "black"),
    )
    axes.set_yticks(list(positions), list(diff.index))
    axes.axvline(0, color="#6b7280", linestyle="--", linewidth=1)
    axes.set_xlabel(f"Δ {metric} ({treatment} − {control})")
    axes.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    return figure


def render_panel_figures(
    metrics: pd.DataFrame,
    contrasts: pd.DataFrame,
    out_dir: Path,
    *,
    lr: float,
    block: int,
    primary_forecaster: str,
) -> list[Path]:
    """Write every panel figure as PDF into ``out_dir`` and return their paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    figures: dict[str, Figure] = {}
    for controller in contrasts["controller"].unique():
        arm_lr = RAW_LR if controller == "raw" else lr
        figures[f"panel_forest_leaf_mean_{controller}.pdf"] = panel_forest_figure(
            contrasts,
            level="leaf_mean",
            controller=controller,
            lr=arm_lr,
            block=block,
            primary_forecaster=primary_forecaster,
        )
        figures[f"panel_forest_us_{controller}.pdf"] = panel_forest_figure(
            contrasts,
            level="us",
            controller=controller,
            lr=arm_lr,
            block=block,
            primary_forecaster=primary_forecaster,
        )
        figures[f"leaf_contrasts_{controller}.pdf"] = leaf_contrast_figure(
            metrics,
            forecaster=primary_forecaster,
            controller=controller,
            lr=arm_lr,
        )
    paths = []
    for name, figure in figures.items():
        path = out_dir / name
        figure.savefig(path)
        paths.append(path)
    return paths
