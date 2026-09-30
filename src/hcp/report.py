# ABOUTME: PDF figures for a run: rolling US coverage and intervals per controller, and a contrast forest plot.
# ABOUTME: Uses the object-oriented matplotlib API (Figure + Agg canvas) so no global pyplot state leaks.
"""Report stage. Called only by ``run.write_artifacts``."""

from pathlib import Path

import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from hcp.evaluate import coverage_indicator, rolling_coverage

RAW_LR = 0.0
RECONCILER_COLORS = {
    "none": "#6b7280",
    "bottom_up": "#2563eb",
    "mint_shrink": "#d97706",
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
