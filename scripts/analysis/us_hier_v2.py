# ABOUTME: Post-hoc diagnostics (D19) for a US-design run: noise floor, location vs scale, PI warm start.
# ABOUTME: Reads results/<run_id>/ and writes one JSON and two PDF figures under --out (docs/analysis/).
"""Exploratory diagnostics of a finished US-design run (not confirmatory).

Usage: ``uv run python -m scripts.analysis.us_hier_v2 --run-dir results/us-hier-v2``.

Everything is scored on the run's evaluation window ``[t0:]``. "Cold" PI starts at
``t0`` exactly as ``run_experiment`` does; "warm" PI runs from week 0 and is then
sliced to ``[t0:]``, so its offsets have already adapted when scoring begins.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from hcp.config import RunConfig, controller_config
from hcp.conformal import conformalize
from hcp.data import HierarchyBundle, load_hierarchy
from hcp.evaluate import coverage_deviation_floor, evaluate
from hcp.reconcile import evaluation_start, reconcile_bundle
from hcp.report import RAW_LR, RECONCILER_COLORS

LOCATION_SCALE = ("none", "bottom_up", "mint_shrink", "mint_median")
WARM_START = ("none", "bottom_up")
START_COLORS = {"cold": "#2a78d6", "warm": "#eb6834"}
CONTROLLER_STYLES = {"raw": "-", "pi": "--"}
PDF_METADATA = {"CreationDate": None}  # byte-stable PDFs across reruns


def _simulate_floor(
    n_weeks: int, cfg: RunConfig, *, n_sim: int, seed: int
) -> np.ndarray:
    rng = np.random.default_rng(np.random.SeedSequence(seed))
    return coverage_deviation_floor(
        n_weeks, alpha=cfg.alpha, window=cfg.rolling_window, n_sim=n_sim, rng=rng
    )


def _pi_starts(
    bundle: HierarchyBundle, cfg: RunConfig, reconciler: str
) -> dict[str, tuple[np.ndarray, ...]]:
    """``(y, q_lo, q_hi, lo, hi)`` on ``[t0:]`` for PI started at ``t0`` and at week 0."""
    t0 = evaluation_start(bundle, cfg.mint_warmup)
    q_us = reconcile_bundle(bundle, reconciler, cfg.mint_warmup).q[0]
    y = bundle.y[0]
    window = (y[t0:], q_us[t0:, 0], q_us[t0:, 2])
    cold = conformalize(*window, "pi", controller_config(cfg, cfg.lr))
    lo, hi = conformalize(
        y, q_us[:, 0], q_us[:, 2], "pi", controller_config(cfg, cfg.lr)
    )
    return {"cold": (*window, *cold), "warm": (*window, lo[t0:], hi[t0:])}


def diagnose(bundle: HierarchyBundle, cfg: RunConfig, *, n_sim: int, seed: int) -> dict:
    """Compute the D19 diagnostics on the run's evaluation window.

    Args:
        bundle: The hierarchy the run was computed on.
        cfg: The run's configuration.
        n_sim: Number of simulated i.i.d.-coverage series for the noise floor.
        seed: Root seed for the floor simulation (``cfg.root_seed``).

    Returns:
        ``run_id``, ``n_weeks``, ``evaluation_start``, ``floor`` (mean and 95%
        percentile range), ``location_scale`` (raw-interval metrics and median MAE per
        reconciler) and ``warm_start`` (PI metrics cold vs warm per reconciler).
    """
    t0 = evaluation_start(bundle, cfg.mint_warmup)
    y = bundle.y[0, t0:]
    metrics = {"alpha": cfg.alpha, "window": cfg.rolling_window}
    floor = _simulate_floor(len(y), cfg, n_sim=n_sim, seed=seed)
    low, high = np.percentile(floor, [2.5, 97.5])
    location_scale = {}
    for reconciler in LOCATION_SCALE:
        q_us = reconcile_bundle(bundle, reconciler, cfg.mint_warmup).q[0, t0:]
        location_scale[reconciler] = {
            **evaluate(y, q_us[:, 0], q_us[:, 2], **metrics),
            "median_mae": float(np.mean(np.abs(y - q_us[:, 1]))),
        }
    warm_start = {}
    for reconciler in WARM_START:
        starts = _pi_starts(bundle, cfg, reconciler)
        warm_start[reconciler] = {
            start: evaluate(y_w, lo, hi, **metrics)
            for start, (y_w, _q_lo, _q_hi, lo, hi) in starts.items()
        }
    return {
        "run_id": cfg.run_id,
        "n_weeks": len(y),
        "evaluation_start": str(pd.Timestamp(bundle.dates[t0]).date()),
        "floor": {
            "n_sim": n_sim,
            "mean": float(floor.mean()),
            "p2.5": float(low),
            "p97.5": float(high),
        },
        "location_scale": location_scale,
        "warm_start": warm_start,
    }


def _figure(width: float, height: float) -> Figure:
    figure = Figure(figsize=(width, height))
    FigureCanvasAgg(figure)
    return figure


def floor_figure(floor: np.ndarray, metrics: pd.DataFrame, *, lr: float) -> Figure:
    """Histogram of the simulated floor with each arm's observed coverage deviation."""
    figure = _figure(7.0, 3.5)
    axes = figure.add_subplot()
    axes.hist(floor, bins=40, color="#e5e7eb", linewidth=0, label="i.i.d. 80% floor")
    axes.axvline(floor.mean(), color="black", linestyle=":", label="floor mean")
    for row in metrics.itertuples():
        if (row.controller, row.lr) not in {("raw", RAW_LR), ("pi", lr)}:
            continue
        axes.axvline(
            row.coverage_deviation,
            color=RECONCILER_COLORS.get(row.reconciler),
            linestyle=CONTROLLER_STYLES[row.controller],
            linewidth=2,
            label=f"{row.reconciler}, {row.controller}",
        )
    axes.set_xlabel("Coverage deviation, mean |rolling coverage − 0.8|")
    axes.set_ylabel("Simulated series")
    axes.spines[["top", "right"]].set_visible(False)
    axes.legend(frameon=False, fontsize=8)
    figure.tight_layout()
    return figure


def warm_start_figure(
    weeks: np.ndarray, starts: dict[str, tuple[np.ndarray, ...]], *, lr: float
) -> Figure:
    """PI offsets ``q_lo − lo`` and ``hi − q_hi`` over the window, cold vs warm."""
    figure = _figure(8.0, 4.5)
    lower, upper = figure.subplots(2, 1, sharex=True)
    for start, (_y, q_lo, q_hi, lo, hi) in starts.items():
        style = {"color": START_COLORS[start], "linewidth": 2, "label": start}
        lower.plot(weeks, q_lo - lo, **style)
        upper.plot(weeks, hi - q_hi, **style)
    for axes, label in (
        (lower, r"Lower offset $q_{0.1} - lo$"),
        (upper, r"Upper offset $hi - q_{0.9}$"),
    ):
        axes.axhline(0, color="#6b7280", linewidth=1)
        axes.set_ylabel(label, fontsize=8)
        axes.spines[["top", "right"]].set_visible(False)
    upper.set_xlabel("Target week")
    lower.set_title(f"US PI offsets, none, lr={lr}")
    lower.legend(frameon=False)
    figure.tight_layout()
    return figure


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; returns 0 on success."""
    parser = argparse.ArgumentParser(description="Post-hoc diagnostics (D19)")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("docs/analysis"))
    parser.add_argument("--n-sim", type=int, default=10_000)
    args = parser.parse_args(argv)
    manifest = json.loads((args.run_dir / "manifest.json").read_text())
    cfg = RunConfig(**manifest["config"])
    bundle = load_hierarchy(args.data or cfg.data_path, manifest["input_sha256"])
    result = diagnose(bundle, cfg, n_sim=args.n_sim, seed=cfg.root_seed)
    figures_dir = args.out / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    (args.out / f"{cfg.run_id}-diagnostics.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    t0 = evaluation_start(bundle, cfg.mint_warmup)
    floor = _simulate_floor(
        result["n_weeks"], cfg, n_sim=args.n_sim, seed=cfg.root_seed
    )
    metrics = pd.read_parquet(args.run_dir / "metrics.parquet")
    floor_figure(floor, metrics, lr=cfg.lr).savefig(
        figures_dir / f"{cfg.run_id}-floor.pdf", metadata=PDF_METADATA
    )
    warm_start_figure(
        bundle.dates[t0:], _pi_starts(bundle, cfg, "none"), lr=cfg.lr
    ).savefig(figures_dir / f"{cfg.run_id}-warm-start.pdf", metadata=PDF_METADATA)
    print(f"Diagnostics: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
