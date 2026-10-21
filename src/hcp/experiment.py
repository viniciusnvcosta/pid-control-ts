# ABOUTME: Numeric Phase 1 experiment stage for hierarchical reconciliation and conformal evaluation.
# ABOUTME: Runs all registered arms in memory and computes paired bootstrap contrasts without file I/O.
"""In-memory Phase 1 experiment stage."""

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from hcp.config import RunConfig, controller_config
from hcp.conformal import conformalize
from hcp.data import HierarchyBundle
from hcp.evaluate import METRIC_NAMES, evaluate
from hcp.reconcile import evaluation_start, reconcile_bundle
from hcp.report import RAW_LR
from hcp.stats import draw_blocks, holm, summarize_diff

PRIMARY = {
    "treatment": "mint_shrink",
    "control": "none",
    "controller": "pi",
    "metric": "coverage_deviation",
}
COMPARISONS = (("mint_shrink", "none"), ("bottom_up", "none"))


@dataclass(frozen=True)
class ExperimentResult:
    """In-memory outputs of one run."""

    intervals: pd.DataFrame
    metrics: pd.DataFrame
    contrasts: pd.DataFrame
    diagnostics: dict


def _arm(
    intervals: pd.DataFrame, reconciler: str, controller: str, lr: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = intervals[
        (intervals["reconciler"] == reconciler)
        & (intervals["controller"] == controller)
        & (intervals["lr"] == lr)
    ]
    return rows["y"].to_numpy(), rows["lo"].to_numpy(), rows["hi"].to_numpy()


def _contrasts(cfg: RunConfig, intervals: pd.DataFrame) -> pd.DataFrame:
    n = intervals["week"].nunique()
    children = np.random.SeedSequence(cfg.root_seed).spawn(len(cfg.blocks))
    rows = []
    for block, child in zip(cfg.blocks, children, strict=True):
        resamples = draw_blocks(n, block, cfg.n_boot, np.random.default_rng(child))
        for controller in cfg.controllers:
            lr = RAW_LR if controller == "raw" else cfg.lr
            arms = {r: _arm(intervals, r, controller, lr) for r in cfg.reconcilers}
            full = {
                r: evaluate(*arm, alpha=cfg.alpha, window=cfg.rolling_window)
                for r, arm in arms.items()
            }
            boot = {
                r: pd.DataFrame(
                    [
                        evaluate(
                            y[ix],
                            lo[ix],
                            hi[ix],
                            alpha=cfg.alpha,
                            window=cfg.rolling_window,
                        )
                        for ix in resamples
                    ]
                )
                for r, (y, lo, hi) in arms.items()
            }
            for treatment, control in COMPARISONS:
                if treatment not in arms:
                    continue
                for metric in METRIC_NAMES:
                    contrast = summarize_diff(
                        full[treatment][metric] - full[control][metric],
                        (boot[treatment][metric] - boot[control][metric]).to_numpy(),
                    )
                    rows.append(
                        {
                            "block": block,
                            "controller": controller,
                            "lr": lr,
                            "treatment": treatment,
                            "control": control,
                            "metric": metric,
                            **asdict(contrast),
                        }
                    )
    frame = pd.DataFrame(rows)
    frame["primary"] = (frame["block"] == cfg.blocks[0]) & np.logical_and.reduce(
        [frame[key] == value for key, value in PRIMARY.items()]
    )
    frame["p_holm"] = frame["p_value"]
    secondary = ~frame["primary"]
    frame.loc[secondary, "p_holm"] = (
        frame[secondary]
        .groupby(["block", "controller"])["p_value"]
        .transform(lambda p: holm(p.to_numpy()))
    )
    return frame


def run_experiment(cfg: RunConfig, bundle: HierarchyBundle) -> ExperimentResult:
    """Run every arm in memory and compute metrics and contrasts."""
    t0 = evaluation_start(bundle, cfg.mint_warmup)
    weeks = bundle.dates[t0:]
    y = bundle.y[0, t0:]
    gap = bundle.y[0, t0:] - bundle.y[1:, t0:].sum(axis=0)
    diagnostics: dict = {
        "n_weeks": len(weeks),
        "evaluation_start": str(pd.Timestamp(weeks[0]).date()),
        "imputed_cells": int(bundle.imputed.sum()),
        "truth_incoherence": {
            "median": float(np.nanmedian(gap)),
            "max_abs": float(np.nanmax(np.abs(gap))),
        },
        "crossings": {},
        "root_crossings": {},
    }
    frames = []
    for reconciler in cfg.reconcilers:
        reconciled = reconcile_bundle(bundle, reconciler, cfg.mint_warmup)
        diagnostics["crossings"][reconciler] = reconciled.crossings
        diagnostics["root_crossings"][reconciler] = reconciled.root_crossings
        if reconciler == "mint_shrink":
            diagnostics["mean_shrinkage"] = float(np.mean(reconciled.shrinkage[t0:]))
        q_us = reconciled.q[0, t0:, :]
        for controller in cfg.controllers:
            for lr in (RAW_LR,) if controller == "raw" else cfg.lr_grid:
                lo, hi = conformalize(
                    y, q_us[:, 0], q_us[:, 2], controller, controller_config(cfg, lr)
                )
                frames.append(
                    pd.DataFrame(
                        {
                            "week": weeks,
                            "reconciler": reconciler,
                            "controller": controller,
                            "lr": lr,
                            "y": y,
                            "q_lo": q_us[:, 0],
                            "q_med": q_us[:, 1],
                            "q_hi": q_us[:, 2],
                            "lo": lo,
                            "hi": hi,
                        }
                    )
                )
    intervals = pd.concat(frames, ignore_index=True)
    metrics = pd.DataFrame(
        [
            {
                "reconciler": reconciler,
                "controller": controller,
                "lr": lr,
                **evaluate(
                    group["y"].to_numpy(),
                    group["lo"].to_numpy(),
                    group["hi"].to_numpy(),
                    alpha=cfg.alpha,
                    window=cfg.rolling_window,
                ),
            }
            for (reconciler, controller, lr), group in intervals.groupby(
                ["reconciler", "controller", "lr"], sort=False
            )
        ]
    )
    return ExperimentResult(intervals, metrics, _contrasts(cfg, intervals), diagnostics)
