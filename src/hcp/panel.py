# ABOUTME: Panel stage: reconciles and conformalizes every scored node per forecaster,
# ABOUTME: builds node/level metrics, paired block-bootstrap contrasts, and a cross-forecaster summary.
"""Panel stage (decisions D20-D26).

Reconciles and conformalizes every scored node of every forecaster's bundle on that
forecaster's own evaluation window (D21), then builds node- and level-level metrics
(D23, D25) and contrasts each reconciler against ``none`` with a paired moving-block
bootstrap whose week-index draw is shared by every node (D20). Holm adjustment runs
within (block, forecaster, level, controller); the primary contrast (D20) is
unadjusted. ``summarize_forecasters`` gives a descriptive cross-forecaster summary
(D24) at the first block length. ``mint_median`` (D26) is one of the panel reconcilers.
"""

from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace

import numpy as np
import pandas as pd

from hcp.conformal import ControllerConfig, conformalize
from hcp.data import TERRITORIES, HierarchyBundle

# LEVELS, PANEL_METRICS, ForecasterArms and node_scale are re-exported here as part of
# the panel API: tests and later tasks import them from hcp.panel.
from hcp.panel_metrics import (
    LEVELS,
    PANEL_METRICS,
    ForecasterArms,
    intervals_frame,
    level_metrics_frame,
    level_values,
    metrics_frame,
    node_metrics,
    node_scale,
)
from hcp.reconcile import evaluation_start, reconcile_bundle
from hcp.report import RAW_LR
from hcp.stats import draw_blocks, holm, summarize_diff

SUMMARY_METRICS = ("coverage_deviation", "scaled_interval_score")
COMPARISONS = (("mint_shrink", "none"), ("bottom_up", "none"), ("mint_median", "none"))
PRIMARY = {
    "level": "leaf_mean",
    "controller": "pi",
    "treatment": "mint_shrink",
    "control": "none",
    "metric": "coverage_deviation",
}


@dataclass(frozen=True)
class PanelConfig:
    """One pre-registered panel configuration; see docs/protocol-decisions.md D20-D26.

    Attributes:
        controller: Base controller hyperparameters; ``controller.lr`` is the lr used
            for every contrast.
        lr_grid: Learning rates swept for the ``pi`` controller's metrics.
        reconcilers: Reconciliation methods to run, must include ``"none"``.
        controllers: Controllers to run.
        mint_warmup: Minimum known complete error weeks before MinT is defined.
        rolling_window: Rolling-coverage window in weeks.
        blocks: Moving-block bootstrap block lengths.
        n_boot: Bootstrap resamples per block length.
        root_seed: Seeds every forecaster's bootstrap draws and the cross-forecaster
            summary.
        primary_forecaster: Must be a key of the ``bundles`` mapping passed to
            ``run_panel``.
    """

    controller: ControllerConfig
    lr_grid: tuple[float, ...]
    reconcilers: tuple[str, ...]
    controllers: tuple[str, ...]
    mint_warmup: int
    rolling_window: int
    blocks: tuple[int, ...]
    n_boot: int
    root_seed: int
    primary_forecaster: str


@dataclass(frozen=True)
class PanelResult:
    """In-memory outputs of one panel run.

    Attributes:
        intervals: Columns ``forecaster, node, week, reconciler, controller, lr, y,
            q_lo, q_med, q_hi, lo, hi``.
        metrics: Columns ``forecaster, node, reconciler, controller, lr, *METRIC_NAMES,
            scaled_mean_width, scaled_interval_score``.
        level_metrics: Columns ``forecaster, level, reconciler, controller, lr,
            *PANEL_METRICS``.
        contrasts: Columns ``block, forecaster, level, controller, lr, treatment,
            control, metric, estimate, low, high, p_value, primary, p_holm``.
        summary: Columns ``level, controller, lr, treatment, control, metric,
            n_forecasters, n_improved, mean_estimate, low, high``.
        diagnostics: Forecaster -> ``ForecasterArms.diagnostics``.
    """

    intervals: pd.DataFrame
    metrics: pd.DataFrame
    level_metrics: pd.DataFrame
    contrasts: pd.DataFrame
    summary: pd.DataFrame
    diagnostics: dict


def forecaster_arms(bundle: HierarchyBundle, cfg: PanelConfig) -> ForecasterArms:
    """Reconcile and conformalize every scored node on the forecaster's evaluation window.

    A node is scored only when its truth is complete on the window (D25); unscored
    nodes are dropped and recorded in ``diagnostics["unscored_nodes"]``.
    """
    t0 = evaluation_start(bundle, cfg.mint_warmup)
    window_y = bundle.y[:, t0:]
    scored = np.isfinite(window_y).all(axis=1)
    nodes = tuple(name for name, keep in zip(bundle.nodes, scored, strict=True) if keep)
    y = window_y[scored]
    endpoint = np.array(
        [i > 0 and name not in TERRITORIES for i, name in enumerate(nodes)]
    )
    q, intervals, crossings, root_crossings, shrinkage = {}, {}, {}, {}, {}
    for reconciler in cfg.reconcilers:
        rec = reconcile_bundle(bundle, reconciler, cfg.mint_warmup)
        crossings[reconciler] = rec.crossings
        root_crossings[reconciler] = rec.root_crossings
        if np.isfinite(rec.shrinkage[t0:]).all():
            shrinkage[reconciler] = float(np.mean(rec.shrinkage[t0:]))
        q[reconciler] = rec.q[scored, t0:, :]
        for controller in cfg.controllers:
            for lr in (RAW_LR,) if controller == "raw" else cfg.lr_grid:
                settings = replace(cfg.controller, lr=lr)
                bounds = [
                    conformalize(
                        y[i],
                        q[reconciler][i, :, 0],
                        q[reconciler][i, :, 2],
                        controller,
                        settings,
                    )
                    for i in range(len(nodes))
                ]
                intervals[(reconciler, controller, lr)] = (
                    np.array([lo for lo, _ in bounds]),
                    np.array([hi for _, hi in bounds]),
                )
    diagnostics = {
        "n_weeks": int(y.shape[1]),
        "evaluation_start": str(pd.Timestamp(bundle.dates[t0]).date()),
        "imputed_cells": int(bundle.imputed.sum()),
        "leaf_imputed_share": float(bundle.imputed[1:].mean()),
        "unscored_nodes": [
            n for n, keep in zip(bundle.nodes, scored, strict=True) if not keep
        ],
        "crossings": crossings,
        "root_crossings": root_crossings,
        "mean_shrinkage": shrinkage,
    }
    return ForecasterArms(
        bundle.dates[t0:], nodes, y, q, intervals, endpoint, diagnostics
    )


def forecaster_contrasts(
    arms: ForecasterArms, cfg: PanelConfig, seed: np.random.SeedSequence
) -> pd.DataFrame:
    """Paired block-bootstrap contrasts at ``cfg.controller.lr``; one week-index draw per block is shared by every node."""
    alpha, window = cfg.controller.alpha, cfg.rolling_window
    scale = node_scale(arms.y)
    rows = []
    for block, child in zip(cfg.blocks, seed.spawn(len(cfg.blocks)), strict=True):
        ix = draw_blocks(
            arms.y.shape[1], block, cfg.n_boot, np.random.default_rng(child)
        )
        for controller in cfg.controllers:
            lr = RAW_LR if controller == "raw" else cfg.controller.lr
            full, boot = {}, {}
            for reconciler in cfg.reconcilers:
                lo, hi = arms.intervals[(reconciler, controller, lr)]
                full[reconciler] = level_values(
                    node_metrics(arms.y, lo, hi, scale, alpha=alpha, window=window),
                    arms.endpoint,
                )
                per_node = [
                    node_metrics(
                        arms.y[i][ix],
                        lo[i][ix],
                        hi[i][ix],
                        scale[i],
                        alpha=alpha,
                        window=window,
                    )
                    for i in range(len(arms.nodes))
                ]
                stacked = {
                    m: np.stack([values[m] for values in per_node])
                    for m in PANEL_METRICS
                }
                boot[reconciler] = level_values(stacked, arms.endpoint)
            for treatment, control in COMPARISONS:
                if treatment not in cfg.reconcilers:
                    continue
                for level in LEVELS:
                    for metric in PANEL_METRICS:
                        contrast = summarize_diff(
                            float(
                                full[treatment][level][metric]
                                - full[control][level][metric]
                            ),
                            boot[treatment][level][metric]
                            - boot[control][level][metric],
                        )
                        rows.append(
                            {
                                "block": block,
                                "level": level,
                                "controller": controller,
                                "lr": lr,
                                "treatment": treatment,
                                "control": control,
                                "metric": metric,
                                **asdict(contrast),
                            }
                        )
    return pd.DataFrame(rows)


def summarize_forecasters(
    contrasts: pd.DataFrame, *, block: int, n_boot: int, rng: np.random.Generator
) -> pd.DataFrame:
    """Descriptive cross-forecaster summary (D24); one forecaster resample draw is shared by every row."""
    rows = contrasts[
        (contrasts["block"] == block) & contrasts["metric"].isin(SUMMARY_METRICS)
    ]
    forecasters = list(pd.unique(rows["forecaster"]))
    draws = rng.integers(0, len(forecasters), size=(n_boot, len(forecasters)))
    keys = ["level", "controller", "lr", "treatment", "control", "metric"]
    out = []
    for key, group in rows.groupby(keys, sort=False):
        estimates = (
            group.set_index("forecaster").loc[forecasters, "estimate"].to_numpy()
        )
        low, high = np.quantile(estimates[draws].mean(axis=1), [0.025, 0.975])
        out.append(
            {
                **dict(zip(keys, key, strict=True)),
                "n_forecasters": len(forecasters),
                "n_improved": int((estimates < 0).sum()),
                "mean_estimate": float(estimates.mean()),
                "low": float(low),
                "high": float(high),
            }
        )
    return pd.DataFrame(out)


def run_panel(bundles: Mapping[str, HierarchyBundle], cfg: PanelConfig) -> PanelResult:
    """Run every forecaster's arms, metrics and contrasts, and pool them into one result.

    Raises:
        ValueError: If ``cfg.primary_forecaster`` is not a key of ``bundles``.
    """
    if cfg.primary_forecaster not in bundles:
        raise ValueError(
            f"primary forecaster {cfg.primary_forecaster!r} is not in bundles"
        )
    seeds = np.random.SeedSequence(cfg.root_seed).spawn(len(bundles) + 1)
    intervals, metrics, level_metrics, contrasts = [], [], [], []
    diagnostics = {}
    for (forecaster, bundle), seed in zip(bundles.items(), seeds[:-1], strict=True):
        arms = forecaster_arms(bundle, cfg)
        diagnostics[forecaster] = arms.diagnostics
        alpha, window = cfg.controller.alpha, cfg.rolling_window
        intervals.append(intervals_frame(forecaster, arms))
        metrics.append(metrics_frame(forecaster, arms, alpha=alpha, window=window))
        level_metrics.append(
            level_metrics_frame(forecaster, arms, alpha=alpha, window=window)
        )
        contrast = forecaster_contrasts(arms, cfg, seed)
        contrast.insert(1, "forecaster", forecaster)
        contrasts.append(contrast)
    contrasts = pd.concat(contrasts, ignore_index=True)
    contrasts["primary"] = (
        (contrasts["forecaster"] == cfg.primary_forecaster)
        & (contrasts["block"] == cfg.blocks[0])
        & (contrasts["lr"] == cfg.controller.lr)
        & np.logical_and.reduce(
            [contrasts[key] == value for key, value in PRIMARY.items()]
        )
    )
    contrasts["p_holm"] = contrasts["p_value"]
    secondary = ~contrasts["primary"]
    contrasts.loc[secondary, "p_holm"] = (
        contrasts[secondary]
        .groupby(["block", "forecaster", "level", "controller"])["p_value"]
        .transform(lambda p: holm(p.to_numpy()))
    )
    summary = summarize_forecasters(
        contrasts,
        block=cfg.blocks[0],
        n_boot=cfg.n_boot,
        rng=np.random.default_rng(seeds[-1]),
    )
    return PanelResult(
        intervals=pd.concat(intervals, ignore_index=True),
        metrics=pd.concat(metrics, ignore_index=True),
        level_metrics=pd.concat(level_metrics, ignore_index=True),
        contrasts=contrasts,
        summary=summary,
        diagnostics=diagnostics,
    )
