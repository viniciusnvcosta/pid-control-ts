# ABOUTME: Evaluates generated and reference forecasts on one common hierarchy window.
# ABOUTME: Reuses panel reconciliation, conformal metrics, paired block bootstrap, and Holm correction without I/O.
"""Common-window model benchmark evaluation."""

from collections.abc import Mapping
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from hcp.data import ROOT, TERRITORIES, HierarchyBundle
from hcp.panel import PanelConfig, forecaster_arms, forecaster_contrasts
from hcp.panel_metrics import (
    PANEL_METRICS,
    ForecasterArms,
    intervals_frame,
    level_metrics_frame,
    level_values,
    metrics_frame,
    node_metrics,
    node_scale,
)
from hcp.reconcile import evaluation_start
from hcp.report import RAW_LR
from hcp.stats import draw_blocks, holm, summarize_diff

MODEL_METRICS = ("coverage_deviation", "scaled_interval_score")
MODEL_RECONCILERS = ("none", "mint_shrink")


@dataclass(frozen=True)
class BenchmarkConfig:
    """Configuration for common-window hierarchy and model comparisons."""

    panel: PanelConfig
    models: tuple[str, ...]
    reference_model: str
    min_eval_weeks: int = 60
    seed_spawn_key: tuple[int, ...] = (1,)

    def __post_init__(self) -> None:
        if not self.models or len(set(self.models)) != len(self.models):
            raise ValueError("benchmark models must be non-empty and unique")
        if self.reference_model in self.models:
            raise ValueError("reference model must not collide with generated models")
        if self.min_eval_weeks <= 0:
            raise ValueError("min_eval_weeks must be positive")
        if any(value < 0 for value in self.seed_spawn_key):
            raise ValueError("seed spawn keys must be non-negative")


@dataclass(frozen=True)
class BenchmarkResult:
    """All in-memory benchmark forecast, metric, and contrast tables."""

    base_forecasts: pd.DataFrame
    intervals: pd.DataFrame
    metrics: pd.DataFrame
    level_metrics: pd.DataFrame
    hierarchy_contrasts: pd.DataFrame
    model_contrasts: pd.DataFrame
    diagnostics: dict


def _required_models(cfg: BenchmarkConfig) -> tuple[str, ...]:
    return (*cfg.models, cfg.reference_model)


def _validate_aligned(
    bundles: Mapping[str, HierarchyBundle], cfg: BenchmarkConfig
) -> HierarchyBundle:
    required = _required_models(cfg)
    if set(bundles) != set(required):
        raise ValueError(
            f"model keys must be exactly {list(required)!r}; got {list(bundles)!r}"
        )
    reference = bundles[cfg.reference_model]
    expected_shapes = {
        "y": (len(reference.nodes), len(reference.dates)),
        "q": (len(reference.nodes), len(reference.dates), 3),
        "imputed": (len(reference.nodes), len(reference.dates)),
    }
    for model in required:
        bundle = bundles[model]
        if bundle.nodes != reference.nodes:
            raise ValueError(f"model {model!r} nodes do not match the reference")
        if not np.array_equal(bundle.dates, reference.dates):
            raise ValueError(f"model {model!r} target dates do not match the reference")
        if not np.array_equal(bundle.forecast_dates, reference.forecast_dates):
            raise ValueError(
                f"model {model!r} forecast dates do not match the reference"
            )
        for name, shape in expected_shapes.items():
            if getattr(bundle, name).shape != shape:
                raise ValueError(f"model {model!r} {name} has invalid shape")
        if not np.array_equal(bundle.y, reference.y, equal_nan=True):
            raise ValueError(f"model {model!r} truth does not match the reference")
    if not reference.nodes or reference.nodes[0] != ROOT:
        raise ValueError("aligned benchmark nodes must begin with the US root")
    return reference


def common_evaluation_index(
    bundles: Mapping[str, HierarchyBundle], cfg: BenchmarkConfig
) -> int:
    """Return the single latest valid evaluation start shared by every model.

    Args:
        bundles: Exactly the generated model bundles and configured reference.
        cfg: Benchmark and reused panel settings.

    Returns:
        Integer index into the shared target-date grid.

    Raises:
        ValueError: On alignment errors, non-finite scored data, or fewer than the
            configured number of common weeks.
    """
    reference = _validate_aligned(bundles, cfg)
    starts = [
        evaluation_start(bundles[model], cfg.panel.mint_warmup)
        for model in _required_models(cfg)
    ]
    start = max(starts)
    invalid = np.zeros(len(reference.dates), dtype=bool)
    for model in _required_models(cfg):
        invalid |= ~np.isfinite(bundles[model].q).all(axis=(0, 2))
    endpoint = [
        index
        for index, node in enumerate(reference.nodes)
        if index == 0 or node not in TERRITORIES
    ]
    invalid |= ~np.isfinite(reference.y[endpoint]).all(axis=0)
    bad_at_or_after = np.flatnonzero(invalid & (np.arange(len(invalid)) >= start))
    if len(bad_at_or_after):
        start = int(bad_at_or_after[-1] + 1)
    if start >= len(reference.dates):
        raise ValueError("no remaining window has finite forecasts and endpoint truth")
    remaining = len(reference.dates) - start
    if remaining < cfg.min_eval_weeks:
        raise ValueError(
            f"fewer than {cfg.min_eval_weeks} common evaluation weeks remain"
        )
    return start


def _base_forecasts(model: str, bundle: HierarchyBundle) -> pd.DataFrame:
    nodes = np.repeat(bundle.nodes, len(bundle.dates))
    weeks = np.tile(bundle.dates, len(bundle.nodes))
    forecast_dates = np.tile(bundle.forecast_dates, len(bundle.nodes))
    return pd.DataFrame(
        {
            "model": model,
            "node": nodes,
            "week": weeks,
            "forecast_date": forecast_dates,
            "q_lo": bundle.q[:, :, 0].reshape(-1),
            "q_med": bundle.q[:, :, 1].reshape(-1),
            "q_hi": bundle.q[:, :, 2].reshape(-1),
        }
    )


def _bootstrap_levels(
    arms: ForecasterArms,
    reconciler: str,
    controller: str,
    lr: float,
    indices: np.ndarray | None,
    cfg: PanelConfig,
) -> dict[str, dict[str, np.ndarray]]:
    lo, hi = arms.intervals[(reconciler, controller, lr)]
    scale = node_scale(arms.y)
    if indices is None:
        values = node_metrics(
            arms.y,
            lo,
            hi,
            scale,
            alpha=cfg.controller.alpha,
            window=cfg.rolling_window,
        )
    else:
        per_node = [
            node_metrics(
                arms.y[index][indices],
                lo[index][indices],
                hi[index][indices],
                scale[index],
                alpha=cfg.controller.alpha,
                window=cfg.rolling_window,
            )
            for index in range(len(arms.nodes))
        ]
        values = {
            metric: np.stack([node[metric] for node in per_node])
            for metric in PANEL_METRICS
        }
    return level_values(values, arms.endpoint)


def _model_contrasts(
    arms: Mapping[str, ForecasterArms],
    cfg: BenchmarkConfig,
    seed: np.random.SeedSequence,
) -> pd.DataFrame:
    rows = []
    for block, child in zip(
        cfg.panel.blocks, seed.spawn(len(cfg.panel.blocks)), strict=True
    ):
        indices = draw_blocks(
            next(iter(arms.values())).y.shape[1],
            block,
            cfg.panel.n_boot,
            np.random.default_rng(child),
        )
        for controller in cfg.panel.controllers:
            lr = RAW_LR if controller == "raw" else cfg.panel.controller.lr
            for reconciler in MODEL_RECONCILERS:
                full = {
                    model: _bootstrap_levels(
                        model_arms, reconciler, controller, lr, None, cfg.panel
                    )
                    for model, model_arms in arms.items()
                }
                boot = {
                    model: _bootstrap_levels(
                        model_arms, reconciler, controller, lr, indices, cfg.panel
                    )
                    for model, model_arms in arms.items()
                }
                for model in cfg.models:
                    for level in ("leaf_mean", "us"):
                        for metric in MODEL_METRICS:
                            contrast = summarize_diff(
                                float(
                                    full[model][level][metric]
                                    - full[cfg.reference_model][level][metric]
                                ),
                                boot[model][level][metric]
                                - boot[cfg.reference_model][level][metric],
                            )
                            rows.append(
                                {
                                    "block": block,
                                    "model": model,
                                    "reference_model": cfg.reference_model,
                                    "level": level,
                                    "controller": controller,
                                    "lr": lr,
                                    "reconciler": reconciler,
                                    "metric": metric,
                                    **asdict(contrast),
                                }
                            )
    frame = pd.DataFrame(rows)
    frame["p_holm"] = frame.groupby(["block", "level", "controller", "reconciler"])[
        "p_value"
    ].transform(lambda values: holm(values.to_numpy()))
    return frame


def run_benchmark(
    bundles: Mapping[str, HierarchyBundle], cfg: BenchmarkConfig
) -> BenchmarkResult:
    """Evaluate hierarchy arms and model-reference contrasts on one common window."""
    start = common_evaluation_index(bundles, cfg)
    order = _required_models(cfg)
    root = np.random.SeedSequence(cfg.panel.root_seed, spawn_key=cfg.seed_spawn_key)
    seeds = root.spawn(len(order) + 1)
    arms = {
        model: forecaster_arms(bundles[model], cfg.panel, evaluation_index=start)
        for model in order
    }
    intervals = []
    metrics = []
    level_metrics = []
    hierarchy = []
    for model, seed in zip(order, seeds[:-1], strict=True):
        model_arms = arms[model]
        intervals.append(intervals_frame(model, model_arms))
        metrics.append(
            metrics_frame(
                model,
                model_arms,
                alpha=cfg.panel.controller.alpha,
                window=cfg.panel.rolling_window,
            )
        )
        level_metrics.append(
            level_metrics_frame(
                model,
                model_arms,
                alpha=cfg.panel.controller.alpha,
                window=cfg.panel.rolling_window,
            )
        )
        contrasts = forecaster_contrasts(model_arms, cfg.panel, seed)
        contrasts.insert(1, "model", model)
        hierarchy.append(contrasts)
    hierarchy_frame = pd.concat(hierarchy, ignore_index=True)
    hierarchy_frame["primary"] = False
    hierarchy_frame["p_holm"] = hierarchy_frame.groupby(
        ["block", "model", "level", "controller"]
    )["p_value"].transform(lambda values: holm(values.to_numpy()))

    def named(frames: list[pd.DataFrame]) -> pd.DataFrame:
        return pd.concat(frames, ignore_index=True).rename(
            columns={"forecaster": "model"}
        )

    reference = bundles[cfg.reference_model]
    return BenchmarkResult(
        base_forecasts=pd.concat(
            [_base_forecasts(model, bundles[model]) for model in order],
            ignore_index=True,
        ),
        intervals=named(intervals),
        metrics=named(metrics),
        level_metrics=named(level_metrics),
        hierarchy_contrasts=hierarchy_frame,
        model_contrasts=_model_contrasts(arms, cfg, seeds[-1]),
        diagnostics={
            "evaluation_start": str(pd.Timestamp(reference.dates[start]).date()),
            "evaluation_end": str(pd.Timestamp(reference.dates[-1]).date()),
            "n_weeks": len(reference.dates) - start,
            "nodes": list(reference.nodes),
            "model_order": list(order),
            "models": {model: arms[model].diagnostics for model in order},
        },
    )
