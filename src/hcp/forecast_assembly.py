# ABOUTME: Assembles pure origin batches into registered model forecast bundles.
# ABOUTME: Validates batch identities and preserves ordered diagnostics without I/O.
"""Deterministic in-memory forecast batch assembly helpers."""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from hcp.benchmark_batch import OriginForecastBatch
from hcp.forecast_diagnostics import _diagnostic_frame
from hcp.forecast_preparation import _bundle, _long_forecasts
from hcp.forecast_types import DiagnosticsTables, ForecastPlan, ForecastResult

if TYPE_CHECKING:
    from hcp.forecast import ForecastConfig


def _seed_key(plan: ForecastPlan, model: str, target_index: int) -> tuple[int, ...]:
    return plan.config.seed_spawn_key + (plan.config.models.index(model), target_index)


def _source_kind(model: str, cfg: ForecastConfig) -> str | None:
    if cfg.protocol_version == 1:
        return None
    return "native_quantiles" if model == "qr_l1" else "sample_derived"


def _tables(rows: dict[str, list[dict]]) -> DiagnosticsTables:
    empty = DiagnosticsTables()
    return DiagnosticsTables(
        **{
            name: _diagnostic_frame(name, values, getattr(empty, name).columns)
            for name, values in rows.items()
        }
    )


def _assemble_model_bundles(
    plan: ForecastPlan,
    batches: tuple[OriginForecastBatch, ...],
    cfg: ForecastConfig,
) -> ForecastResult:
    """Assemble complete origin batches into registered model and target order.

    Args:
        plan: Validated causal forecast plan.
        batches: Every eligible batch exactly once, in any order.
        cfg: The configuration used to prepare the plan.

    Returns:
        Hierarchy bundles, long forecasts, summaries and diagnostic tables.

    Raises:
        ValueError: If a batch is missing, duplicated, or differs from the plan.
    """
    if cfg != plan.config:
        raise ValueError("configuration differs from forecast plan")
    indexed = {}
    for batch in batches:
        key = (batch.model, batch.target_index)
        if key not in plan.eligible_keys or key in indexed:
            raise ValueError("unexpected or duplicate model/origin batch")
        index = batch.target_index
        if (
            batch.origin != plan.origins[index]
            or batch.target != plan.targets[index]
            or batch.nodes != plan.actuals.nodes
            or batch.seed_spawn_key != _seed_key(plan, *key)
            or batch.source_kind != _source_kind(batch.model, cfg)
            or not np.isfinite(batch.quantiles).all()
            or np.any(np.diff(batch.quantiles, axis=-1) < 0)
        ):
            raise ValueError("batch identity or quantiles differ from forecast plan")
        indexed[key] = batch
    if set(indexed) != set(plan.eligible_keys):
        raise ValueError("missing eligible model/origin batches")
    bundles = {}
    frames = []
    model_diagnostics = {}
    diagnostic_rows = {
        name: []
        for name in ("forecast_diagnostics", "tuning", "quantile_repairs", "warnings")
    }
    for model in cfg.models:
        q = np.full((len(plan.actuals.nodes), len(plan.targets), 3), np.nan)
        warning_counts: Counter[str] = Counter()
        interpolation = []
        elapsed = 0.0
        fit_count = 0
        for index in range(len(plan.targets)):
            batch = indexed.get((model, index))
            if batch is None:
                continue
            q[:, index, :] = batch.quantiles
            warning_counts.update(batch.warning_counts)
            interpolation.extend(dict(record) for record in batch.interpolation)
            elapsed += batch.elapsed_seconds
            fit_count += 1
            for name, rows in diagnostic_rows.items():
                rows.extend(
                    getattr(batch.diagnostics, name).to_dict("records")
                )
        bundle = _bundle(
            plan.actuals, plan.targets, plan.origins, plan.target_indices, q
        )
        bundles[model] = bundle
        frames.append(_long_forecasts(model, bundle, _source_kind(model, cfg)))
        model_diagnostics[model] = {
            "fit_count": fit_count,
            "warning_counts": dict(sorted(warning_counts.items())),
            "elapsed_seconds": elapsed,
            "interpolation": interpolation,
        }
    return ForecastResult(
        bundles=bundles,
        base_forecasts=pd.concat(frames, ignore_index=True),
        diagnostics={
            "models": model_diagnostics,
            "seed": {
                "root_seed": cfg.root_seed,
                "spawn_key": list(cfg.seed_spawn_key),
                "model_order": list(cfg.models),
            },
        },
        tables=_tables(diagnostic_rows),
    )
