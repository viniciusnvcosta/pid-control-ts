# ABOUTME: Builds v2 stage-specific quantile repair and coherence evidence.
# ABOUTME: Retains projected and sorted values without changing reconciliation formulas.
"""Pure diagnostic records for v2 reconciliation stages."""

import json
from collections.abc import Mapping

import numpy as np
import pandas as pd

from hcp.data import HierarchyBundle
from hcp.forecast_assembly import _tables
from hcp.forecast_types import DiagnosticsTables
from hcp.reconcile import Reconciled, crossing_mask


def reconciliation_tables(
    model: str, bundle: HierarchyBundle, reconciled: Mapping[str, Reconciled]
) -> DiagnosticsTables:
    """Record repairs and all defined per-origin post-repair root residuals.

    Args:
        model: Canonical forecast model key.
        bundle: Node-ordered base forecasts and associated dates.
        reconciled: Evidence-enabled reconciliations by method.

    Returns:
        Canonical repair records and coherence/adjustment event summaries.
    """
    repairs, events = [], []
    for method, result in reconciled.items():
        before = result.pre_repair
        if before is None:
            raise ValueError("reconciliation requires pre-repair evidence")
        changed = np.isfinite(before).all(axis=-1) & np.any(before != result.q, axis=-1)
        crossed = crossing_mask(before)
        for node, origin in np.argwhere(changed):
            adjustment = np.abs(result.q[node, origin] - before[node, origin])
            repairs.append(
                {
                    "model": model,
                    "node": bundle.nodes[node],
                    "forecast_date": pd.Timestamp(bundle.forecast_dates[origin]),
                    "target_date": pd.Timestamp(bundle.dates[origin]),
                    "reconciler": method,
                    "stage": "reconciliation",
                    "before": before[node, origin].tolist(),
                    "after": result.q[node, origin].tolist(),
                    "count": 1,
                    "crossing": bool(crossed[node, origin]),
                    "max_absolute_adjustment": float(adjustment.max()),
                    "mean_absolute_adjustment": float(adjustment.mean()),
                }
            )
        for origin in range(len(bundle.dates)):
            if not np.isfinite(result.q[:, origin]).all():
                continue
            residual = result.q[0, origin] - result.q[1:, origin].sum(axis=0)
            pre_residual = before[0, origin] - before[1:, origin].sum(axis=0)
            adjustment = np.abs(result.q[:, origin] - before[:, origin])
            identity = {
                "model": model,
                "node": "us",
                "forecast_date": pd.Timestamp(bundle.forecast_dates[origin]),
                "target_date": pd.Timestamp(bundle.dates[origin]),
                "phase": "reconciliation",
                "reconciler": method,
            }
            for name, value in {
                "pre_repair_coherence_residual": pre_residual.tolist(),
                "post_repair_coherence_residual": residual.tolist(),
                "repair_summary": {
                    "adjusted_cells": int(changed[:, origin].sum()),
                    "crossing_cells": int(crossed[:, origin].sum()),
                    "root_crossings": int(crossed[0, origin]),
                    "max_absolute_adjustment": float(adjustment.max()),
                    "mean_absolute_adjustment": float(adjustment.mean()),
                },
            }.items():
                events.append(
                    {
                        **identity,
                        "name": name,
                        "value": json.dumps(value, sort_keys=True, allow_nan=False),
                    }
                )
    return _tables(
        {
            "quantile_repairs": repairs,
            "forecast_diagnostics": events,
            "tuning": [],
            "warnings": [],
        }
    )
