# ABOUTME: Paired factorial and model-reference inference for benchmark protocol v2.
# ABOUTME: Reuses shared moving-block draws and the established contrast/Holm helpers.
"""V2 benchmark contrasts over fixed completed weekly intervals."""

from collections.abc import Mapping
from hashlib import sha256

import numpy as np
import pandas as pd

from hcp.panel_metrics import ForecasterArms, node_scale
from hcp.stats import holm, summarize_diff

CONTRASTS = {
    "pi_effect": lambda a, b, c, d: b - a,
    "reconciliation_effect": lambda a, b, c, d: c - a,
    "combined_effect": lambda a, b, c, d: d - a,
    "reconciliation_given_pi": lambda a, b, c, d: d - b,
    "pi_given_reconciliation": lambda a, b, c, d: d - c,
    "interaction": lambda a, b, c, d: (d - c) - (b - a),
}
PRINCIPAL_METRICS = ("marginal_coverage_error", "scaled_interval_score")
ARM_CELLS = {
    "A": ("none", "raw"),
    "B": ("none", "pi_norm"),
    "C": ("mint_median", "raw"),
    "D": ("mint_median", "pi_norm"),
}


def factorial_contrasts(
    metric_values: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Calculate the six registered A-D factorial contrasts.

    Args:
        metric_values: A, B, C, and D arrays with matching shapes.

    Returns:
        Contrast key to elementwise estimate array.
    """
    if set(metric_values) != set("ABCD"):
        raise ValueError("metric_values must contain exactly A, B, C, and D")
    arrays = {key: np.asarray(metric_values[key], dtype=float) for key in "ABCD"}
    if len({array.shape for array in arrays.values()}) != 1:
        raise ValueError("A-D metric arrays must have matching shapes")
    a, b, c, d = (arrays[key] for key in "ABCD")
    return {key: fn(a, b, c, d) for key, fn in CONTRASTS.items()}


def _arm_values(
    arms: ForecasterArms,
    reconciler: str,
    controller: str,
    indices: np.ndarray | None,
    *,
    alpha: float,
    window: int,
) -> dict[str, np.ndarray]:
    lr = 0.0 if controller == "raw" else 0.1
    lo, hi = arms.intervals[(reconciler, controller, lr)]
    endpoint = arms.endpoint
    y = arms.y[endpoint]
    lo, hi = lo[endpoint], hi[endpoint]
    if indices is not None:
        y, lo, hi = y[:, indices], lo[:, indices], hi[:, indices]
    covered = (lo <= y) & (y <= hi)
    coverage_error = np.abs(covered.mean(axis=-1) - (1 - alpha))
    scale = node_scale(arms.y)[endpoint]
    score = (
        (hi - lo)
        + (2 / alpha) * (lo - y) * (y < lo)
        + (2 / alpha) * (y - hi) * (y > hi)
    )
    scaled_score = score.mean(axis=-1) / scale.reshape(
        (len(scale),) + (1,) * (score.ndim - 2)
    )
    return {
        "marginal_coverage_error": coverage_error.mean(axis=0),
        "scaled_interval_score": scaled_score.mean(axis=0),
    }


def v2_contrast_tables(
    arms: Mapping[str, ForecasterArms],
    draws: Mapping[int, np.ndarray],
    *,
    alpha: float,
    window: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the registered intervention and model-reference inference families.

    Args:
        arms: Seven generated model families plus the ``covidhub`` reference.
        draws: One week-index matrix per block length, shared by all calculations.
        alpha: Marginal interval miscoverage target.
        window: Rolling metric window (accepted for protocol compatibility).

    Returns:
        Intervention table (96 rows per block) and reference table (56 rows per block).
    """
    if "covidhub" not in arms or len(arms) != 8:
        raise ValueError("v2 inference requires seven generated models and covidhub")
    generated = [name for name in arms if name != "covidhub"]
    intervention_rows, reference_rows = [], []
    for block, indices in draws.items():
        indices = np.asarray(indices, dtype=int)
        if indices.ndim != 2 or any(
            ix.min() < 0 or ix.max() >= next(iter(arms.values())).y.shape[1]
            for ix in indices
        ):
            raise ValueError(
                "bootstrap indices must be a valid [resamples, weeks] matrix"
            )
        fingerprint = sha256(indices.tobytes()).hexdigest()
        full = {model: {} for model in arms}
        boot = {model: {} for model in arms}
        for model, model_arms in arms.items():
            for label, (reconciler, controller) in ARM_CELLS.items():
                full[model][label] = _arm_values(
                    model_arms, reconciler, controller, None, alpha=alpha, window=window
                )
                boot[model][label] = _arm_values(
                    model_arms,
                    reconciler,
                    controller,
                    indices,
                    alpha=alpha,
                    window=window,
                )
        for metric in PRINCIPAL_METRICS:
            for model in arms:
                effects = factorial_contrasts(
                    {cell: full[model][cell][metric] for cell in ARM_CELLS}
                )
                boot_effects = factorial_contrasts(
                    {cell: boot[model][cell][metric] for cell in ARM_CELLS}
                )
                for contrast, estimate in effects.items():
                    summary = summarize_diff(float(estimate), boot_effects[contrast])
                    intervention_rows.append(
                        {
                            "block": block,
                            "model": model,
                            "metric": metric,
                            "contrast": contrast,
                            "level": "leaf_mean",
                            "family": "intervention",
                            "family_id": f"{block}:intervention",
                            "family_size": 96,
                            "draw_fingerprint": fingerprint,
                            **summary.__dict__,
                        }
                    )
            for model in generated:
                for cell in ARM_CELLS:
                    estimate = float(
                        full[model][cell][metric] - full["covidhub"][cell][metric]
                    )
                    diff = boot[model][cell][metric] - boot["covidhub"][cell][metric]
                    summary = summarize_diff(estimate, diff)
                    reference_rows.append(
                        {
                            "block": block,
                            "model": model,
                            "reference_model": "covidhub",
                            "arm": cell,
                            "metric": metric,
                            "level": "leaf_mean",
                            "family": "model_reference",
                            "family_id": f"{block}:model_reference",
                            "family_size": 56,
                            "draw_fingerprint": fingerprint,
                            **summary.__dict__,
                        }
                    )
    intervention = pd.DataFrame(intervention_rows)
    reference = pd.DataFrame(reference_rows)
    if not intervention.empty:
        intervention["p_holm"] = intervention.groupby("block")["p_value"].transform(
            lambda p: holm(p.to_numpy())
        )
    if not reference.empty:
        reference["p_holm"] = reference.groupby("block")["p_value"].transform(
            lambda p: holm(p.to_numpy())
        )
    return intervention, reference
