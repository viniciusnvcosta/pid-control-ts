# ABOUTME: Forecaster eligibility rule for the panel (D21): node match, evaluation-window length,
# ABOUTME: leaf imputation share, and near-duplicate exclusion, applied to a mapping of bundles.
"""Eligibility stage (decision D21).

A forecaster is eligible when its bundle builds with the expected nodes, has enough
evaluation weeks after the MinT warm-up, and keeps leaf imputation below a share. Among
near-duplicate pairs (almost every shared quantile cell equal), only the one with more
evaluation weeks is kept, ties broken by name.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from hcp.data import HierarchyBundle
from hcp.reconcile import evaluation_start

_COLUMNS = ("forecaster", "n_eval_weeks", "leaf_imputed_share", "eligible", "reason")


@dataclass(frozen=True)
class EligibilityRule:
    """Thresholds for D21 forecaster eligibility.

    Attributes:
        nodes: The expected ``HierarchyBundle.nodes``.
        mint_warmup: Warm-up passed to ``evaluation_start``.
        min_eval_weeks: Minimum evaluation weeks required.
        max_leaf_imputed: Maximum tolerated share of imputed leaf cells.
        duplicate_share: Above this ``duplicate_share``, a forecaster is a near-duplicate.
    """

    nodes: tuple[str, ...]
    mint_warmup: int
    min_eval_weeks: int = 60
    max_leaf_imputed: float = 0.05
    duplicate_share: float = 0.95


def evaluation_weeks(bundle: HierarchyBundle, mint_warmup: int) -> int:
    """Number of evaluation weeks, 0 if ``bundle`` never reaches the warm-up."""
    try:
        start = evaluation_start(bundle, mint_warmup)
    except ValueError:
        return 0
    return len(bundle.dates) - start


def duplicate_share(a: HierarchyBundle, b: HierarchyBundle) -> float:
    """Share of the shared ``(node, week)`` cells where all three quantiles are close.

    Returns:
        0.0 when ``a`` and ``b`` share no target week.
    """
    a_shared = np.isin(a.dates, b.dates)
    if not a_shared.any():
        return 0.0
    b_shared = np.isin(b.dates, a.dates)
    qa = a.q[:, a_shared]
    qb = b.q[:, b_shared]
    return float(np.isclose(qa, qb).all(axis=-1).mean())


def _row(
    name: str, n_eval_weeks: int, leaf_imputed_share: float, eligible: bool, reason: str
) -> dict[str, object]:
    return {
        "forecaster": name,
        "n_eval_weeks": n_eval_weeks,
        "leaf_imputed_share": leaf_imputed_share,
        "eligible": eligible,
        "reason": reason,
    }


def eligibility(
    bundles: Mapping[str, HierarchyBundle | None], rule: EligibilityRule
) -> pd.DataFrame:
    """D21 eligibility table, one row per key of ``bundles`` in input order."""
    rows: dict[str, dict[str, object]] = {}
    candidates: list[str] = []
    for name, bundle in bundles.items():
        if bundle is None:
            rows[name] = _row(name, 0, float("nan"), False, "no complete week")
            continue
        n_eval_weeks = evaluation_weeks(bundle, rule.mint_warmup)
        leaves_imputed = bundle.imputed[1:]
        leaf_imputed_share = (
            float(leaves_imputed.mean()) if leaves_imputed.size else float("nan")
        )
        if bundle.nodes != rule.nodes:
            reason = "nodes differ"
        elif n_eval_weeks < rule.min_eval_weeks:
            reason = f"{n_eval_weeks} evaluation weeks < {rule.min_eval_weeks}"
        elif leaf_imputed_share > rule.max_leaf_imputed:
            reason = (
                f"leaf imputation {leaf_imputed_share:.3f} > {rule.max_leaf_imputed}"
            )
        else:
            reason = ""
        rows[name] = _row(name, n_eval_weeks, leaf_imputed_share, reason == "", reason)
        if reason == "":
            candidates.append(name)

    kept: list[str] = []
    for name in sorted(candidates, key=lambda n: (-rows[n]["n_eval_weeks"], n)):
        duplicate_of = next(
            (
                other
                for other in kept
                if duplicate_share(bundles[name], bundles[other]) > rule.duplicate_share
            ),
            None,
        )
        if duplicate_of is None:
            kept.append(name)
        else:
            rows[name]["eligible"] = False
            rows[name]["reason"] = f"duplicate of {duplicate_of}"

    return pd.DataFrame([rows[name] for name in bundles], columns=list(_COLUMNS))


def require_eligible(table: pd.DataFrame, forecasters: Sequence[str]) -> None:
    """Raise if any of ``forecasters`` is unknown to or ineligible in ``table``.

    Raises:
        ValueError: Listing the unknown or ineligible names.
    """
    indexed = table.set_index("forecaster")["eligible"]
    bad = [name for name in forecasters if not indexed.get(name, False)]
    if bad:
        raise ValueError(f"ineligible forecasters: {bad}")
