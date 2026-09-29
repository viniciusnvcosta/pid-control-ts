# ABOUTME: Summing matrix, bottom-up and MinT(Shrink) reconciliation of the base quantile forecasts.
# ABOUTME: MinT uses a causal expanding covariance of median errors known by each forecast date (D7).
"""Reconciliation stage (decisions D2, D5, D7, D8).

``summing_matrix``, ``shrinkage_covariance``, bottom-up and MinT(Shrink) are ported from
net-sci-epi ``src/headd_l0/reconcile.py`` and generalized to any number of leaves. MinT
follows Wickramasuriya, Athanasopoulos & Hyndman (2019), shrinking the error covariance
towards its diagonal with the Schafer & Strimmer (2005) lambda. Each quantile level is
reconciled with the same projection; levels that cross afterwards are re-sorted.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from hcp.data import HierarchyBundle


@dataclass(frozen=True)
class Reconciled:
    """Reconciled quantiles for every node.

    Attributes:
        q: ``[n_nodes, T, 3]``; NaN in weeks before the MinT warm-up.
        crossings: Number of (node, week) cells whose levels had to be re-sorted.
        shrinkage: MinT lambda per week ``[T]``; NaN where no covariance was estimated.
    """

    q: np.ndarray
    crossings: int
    shrinkage: np.ndarray


def summing_matrix(n_leaves: int) -> np.ndarray:
    """Return S ``[n_leaves + 1, n_leaves]``: a row of ones, then the identity."""
    return np.vstack([np.ones((1, n_leaves)), np.eye(n_leaves)])


def shrinkage_covariance(errors: np.ndarray) -> tuple[np.ndarray, float]:
    """Shrink the error covariance towards its diagonal (MinT(Shrink)).

    Args:
        errors: Training errors ``[n_train, n_series]``.

    Returns:
        ``(W, lambda)`` with ``W = lambda diag(W1) + (1 - lambda) W1`` and a floor of
        ``1e-8 * max(trace(W) / n_series, 1)`` on the diagonal. A series with zero
        variance contributes zero correlation; lambda is 1 when every correlation is 0.
    """
    n, p = errors.shape
    w1 = errors.T @ errors / n
    sd = np.sqrt(np.diag(w1))
    scale = np.where(sd > 0, sd, 1.0)
    xs = np.where(sd > 0, errors / scale, 0.0)
    correlation = xs.T @ xs / n
    variance = (xs.T**2 @ xs**2 - (xs.T @ xs) ** 2 / n) / (n * (n - 1))
    off = ~np.eye(p, dtype=bool)
    denominator = float((correlation[off] ** 2).sum())
    lam = (
        1.0
        if denominator == 0
        else min(max(variance[off].sum() / denominator, 0.0), 1.0)
    )
    covariance = lam * np.diag(np.diag(w1)) + (1 - lam) * w1
    floor = 1e-8 * max(np.trace(covariance) / p, 1.0)
    np.fill_diagonal(covariance, np.maximum(np.diag(covariance), floor))
    return covariance, lam


def _none(
    base: np.ndarray, _S: np.ndarray, _covariance: np.ndarray | None
) -> np.ndarray:
    return base.copy()


def _bottom_up(
    base: np.ndarray, S: np.ndarray, _covariance: np.ndarray | None
) -> np.ndarray:
    return S @ base[1:]


def _mint_shrink(
    base: np.ndarray, S: np.ndarray, covariance: np.ndarray | None
) -> np.ndarray:
    if covariance is None:
        raise ValueError("mint_shrink needs the training error covariance")
    weighted = np.linalg.solve(covariance, S)  # W^-1 S
    gain = np.linalg.solve(S.T @ weighted, weighted.T)  # (S' W^-1 S)^-1 S' W^-1
    return S @ (gain @ base)


RECONCILERS: dict[
    str, Callable[[np.ndarray, np.ndarray, np.ndarray | None], np.ndarray]
] = {
    "none": _none,
    "bottom_up": _bottom_up,
    "mint_shrink": _mint_shrink,
}


def reconcile(
    base: np.ndarray,
    S: np.ndarray,
    method: str,
    *,
    covariance: np.ndarray | None = None,
) -> np.ndarray:
    """Project base forecasts ``[n_nodes, ...]`` (root first) onto the coherent subspace.

    Raises:
        ValueError: On an unknown method, or ``mint_shrink`` without a covariance.
    """
    if method not in RECONCILERS:
        raise ValueError(f"unknown reconciliation method: {method}")
    return RECONCILERS[method](base, S, covariance)


def available_errors(bundle: HierarchyBundle, t: int) -> np.ndarray:
    """Median errors ``[n_known, n_nodes]`` of targets reported by ``forecast_dates[t]``.

    Only weeks where every node has a finite error are returned.
    """
    known = bundle.dates <= bundle.forecast_dates[t]
    errors = (bundle.y - bundle.q[:, :, 1]).T[known]
    return errors[np.isfinite(errors).all(axis=1)]


def evaluation_start(bundle: HierarchyBundle, warmup: int) -> int:
    """First week index whose forecast date already knows ``warmup`` complete error weeks.

    Raises:
        ValueError: If no week reaches the warm-up.
    """
    for t in range(len(bundle.dates)):
        if len(available_errors(bundle, t)) >= warmup:
            return t
    raise ValueError(
        f"fewer than {warmup} complete error weeks before the last forecast"
    )


def reconcile_bundle(bundle: HierarchyBundle, method: str, warmup: int) -> Reconciled:
    """Reconcile every week; MinT re-estimates its covariance causally each week.

    Args:
        bundle: Data stage output.
        method: A key of ``RECONCILERS``.
        warmup: Minimum number of known complete error weeks before MinT is defined.

    Raises:
        ValueError: On an unknown method or ``warmup < 2``.
    """
    if method not in RECONCILERS:
        raise ValueError(f"unknown reconciliation method: {method}")
    if warmup < 2:
        raise ValueError("warmup must be at least 2 weeks")
    S = summing_matrix(len(bundle.nodes) - 1)
    out = np.full(bundle.q.shape, np.nan)
    shrinkage = np.full(len(bundle.dates), np.nan)
    for t in range(len(bundle.dates)):
        covariance = None
        if method == "mint_shrink":
            errors = available_errors(bundle, t)
            if len(errors) < warmup:
                continue
            covariance, shrinkage[t] = shrinkage_covariance(errors)
        out[:, t, :] = reconcile(bundle.q[:, t, :], S, method, covariance=covariance)
    defined = np.isfinite(out).all(axis=-1)
    crossed = (np.diff(out, axis=-1) < 0).any(axis=-1) & defined
    return Reconciled(
        q=np.sort(out, axis=-1), crossings=int(crossed.sum()), shrinkage=shrinkage
    )
