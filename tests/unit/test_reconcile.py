# ABOUTME: Unit tests for reconciliation: summing matrix, bottom-up, MinT(Shrink) and the causal online bundle pass.
# ABOUTME: Checks coherence, projection, prefix invariance and that unknown truth never leaks in.
from dataclasses import replace

import numpy as np
import pytest

from hcp.reconcile import (
    RECONCILERS,
    available_errors,
    crossing_mask,
    evaluation_start,
    reconcile,
    reconcile_bundle,
    shrinkage_covariance,
    summing_matrix,
)

WARMUP = 5
T0 = 8  # synthetic bundle: 5 complete error rows are known first at t = 8


def _truncate(bundle, k: int):
    return replace(
        bundle,
        dates=bundle.dates[:k],
        forecast_dates=bundle.forecast_dates[:k],
        y=bundle.y[:, :k],
        q=bundle.q[:, :k],
        imputed=bundle.imputed[:, :k],
    )


def test_registry_names() -> None:
    assert set(RECONCILERS) == {"none", "bottom_up", "mint_shrink"}


def test_summing_matrix_is_ones_over_identity() -> None:
    S = summing_matrix(3)
    assert S.shape == (4, 3)
    np.testing.assert_array_equal(S[0], np.ones(3))
    np.testing.assert_array_equal(S[1:], np.eye(3))


def test_none_returns_a_copy_of_base() -> None:
    base = np.arange(12.0).reshape(4, 3)
    out = reconcile(base, summing_matrix(3), "none")
    np.testing.assert_array_equal(out, base)
    assert out is not base


@pytest.mark.parametrize("seed", range(5))
def test_bottom_up_sums_leaves(seed: int) -> None:
    base = np.random.default_rng(seed).normal(size=(4, 3))
    out = reconcile(base, summing_matrix(3), "bottom_up")
    np.testing.assert_allclose(out[0], base[1:].sum(axis=0))
    np.testing.assert_allclose(out[1:], base[1:])


@pytest.mark.parametrize("seed", range(5))
def test_mint_is_coherent_and_fixes_coherent_input(seed: int) -> None:
    rng = np.random.default_rng(seed)
    S = summing_matrix(3)
    covariance, lam = shrinkage_covariance(rng.normal(size=(30, 4)))
    out = reconcile(rng.normal(size=(4, 3)), S, "mint_shrink", covariance=covariance)
    np.testing.assert_allclose(out[0], out[1:].sum(axis=0))
    coherent = S @ rng.normal(size=(3, 3))
    np.testing.assert_allclose(
        reconcile(coherent, S, "mint_shrink", covariance=covariance), coherent
    )
    assert 0.0 <= lam <= 1.0


def test_shrinkage_covariance_is_symmetric_positive_definite() -> None:
    covariance, _ = shrinkage_covariance(np.random.default_rng(0).normal(size=(20, 6)))
    np.testing.assert_allclose(covariance, covariance.T)
    assert np.linalg.eigvalsh(covariance).min() > 0


def test_unknown_method_raises() -> None:
    with pytest.raises(ValueError, match="unknown reconciliation method"):
        reconcile(np.zeros((4, 3)), summing_matrix(3), "nope")


def test_mint_without_covariance_raises() -> None:
    with pytest.raises(ValueError, match="covariance"):
        reconcile(np.zeros((4, 3)), summing_matrix(3), "mint_shrink")


def test_available_errors_uses_targets_known_by_forecast_date(hier_bundle) -> None:
    assert available_errors(hier_bundle, T0).shape == (WARMUP, 4)


def test_evaluation_start_is_first_week_with_warmup_errors(hier_bundle) -> None:
    assert evaluation_start(hier_bundle, WARMUP) == T0


def test_evaluation_start_raises_without_enough_history(hier_bundle) -> None:
    with pytest.raises(ValueError, match="complete error weeks"):
        evaluation_start(hier_bundle, 1000)


def test_warmup_below_two_raises(hier_bundle) -> None:
    with pytest.raises(ValueError, match="warmup"):
        reconcile_bundle(hier_bundle, "mint_shrink", 1)


def test_crossing_mask_uses_relative_tolerance_and_ignores_nan() -> None:
    q = np.array(
        [
            [[1.0, 2.0, 2.0 - 1e-12]],  # ~1e-12 inversion: below tolerance
            [[1.0, 2.0, 1.0]],  # 1.0 inversion: above tolerance
            [[1.0, np.nan, 1.0]],  # NaN cell: never counted
        ]
    )
    np.testing.assert_array_equal(crossing_mask(q), [[False], [True], [False]])


def test_none_bundle_equals_base(hier_bundle) -> None:
    rec = reconcile_bundle(hier_bundle, "none", WARMUP)
    np.testing.assert_array_equal(rec.q, hier_bundle.q)
    assert rec.crossings == 0
    assert rec.root_crossings == 0


@pytest.mark.parametrize("method", ["bottom_up", "mint_shrink"])
def test_root_crossings_counts_only_row_zero(hier_bundle, method: str) -> None:
    rec = reconcile_bundle(hier_bundle, method, WARMUP)
    if method == "bottom_up":
        assert rec.root_crossings == 0
    assert rec.root_crossings <= rec.crossings


@pytest.mark.parametrize("method", ["bottom_up", "mint_shrink"])
def test_bundle_is_coherent_and_sorted_where_defined(hier_bundle, method: str) -> None:
    rec = reconcile_bundle(hier_bundle, method, WARMUP)
    defined = np.isfinite(rec.q).all(axis=(0, 2))
    assert (np.diff(rec.q[:, defined], axis=-1) >= 0).all()
    if rec.crossings == 0:
        np.testing.assert_allclose(rec.q[0, defined], rec.q[1:, defined].sum(axis=0))


def test_mint_is_undefined_before_evaluation_start(hier_bundle) -> None:
    rec = reconcile_bundle(hier_bundle, "mint_shrink", WARMUP)
    assert np.isnan(rec.q[:, :T0]).all()
    assert np.isfinite(rec.q[:, T0:]).all()
    assert np.isnan(rec.shrinkage[:T0]).all()
    assert np.isfinite(rec.shrinkage[T0:]).all()


def test_mint_is_prefix_invariant(hier_bundle) -> None:
    full = reconcile_bundle(hier_bundle, "mint_shrink", WARMUP)
    part = reconcile_bundle(_truncate(hier_bundle, 15), "mint_shrink", WARMUP)
    np.testing.assert_array_equal(part.q, full.q[:, :15])


def test_mint_ignores_truth_unknown_at_forecast_date(hier_bundle) -> None:
    t = 12
    y = hier_bundle.y.copy()
    y[:, t - 3 :] = (
        1e9  # targets less than 4 weeks before t are unknown at its forecast date
    )
    leaked = reconcile_bundle(replace(hier_bundle, y=y), "mint_shrink", WARMUP)
    clean = reconcile_bundle(hier_bundle, "mint_shrink", WARMUP)
    np.testing.assert_array_equal(leaked.q[:, t], clean.q[:, t])
