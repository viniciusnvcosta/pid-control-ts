# ABOUTME: Synthetic tests for paired v2 factorial and model-reference inference.
# ABOUTME: Guards formulas, family sizes, signs, shared draws, and empty-set coverage.
import numpy as np

from hcp.benchmark_stats import factorial_contrasts, v2_contrast_tables
from hcp.panel_metrics import ForecasterArms


def test_factorial_contrast_formulas() -> None:
    result = factorial_contrasts(
        {k: np.array([v]) for k, v in zip("ABCD", (10, 8, 9, 6), strict=True)}
    )
    assert {k: v.item() for k, v in result.items()} == {
        "pi_effect": -2,
        "reconciliation_effect": -1,
        "combined_effect": -4,
        "reconciliation_given_pi": -2,
        "pi_given_reconciliation": -3,
        "interaction": -1,
    }


def test_all_zero_factorial_contrasts_remain_zero() -> None:
    values = {key: np.zeros(9) for key in "ABCD"}
    assert all(
        np.count_nonzero(value) == 0 for value in factorial_contrasts(values).values()
    )


def _arms(offset: float) -> ForecasterArms:
    weeks, nodes = 24, ("us", "aa", "bb")
    y = np.vstack(
        [np.arange(weeks) + 3.0, np.arange(weeks) + 1.0, np.arange(weeks) + 2.0]
    )
    endpoint = np.array([False, True, True])
    intervals = {}
    for rec in ("none", "bottom_up", "mint_shrink", "mint_median"):
        for ctl in ("raw", "pi_norm", "pi"):
            lo = y - 2 + offset
            hi = y + 2 + offset
            if ctl in ("pi_norm", "pi"):
                lo = y + 2 + offset
                hi = y - 2 + offset
            intervals[(rec, ctl, 0.0 if ctl == "raw" else 0.1)] = (lo, hi)
    return ForecasterArms(np.arange(weeks), nodes, y, {}, intervals, endpoint, {})


def test_contrast_tables_have_joint_family_sizes_and_shared_draws() -> None:
    names = (*[f"m{i}" for i in range(7)], "covidhub")
    arms = {name: _arms(float(i)) for i, name in enumerate(names)}
    indices = np.tile(np.arange(24), (8, 1))
    draws = {b: indices for b in (8, 4, 12)}
    intervention, reference = v2_contrast_tables(arms, draws, alpha=0.2, window=5)
    assert intervention.groupby("block").size().eq(96).all()
    assert reference.groupby("block").size().eq(56).all()
    assert intervention.groupby("block")["family_size"].first().eq(96).all()
    assert reference.groupby("block")["family_size"].first().eq(56).all()
    row = reference.query(
        "model == 'm0' and arm == 'A' and metric == 'scaled_interval_score'"
    ).iloc[0]
    assert row.estimate < 0
    empty_set_effect = intervention.query(
        "model == 'm0' and contrast == 'pi_effect' and metric == 'marginal_coverage_error'"
    ).iloc[0]
    assert np.isclose(empty_set_effect.estimate, 0.6)
    assert not intervention.level.eq("us").any()
    for block in draws:
        assert intervention.query("block == @block")["draw_fingerprint"].nunique() == 1
        assert reference.query("block == @block")["draw_fingerprint"].nunique() == 1
        assert (
            intervention.query("block == @block")["draw_fingerprint"].iloc[0]
            == reference.query("block == @block")["draw_fingerprint"].iloc[0]
        )


def test_resampling_recomputes_absolute_coverage_error_and_preserves_empty_sets() -> (
    None
):
    arms = {f"m{i}": _arms(0.0) for i in range(7)} | {"covidhub": _arms(1.0)}
    ix = np.tile(np.arange(24), (5, 1))
    a, b = v2_contrast_tables(arms, {8: ix}, alpha=0.2, window=5)
    assert "marginal_coverage_error" in set(a.metric)
    assert a.estimate.notna().all() and b.estimate.notna().all()
    pi_effects = a[
        (a.metric == "marginal_coverage_error") & (a.contrast == "pi_effect")
    ]
    assert np.isclose(pi_effects.estimate, 0.6).all()
