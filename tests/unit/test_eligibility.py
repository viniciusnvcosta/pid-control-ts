# ABOUTME: Unit tests for the D21 forecaster eligibility rule on the synthetic panel.
# ABOUTME: Covers node mismatch, short windows, leaf imputation, duplicates and require_eligible.
import pytest

from hcp.data import FORECASTER, build_bundles
from hcp.eligibility import (
    EligibilityRule,
    duplicate_share,
    eligibility,
    require_eligible,
)
from tests.helpers import ALT_FORECASTER, DUP_FORECASTER, SHORT_FORECASTER

RULE = EligibilityRule(nodes=("us", "aa", "bb", "cc"), mint_warmup=5, min_eval_weeks=10)


@pytest.fixture
def bundles(panel_frame):
    return build_bundles(panel_frame)


@pytest.fixture
def table(bundles):
    return eligibility(bundles, RULE)


def test_forecaster_and_alt_are_eligible(table) -> None:
    indexed = table.set_index("forecaster")
    assert indexed.loc[FORECASTER, "eligible"]
    assert indexed.loc[FORECASTER, "reason"] == ""
    assert indexed.loc[ALT_FORECASTER, "eligible"]
    assert indexed.loc[ALT_FORECASTER, "reason"] == ""


def test_dup_forecaster_is_a_duplicate_of_forecaster(table) -> None:
    row = table.set_index("forecaster").loc[DUP_FORECASTER]
    assert not row["eligible"]
    assert row["reason"] == f"duplicate of {FORECASTER}"


def test_short_forecaster_is_ineligible(table) -> None:
    row = table.set_index("forecaster").loc[SHORT_FORECASTER]
    assert not row["eligible"]
    assert row["reason"] == "no complete week" or row["reason"].startswith(
        "0 evaluation weeks"
    )


def test_extra_node_makes_every_row_nodes_differ(bundles) -> None:
    rule = EligibilityRule(
        nodes=("us", "aa", "bb", "cc", "dd"), mint_warmup=5, min_eval_weeks=10
    )
    table = eligibility(bundles, rule)
    for _, row in table.iterrows():
        assert row["reason"] in ("nodes differ", "no complete week")
        assert not row["eligible"]


def test_tight_leaf_imputed_threshold_excludes_forecaster(bundles) -> None:
    rule = EligibilityRule(
        nodes=("us", "aa", "bb", "cc"),
        mint_warmup=5,
        min_eval_weeks=10,
        max_leaf_imputed=0.01,
    )
    row = eligibility(bundles, rule).set_index("forecaster").loc[FORECASTER]
    assert not row["eligible"]
    assert row["reason"].startswith("leaf imputation 0.0")


def test_duplicate_share_identity_and_distinct(bundles) -> None:
    base = bundles[FORECASTER]
    alt = bundles[ALT_FORECASTER]
    assert duplicate_share(base, base) == 1.0
    assert duplicate_share(base, alt) == 0.0


def test_require_eligible_passes_for_eligible_names(table) -> None:
    require_eligible(table, [FORECASTER, ALT_FORECASTER])


def test_require_eligible_raises_for_duplicate(table) -> None:
    with pytest.raises(ValueError, match="ineligible forecasters"):
        require_eligible(table, [DUP_FORECASTER])


def test_require_eligible_raises_for_unknown_name(table) -> None:
    with pytest.raises(ValueError, match="ineligible forecasters"):
        require_eligible(table, ["unknown"])


def test_table_columns_and_input_order(bundles, table) -> None:
    assert list(table.columns) == [
        "forecaster",
        "n_eval_weeks",
        "leaf_imputed_share",
        "eligible",
        "reason",
    ]
    assert list(table["forecaster"]) == list(bundles.keys())
