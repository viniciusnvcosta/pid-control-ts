# ABOUTME: Integration test reproducing the D21 forecaster eligibility table on the real deaths.csv.
# ABOUTME: Needs data/deaths.csv (gitignored); run with `uv run pytest -m integration`.
import pytest

from hcp.data import FORECASTER, build_bundles, read_forecasts
from hcp.eligibility import EligibilityRule, eligibility
from tests.helpers import REAL_DATA, REAL_SHA256

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not REAL_DATA.exists(), reason="data/deaths.csv not downloaded"),
]

D21_ELIGIBLE = [
    "CEID-Walk",
    "COVIDhub-4_week_ensemble",
    "COVIDhub-baseline",
    "COVIDhub-ensemble",
    "COVIDhub-trained_ensemble",
    "JHUAPL-Bucky",
    "RobertWalraven-ESG",
    "UMass-MechBayes",
]


def test_eligibility_reproduces_d21() -> None:
    bundles = build_bundles(read_forecasts(REAL_DATA, REAL_SHA256))
    rule = EligibilityRule(nodes=bundles[FORECASTER].nodes, mint_warmup=20)
    table = eligibility(bundles, rule).set_index("forecaster")
    assert sorted(table.index[table["eligible"]]) == D21_ELIGIBLE
    assert (
        table.loc["COVIDhub_CDC-ensemble", "reason"] == "duplicate of COVIDhub-ensemble"
    )
    assert table.loc["JHU_IDD-CovidSP", "reason"].startswith("leaf imputation")
