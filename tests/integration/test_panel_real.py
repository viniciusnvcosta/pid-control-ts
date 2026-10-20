# ABOUTME: Integration tests reproducing the D21 eligibility table and the D27 panel
# ABOUTME: regression anchor against Phase 1's us-hier-v2 metrics, on the real deaths.csv.
import numpy as np
import pandas as pd
import pytest

from hcp.conformal import ControllerConfig
from hcp.data import FORECASTER, build_bundle, build_bundles, read_forecasts
from hcp.eligibility import EligibilityRule, eligibility
from hcp.evaluate import METRIC_NAMES
from hcp.panel import PanelConfig, run_panel
from tests.helpers import D21_ELIGIBLE, REAL_DATA, REAL_SHA256, ROOT_DIR

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not REAL_DATA.exists(), reason="data/deaths.csv not downloaded"),
]

V2_METRICS = ROOT_DIR / "results" / "us-hier-v2" / "metrics.parquet"


def test_eligibility_reproduces_d21() -> None:
    bundles = build_bundles(read_forecasts(REAL_DATA, REAL_SHA256))
    rule = EligibilityRule(nodes=bundles[FORECASTER].nodes, mint_warmup=20)
    table = eligibility(bundles, rule).set_index("forecaster")
    assert sorted(table.index[table["eligible"]]) == D21_ELIGIBLE
    assert (
        table.loc["COVIDhub_CDC-ensemble", "reason"] == "duplicate of COVIDhub-ensemble"
    )
    assert table.loc["JHU_IDD-CovidSP", "reason"].startswith("leaf imputation")


@pytest.mark.skipif(not V2_METRICS.exists(), reason="results/us-hier-v2 not run")
def test_panel_us_node_reproduces_v2_metrics() -> None:
    bundle = build_bundle(read_forecasts(REAL_DATA, REAL_SHA256))
    cfg = PanelConfig(
        controller=ControllerConfig(),
        lr_grid=(1.0, 0.5, 0.1, 0.05),
        reconcilers=("none", "bottom_up", "mint_shrink"),
        controllers=("raw", "pi"),
        mint_warmup=20,
        rolling_window=10,
        blocks=(8,),
        n_boot=10,
        root_seed=1,
        primary_forecaster=FORECASTER,
    )
    result = run_panel({FORECASTER: bundle}, cfg)
    us_metrics = result.metrics[result.metrics["node"] == "us"]
    v2_metrics = pd.read_parquet(V2_METRICS)
    merged = us_metrics.merge(
        v2_metrics, on=["reconciler", "controller", "lr"], suffixes=("_panel", "_v2")
    )
    assert len(merged) == 15
    for metric in METRIC_NAMES:
        np.testing.assert_allclose(
            merged[f"{metric}_panel"], merged[f"{metric}_v2"], rtol=1e-10
        )
