# ABOUTME: Integration sanity check of the full experiment on the real deaths.csv (raw and PI arms only).
# ABOUTME: The unreconciled base US interval must cover roughly its nominal 80% on the evaluation window.
from pathlib import Path

import pytest

from hcp.data import load_hierarchy
from hcp.run import RunConfig, run_experiment
from tests.helpers import REAL_DATA, REAL_SHA256

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not REAL_DATA.exists(), reason="data/deaths.csv not downloaded"),
]


def test_real_run_sanity() -> None:
    cfg = RunConfig(
        run_id="it",
        root_seed=1,
        data_path=REAL_DATA,
        output_dir=Path("unused"),
        controllers=("raw", "pi"),
        lr_grid=(0.1,),
        blocks=(8,),
        n_boot=50,
    )
    result = run_experiment(cfg, load_hierarchy(REAL_DATA, REAL_SHA256))
    metrics = result.metrics.set_index(["reconciler", "controller"])
    assert 0.75 <= metrics.loc[("none", "raw"), "marginal_coverage"] <= 0.9
    assert result.diagnostics["n_weeks"] > 100
    assert result.contrasts["primary"].sum() == 1
