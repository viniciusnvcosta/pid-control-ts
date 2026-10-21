# ABOUTME: End-to-end test of the post-hoc diagnostics script on a synthetic US-design run.
# ABOUTME: Builds the run with hcp.run, then checks the diagnostics JSON keys, floor and figures.
import json
from pathlib import Path

import pytest

from hcp.data import file_sha256
from hcp.run import main as run_main
from scripts.analysis.us_hier_v2 import main
from tests.helpers import write_config

RECONCILERS = {"none", "bottom_up", "mint_shrink", "mint_median"}


def _run(tmp_path: Path, csv: Path) -> Path:
    config = write_config(
        tmp_path / "e2e.toml",
        {
            "run_id": "e2e",
            "root_seed": 7,
            "data_path": str(csv),
            "output_dir": str(tmp_path / "results"),
            "expected_sha256": file_sha256(csv),
            "lr": 0.1,
            "lr_grid": [0.1, 0.5],
            "controllers": ["raw", "pi"],
            "T_burnin": 2,
            "mint_warmup": 5,
            "rolling_window": 5,
            "blocks": [4],
            "n_boot": 20,
        },
    )
    assert run_main([str(config)]) == 0
    return tmp_path / "results" / "e2e"


def test_diagnostics_script_writes_json_and_figures(
    tmp_path: Path, hier_csv: Path
) -> None:
    run_dir = _run(tmp_path, hier_csv)
    out = tmp_path / "analysis"
    assert main(["--run-dir", str(run_dir), "--out", str(out), "--n-sim", "200"]) == 0
    diagnostics = json.loads((out / "e2e-diagnostics.json").read_text())
    assert set(diagnostics) == {
        "run_id",
        "n_weeks",
        "evaluation_start",
        "floor",
        "location_scale",
        "warm_start",
    }
    assert diagnostics["run_id"] == "e2e"
    assert set(diagnostics["location_scale"]) == RECONCILERS
    assert set(diagnostics["warm_start"]) == {"none", "bottom_up"}
    assert set(diagnostics["warm_start"]["none"]) == {"cold", "warm"}
    floor = diagnostics["floor"]
    assert floor["n_sim"] == 200
    assert 0 < floor["mean"] < 1
    location_scale = diagnostics["location_scale"]
    assert location_scale["mint_median"]["mean_width"] == pytest.approx(
        location_scale["none"]["mean_width"]
    )
    for name in ("e2e-floor.pdf", "e2e-warm-start.pdf"):
        figure = out / "figures" / name
        assert figure.is_file(), name
        assert figure.stat().st_size > 0, name
