# ABOUTME: End-to-end test of the CLI panel design on the synthetic panel hierarchy.
# ABOUTME: Verifies every panel artifact, manifest fields, eligibility, and the no-overwrite rule.
import json
from pathlib import Path

import pandas as pd
import pytest

from hcp.data import FORECASTER, file_sha256
from hcp.run import main
from tests.helpers import ALT_FORECASTER, DUP_FORECASTER, panel_rows, write_config


def _config(tmp_path: Path, csv: Path, forecasters: tuple[str, ...]) -> Path:
    return write_config(
        tmp_path / "panel.toml",
        {
            "run_id": "panel-e2e",
            "root_seed": 7,
            "data_path": str(csv),
            "output_dir": str(tmp_path / "results"),
            "expected_sha256": file_sha256(csv),
            "design": "panel",
            "forecasters": list(forecasters),
            "controllers": ["raw", "pi"],
            "reconcilers": ["none", "bottom_up", "mint_shrink", "mint_median"],
            "lr": 0.1,
            "lr_grid": [0.1, 0.5],
            "T_burnin": 2,
            "mint_warmup": 5,
            "rolling_window": 5,
            "blocks": [4],
            "n_boot": 20,
            "min_eval_weeks": 10,
        },
    )


def test_panel_cli_writes_every_artifact(tmp_path: Path, panel_csv: Path) -> None:
    config = _config(tmp_path, panel_csv, (FORECASTER, ALT_FORECASTER))
    assert main([str(config)]) == 0
    out = tmp_path / "results" / "panel-e2e"
    for name in (
        "manifest.json",
        "intervals.parquet",
        "metrics.parquet",
        "level_metrics.parquet",
        "contrasts.parquet",
        "summary.parquet",
        "eligibility.parquet",
    ):
        assert (out / name).is_file(), name
    figures = {path.name for path in (out / "figures").glob("*.pdf")}
    assert figures == {
        "panel_forest_leaf_mean_raw.pdf",
        "panel_forest_leaf_mean_pi.pdf",
        "panel_forest_us_raw.pdf",
        "panel_forest_us_pi.pdf",
        "leaf_contrasts_raw.pdf",
        "leaf_contrasts_pi.pdf",
    }
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["config"]["design"] == "panel"
    assert manifest["forecasters"] == [FORECASTER, ALT_FORECASTER]
    dup_row = next(
        row for row in manifest["eligibility"] if row["forecaster"] == DUP_FORECASTER
    )
    assert dup_row["reason"].startswith("duplicate of")
    assert manifest["primary"]["forecaster"] == FORECASTER
    contrasts = pd.read_parquet(out / "contrasts.parquet")
    assert contrasts["primary"].sum() == 1
    summary = pd.read_parquet(out / "summary.parquet")
    assert (summary["n_forecasters"] == 2).all()


def test_panel_cli_rejects_ineligible_forecaster(
    tmp_path: Path, panel_csv: Path
) -> None:
    config = _config(tmp_path, panel_csv, (FORECASTER, DUP_FORECASTER))
    with pytest.raises(ValueError, match="ineligible"):
        main([str(config)])


def test_panel_cli_refuses_to_overwrite_a_run(tmp_path: Path, panel_csv: Path) -> None:
    config = _config(tmp_path, panel_csv, (FORECASTER, ALT_FORECASTER))
    assert main([str(config)]) == 0
    with pytest.raises(FileExistsError):
        main([str(config)])


def test_panel_cli_raises_value_error_when_primary_forecaster_has_no_rows(
    tmp_path: Path,
) -> None:
    # No bundle can be built for FORECASTER when the CSV has zero rows for it at
    # all (as opposed to rows that fail to build a bundle, which build_bundles
    # still maps to None); main must still raise ValueError, not KeyError.
    rows = [row for row in panel_rows() if row["forecaster"] != FORECASTER]
    csv = tmp_path / "panel-no-primary.csv"
    pd.DataFrame(rows).to_csv(csv)  # index column mimics the real leading ""
    config = _config(tmp_path, csv, (FORECASTER, ALT_FORECASTER))
    with pytest.raises(ValueError, match="has no usable bundle"):
        main([str(config)])
