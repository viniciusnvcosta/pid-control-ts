# ABOUTME: End-to-end test of the CLI on the synthetic hierarchy: TOML in, results/<run_id>/ artifacts out.
# ABOUTME: Verifies every artifact, manifest fields, the arm grid and the no-overwrite rule.
import json
from pathlib import Path

import pandas as pd
import pytest

from hcp.data import file_sha256
from hcp.run import main
from tests.helpers import write_config


def _config(tmp_path: Path, csv: Path) -> Path:
    return write_config(
        tmp_path / "e2e.toml",
        {
            "run_id": "e2e",
            "root_seed": 7,
            "data_path": str(csv),
            "output_dir": str(tmp_path / "results"),
            "expected_sha256": file_sha256(csv),
            "lr": 0.1,
            "lr_grid": [0.1, 0.5],
            "T_burnin": 2,
            "mint_warmup": 5,
            "rolling_window": 5,
            "blocks": [4],
            "n_boot": 20,
        },
    )


def test_cli_writes_every_artifact(tmp_path: Path, hier_csv: Path) -> None:
    assert main([str(_config(tmp_path, hier_csv))]) == 0
    out = tmp_path / "results" / "e2e"
    for name in (
        "manifest.json",
        "intervals.parquet",
        "metrics.parquet",
        "contrasts.parquet",
    ):
        assert (out / name).is_file(), name
    figures = {path.name for path in (out / "figures").glob("*.pdf")}
    assert {
        "rolling_coverage_pi.pdf",
        "rolling_coverage_pid_theta.pdf",
        "contrasts_coverage_deviation.pdf",
    } <= figures
    manifest = json.loads((out / "manifest.json").read_text())
    expected_keys = {
        "config",
        "git_sha",
        "git_dirty",
        "timestamp_utc",
        "input_sha256",
        "seed",
        "versions",
        "diagnostics",
    }
    assert expected_keys <= manifest.keys()
    assert manifest["input_sha256"] == file_sha256(hier_csv)
    assert manifest["diagnostics"]["imputed_cells"] == 1
    intervals = pd.read_parquet(out / "intervals.parquet")
    assert len(intervals) == 3 * (1 + 2 + 2) * (26 - 8)
    contrasts = pd.read_parquet(out / "contrasts.parquet")
    primary = contrasts[contrasts["primary"]]
    assert len(primary) == 1
    assert primary["metric"].item() == "coverage_deviation"


def test_cli_refuses_to_overwrite_a_run(tmp_path: Path, hier_csv: Path) -> None:
    config = _config(tmp_path, hier_csv)
    assert main([str(config)]) == 0
    with pytest.raises(FileExistsError):
        main([str(config)])
