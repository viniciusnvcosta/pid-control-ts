# ABOUTME: Unit tests for run configuration validation and the in-memory experiment on synthetic data.
# ABOUTME: Checks config errors, arm grid, raw/none parity with base quantiles, and the single primary contrast.
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from hcp.run import RunConfig, load_config, run_experiment
from tests.helpers import write_config

BASE = {
    "run_id": "unit",
    "root_seed": 1,
    "data_path": "unused.csv",
    "output_dir": "results",
}
SMALL = {
    "controllers": ("raw", "pi"),
    "lr_grid": (0.1,),
    "T_burnin": 2,
    "mint_warmup": 5,
    "rolling_window": 5,
    "blocks": (4,),
    "n_boot": 10,
}


def test_load_config_converts_lists_and_paths(tmp_path: Path) -> None:
    path = write_config(
        tmp_path / "c.toml", {**BASE, "blocks": [8, 4], "lr_grid": [0.1, 0.5]}
    )
    cfg = load_config(path)
    assert cfg.blocks == (8, 4)
    assert cfg.lr_grid == (0.1, 0.5)
    assert isinstance(cfg.data_path, Path)


def test_unknown_key_raises(tmp_path: Path) -> None:
    path = write_config(tmp_path / "c.toml", {**BASE, "surprise": 1})
    with pytest.raises(ValueError, match="invalid run configuration"):
        load_config(path)


@pytest.mark.parametrize(
    "override",
    [
        {"run_id": "a/b"},
        {"root_seed": -1},
        {"root_seed": True},
        {"reconcilers": ("bottom_up", "mint_shrink")},
        {"reconcilers": ("none", "nope")},
        {"controllers": ("nope",)},
        {"lr": 0.3},
        {"blocks": ()},
        {"n_boot": 0},
    ],
)
def test_invalid_config_raises(override: dict) -> None:
    with pytest.raises(ValueError):
        RunConfig(**{**BASE, **override})


def test_run_experiment_arm_grid_and_parity(hier_bundle) -> None:
    cfg = RunConfig(**BASE, **SMALL)
    result = run_experiment(cfg, hier_bundle)
    n_weeks = 26 - 8
    assert len(result.intervals) == 3 * (1 + 1) * n_weeks
    assert np.isfinite(result.intervals[["lo", "hi"]].to_numpy()).all()
    raw = result.intervals.query("reconciler == 'none' and controller == 'raw'")
    np.testing.assert_array_equal(raw["lo"], hier_bundle.q[0, 8:, 0])
    np.testing.assert_array_equal(raw["hi"], hier_bundle.q[0, 8:, 2])
    assert len(result.metrics) == 6
    assert result.contrasts["primary"].sum() == 1
    assert result.diagnostics["n_weeks"] == n_weeks
    assert result.diagnostics["imputed_cells"] == 1
    assert set(result.diagnostics["crossings"]) == {"none", "bottom_up", "mint_shrink"}


def test_truth_incoherence_is_scoped_to_the_evaluation_window(hier_bundle) -> None:
    # week index 2 is before t0 = 8, so a leaf-sum gap there must not leak into
    # a diagnostic that is supposed to describe the evaluation window only.
    y = hier_bundle.y.copy()
    y[0, 2] += 100.0
    bundle = replace(hier_bundle, y=y)
    cfg = RunConfig(**BASE, **SMALL)
    result = run_experiment(cfg, bundle)
    assert result.diagnostics["truth_incoherence"]["max_abs"] == 0.0
