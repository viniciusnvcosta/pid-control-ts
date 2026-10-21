# ABOUTME: Unit tests for run configuration validation and the in-memory experiment on synthetic data.
# ABOUTME: Checks config errors, arm grid, raw/none parity with base quantiles, and the single primary contrast.
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from hcp.config import BenchmarkSettings, benchmark_configs, config_manifest
from hcp.conformal import ControllerConfig
from hcp.data import FORECASTER
from hcp.run import (
    PRIMARY,
    RunConfig,
    _manifest,
    load_config,
    panel_config,
    run_experiment,
)
from tests.helpers import D21_ELIGIBLE, ROOT_DIR, write_config

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
        {"design": "both"},
        {"forecasters": ("x",)},
        {
            "design": "panel",
            "controllers": ("raw", "pi", "pid_theta"),
            "forecasters": (FORECASTER,),
        },
        {
            "design": "panel",
            "controllers": ("raw", "pi"),
            "forecasters": ("x",),
        },
        {
            "design": "panel",
            "controllers": ("raw", "pi"),
            "forecasters": (FORECASTER, FORECASTER),
        },
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
    assert set(result.diagnostics["root_crossings"]) == {
        "none",
        "bottom_up",
        "mint_shrink",
    }


def test_truth_incoherence_is_scoped_to_the_evaluation_window(hier_bundle) -> None:
    # week index 2 is before t0 = 8, so a leaf-sum gap there must not leak into
    # a diagnostic that is supposed to describe the evaluation window only.
    y = hier_bundle.y.copy()
    y[0, 2] += 100.0
    bundle = replace(hier_bundle, y=y)
    cfg = RunConfig(**BASE, **SMALL)
    result = run_experiment(cfg, bundle)
    assert result.diagnostics["truth_incoherence"]["max_abs"] == 0.0


def test_panel_config_maps_fields() -> None:
    cfg = RunConfig(
        **BASE,
        **SMALL,
        design="panel",
        forecasters=(FORECASTER, "synth-alt"),
    )
    panel_cfg = panel_config(cfg)
    assert panel_cfg.controller == ControllerConfig(
        alpha=cfg.alpha,
        lr=cfg.lr,
        Csat=cfg.Csat,
        KI=cfg.KI,
        T_burnin=cfg.T_burnin,
        ahead=cfg.ahead,
    )
    assert panel_cfg.lr_grid == cfg.lr_grid
    assert panel_cfg.reconcilers == cfg.reconcilers
    assert panel_cfg.controllers == cfg.controllers
    assert panel_cfg.mint_warmup == cfg.mint_warmup
    assert panel_cfg.rolling_window == cfg.rolling_window
    assert panel_cfg.blocks == cfg.blocks
    assert panel_cfg.n_boot == cfg.n_boot
    assert panel_cfg.root_seed == cfg.root_seed
    assert panel_cfg.primary_forecaster == FORECASTER


def test_panel_leaves_config_is_the_preregistered_run() -> None:
    cfg = load_config(ROOT_DIR / "configs" / "panel_leaves.toml")
    assert cfg.design == "panel"
    assert sorted(cfg.forecasters) == D21_ELIGIBLE
    assert cfg.forecasters[0] == FORECASTER


def test_model_benchmark_config_is_the_preregistered_protocol() -> None:
    cfg = load_config(ROOT_DIR / "configs" / "model_benchmarks.toml")
    assert cfg.design == "benchmark"
    assert cfg.run_id == "model-benchmarks-v1"
    assert cfg.root_seed == 20261021
    assert cfg.benchmark == BenchmarkSettings()
    forecast_cfg, benchmark_cfg = benchmark_configs(cfg)
    assert forecast_cfg.models == ("arima", "sarima", "deepar")
    assert forecast_cfg.quantiles == (0.1, 0.5, 0.9)
    assert forecast_cfg.min_train_weeks == 60
    assert forecast_cfg.refit_every == 1
    assert forecast_cfg.settings.n_samples == 500
    assert forecast_cfg.settings.arima_order == (3, 0, 0)
    assert forecast_cfg.settings.sarima_seasonal_order == (1, 0, 0, 52)
    assert forecast_cfg.settings.deepar_model == "LSTM"
    assert forecast_cfg.settings.deepar_input_chunk_length == 12
    assert forecast_cfg.settings.deepar_training_length == 16
    assert forecast_cfg.settings.deepar_hidden_dim == 25
    assert forecast_cfg.settings.deepar_n_rnn_layers == 1
    assert forecast_cfg.settings.deepar_dropout == 0.0
    assert forecast_cfg.settings.deepar_batch_size == 32
    assert forecast_cfg.settings.deepar_n_epochs == 10
    assert benchmark_cfg.reference_model == "covidhub"
    assert cfg.benchmark.reference_forecaster == FORECASTER
    assert benchmark_cfg.min_eval_weeks == 60
    assert benchmark_cfg.panel.reconcilers == (
        "none",
        "bottom_up",
        "mint_shrink",
        "mint_median",
    )
    assert benchmark_cfg.panel.controllers == ("raw", "pi")
    assert benchmark_cfg.panel.controller.lr == 0.1
    assert benchmark_cfg.panel.blocks == (8, 4, 12)
    assert benchmark_cfg.panel.n_boot == 2000
    assert forecast_cfg.seed_spawn_key == (0,)
    assert benchmark_cfg.seed_spawn_key == (1,)
    assert forecast_cfg.seed_spawn_key != benchmark_cfg.seed_spawn_key


def test_benchmark_settings_are_required_only_for_benchmark_design() -> None:
    with pytest.raises(ValueError, match="benchmark settings"):
        RunConfig(**BASE, benchmark=BenchmarkSettings())
    with pytest.raises(ValueError, match="benchmark settings"):
        RunConfig(**BASE, design="benchmark", controllers=("raw", "pi"))


@pytest.mark.parametrize(
    "override",
    [
        {"models": ("arima", "unknown", "deepar")},
        {"models": ("arima", "arima", "deepar")},
        {"ahead": 1},
        {"refit_every": 2},
        {"quantiles": (0.5, 0.1, 0.9)},
        {"n_samples": 499},
        {"arima_order": (2, 0, 0)},
        {"sarima_seasonal_order": (1, 0, 0, 12)},
        {"deepar_model": "GRU"},
        {"deepar_input_chunk_length": 11},
        {"deepar_training_length": 15},
        {"deepar_hidden_dim": 24},
        {"deepar_n_rnn_layers": 2},
        {"deepar_dropout": 0.1},
        {"deepar_batch_size": 16},
        {"deepar_n_epochs": 9},
        {"reference_model": "arima"},
    ],
)
def test_invalid_benchmark_model_settings_raise(override: dict) -> None:
    with pytest.raises(ValueError):
        BenchmarkSettings(**override)


@pytest.mark.parametrize(
    "override",
    [
        {"controllers": ("raw", "pi", "pid_theta")},
        {"reconcilers": ("bottom_up", "mint_shrink", "mint_median")},
    ],
)
def test_invalid_benchmark_run_arms_raise(override: dict) -> None:
    values = {
        **BASE,
        "design": "benchmark",
        "controllers": ("raw", "pi"),
        "reconcilers": ("none", "bottom_up", "mint_shrink", "mint_median"),
        "benchmark": BenchmarkSettings(),
        **override,
    }
    with pytest.raises(ValueError):
        RunConfig(**values)


def test_existing_config_manifest_shape_omits_absent_benchmark() -> None:
    for name in ("us_hier.toml", "panel_leaves.toml"):
        manifest = config_manifest(load_config(ROOT_DIR / "configs" / name))
        assert "benchmark" not in manifest


def test_us_design_manifest_keys_unchanged() -> None:
    cfg = RunConfig(**BASE, **SMALL)
    manifest = _manifest(
        cfg,
        "deadbeef",
        primary=PRIMARY,
        seed={"root_seed": cfg.root_seed, "block_children": len(cfg.blocks)},
        diagnostics={},
    )
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
    assert "benchmark" not in manifest["config"]
