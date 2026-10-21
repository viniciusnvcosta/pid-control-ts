# ABOUTME: End-to-end tests for atomic probabilistic-model benchmark orchestration and artifacts.
# ABOUTME: Covers writer completeness, manifest provenance, stale staging, failures, and the benchmark CLI.
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import hcp.run as run_module
from hcp.benchmark import BenchmarkResult
from hcp.config import BenchmarkSettings
from hcp.data import FORECASTER, file_sha256
from hcp.forecast import FORECASTERS
from hcp.run import RunConfig, main, write_benchmark_artifacts

MODELS = ("arima", "sarima", "deepar", "covidhub")


def _config(tmp_path: Path) -> RunConfig:
    return RunConfig(
        run_id="benchmark-e2e",
        root_seed=7,
        data_path=tmp_path / "input.csv",
        output_dir=tmp_path / "results",
        design="benchmark",
        controllers=("raw", "pi"),
        reconcilers=("none", "bottom_up", "mint_shrink", "mint_median"),
        benchmark=BenchmarkSettings(),
    )


def _result() -> BenchmarkResult:
    hierarchy = pd.DataFrame(
        [
            {
                "block": 8,
                "model": model,
                "level": level,
                "controller": controller,
                "lr": 0.0 if controller == "raw" else 0.1,
                "treatment": treatment,
                "control": "none",
                "metric": "coverage_deviation",
                "estimate": 0.0,
                "low": -0.1,
                "high": 0.1,
                "p_value": 1.0,
                "primary": False,
                "p_holm": 1.0,
            }
            for model in MODELS
            for level in ("leaf_mean", "us")
            for controller in ("raw", "pi")
            for treatment in ("bottom_up", "mint_shrink", "mint_median")
        ]
    )
    level_metrics = pd.DataFrame(
        [
            {
                "model": model,
                "level": level,
                "reconciler": reconciler,
                "controller": "pi",
                "lr": 0.1,
                "coverage_deviation": 0.01,
                "scaled_interval_score": 1.0,
            }
            for model in MODELS
            for level in ("leaf_mean", "us")
            for reconciler in ("none", "mint_shrink")
        ]
    )
    placeholder = pd.DataFrame({"model": ["arima"], "value": [1.0]})
    return BenchmarkResult(
        base_forecasts=placeholder,
        intervals=placeholder,
        metrics=placeholder,
        level_metrics=level_metrics,
        hierarchy_contrasts=hierarchy,
        model_contrasts=placeholder,
        diagnostics={
            "evaluation_start": "2021-01-02",
            "evaluation_end": "2022-03-05",
            "n_weeks": 61,
            "nodes": ["us", "aa"],
            "model_order": list(MODELS),
            "models": {model: {"crossings": 0} for model in MODELS},
        },
    )


def _forecast_diagnostics() -> dict:
    return {
        "models": {
            model: {
                "fit_count": 2,
                "warning_counts": {},
                "elapsed_seconds": 0.25,
                "interpolation": [{"node": "aa", "interpolated_observations": 1}],
            }
            for model in MODELS[:-1]
        },
        "seed": {"root_seed": 7, "spawn_key": [0], "model_order": list(MODELS[:-1])},
        "canonical_truth": {"finite_disagreements": 0},
    }


def test_benchmark_writer_creates_complete_atomic_artifact_set(tmp_path: Path) -> None:
    cfg = _config(tmp_path)

    out = write_benchmark_artifacts(cfg, _result(), _forecast_diagnostics(), "abc123")

    assert out == tmp_path / "results" / "benchmark-e2e"
    assert {path.name for path in out.glob("*.parquet")} == {
        "base_forecasts.parquet",
        "intervals.parquet",
        "metrics.parquet",
        "level_metrics.parquet",
        "hierarchy_contrasts.parquet",
        "model_contrasts.parquet",
    }
    assert {path.name for path in (out / "figures").glob("*.pdf")} == {
        "forest-leaf-mean-pi.pdf",
        "forest-us-pi.pdf",
        "coverage-score-tradeoff.pdf",
        "hierarchy-effect.pdf",
    }
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["config"]["design"] == "benchmark"
    assert manifest["input_sha256"] == "abc123"
    assert {"darts", "torch", "lightning"} <= manifest["versions"].keys()
    assert manifest["execution"] == {
        "device": "cpu",
        "devices": 1,
        "data_loader_workers": 0,
        "deterministic_algorithms": True,
    }
    assert manifest["seed"]["forecast"]["spawn_key"] == [0]
    assert manifest["seed"]["evaluation"]["spawn_key"] == [1]
    assert manifest["models"]["classes"] == {
        "arima": "darts.models.ARIMA",
        "sarima": "darts.models.ARIMA",
        "deepar": "darts.models.RNNModel",
    }
    assert manifest["models"]["hyperparameters"]["n_samples"] == 500
    assert manifest["diagnostics"]["benchmark"]["n_weeks"] == 61
    assert manifest["diagnostics"]["forecast"]["models"]["arima"]["fit_count"] == 2
    assert manifest["diagnostics"]["forecast"]["canonical_truth"] == {
        "finite_disagreements": 0
    }
    assert "final revised truth" in manifest["limitations"]["D15"]
    assert manifest["analysis_note"] == "exploratory benchmark; no unadjusted primary"
    assert not out.with_name(f".{out.name}.partial").exists()


def test_benchmark_writer_refuses_stale_staging_and_never_publishes_partial_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _config(tmp_path)
    out = cfg.output_dir / cfg.run_id
    staging = out.with_name(f".{out.name}.partial")
    staging.mkdir(parents=True)

    with pytest.raises(FileExistsError):
        write_benchmark_artifacts(cfg, _result(), _forecast_diagnostics(), "abc123")
    assert not out.exists()

    staging.rmdir()

    def fail_figures(*args, **kwargs):
        raise RuntimeError("figure generation failed")

    monkeypatch.setattr("hcp.run.benchmark_figures", fail_figures)
    with pytest.raises(RuntimeError, match="figure generation failed"):
        write_benchmark_artifacts(cfg, _result(), _forecast_diagnostics(), "abc123")
    assert not out.exists()


def _benchmark_csv(path: Path) -> Path:
    rows = []
    for index, target in enumerate(pd.date_range("2019-01-05", periods=132, freq="7D")):
        aa = 20.0 + index / 10.0 + np.sin(index / 5.0)
        bb = 30.0 + index / 20.0 + np.cos(index / 7.0)
        for node_index, (node, actual) in enumerate(
            (("us", aa + bb), ("aa", aa), ("bb", bb))
        ):
            median = actual + (node_index + 1) * np.sin(index / 4.0)
            rows.append(
                {
                    "ahead": 4,
                    "geo_value": node,
                    "forecaster": FORECASTER,
                    "forecast_date": target - pd.Timedelta(days=26),
                    "target_end_date": target,
                    "forecast_0.1": median - 3.0,
                    "forecast_0.5": median,
                    "forecast_0.9": median + 3.0,
                    "actual": actual,
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _benchmark_toml(tmp_path: Path, csv: Path) -> Path:
    path = tmp_path / "benchmark.toml"
    path.write_text(
        "\n".join(
            [
                'run_id = "benchmark-cli"',
                "root_seed = 7",
                f'data_path = "{csv}"',
                f'output_dir = "{tmp_path / "results"}"',
                f'expected_sha256 = "{file_sha256(csv)}"',
                'design = "benchmark"',
                'controllers = ["raw", "pi"]',
                'reconcilers = ["none", "bottom_up", "mint_shrink", "mint_median"]',
                "lr = 0.1",
                "lr_grid = [0.1]",
                "T_burnin = 2",
                "mint_warmup = 2",
                "rolling_window = 5",
                "blocks = [4]",
                "n_boot = 5",
                "min_eval_weeks = 60",
                "[benchmark]",
            ]
        )
        + "\n"
    )
    return path


def test_benchmark_cli_uses_real_data_boundaries_and_writes_common_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    csv = _benchmark_csv(tmp_path / "benchmark.csv")
    config = _benchmark_toml(tmp_path, csv)
    calls = []
    io_calls = {"read": 0, "hash": 0}
    real_read = run_module.read_forecasts
    real_hash = run_module.file_sha256

    def counted_read(*args, **kwargs):
        io_calls["read"] += 1
        return real_read(*args, **kwargs)

    def counted_hash(*args, **kwargs):
        io_calls["hash"] += 1
        return real_hash(*args, **kwargs)

    def fake_adapter(histories, settings, seed):
        calls.append((tuple(history.node for history in histories), seed.spawn_key))
        samples = np.empty((len(histories), settings.ahead, settings.n_samples))
        offsets = np.linspace(-4.0, 4.0, settings.n_samples)
        for row, history in enumerate(histories):
            center = history.values[-1] * (0.96 + 0.01 * row)
            center += (row + 1) * np.sin(history.values[-1] / 3.0)
            samples[row] = center + offsets
        return samples

    for model in tuple(FORECASTERS):
        monkeypatch.setitem(FORECASTERS, model, fake_adapter)
    monkeypatch.setattr(run_module, "read_forecasts", counted_read)
    monkeypatch.setattr(run_module, "file_sha256", counted_hash)

    assert main([str(config)]) == 0

    out = tmp_path / "results" / "benchmark-cli"
    expected_columns = {
        "base_forecasts.parquet": {
            "model",
            "node",
            "week",
            "forecast_date",
            "q_lo",
            "q_med",
            "q_hi",
        },
        "intervals.parquet": {"model", "node", "week", "reconciler", "controller"},
        "metrics.parquet": {"model", "node", "reconciler", "controller"},
        "level_metrics.parquet": {"model", "level", "reconciler", "controller"},
        "hierarchy_contrasts.parquet": {"model", "treatment", "control", "p_holm"},
        "model_contrasts.parquet": {"model", "reference_model", "p_holm"},
    }
    for name, columns in expected_columns.items():
        assert columns <= set(pd.read_parquet(out / name).columns)
    manifest = json.loads((out / "manifest.json").read_text())
    diagnostics = manifest["diagnostics"]["benchmark"]
    assert diagnostics["model_order"] == ["arima", "sarima", "deepar", "covidhub"]
    assert diagnostics["n_weeks"] >= 60
    assert diagnostics["nodes"] == ["us", "aa", "bb"]
    assert len(calls) > 0
    assert io_calls == {"read": 1, "hash": 1}
    fit_calls = len(calls)
    with pytest.raises(FileExistsError):
        main([str(config)])
    assert len(calls) == fit_calls
    assert io_calls == {"read": 1, "hash": 1}
