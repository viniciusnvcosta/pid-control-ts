# ABOUTME: Validates isolated v2 benchmark configuration and legacy version selection.
# ABOUTME: Guards generated model order and rejects invalid v2 settings before fitting.

from dataclasses import replace

import pytest

from hcp.config import (
    BenchmarkSettings,
    benchmark_configs,
    config_manifest,
    load_config,
)
from hcp.config_benchmark_v2 import BenchmarkV2Settings
from tests.helpers import ROOT_DIR


def test_legacy_config_keeps_v1_settings() -> None:
    cfg = load_config(ROOT_DIR / "configs/model_benchmarks.toml")
    assert isinstance(cfg.benchmark, BenchmarkSettings)
    forecast, _ = benchmark_configs(cfg)
    assert cfg.benchmark.models == ("arima", "sarima", "deepar")
    assert forecast.protocol_version == 1


def test_v2_config_has_registered_order_and_defaults() -> None:
    cfg = load_config(ROOT_DIR / "configs/model_benchmarks_v2.toml")
    assert isinstance(cfg.benchmark, BenchmarkV2Settings)
    forecast, benchmark = benchmark_configs(cfg)
    assert forecast.protocol_version == 2
    assert forecast.models == (
        "arima",
        "sarima",
        "deepar",
        "prophet",
        "transformer",
        "qr_l1",
        "deepar_enhanced",
    )
    assert forecast.settings.ahead == 4 and forecast.settings.n_samples == 500
    assert forecast.v2_settings.neural_max_epochs == 100
    assert benchmark.reference_model == "covidhub"
    assert cfg.benchmark.execution.device == "cuda:0"
    assert cfg.benchmark.execution.required_gpu_name == "NVIDIA GeForce RTX 4080"
    assert cfg.benchmark.execution.cublas_workspace_config == ":4096:8"
    assert cfg.benchmark.execution.allow_tf32 is False
    assert cfg.benchmark.execution.precision == "32-true"
    assert config_manifest(cfg)["benchmark"]["execution"] == {
        "device": "cuda:0",
        "required_gpu_name": "NVIDIA GeForce RTX 4080",
        "cublas_workspace_config": ":4096:8",
        "allow_tf32": False,
        "precision": "32-true",
    }


@pytest.mark.parametrize(
    "execution",
    [
        {
            "device": "cuda:1",
            "required_gpu_name": "NVIDIA GeForce RTX 4080",
            "cublas_workspace_config": ":4096:8",
            "allow_tf32": False,
            "precision": "32-true",
        },
        {
            "device": "cuda:0",
            "required_gpu_name": "NVIDIA GeForce RTX 4080",
            "cublas_workspace_config": ":4096:8",
            "allow_tf32": True,
            "precision": "32-true",
        },
        {
            "device": "cuda:0",
            "required_gpu_name": "NVIDIA GeForce RTX 4080",
            "cublas_workspace_config": ":4096:8",
            "allow_tf32": False,
            "precision": "16-mixed",
        },
        {
            "device": "cuda:0",
            "required_gpu_name": "NVIDIA RTX 4090",
            "cublas_workspace_config": ":4096:8",
            "allow_tf32": False,
            "precision": "32-true",
        },
        {
            "device": "cuda:0",
            "required_gpu_name": "NVIDIA GeForce RTX 4080",
            "cublas_workspace_config": ":8192:8",
            "allow_tf32": False,
            "precision": "32-true",
        },
    ],
)
def test_v2_rejects_nonregistered_execution_mapping(execution) -> None:
    with pytest.raises(ValueError):
        BenchmarkV2Settings(execution=execution)


def test_v2_cpu_execution_is_available_only_as_explicit_fixture() -> None:
    try:
        from hcp.benchmark_batch import NeuralExecution
    except ImportError:
        pytest.fail("benchmark batch contracts are not implemented")

    assert NeuralExecution.cpu().device == "cpu"
    with pytest.raises(ValueError):
        BenchmarkV2Settings(execution=NeuralExecution.cpu())


def test_invalid_protocol_and_v2_settings_fail_before_forecasting() -> None:
    with pytest.raises(ValueError):
        BenchmarkV2Settings(protocol_version=1)
    with pytest.raises(ValueError):
        BenchmarkV2Settings(models=("arima", "qr_l1"))
    with pytest.raises(ValueError):
        BenchmarkV2Settings(forecast_seed_spawn_key=(5,))


def test_invalid_v2_arms_and_root_seed_fail_before_forecasting() -> None:
    cfg = load_config(ROOT_DIR / "configs/model_benchmarks_v2.toml")
    with pytest.raises(ValueError, match="controllers"):
        replace(cfg, controllers=("raw", "pi"))
    with pytest.raises(ValueError, match="root_seed"):
        benchmark_configs(replace(cfg, root_seed=1))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("run_id", "other-id"),
        ("data_path", ROOT_DIR / "data/other.csv"),
        ("expected_sha256", "0" * 64),
        ("root_seed", 20261022),
        ("alpha", 0.1),
        ("lr", 0.05),
        ("lr_grid", (0.1, 0.05)),
        ("Csat", 3.0),
        ("KI", 999.0),
        ("T_burnin", 6),
        ("ahead", 5),
        ("mint_warmup", 21),
        ("rolling_window", 11),
        ("blocks", (4, 8, 12)),
        ("n_boot", 1999),
        ("min_eval_weeks", 61),
    ],
)
def test_v2_rejects_changed_registered_protocol_constants(field, value) -> None:
    cfg = load_config(ROOT_DIR / "configs/model_benchmarks_v2.toml")
    with pytest.raises(ValueError):
        replace(cfg, **{field: value})
