# ABOUTME: Validates isolated v2 benchmark configuration and legacy version selection.
# ABOUTME: Guards generated model order and rejects invalid v2 settings before fitting.

from dataclasses import replace

import pytest

from hcp.config import BenchmarkSettings, benchmark_configs, load_config
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
