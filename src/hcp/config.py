# ABOUTME: Experiment configuration: RunConfig, TOML loading, and the controller/panel config mappers.
# ABOUTME: Read-only: this module only parses TOML into a validated RunConfig and never writes files.
"""Run configuration.

Validates ``RunConfig`` (see docs/protocol-decisions.md) and maps it onto the
``ControllerConfig`` and ``PanelConfig`` consumed by ``hcp.conformal`` and ``hcp.panel``.
"""

import tomllib
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from hcp.benchmark import BenchmarkConfig
from hcp.config_benchmark_v2 import BenchmarkV2Settings
from hcp.conformal import CONTROLLERS, ControllerConfig
from hcp.data import FORECASTER
from hcp.forecast import ForecastConfig
from hcp.forecast_models import ForecastSettings
from hcp.panel import PanelConfig
from hcp.reconcile import RECONCILERS


@dataclass(frozen=True)
class BenchmarkSettings:
    """Frozen pre-registered probabilistic model benchmark settings."""

    models: tuple[str, ...] = ("arima", "sarima", "deepar")
    reference_model: str = "covidhub"
    reference_forecaster: str = FORECASTER
    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9)
    n_samples: int = 500
    min_train_weeks: int = 60
    refit_every: int = 1
    ahead: int = 4
    arima_order: tuple[int, int, int] = (3, 0, 0)
    sarima_seasonal_order: tuple[int, int, int, int] = (1, 0, 0, 52)
    deepar_model: str = "LSTM"
    deepar_input_chunk_length: int = 12
    deepar_training_length: int = 16
    deepar_hidden_dim: int = 25
    deepar_n_rnn_layers: int = 1
    deepar_dropout: float = 0.0
    deepar_batch_size: int = 32
    deepar_n_epochs: int = 10
    min_eval_weeks: int = 60
    forecast_seed_spawn_key: tuple[int, ...] = (0,)
    evaluation_seed_spawn_key: tuple[int, ...] = (1,)

    def __post_init__(self) -> None:
        for name in (
            "models",
            "quantiles",
            "arima_order",
            "sarima_seasonal_order",
            "forecast_seed_spawn_key",
            "evaluation_seed_spawn_key",
        ):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        fixed = {
            "models": ("arima", "sarima", "deepar"),
            "reference_model": "covidhub",
            "reference_forecaster": FORECASTER,
            "quantiles": (0.1, 0.5, 0.9),
            "n_samples": 500,
            "min_train_weeks": 60,
            "refit_every": 1,
            "ahead": 4,
            "arima_order": (3, 0, 0),
            "sarima_seasonal_order": (1, 0, 0, 52),
            "deepar_model": "LSTM",
            "deepar_input_chunk_length": 12,
            "deepar_training_length": 16,
            "deepar_hidden_dim": 25,
            "deepar_n_rnn_layers": 1,
            "deepar_dropout": 0.0,
            "deepar_batch_size": 32,
            "deepar_n_epochs": 10,
            "min_eval_weeks": 60,
            "forecast_seed_spawn_key": (0,),
            "evaluation_seed_spawn_key": (1,),
        }
        for name, expected in fixed.items():
            if getattr(self, name) != expected:
                raise ValueError(
                    f"benchmark {name} must be fixed at {expected!r}, "
                    f"got {getattr(self, name)!r}"
                )
        if self.reference_model in self.models:
            raise ValueError(
                "benchmark reference model collides with a generated model"
            )


@dataclass(frozen=True)
class RunConfig:
    """One pre-registered experiment; see docs/protocol-decisions.md."""

    run_id: str
    root_seed: int
    data_path: Path
    output_dir: Path
    expected_sha256: str | None = None
    alpha: float = 0.2
    lr: float = 0.1
    lr_grid: tuple[float, ...] = (1.0, 0.5, 0.1, 0.05)
    Csat: float = 2.0
    KI: float = 1000.0
    T_burnin: int = 5
    ahead: int = 4
    reconcilers: tuple[str, ...] = ("none", "bottom_up", "mint_shrink")
    controllers: tuple[str, ...] = ("raw", "pi", "pid_theta")
    mint_warmup: int = 20
    rolling_window: int = 10
    blocks: tuple[int, ...] = (8, 4, 12)
    n_boot: int = 2000
    design: str = "us"
    forecasters: tuple[str, ...] = (FORECASTER,)
    min_eval_weeks: int = 60
    max_leaf_imputed: float = 0.05
    duplicate_share: float = 0.95
    benchmark: BenchmarkSettings | BenchmarkV2Settings | None = None

    def __post_init__(self) -> None:
        if not self.run_id or Path(self.run_id).name != self.run_id:
            raise ValueError("run_id must be a bare directory name")
        if (
            isinstance(self.root_seed, bool)
            or not isinstance(self.root_seed, int)
            or self.root_seed < 0
        ):
            raise ValueError("root_seed must be a non-negative integer")
        for name in ("lr_grid", "reconcilers", "controllers", "blocks", "forecasters"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        for name in ("data_path", "output_dir"):
            object.__setattr__(self, name, Path(getattr(self, name)).expanduser())
        if isinstance(self.benchmark, dict):
            settings_type = (
                BenchmarkV2Settings
                if self.benchmark.get("protocol_version") == 2
                else BenchmarkSettings
            )
            object.__setattr__(self, "benchmark", settings_type(**self.benchmark))
        if self.design not in ("us", "panel", "benchmark"):
            raise ValueError(
                f"design must be 'us', 'panel', or 'benchmark', got {self.design!r}"
            )
        if self.design == "benchmark" and self.benchmark is None:
            raise ValueError("design 'benchmark' requires benchmark settings")
        if self.design != "benchmark" and self.benchmark is not None:
            raise ValueError("benchmark settings require design 'benchmark'")
        if self.design == "us":
            if self.forecasters != (FORECASTER,):
                raise ValueError("design 'us' requires forecasters == (FORECASTER,)")
        elif self.design == "panel":
            if "pid_theta" in self.controllers:
                raise ValueError("pid_theta is rejected for design 'panel' (D22)")
            if FORECASTER not in self.forecasters:
                raise ValueError(
                    f"forecasters must include the primary forecaster {FORECASTER!r}"
                )
            if len(set(self.forecasters)) != len(self.forecasters):
                raise ValueError("forecasters must not contain duplicates")
        else:
            expected_reconcilers = (
                "none",
                "bottom_up",
                "mint_shrink",
                "mint_median",
            )
            if self.reconcilers != expected_reconcilers:
                raise ValueError(
                    f"benchmark reconcilers must be {expected_reconcilers!r}"
                )
            assert self.benchmark is not None
            expected_controllers = (
                ("raw", "pi", "pi_norm")
                if isinstance(self.benchmark, BenchmarkV2Settings)
                else ("raw", "pi")
            )
            if self.controllers != expected_controllers:
                raise ValueError(
                    f"benchmark controllers must be {expected_controllers!r}"
                )
            if self.ahead != self.benchmark.ahead:
                raise ValueError("run and benchmark horizons must match")
            if self.min_eval_weeks != self.benchmark.min_eval_weeks:
                raise ValueError(
                    "run and benchmark minimum evaluation weeks must match"
                )
        if "none" not in self.reconcilers:
            raise ValueError("reconcilers must include the 'none' control")
        if unknown := set(self.reconcilers) - set(RECONCILERS):
            raise ValueError(f"unknown reconcilers: {sorted(unknown)}")
        if unknown := set(self.controllers) - set(CONTROLLERS):
            raise ValueError(f"unknown controllers: {sorted(unknown)}")
        if self.lr not in self.lr_grid:
            raise ValueError("lr must be one of lr_grid")
        if not self.blocks:
            raise ValueError("blocks must name at least one block length")
        if self.n_boot < 1:
            raise ValueError("n_boot must be positive")


def load_config(path: Path) -> RunConfig:
    """Read a TOML file into a validated ``RunConfig``.

    Raises:
        ValueError: On unknown or missing keys, or invalid values.
    """
    with Path(path).open("rb") as handle:
        values = tomllib.load(handle)
    try:
        if "benchmark" in values:
            settings_type = (
                BenchmarkV2Settings
                if values["benchmark"].get("protocol_version") == 2
                else BenchmarkSettings
            )
            values["benchmark"] = settings_type(**values["benchmark"])
        return RunConfig(**values)
    except TypeError as exc:
        raise ValueError(f"invalid run configuration: {exc}") from exc


def controller_config(cfg: RunConfig, lr: float) -> ControllerConfig:
    return ControllerConfig(
        alpha=cfg.alpha,
        lr=lr,
        Csat=cfg.Csat,
        KI=cfg.KI,
        T_burnin=cfg.T_burnin,
        ahead=cfg.ahead,
    )


def panel_config(cfg: RunConfig) -> PanelConfig:
    """Map a panel ``RunConfig`` onto the ``PanelConfig`` consumed by ``run_panel``."""
    return PanelConfig(
        controller=controller_config(cfg, cfg.lr),
        lr_grid=cfg.lr_grid,
        reconcilers=cfg.reconcilers,
        controllers=cfg.controllers,
        mint_warmup=cfg.mint_warmup,
        rolling_window=cfg.rolling_window,
        blocks=cfg.blocks,
        n_boot=cfg.n_boot,
        root_seed=cfg.root_seed,
        primary_forecaster=FORECASTER,
    )


def benchmark_configs(cfg: RunConfig) -> tuple[ForecastConfig, BenchmarkConfig]:
    """Map a benchmark run onto its numeric forecast and evaluation configs.

    Args:
        cfg: A validated benchmark run configuration.

    Returns:
        Forecast generation and common-window evaluation configurations.

    Raises:
        ValueError: If ``cfg`` is not a benchmark design.
    """
    if cfg.design != "benchmark" or cfg.benchmark is None:
        raise ValueError("benchmark_configs requires design 'benchmark'")
    settings = cfg.benchmark
    if isinstance(settings, BenchmarkV2Settings):
        if cfg.root_seed != 20261021:
            raise ValueError("v2 root_seed must be 20261021")
        forecast = ForecastConfig(
            models=settings.models,
            quantiles=settings.quantiles,
            min_train_weeks=settings.min_train_weeks,
            refit_every=settings.refit_every,
            root_seed=cfg.root_seed,
            seed_spawn_key=settings.forecast_seed_spawn_key,
            settings=settings.model_settings,
            protocol_version=2,
            v2_settings=settings.model_settings,
        )
        panel = replace(panel_config(cfg), primary_forecaster=settings.reference_model)
        benchmark = BenchmarkConfig(
            panel=panel,
            models=settings.models,
            reference_model=settings.reference_model,
            min_eval_weeks=settings.min_eval_weeks,
            seed_spawn_key=settings.evaluation_seed_spawn_key,
        )
        return forecast, benchmark
    forecast_settings = ForecastSettings(
        ahead=settings.ahead,
        n_samples=settings.n_samples,
        arima_order=settings.arima_order,
        sarima_seasonal_order=settings.sarima_seasonal_order,
        deepar_model=settings.deepar_model,
        deepar_input_chunk_length=settings.deepar_input_chunk_length,
        deepar_training_length=settings.deepar_training_length,
        deepar_hidden_dim=settings.deepar_hidden_dim,
        deepar_n_rnn_layers=settings.deepar_n_rnn_layers,
        deepar_dropout=settings.deepar_dropout,
        deepar_batch_size=settings.deepar_batch_size,
        deepar_n_epochs=settings.deepar_n_epochs,
    )
    forecast = ForecastConfig(
        models=settings.models,
        quantiles=settings.quantiles,
        min_train_weeks=settings.min_train_weeks,
        refit_every=settings.refit_every,
        root_seed=cfg.root_seed,
        seed_spawn_key=settings.forecast_seed_spawn_key,
        settings=forecast_settings,
    )
    panel = replace(panel_config(cfg), primary_forecaster=settings.reference_model)
    benchmark = BenchmarkConfig(
        panel=panel,
        models=settings.models,
        reference_model=settings.reference_model,
        min_eval_weeks=settings.min_eval_weeks,
        seed_spawn_key=settings.evaluation_seed_spawn_key,
    )
    return forecast, benchmark


def config_manifest(cfg: RunConfig) -> dict:
    """Return a serializable config mapping without absent optional sections."""
    values = asdict(cfg)
    if cfg.benchmark is None:
        values.pop("benchmark")
    return values
