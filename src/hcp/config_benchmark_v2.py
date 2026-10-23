# ABOUTME: Frozen protocol-v2 benchmark configuration, isolated from v1 constraints.
# ABOUTME: Validates registered model order, seed streams, and native forecast settings.
"""Version-two model benchmark settings."""

from dataclasses import dataclass, field

from hcp.data import FORECASTER
from hcp.forecast_types import V2ModelSettings

V2_MODELS = (
    "arima",
    "sarima",
    "deepar",
    "prophet",
    "transformer",
    "qr_l1",
    "deepar_enhanced",
)


@dataclass(frozen=True)
class BenchmarkV2Settings:
    """Immutable registered v2 model and generation settings."""

    protocol_version: int = 2
    models: tuple[str, ...] = V2_MODELS
    reference_model: str = "covidhub"
    reference_forecaster: str = FORECASTER
    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9)
    n_samples: int = 500
    min_train_weeks: int = 60
    refit_every: int = 1
    ahead: int = 4
    min_eval_weeks: int = 60
    forecast_seed_spawn_key: tuple[int, ...] = (0,)
    evaluation_seed_spawn_key: tuple[int, ...] = (1,)
    model_settings: V2ModelSettings = field(default_factory=V2ModelSettings)

    def __post_init__(self) -> None:
        for name in (
            "models",
            "quantiles",
            "forecast_seed_spawn_key",
            "evaluation_seed_spawn_key",
        ):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if isinstance(self.model_settings, dict):
            object.__setattr__(
                self, "model_settings", V2ModelSettings(**self.model_settings)
            )
        if self.protocol_version != 2:
            raise ValueError("BenchmarkV2Settings requires protocol_version=2")
        if self.models != V2_MODELS:
            raise ValueError(
                f"v2 models must be registered in exact order {V2_MODELS!r}"
            )
        if (
            self.reference_model != "covidhub"
            or self.reference_forecaster != FORECASTER
        ):
            raise ValueError("v2 reference must be covidhub / COVIDhub-4_week_ensemble")
        if self.quantiles != (0.1, 0.5, 0.9) or self.n_samples != 500:
            raise ValueError("v2 requires quantiles (0.1, 0.5, 0.9) and 500 paths")
        if (
            self.ahead,
            self.min_train_weeks,
            self.refit_every,
            self.min_eval_weeks,
        ) != (4, 60, 1, 60):
            raise ValueError(
                "v2 requires ahead=4, min_train_weeks=60, weekly refits, min_eval_weeks=60"
            )
        if self.forecast_seed_spawn_key != (0,) or self.evaluation_seed_spawn_key != (
            1,
        ):
            raise ValueError(
                "v2 seed spawn keys must be forecast=(0,), evaluation=(1,)"
            )
        if (self.model_settings.ahead, self.model_settings.n_samples) != (4, 500):
            raise ValueError("v2 model settings require ahead=4 and n_samples=500")
        if self.model_settings != V2ModelSettings():
            raise ValueError("v2 model settings must match the registered defaults")
