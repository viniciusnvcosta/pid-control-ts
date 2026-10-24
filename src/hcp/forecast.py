# ABOUTME: Generates causal rolling-origin probabilistic forecasts on the registered hierarchy grid.
# ABOUTME: Prepares prefix-only histories, assigns deterministic seeds, and returns immutable bundles without I/O.
"""Causal benchmark forecast orchestration."""

import time
import warnings
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from hcp.benchmark_batch import NeuralExecution, OriginForecastBatch
from hcp.benchmark_data import ActualPanel
from hcp.forecast_assembly import (
    _assemble_model_bundles,
    _seed_key,
    _source_kind,
    _tables,
)
from hcp.forecast_classical import (
    arima_prediction,
    prophet_prediction,
    sarima_prediction,
)
from hcp.forecast_deepar import enhanced_deepar_prediction
from hcp.forecast_models import (
    ForecastSettings,
    PreparedHistory,
    arima_samples,
    deepar_samples,
    sarima_samples,
)
from hcp.forecast_neural import transformer_prediction
from hcp.forecast_preparation import (
    _prepare_origin,
    _validate_inputs,
)
from hcp.forecast_quantile import qr_prediction
from hcp.forecast_types import (
    ForecastPlan,
    ForecastResult,
    NativePrediction,
    V2ModelSettings,
    prediction_quantiles,
)

ForecastAdapter = Callable[
    [
        tuple[PreparedHistory, ...],
        ForecastSettings,
        np.random.SeedSequence,
        NeuralExecution,
    ],
    np.ndarray | NativePrediction,
]


def _arima_adapter(histories, settings, seed, execution):
    return (
        arima_prediction(histories, settings, seed)
        if isinstance(settings, V2ModelSettings)
        else arima_samples(histories, settings, seed)
    )


def _sarima_adapter(histories, settings, seed, execution):
    return (
        sarima_prediction(histories, settings, seed)
        if isinstance(settings, V2ModelSettings)
        else sarima_samples(histories, settings, seed)
    )


def _deepar_adapter(histories, settings, seed, execution):
    samples = deepar_samples(histories, settings, seed, execution)
    if isinstance(settings, V2ModelSettings):
        return NativePrediction(
            tuple(h.node for h in histories), settings.ahead, "samples", samples=samples
        )
    return samples


def _prophet_adapter(histories, settings, seed, execution):
    return prophet_prediction(histories, settings, seed)


def _qr_adapter(histories, settings, seed, execution):
    return qr_prediction(histories, settings, seed)


FORECASTERS: dict[str, ForecastAdapter] = {
    "arima": _arima_adapter,
    "sarima": _sarima_adapter,
    "deepar": _deepar_adapter,
    "prophet": _prophet_adapter,
    "transformer": transformer_prediction,
    "qr_l1": _qr_adapter,
    "deepar_enhanced": enhanced_deepar_prediction,
}


@dataclass(frozen=True)
class ForecastConfig:
    """Validated rolling-origin forecast configuration."""

    models: tuple[str, ...] = ("arima", "sarima", "deepar")
    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9)
    min_train_weeks: int = 60
    refit_every: int = 1
    root_seed: int = 0
    seed_spawn_key: tuple[int, ...] = (0,)
    settings: ForecastSettings | V2ModelSettings = field(
        default_factory=ForecastSettings
    )
    execution: NeuralExecution = field(default_factory=NeuralExecution.cpu)
    protocol_version: int = 1
    v2_settings: V2ModelSettings | None = None

    def __post_init__(self) -> None:
        if not self.models or len(set(self.models)) != len(self.models):
            raise ValueError("models must be non-empty and unique")
        unknown = set(self.models) - set(FORECASTERS)
        if unknown:
            raise ValueError(f"unknown model keys: {sorted(unknown)}")
        if (
            len(self.quantiles) != 3
            or any(not 0.0 < value < 1.0 for value in self.quantiles)
            or tuple(sorted(self.quantiles)) != self.quantiles
        ):
            raise ValueError("quantiles must be three strictly ordered probabilities")
        if self.min_train_weeks <= 0:
            raise ValueError("min_train_weeks must be positive")
        if self.refit_every != 1:
            raise ValueError("benchmark forecasting requires weekly refits")
        if self.settings.ahead != 4:
            raise ValueError("benchmark forecasting requires ahead=4")
        if self.settings.n_samples <= 0:
            raise ValueError("n_samples must be positive")
        if self.root_seed < 0 or any(value < 0 for value in self.seed_spawn_key):
            raise ValueError("seed values must be non-negative")
        if self.protocol_version not in (1, 2):
            raise ValueError("protocol_version must be 1 or 2")
        if (
            self.protocol_version == 1
            and self.models != ("arima", "sarima", "deepar")
            and set(self.models)
            & {"prophet", "transformer", "qr_l1", "deepar_enhanced"}
        ):
            raise ValueError("v2 models require protocol_version=2")
        if self.protocol_version == 2:
            expected = (
                "arima",
                "sarima",
                "deepar",
                "prophet",
                "transformer",
                "qr_l1",
                "deepar_enhanced",
            )
            if self.models != expected or self.v2_settings is None:
                raise ValueError(
                    "v2 requires its registered model order and model settings"
                )
            if self.settings is not self.v2_settings:
                raise ValueError(
                    "v2 settings must be shared by settings and v2_settings"
                )


def plan_model_forecasts(
    actuals: ActualPanel,
    dates: np.ndarray,
    forecast_dates: np.ndarray,
    cfg: ForecastConfig,
) -> ForecastPlan:
    """Validate a weekly grid and prepare causal histories without fitting.

    Args:
        actuals: Canonical truth used for expanding histories.
        dates: Weekly target dates.
        forecast_dates: Information cutoffs for those targets.
        cfg: Registered forecast settings and deterministic seed addresses.

    Returns:
        An owned plan whose eligible keys follow model-major/origin-minor order.
    """
    targets, origins, target_index = _validate_inputs(actuals, dates, forecast_dates)
    prepared = [
        _prepare_origin(actuals, origin, cfg.min_train_weeks) for origin in origins
    ]
    for target, item in zip(targets, prepared, strict=True):
        if item is not None and any(
            history.dates[-1] != target - np.timedelta64(28, "D") for history in item[0]
        ):
            raise ValueError("prepared history violates the horizon-four cutoff")
    if not any(item is not None for item in prepared):
        raise ValueError("no eligible origin has sufficient history for every node")

    arrays = [
        np.array(array, copy=True)
        for array in (actuals.dates, actuals.y, actuals.observed)
    ]
    for array in arrays:
        array.setflags(write=False)
    owned_actuals = ActualPanel(arrays[0], tuple(actuals.nodes), arrays[1], arrays[2])
    return ForecastPlan(
        owned_actuals,
        targets,
        origins,
        target_index,
        tuple(prepared),
        tuple(
            (model, index)
            for model in cfg.models
            for index, item in enumerate(prepared)
            if item is not None
        ),
        cfg,
    )


def generate_origin_batch(
    plan: ForecastPlan,
    model: str,
    target_index: int,
    cfg: ForecastConfig,
) -> OriginForecastBatch:
    """Execute one eligible key using its fixed original seed address.

    Args:
        plan: Validated causal forecast plan.
        model: Registered model key.
        target_index: Original position in the target grid.
        cfg: The configuration used to prepare the plan.

    Returns:
        Owned native-unit quantiles and normalized diagnostics for one origin.
    """
    if cfg != plan.config:
        raise ValueError("configuration differs from forecast plan")
    if (
        isinstance(target_index, bool)
        or (model, target_index) not in plan.eligible_keys
    ):
        raise ValueError("unknown or ineligible model/origin key")
    model_name = model
    item = plan.prepared[target_index]
    adapter = FORECASTERS[model_name]
    origin_seed = np.random.SeedSequence(
        cfg.root_seed, spawn_key=_seed_key(plan, model, target_index)
    )
    diagnostic_rows = {
        name: []
        for name in ("forecast_diagnostics", "tuning", "quantile_repairs", "warnings")
    }
    warning_counts: Counter[str] = Counter()
    q = np.full((len(plan.actuals.nodes), 3), np.nan)
    started = time.perf_counter()
    histories, records = item
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            output = adapter(histories, cfg.settings, origin_seed, cfg.execution)
    except Exception as exc:
        raise RuntimeError(
            f"model {model_name!r} failed at origin "
            f"{pd.Timestamp(plan.origins[target_index]).date()}: {exc}"
        ) from exc
    warning_counts.update(warning.category.__name__ for warning in caught)
    origin = pd.Timestamp(plan.origins[target_index])
    for warning in caught:
        diagnostic_rows["warnings"].append(
            {
                "category": warning.category.__name__,
                "message": str(warning.message),
                "source": f"{warning.filename}:{warning.lineno}",
                "model": model_name,
                "node": None,
                "origin": origin,
                "candidate": None,
                "phase": "outer_fit",
                "count": 1,
            }
        )
    if isinstance(output, NativePrediction):
        if (
            output.nodes != tuple(history.node for history in histories)
            or output.ahead != cfg.settings.ahead
        ):
            raise ValueError(
                "native prediction nodes or horizon differ from prepared histories"
            )
        samples = output.samples
        q = prediction_quantiles(output)
        if output.diagnostics:
            warning_counts.update(output.diagnostics.get("warning_counts", {}))
            for key in (
                "forecast_diagnostics",
                "tuning",
                "quantile_repairs",
                "warnings",
            ):
                for row in output.diagnostics.get(key, ()):
                    diagnostic_rows[key].append(
                        {"model": model_name, "origin": origin, **row}
                    )
    else:
        samples = np.asarray(output)
    if samples is not None:
        expected = (
            len(plan.actuals.nodes),
            cfg.settings.ahead,
            cfg.settings.n_samples,
        )
        if samples.shape != expected:
            raise ValueError(
                f"model {model_name!r} returned shape {samples.shape}, expected {expected}"
            )
        if not np.isfinite(samples).all():
            raise ValueError(f"model {model_name!r} returned non-finite samples")
        if not isinstance(output, NativePrediction):
            q = np.quantile(
                samples[:, cfg.settings.ahead - 1, :], cfg.quantiles, axis=-1
            ).T
    if np.any(np.diff(q, axis=-1) < 0):
        raise ValueError(f"model {model_name!r} returned crossed quantiles")
    interpolation = tuple(
        {"origin": str(pd.Timestamp(plan.origins[target_index]).date()), **record}
        for record in records
    )
    return OriginForecastBatch(
        model,
        target_index,
        plan.targets[target_index],
        plan.origins[target_index],
        origin_seed.spawn_key,
        plan.actuals.nodes,
        q,
        _source_kind(model, cfg),
        _tables(diagnostic_rows),
        dict(sorted(warning_counts.items())),
        interpolation,
        time.perf_counter() - started,
    )


def assemble_model_bundles(
    plan: ForecastPlan,
    batches: tuple[OriginForecastBatch, ...],
    cfg: ForecastConfig,
) -> ForecastResult:
    """Assemble complete origin batches into registered model and target order.

    Args:
        plan: Validated causal forecast plan.
        batches: Every eligible batch exactly once, in any order.
        cfg: The configuration used to prepare the plan.

    Returns:
        Hierarchy bundles, long forecasts, summaries and diagnostic tables.

    Raises:
        ValueError: If a batch is missing, duplicated, or differs from the plan.
    """
    return _assemble_model_bundles(plan, batches, cfg)


def generate_model_bundles(
    actuals: ActualPanel,
    dates: np.ndarray,
    forecast_dates: np.ndarray,
    cfg: ForecastConfig,
) -> ForecastResult:
    """Generate all configured forecasts through the pure origin batch API.

    Args:
        actuals: Canonical truth used to prepare expanding histories.
        dates: Weekly target dates.
        forecast_dates: Information cutoff paired with each target date.
        cfg: Validated model and seed settings.

    Returns:
        Immutable model bundles plus long forecasts and fit diagnostics.
    """
    plan = plan_model_forecasts(actuals, dates, forecast_dates, cfg)
    batches = tuple(
        generate_origin_batch(plan, model, index, cfg)
        for model, index in plan.eligible_keys
    )
    return assemble_model_bundles(plan, batches, cfg)
