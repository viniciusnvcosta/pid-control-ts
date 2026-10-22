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

from hcp.benchmark_data import ActualPanel
from hcp.data import HierarchyBundle
from hcp.forecast_models import (
    ForecastSettings,
    PreparedHistory,
    arima_samples,
    deepar_samples,
    sarima_samples,
)
from hcp.forecast_types import DiagnosticsTables

ForecastAdapter = Callable[
    [tuple[PreparedHistory, ...], ForecastSettings, np.random.SeedSequence],
    np.ndarray,
]
FORECASTERS: dict[str, ForecastAdapter] = {
    "arima": arima_samples,
    "sarima": sarima_samples,
    "deepar": deepar_samples,
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
    settings: ForecastSettings = field(default_factory=ForecastSettings)

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


@dataclass(frozen=True)
class ForecastResult:
    """Generated hierarchy bundles, long forecasts, and fit diagnostics."""

    bundles: dict[str, HierarchyBundle]
    base_forecasts: pd.DataFrame
    diagnostics: dict
    tables: DiagnosticsTables = field(default_factory=DiagnosticsTables)


def _weekly(name: str, values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype="datetime64[ns]")
    if array.ndim != 1 or len(array) == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional date grid")
    if len(array) > 1 and not np.all(np.diff(array) == np.timedelta64(7, "D")):
        raise ValueError(f"{name} must be a strict weekly grid")
    return array


def _validate_inputs(
    actuals: ActualPanel, dates: np.ndarray, forecast_dates: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    actual_dates = _weekly("canonical actual dates", actuals.dates)
    targets = _weekly("target dates", dates)
    origins = _weekly("forecast dates", forecast_dates)
    if origins.shape != targets.shape:
        raise ValueError("forecast-date and target-date grids must have the same shape")
    if np.any(targets - origins != np.timedelta64(26, "D")):
        raise ValueError("benchmark forecasts require the registered 26-day lag")
    expected = (len(actuals.nodes), len(actual_dates))
    if actuals.y.shape != expected or actuals.observed.shape != expected:
        raise ValueError(f"canonical node/date shape must be {expected}")
    target_index = pd.Index(actual_dates).get_indexer(pd.Index(targets))
    if np.any(target_index < 0):
        raise ValueError("target date is unavailable in canonical actuals")
    return targets, origins, target_index


def _prepare_origin(
    actuals: ActualPanel,
    forecast_date: np.datetime64,
    min_train_weeks: int,
) -> tuple[tuple[PreparedHistory, ...], list[dict]] | None:
    prefix = actuals.dates <= forecast_date
    histories = []
    records = []
    for row, node in enumerate(actuals.nodes):
        dates = np.asarray(actuals.dates[prefix], dtype="datetime64[ns]")
        values = np.asarray(actuals.y[row, prefix], dtype=float)
        finite = np.flatnonzero(np.isfinite(values))
        if not len(finite):
            return None
        start = int(finite[0])
        dates = dates[start:].copy()
        values = values[start:].copy()
        raw = int(np.isfinite(values).sum())
        missing = ~np.isfinite(values)
        filled = pd.Series(values).interpolate(limit_area="inside").to_numpy()
        interpolated = int(np.sum(missing & np.isfinite(filled)))
        if len(filled) < min_train_weeks or not np.isfinite(filled).all():
            return None
        dates.setflags(write=False)
        filled.setflags(write=False)
        histories.append(PreparedHistory(node=node, dates=dates, values=filled))
        records.append(
            {
                "node": node,
                "training_start": str(pd.Timestamp(dates[0]).date()),
                "training_end": str(pd.Timestamp(dates[-1]).date()),
                "raw_observations": raw,
                "interpolated_observations": interpolated,
            }
        )
    return tuple(histories), records


def _bundle(
    actuals: ActualPanel,
    dates: np.ndarray,
    forecast_dates: np.ndarray,
    target_index: np.ndarray,
    quantiles: np.ndarray,
) -> HierarchyBundle:
    arrays = {
        "dates": dates.copy(),
        "forecast_dates": forecast_dates.copy(),
        "y": actuals.y[:, target_index].copy(),
        "q": quantiles,
        "imputed": np.zeros(quantiles.shape[:2], dtype=bool),
    }
    for array in arrays.values():
        array.setflags(write=False)
    return HierarchyBundle(nodes=actuals.nodes, **arrays)


def _long_forecasts(model: str, bundle: HierarchyBundle) -> pd.DataFrame:
    rows = []
    for week, forecast_date, q_week in zip(
        bundle.dates, bundle.forecast_dates, bundle.q.transpose(1, 0, 2), strict=True
    ):
        for node, values in zip(bundle.nodes, q_week, strict=True):
            rows.append(
                {
                    "model": model,
                    "node": node,
                    "week": week,
                    "forecast_date": forecast_date,
                    "q_lo": values[0],
                    "q_med": values[1],
                    "q_hi": values[2],
                }
            )
    return pd.DataFrame(rows)


def generate_model_bundles(
    actuals: ActualPanel,
    dates: np.ndarray,
    forecast_dates: np.ndarray,
    cfg: ForecastConfig,
) -> ForecastResult:
    """Generate every configured model on one causal weekly target grid.

    Args:
        actuals: Canonical truth used to prepare expanding histories.
        dates: Weekly target dates.
        forecast_dates: Information cutoff paired with each target date.
        cfg: Validated model and seed settings.

    Returns:
        Immutable model bundles plus long forecasts and fit diagnostics.

    Raises:
        ValueError: On invalid grids, unavailable targets, invalid samples, or no
            origin with sufficient history for every node.
        RuntimeError: If a configured adapter fails to fit or predict.
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

    root = np.random.SeedSequence(cfg.root_seed, spawn_key=cfg.seed_spawn_key)
    bundles: dict[str, HierarchyBundle] = {}
    frames = []
    model_diagnostics = {}
    for model_name, model_seed in zip(
        cfg.models, root.spawn(len(cfg.models)), strict=True
    ):
        adapter = FORECASTERS[model_name]
        q = np.full((len(actuals.nodes), len(targets), 3), np.nan)
        warning_counts: Counter[str] = Counter()
        interpolation = []
        fit_count = 0
        started = time.perf_counter()
        origin_seeds = model_seed.spawn(len(targets))
        for index, (item, origin_seed) in enumerate(
            zip(prepared, origin_seeds, strict=True)
        ):
            if item is None:
                continue
            histories, records = item
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    samples = np.asarray(adapter(histories, cfg.settings, origin_seed))
            except Exception as exc:
                raise RuntimeError(
                    f"model {model_name!r} failed at origin "
                    f"{pd.Timestamp(origins[index]).date()}: {exc}"
                ) from exc
            warning_counts.update(warning.category.__name__ for warning in caught)
            expected = (len(actuals.nodes), cfg.settings.ahead, cfg.settings.n_samples)
            if samples.shape != expected:
                raise ValueError(
                    f"model {model_name!r} returned shape {samples.shape}, expected {expected}"
                )
            if not np.isfinite(samples).all():
                raise ValueError(f"model {model_name!r} returned non-finite samples")
            q[:, index, :] = np.quantile(
                samples[:, cfg.settings.ahead - 1, :], cfg.quantiles, axis=-1
            ).T
            if np.any(np.diff(q[:, index, :], axis=-1) < 0):
                raise ValueError(f"model {model_name!r} returned crossed quantiles")
            interpolation.extend(
                {"origin": str(pd.Timestamp(origins[index]).date()), **record}
                for record in records
            )
            fit_count += 1
        bundle = _bundle(actuals, targets, origins, target_index, q)
        bundles[model_name] = bundle
        frames.append(_long_forecasts(model_name, bundle))
        model_diagnostics[model_name] = {
            "fit_count": fit_count,
            "warning_counts": dict(sorted(warning_counts.items())),
            "elapsed_seconds": time.perf_counter() - started,
            "interpolation": interpolation,
        }
    return ForecastResult(
        bundles=bundles,
        base_forecasts=pd.concat(frames, ignore_index=True),
        diagnostics={
            "models": model_diagnostics,
            "seed": {
                "root_seed": cfg.root_seed,
                "spawn_key": list(cfg.seed_spawn_key),
                "model_order": list(cfg.models),
            },
        },
    )
