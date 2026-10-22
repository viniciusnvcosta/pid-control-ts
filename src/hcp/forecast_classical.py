# ABOUTME: Local classical Darts adapters and seeded weekly Prophet predictive paths.
# ABOUTME: Preserves legacy ARIMA streams and wraps original-unit samples in immutable predictions.
"""Local ARIMA, SARIMA and MAP-fit Prophet benchmark adapters."""

from collections.abc import Iterator
from contextlib import contextmanager
from functools import partial
from typing import Any

import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.models import ARIMA

from hcp.forecast_types import (
    ForecastSettings,
    NativePrediction,
    PreparedHistory,
    V2ModelSettings,
)


def _seed_int(seed: np.random.SeedSequence) -> int:
    return int(seed.generate_state(1, dtype=np.uint32)[0])


def _classical_samples(
    histories: tuple[PreparedHistory, ...],
    settings: ForecastSettings,
    seed: np.random.SeedSequence,
    *,
    seasonal_order: tuple[int, int, int, int],
) -> np.ndarray:
    samples = []
    p, d, q = settings.arima_order
    for history, child in zip(histories, seed.spawn(len(histories)), strict=True):
        random_state = _seed_int(child)
        series = TimeSeries.from_times_and_values(
            pd.DatetimeIndex(history.dates), history.values
        )
        model = ARIMA(
            p=p,
            d=d,
            q=q,
            seasonal_order=seasonal_order,
            random_state=random_state,
        )
        model.fit(series)
        predicted = model.predict(
            settings.ahead,
            num_samples=settings.n_samples,
            random_state=random_state,
        ).all_values(copy=False)
        expected = (settings.ahead, 1, settings.n_samples)
        if predicted.shape != expected:
            raise ValueError(
                f"{history.node} returned samples with shape {predicted.shape}, "
                f"expected {expected}"
            )
        node_samples = predicted[:, 0, :]
        if not np.isfinite(node_samples).all():
            raise ValueError(f"{history.node} returned non-finite forecast samples")
        samples.append(node_samples)
    result = np.stack(samples, axis=0)
    result.setflags(write=False)
    return result


def arima_samples(
    histories: tuple[PreparedHistory, ...],
    settings: ForecastSettings,
    seed: np.random.SeedSequence,
) -> np.ndarray:
    """Fit independent ARIMA(3,0,0) models and return predictive samples."""
    return _classical_samples(histories, settings, seed, seasonal_order=(0, 0, 0, 0))


def sarima_samples(
    histories: tuple[PreparedHistory, ...],
    settings: ForecastSettings,
    seed: np.random.SeedSequence,
) -> np.ndarray:
    """Fit independent fixed annual SARIMA models and return predictive samples."""
    return _classical_samples(
        histories,
        settings,
        seed,
        seasonal_order=settings.sarima_seasonal_order,
    )


def arima_prediction(
    histories: tuple[PreparedHistory, ...],
    settings: ForecastSettings,
    seed: np.random.SeedSequence,
) -> NativePrediction:
    """Return native ARIMA paths without consuming additional seed children.

    Args:
        histories: Weekly node histories in canonical order.
        settings: Fixed classical forecast settings.
        seed: Origin stream, consumed exactly as in the legacy adapter.

    Returns:
        Immutable original-unit predictive paths.
    """
    return NativePrediction(
        nodes=tuple(history.node for history in histories),
        ahead=settings.ahead,
        kind="samples",
        samples=arima_samples(histories, settings, seed),
    )


def sarima_prediction(
    histories: tuple[PreparedHistory, ...],
    settings: ForecastSettings,
    seed: np.random.SeedSequence,
) -> NativePrediction:
    """Return native SARIMA paths without consuming additional seed children.

    Args:
        histories: Weekly node histories in canonical order.
        settings: Fixed classical forecast settings.
        seed: Origin stream, consumed exactly as in the legacy adapter.

    Returns:
        Immutable original-unit predictive paths.
    """
    return NativePrediction(
        nodes=tuple(history.node for history in histories),
        ahead=settings.ahead,
        kind="samples",
        samples=sarima_samples(histories, settings, seed),
    )


@contextmanager
def _numpy_seed(seed: int) -> Iterator[None]:
    previous = np.random.get_state()
    try:
        np.random.set_state(np.random.RandomState(seed).get_state())
        yield
    finally:
        np.random.set_state(previous)


def _seed_prophet_fit(model: Any, seed: int) -> None:
    # Darts 0.47 has no public fit-seed passthrough. Bind only this instance's
    # backend fit: CmdStanPy 1.3 otherwise uses an unseeded default_rng().
    builder = getattr(model, "_model_builder", None)
    if not callable(builder):
        raise TypeError("Darts Prophet lacks a seedable backend builder")

    def seeded_builder(**kwargs: Any) -> Any:
        backend = builder(**kwargs)
        backend.fit = partial(backend.fit, seed=seed)
        return backend

    model._model_builder = seeded_builder


def prophet_prediction(
    histories: tuple[PreparedHistory, ...],
    settings: V2ModelSettings,
    seed: np.random.SeedSequence,
) -> NativePrediction:
    """Fit weekly local Prophet MAP models and return seeded predictive paths.

    Args:
        histories: Original-unit weekly node prefixes in canonical order.
        settings: Explicit Prophet parameters and predictive sample count.
        seed: Origin stream spawning one child per node in fixed order.

    Returns:
        Immutable signed native-unit MAP-fit uncertainty paths and node seeds.

    Raises:
        RuntimeError: Backend fit or prediction fails, retaining node and prefix
            end context and chaining the backend exception.
    """
    # Import only on use: legacy adapters remain importable without Prophet.
    from darts.models import Prophet  # noqa: PLC0415

    samples = []
    node_seeds = []
    for history, child in zip(histories, seed.spawn(len(histories)), strict=True):
        random_state = _seed_int(child)
        node_seeds.append(random_state)
        try:
            with _numpy_seed(random_state):
                series = TimeSeries.from_times_and_values(
                    pd.DatetimeIndex(history.dates), history.values
                )
                model = Prophet(
                    growth=settings.prophet_growth,
                    yearly_seasonality=settings.prophet_yearly_seasonality,
                    weekly_seasonality=settings.prophet_weekly_seasonality,
                    daily_seasonality=settings.prophet_daily_seasonality,
                    n_changepoints=settings.prophet_n_changepoints,
                    changepoint_range=settings.prophet_changepoint_range,
                    changepoint_prior_scale=settings.prophet_changepoint_prior_scale,
                    mcmc_samples=settings.prophet_mcmc_samples,
                    random_state=random_state,
                )
                _seed_prophet_fit(model, random_state)
                model.fit(series)
                predicted = model.predict(
                    settings.ahead,
                    num_samples=settings.n_samples,
                    random_state=random_state,
                ).all_values(copy=False)
            expected = (settings.ahead, 1, settings.n_samples)
            if predicted.shape != expected:
                raise ValueError(
                    f"samples shape {predicted.shape}, expected {expected}"
                )
            if not np.isfinite(predicted).all():
                raise ValueError("non-finite forecast samples")
            samples.append(predicted[:, 0, :])
        except Exception as exc:
            prefix_end = str(pd.Timestamp(history.dates[-1]).date())
            raise RuntimeError(
                f"prophet node {history.node} origin prefix ending {prefix_end} failed"
            ) from exc
    return NativePrediction(
        nodes=tuple(history.node for history in histories),
        ahead=settings.ahead,
        kind="samples",
        samples=np.stack(samples, axis=0),
        diagnostics={"node_seeds": tuple(node_seeds)},
    )
