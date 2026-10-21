# ABOUTME: Probabilistic Darts adapters for the fixed ARIMA, SARIMA, and DeepAR-style benchmarks.
# ABOUTME: Converts prepared weekly histories into deterministic-seed predictive sample arrays without I/O.
"""Fixed probabilistic forecast-model adapters for benchmark generation."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.models import ARIMA


@dataclass(frozen=True)
class PreparedHistory:
    """One model-ready weekly node history.

    Attributes:
        node: Hierarchy node key.
        dates: Weekly observation dates with shape ``[T]``.
        values: Finite observations with shape ``[T]``.
    """

    node: str
    dates: np.ndarray
    values: np.ndarray


@dataclass(frozen=True)
class ForecastSettings:
    """Fixed hyperparameters shared by registered forecast adapters."""

    ahead: int = 4
    n_samples: int = 500
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
