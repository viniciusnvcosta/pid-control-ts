# ABOUTME: Probabilistic Darts adapters for the fixed ARIMA, SARIMA, and DeepAR-style benchmarks.
# ABOUTME: Converts prepared weekly histories into deterministic-seed predictive sample arrays without I/O.
"""Fixed probabilistic forecast-model adapters for benchmark generation."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.models import ARIMA, RNNModel
from darts.utils.likelihood_models import GaussianLikelihood


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


def deepar_samples(
    histories: tuple[PreparedHistory, ...],
    settings: ForecastSettings,
    seed: np.random.SeedSequence,
) -> np.ndarray:
    """Fit one global DeepAR-style LSTM and return original-scale samples."""
    scales = np.array(
        [max(float(np.mean(np.abs(history.values))), 1.0) for history in histories]
    )
    series = [
        TimeSeries.from_times_and_values(
            pd.DatetimeIndex(history.dates), history.values / scale
        )
        for history, scale in zip(histories, scales, strict=True)
    ]
    random_state = _seed_int(seed)
    trainer = {
        "accelerator": "cpu",
        "devices": 1,
        "deterministic": True,
        "logger": False,
        "enable_checkpointing": False,
        "enable_progress_bar": False,
        "enable_model_summary": False,
    }
    model = RNNModel(
        model=settings.deepar_model,
        input_chunk_length=settings.deepar_input_chunk_length,
        training_length=settings.deepar_training_length,
        hidden_dim=settings.deepar_hidden_dim,
        n_rnn_layers=settings.deepar_n_rnn_layers,
        dropout=settings.deepar_dropout,
        likelihood=GaussianLikelihood(),
        batch_size=settings.deepar_batch_size,
        n_epochs=settings.deepar_n_epochs,
        random_state=random_state,
        save_checkpoints=False,
        log_tensorboard=False,
        show_warnings=False,
        pl_trainer_kwargs=trainer,
    )
    loader = {"num_workers": 0}
    model.fit(series, verbose=False, dataloader_kwargs=loader)
    predicted = model.predict(
        settings.ahead,
        series=series,
        num_samples=settings.n_samples,
        verbose=False,
        n_jobs=1,
        dataloader_kwargs=loader,
        random_state=random_state,
    )
    forecasts = [predicted] if isinstance(predicted, TimeSeries) else predicted
    samples = []
    expected = (settings.ahead, 1, settings.n_samples)
    for history, forecast, scale in zip(histories, forecasts, scales, strict=True):
        values = forecast.all_values(copy=False)
        if values.shape != expected:
            raise ValueError(
                f"{history.node} returned samples with shape {values.shape}, "
                f"expected {expected}"
            )
        node_samples = values[:, 0, :] * scale
        if not np.isfinite(node_samples).all():
            raise ValueError(f"{history.node} returned non-finite forecast samples")
        samples.append(node_samples)
    result = np.stack(samples, axis=0)
    result.setflags(write=False)
    return result
