# ABOUTME: Probabilistic Darts adapters for the fixed ARIMA, SARIMA, and DeepAR-style benchmarks.
# ABOUTME: Converts prepared weekly histories into deterministic-seed predictive sample arrays without I/O.
"""Fixed probabilistic forecast-model adapters for benchmark generation."""

import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.models import RNNModel
from darts.utils.likelihood_models import GaussianLikelihood

from hcp.forecast_classical import arima_samples, sarima_samples  # noqa: F401
from hcp.forecast_types import ForecastSettings, PreparedHistory


def _seed_int(seed: np.random.SeedSequence) -> int:
    return int(seed.generate_state(1, dtype=np.uint32)[0])


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
