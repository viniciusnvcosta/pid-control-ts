# ABOUTME: Preserves historical forecast-model imports while exposing neural adapters.
# ABOUTME: Keeps the legacy DeepAR function implemented in its consuming neural module.
"""Compatibility exports for registered forecast-model adapters."""

from darts.models import RNNModel

from hcp.forecast_classical import arima_samples, sarima_samples
from hcp.forecast_neural import deepar_samples, transformer_prediction
from hcp.forecast_types import ForecastSettings, PreparedHistory

__all__ = [
    "ForecastSettings",
    "PreparedHistory",
    "RNNModel",
    "arima_samples",
    "deepar_samples",
    "sarima_samples",
    "transformer_prediction",
]
