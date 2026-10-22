# ABOUTME: Immutable native forecast contracts shared by benchmark adapters.
# ABOUTME: Owns protected payload storage and preserves legacy model settings imports.
"""In-memory forecast contracts with defensive array ownership."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal

import numpy as np
import pandas as pd


def _protected(values: np.ndarray) -> np.ndarray:
    array = np.array(values, copy=True)
    array.setflags(write=False)
    return array


def _frozen_diagnostics(value: object) -> object:
    if isinstance(value, np.ndarray):
        return _protected(value)
    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _frozen_diagnostics(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_frozen_diagnostics(item) for item in value)
    return value


@dataclass(frozen=True)
class PreparedHistory:
    """One model-ready weekly node history.

    Attributes:
        node: Hierarchy node key.
        dates: Weekly observation dates with shape ``[T]``.
        values: Finite observations with shape ``[T]``.
        raw_values: Optional original observations, retaining missing values.
        observed: Optional canonical truth-availability mask aligned to dates.
        forecast_dates: Optional target-associated availability cutoffs.
    """

    node: str
    dates: np.ndarray
    values: np.ndarray
    raw_values: np.ndarray | None = None
    observed: np.ndarray | None = None
    forecast_dates: np.ndarray | None = None

    def __post_init__(self) -> None:
        for name in ("dates", "values", "raw_values", "observed", "forecast_dates"):
            values = getattr(self, name)
            if values is not None:
                object.__setattr__(self, name, _protected(values))


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


@dataclass(frozen=True)
class V2ModelSettings(ForecastSettings):
    """Registered model/selection settings from v2 design sections 3.1-3.6.

    Legacy fields retain their exact names/defaults. Enhanced recurrent input,
    training length, model, dropout, and batch use the inherited DeepAR fields.
    Production v2 validation fixes the registered budget; reduced fixtures may
    override settings without changing the native prediction contract.
    """

    levels: tuple[float, ...] = (0.1, 0.5, 0.9)
    neural_optimizer: str = "Adam"
    neural_learning_rate: float = 0.001
    neural_likelihood: str = "Gaussian"
    prophet_growth: str = "linear"
    prophet_yearly_seasonality: bool = False
    prophet_weekly_seasonality: bool = False
    prophet_daily_seasonality: bool = False
    prophet_n_changepoints: int = 25
    prophet_changepoint_range: float = 0.8
    prophet_changepoint_prior_scale: float = 0.05
    prophet_mcmc_samples: int = 0
    qr_input_length: int = 12
    qr_solver: str = "highs"
    qr_fit_intercept: bool = True
    qr_alphas: tuple[float, ...] = (0.001, 0.01, 0.1, 1.0)
    qr_holdout_targets: int = 12
    qr_min_train_examples: int = 12
    transformer_input_chunk_length: int = 12
    transformer_output_chunk_length: int = 4
    transformer_d_model: int = 32
    transformer_nhead: int = 4
    transformer_num_encoder_layers: int = 2
    transformer_num_decoder_layers: int = 2
    transformer_dim_feedforward: int = 128
    transformer_dropout: float = 0.1
    transformer_activation: str = "relu"
    transformer_norm_type: str = "LayerNorm"
    transformer_positional_encoding: str = "sinusoidal"
    transformer_batch_size: int = 32
    transformer_mc_dropout: bool = False
    enhanced_hidden_dims: tuple[int, ...] = (40, 80, 120)
    enhanced_n_rnn_layers: int = 3
    enhanced_embedding_dim: int = 20
    neural_holdout_targets: int = 12
    neural_validation_origins: int = 9
    neural_max_epochs: int = 100
    neural_patience: int = 10
    neural_min_delta: float = 0.0


@dataclass(frozen=True)
class NativePrediction:
    """Native-unit paths or direct horizon-four quantiles in declared node order."""

    nodes: tuple[str, ...]
    ahead: int
    kind: Literal["samples", "quantiles"]
    samples: np.ndarray | None = None
    quantiles: np.ndarray | None = None
    levels: tuple[float, ...] = (0.1, 0.5, 0.9)
    diagnostics: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        nodes = tuple(self.nodes)
        if not nodes or len(set(nodes)) != len(nodes):
            raise ValueError("nodes must be non-empty and unique")
        if not isinstance(self.ahead, int) or self.ahead != 4:
            raise ValueError("native predictions require ahead=4")
        if tuple(self.levels) != (0.1, 0.5, 0.9):
            raise ValueError("native predictions require levels=(0.1, 0.5, 0.9)")
        if self.kind not in ("samples", "quantiles"):
            raise ValueError("kind must be samples or quantiles")
        payload = self.samples if self.kind == "samples" else self.quantiles
        other = self.quantiles if self.kind == "samples" else self.samples
        if payload is None or other is not None:
            raise ValueError("exactly the tagged payload must be populated")
        array = np.asarray(payload, dtype=float)
        if self.kind == "samples":
            valid_shape = (
                array.ndim == 3
                and array.shape[:2] == (len(nodes), 4)
                and array.shape[2] > 0
            )
        else:
            valid_shape = array.shape == (len(nodes), 3)
        if not valid_shape:
            raise ValueError("native payload has invalid node/horizon shape")
        if not np.isfinite(array).all():
            raise ValueError("native payload must be finite")
        object.__setattr__(self, "nodes", nodes)
        object.__setattr__(self, "levels", tuple(self.levels))
        object.__setattr__(self, self.kind, _protected(array))
        if self.diagnostics is not None:
            object.__setattr__(
                self, "diagnostics", _frozen_diagnostics(self.diagnostics)
            )


def prediction_quantiles(prediction: NativePrediction) -> np.ndarray:
    """Return protected native-unit horizon-four quantiles.

    Args:
        prediction: Validated native paths or direct quantiles.

    Returns:
        Quantiles in declared node order with shape ``[node, 3]``; sample
        quantiles use NumPy's linear empirical interpolation at step four.
    """
    if prediction.kind == "quantiles":
        return _protected(prediction.quantiles)
    return _protected(
        np.quantile(
            prediction.samples[:, prediction.ahead - 1, :],
            prediction.levels,
            axis=-1,
            method="linear",
        ).T
    )


def _empty_diagnostic_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "model": pd.Series(dtype="str"),
            "node": pd.Series(dtype="str"),
            "forecast_date": pd.Series(dtype="datetime64[ns]"),
        }
    )


@dataclass(frozen=True)
class DiagnosticsTables:
    """Owned diagnostic tables with a shared identity schema, even without rows.

    Producer stages extend these identity columns with their event-specific
    fields. The frozen container owns copies; pandas contents remain mutable.
    """

    forecast_diagnostics: pd.DataFrame = field(default_factory=_empty_diagnostic_table)
    tuning: pd.DataFrame = field(default_factory=_empty_diagnostic_table)
    quantile_repairs: pd.DataFrame = field(default_factory=_empty_diagnostic_table)
    warnings: pd.DataFrame = field(default_factory=_empty_diagnostic_table)

    def __post_init__(self) -> None:
        for name in ("forecast_diagnostics", "tuning", "quantile_repairs", "warnings"):
            object.__setattr__(self, name, getattr(self, name).copy(deep=True))
