# ABOUTME: Provides legacy DeepAR and fixed global Transformer forecast adapters.
# ABOUTME: Keeps causal neural selection, scaling, seed streams, and warning evidence in memory.
"""Probabilistic neural forecast adapters for registered benchmark models."""

import numpy as np
import pandas as pd
import torch
from darts import TimeSeries
from darts.models import RNNModel, TransformerModel
from darts.models.forecasting.transformer_model import _TransformerModule
from darts.utils.likelihood_models import GaussianLikelihood

from hcp.forecast_neural_validation import select_neural
from hcp.forecast_types import (
    ForecastSettings,
    NativePrediction,
    PreparedHistory,
    V2ModelSettings,
)
from hcp.forecast_validation import (
    TemporalSplit,
    neural_seed_streams,
    neural_split,
    prepare_prefix,
)
from hcp.forecast_warnings import ForecastWarnings, WarningCandidate

_DARTS_TRANSFORMER = TransformerModel


def _seed_int(seed: np.random.SeedSequence) -> int:
    return int(seed.generate_state(1, dtype=np.uint32)[0])


def deepar_samples(
    histories: tuple[PreparedHistory, ...],
    settings: ForecastSettings,
    seed: np.random.SeedSequence,
) -> np.ndarray:
    """Fit the legacy global Darts LSTM and return original-scale samples."""
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


class _PersistentTransformerModule(_TransformerModule):
    """Darts Transformer module that retains one Adam state over epoch calls."""

    def configure_optimizers(self):
        if not hasattr(self, "_persistent_optimizer"):
            self._persistent_optimizer = super().configure_optimizers()
        return self._persistent_optimizer


class _PersistentTransformer(TransformerModel):
    """Darts Transformer with the same model and a stateful Adam factory."""

    def _create_model(self, train_sample):
        module = super()._create_model(train_sample)
        persistent = _PersistentTransformerModule(
            input_size=module.input_size,
            output_size=module.target_size,
            nr_params=module.nr_params,
            d_model=self.d_model,
            nhead=self.nhead,
            num_encoder_layers=self.num_encoder_layers,
            num_decoder_layers=self.num_decoder_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            activation=self.activation,
            norm_type=self.norm_type,
            custom_encoder=self.custom_encoder,
            custom_decoder=self.custom_decoder,
            **self.pl_module_params,
        )
        persistent.load_state_dict(module.state_dict())
        return persistent


class TransformerCandidate:
    """Stateful direct-Gaussian candidate with causal prefixes and owned best weights."""

    def __init__(
        self,
        histories: tuple[PreparedHistory, ...],
        settings: V2ModelSettings,
        seed: np.random.SeedSequence,
    ):
        histories = _trim_unavailable_tail(histories)
        if not histories:
            raise ValueError("Transformer requires nonempty histories")
        self.histories, self.settings = tuple(histories), settings
        self.split = _split(histories)
        self.best_state = None
        self._build(self.split.training_cutoff, seed)

    def _build(self, cutoff: np.datetime64, seed: np.random.SeedSequence) -> None:
        (initialization,) = seed.spawn(1)
        self.prefixes = tuple(prepare_prefix(h, cutoff) for h in self.histories)
        self.scales = np.array(
            [max(float(np.mean(np.abs(h.values))), 1.0) for h in self.prefixes]
        )
        self.scales.setflags(write=False)
        self.series = _series(self.prefixes, self.scales)
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(_seed_int(initialization))
            model_cls = (
                _PersistentTransformer
                if TransformerModel is _DARTS_TRANSFORMER
                else TransformerModel
            )
            self.model = model_cls(
                input_chunk_length=self.settings.transformer_input_chunk_length,
                output_chunk_length=self.settings.transformer_output_chunk_length,
                d_model=self.settings.transformer_d_model,
                nhead=self.settings.transformer_nhead,
                num_encoder_layers=self.settings.transformer_num_encoder_layers,
                num_decoder_layers=self.settings.transformer_num_decoder_layers,
                dim_feedforward=self.settings.transformer_dim_feedforward,
                dropout=self.settings.transformer_dropout,
                activation=self.settings.transformer_activation,
                norm_type=self.settings.transformer_norm_type,
                likelihood=GaussianLikelihood(),
                optimizer_cls=torch.optim.Adam,
                optimizer_kwargs={"lr": self.settings.neural_learning_rate},
                batch_size=self.settings.transformer_batch_size,
                n_epochs=1,
                random_state=_seed_int(seed),
                save_checkpoints=False,
                log_tensorboard=False,
                show_warnings=False,
                pl_trainer_kwargs=_trainer(),
            )
        self.epochs_trained = 0

    def train_epoch(self) -> None:
        self.model.fit(
            self.series,
            epochs=1,
            verbose=False,
            dataloader_kwargs={"num_workers": 0, "pin_memory": False},
        )
        trainer = getattr(self.model, "trainer", None)
        if trainer is None:
            trainer = getattr(self.model.model, "trainer", None)
        loss = None if trainer is None else trainer.callback_metrics.get("train_loss")
        if (
            loss is None
            or not torch.isfinite(loss).all()
            or any(not torch.isfinite(p).all() for p in self.model.model.parameters())
        ):
            raise ValueError("Transformer training produced nonfinite loss or weights")
        self.epochs_trained += 1

    def retain_best_state(self) -> None:
        self.best_state = {
            k: v.detach().cpu().clone()
            for k, v in self.model.model.state_dict().items()
        }

    def fourth_step_nll(
        self, split: TemporalSplit, seed: np.random.SeedSequence
    ) -> np.ndarray:
        losses = []
        for target, cutoff in zip(
            split.validation_targets, split.validation_cutoffs, strict=True
        ):
            contexts = tuple(prepare_prefix(h, cutoff) for h in self.histories)
            forecasts = self.model.predict(
                4,
                series=_series(contexts, self.scales),
                num_samples=1,
                predict_likelihood_parameters=True,
                mc_dropout=False,
                verbose=False,
                dataloader_kwargs={"num_workers": 0, "pin_memory": False},
                random_state=_seed_int(seed),
            )
            truth = np.array(
                [h.raw_values[np.searchsorted(h.dates, target)] for h in self.histories]
            )
            params = (
                np.stack([f.all_values(copy=False)[3, :, 0] for f in forecasts])
                * self.scales[:, None]
            )
            mu, sigma = params[:, 0], params[:, 1]
            if not np.isfinite(params).all() or np.any(sigma <= 0):
                raise ValueError(
                    "Transformer returned invalid fourth-step Gaussian parameters"
                )
            losses.append(
                0.5 * ((truth - mu) / sigma) ** 2
                + np.log(sigma)
                + 0.5 * np.log(2 * np.pi)
            )
        return np.stack(losses, axis=1)

    def fit_full(self, epochs: int, seed: np.random.SeedSequence) -> None:
        if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 1:
            raise ValueError("full refit requires a positive integer epoch count")
        self._build(self.histories[0].dates[-1] + np.timedelta64(2, "D"), seed)
        for _ in range(epochs):
            self.train_epoch()


def transformer_prediction(
    histories: tuple[PreparedHistory, ...],
    settings: V2ModelSettings,
    seed: np.random.SeedSequence,
) -> NativePrediction:
    """Select/refit the fixed global Transformer and return seeded native paths."""
    histories = _trim_unavailable_tail(histories)
    first = histories[0]
    split = _split(histories)
    streams = neural_seed_streams(seed, 1)
    warnings = ForecastWarnings("transformer", first.forecast_dates[-1])
    try:
        with warnings.capture(32, "construction"):
            candidate = TransformerCandidate(
                histories, settings, streams["training"][0]
            )
        inner_scales = candidate.scales.copy()
        wrapped = WarningCandidate(candidate, 32, warnings)
        selection = select_neural(
            (wrapped,),
            split,
            seed,
            max_epochs=settings.neural_max_epochs,
            patience=settings.neural_patience,
            streams=streams,
        )
        prediction_seed = streams["prediction"][0]
        with warnings.capture(32, "prediction"):
            forecasts = candidate.model.predict(
                4,
                series=candidate.series,
                num_samples=settings.n_samples,
                mc_dropout=False,
                verbose=False,
                n_jobs=1,
                dataloader_kwargs={"num_workers": 0, "pin_memory": False},
                random_state=_seed_int(prediction_seed),
            )
        samples = np.stack(
            [
                f.all_values(copy=False)[:, 0, :] * scale
                for f, scale in zip(forecasts, candidate.scales, strict=True)
            ]
        )
        if (
            samples.shape != (len(histories), 4, settings.n_samples)
            or not np.isfinite(samples).all()
        ):
            raise ValueError("Transformer returned invalid predictive samples")
        samples.setflags(write=False)
        return NativePrediction(
            tuple(h.node for h in histories),
            4,
            "samples",
            samples=samples,
            diagnostics={
                **warnings.diagnostics(),
                "streams": {
                    k: tuple(c.spawn_key for c in v) for k, v in streams.items()
                },
                "tuning": [
                    {**record, "scales": inner_scales.copy()}
                    for record in selection.tuning.to_dict("records")
                ],
                "final_fit": {
                    "scales": candidate.scales.copy(),
                    "epochs": selection.best_epoch,
                },
            },
        )
    except Exception as error:
        warnings.attach(error, {})
        raise


def _trim_unavailable_tail(
    histories: tuple[PreparedHistory, ...],
) -> tuple[PreparedHistory, ...]:
    """Remove dates after the last target observed for every Transformer node."""
    if not histories:
        raise ValueError("Transformer requires nonempty histories")
    last_available = min(
        int(np.flatnonzero(history.observed)[-1])
        for history in histories
        if np.any(history.observed)
    )
    stop = last_available + 1
    return tuple(
        PreparedHistory(
            history.node,
            history.dates[:stop],
            history.values[:stop],
            history.raw_values[:stop],
            history.observed[:stop],
            history.forecast_dates[:stop],
        )
        for history in histories
    )


def _trainer():
    return {
        "accelerator": "cpu",
        "devices": 1,
        "deterministic": True,
        "logger": False,
        "enable_checkpointing": False,
        "enable_progress_bar": False,
        "enable_model_summary": False,
        "num_sanity_val_steps": 0,
    }


def _series(histories, scales):
    return [
        TimeSeries.from_times_and_values(pd.DatetimeIndex(h.dates), h.values / s)
        for h, s in zip(histories, scales, strict=True)
    ]


def _split(histories):
    first = histories[0]
    if any(not np.array_equal(h.dates, first.dates) for h in histories):
        raise ValueError("Transformer histories must share a weekly calendar")
    split = neural_split(
        first.dates, np.stack([h.observed for h in histories]), first.forecast_dates
    )
    if split is None:
        raise ValueError("Transformer requires an eligible neural split")
    return split
