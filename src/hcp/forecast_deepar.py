# ABOUTME: Implements a learned node embedding in the Darts Gaussian recurrent module.
# ABOUTME: Trains causal weighted-window candidates and owns independent selected/refit states.
"""Enhanced DeepAR candidate using Darts loss and public Lightning training."""

import numpy as np
import torch
from darts.models.forecasting.pl_forecasting_module import io_processor
from darts.models.forecasting.rnn_model import CustomRNNModule
from darts.utils.likelihood_models import GaussianLikelihood
from pytorch_lightning import LightningModule, Trainer
from torch import nn
from torch.utils.data import DataLoader

from hcp.benchmark_batch import NeuralExecution
from hcp.forecast_deepar_data import (
    CovariateTransform,
    EnhancedWindowDataset,
    collate_windows,
    seed_integer,
    weighted_window_sampler,
)
from hcp.forecast_deepar_rollout import recurrent_rollout
from hcp.forecast_neural_validation import (
    CandidateSelection,
    CandidateTrainingResult,
    NeuralEpochCallback,
    PhaseCapture,
    fit_candidate_selection,
    gaussian_mixture_nll,
    select_neural,
)
from hcp.forecast_types import NativePrediction, PreparedHistory, V2ModelSettings
from hcp.forecast_validation import (
    TemporalSplit,
    neural_seed_streams,
    neural_split,
    prepare_prefix,
)
from hcp.forecast_warnings import ForecastWarnings, WarningCandidate, quiet_neural_logs


class EnhancedRNNModule(CustomRNNModule):
    """Three-layer LSTM with learned 20-dimensional fixed-vocabulary node IDs."""

    def __init__(self, n_nodes: int, hidden_dim: int, learning_rate: float = 0.001):
        super().__init__(
            input_size=3,
            hidden_dim=hidden_dim,
            num_layers=3,
            target_size=1,
            nr_params=2,
            dropout=0.0,
            input_chunk_length=12,
            output_chunk_length=1,
            likelihood=GaussianLikelihood(),
            loss_fn=nn.MSELoss(),  # Darts' mutable default would leak eval mode.
            optimizer_kwargs={"lr": learning_rate},
        )
        self.embedding = nn.Embedding(n_nodes, 20)
        self.rnn = nn.LSTM(23, hidden_dim, 3, batch_first=True, dropout=0.0)
        self.projection = nn.Linear(hidden_dim, 2)
        self._optimizer = None

    @io_processor
    def forward(
        self, x_in: tuple, h: tuple[torch.Tensor, torch.Tensor] | None = None
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Map scaled targets/known covariates and node IDs to Gaussian raw outputs."""
        x, _, static, _ = x_in
        identity = self.embedding(static[:, 0, 0].long())
        inputs = torch.cat((x, identity[:, None, :].expand(-1, x.shape[1], -1)), dim=-1)
        output, hidden = self.rnn(inputs, h)
        return self.projection(output).unsqueeze(2), hidden

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Preserve Adam state across public one-epoch Trainer.fit invocations."""
        if self._optimizer is None:
            self._optimizer = torch.optim.Adam(
                self.parameters(), **self.optimizer_kwargs
            )
        return self._optimizer


class EnhancedCandidate:
    """Stateful candidate retaining raw outer inputs and isolated inner fit state.

    Attributes:
        module: Actual Darts module used by Lightning and recurrent rollouts.
        scales: Training-only nu in fixed node order.
        best_state: Independent CPU tensors copied at a strict NLL improvement.
    """

    def __init__(
        self,
        histories: tuple[PreparedHistory, ...],
        settings: V2ModelSettings,
        width: int,
        seed: np.random.SeedSequence,
        execution: NeuralExecution,
    ):
        if not histories or width not in settings.enhanced_hidden_dims:
            raise ValueError(
                "enhanced candidate requires histories and a registered width"
            )
        split = _enhanced_split(histories)
        self.histories, self.settings, self.width = tuple(histories), settings, width
        self.execution = execution
        self.best_state: dict[str, torch.Tensor] | None = None
        self._build(split.training_cutoff, seed)

    def _build(self, cutoff: np.datetime64, seed: np.random.SeedSequence) -> None:
        initialization, sampler_seed, loader_seed = seed.spawn(3)
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        self.prefixes = tuple(prepare_prefix(h, cutoff) for h in self.histories)
        self.scales = np.array([1.0 + np.mean(np.abs(h.values)) for h in self.prefixes])
        self.scales.setflags(write=False)
        self.transforms = tuple(CovariateTransform.fit(h) for h in self.prefixes)
        with torch.random.fork_rng(
            devices=[0] if self.execution.device == "cuda:0" else []
        ):
            torch.manual_seed(seed_integer(initialization))
            self.module = EnhancedRNNModule(
                len(self.histories), self.width, self.settings.neural_learning_rate
            )
        self.dataset = EnhancedWindowDataset(
            self.prefixes,
            self.scales,
            self.transforms,
            self.settings.deepar_training_length,
        )
        sampler = weighted_window_sampler(
            self.dataset.window_counts, self.scales, sampler_seed
        )
        self.loader = DataLoader(
            self.dataset,
            batch_size=self.settings.deepar_batch_size,
            sampler=sampler,
            num_workers=0,
            pin_memory=False,
            collate_fn=collate_windows,
            generator=torch.Generator().manual_seed(seed_integer(loader_seed)),
        )
        self.epochs_trained = 0

    def fit_selection(
        self,
        split: TemporalSplit,
        validation_seed: np.random.SeedSequence,
        max_epochs: int,
        patience: int,
        *,
        capture: PhaseCapture | None = None,
    ) -> CandidateTrainingResult:
        """Train the inner candidate in one Trainer/fit lifecycle."""
        return fit_candidate_selection(
            self, split, validation_seed, max_epochs, patience, capture
        )

    def train_epoch(self) -> None:
        """Advance one epoch for characterization callers."""
        self._fit_epochs(1)

    def _fit_epochs(
        self, epochs: int, selection: CandidateSelection | None = None
    ) -> None:
        self.module.train()
        with quiet_neural_logs():
            trainer = Trainer(
                accelerator="gpu" if self.execution.device == "cuda:0" else "cpu",
                devices=[0] if self.execution.device == "cuda:0" else 1,
                precision=self.execution.precision,
                max_epochs=epochs,
                deterministic=True,
                logger=False,
                enable_checkpointing=False,
                enable_progress_bar=False,
                enable_model_summary=False,
                num_sanity_val_steps=0,
                callbacks=[NeuralEpochCallback(self, epochs, selection)],
            )
            trainer.fit(self.module, train_dataloaders=self.loader)
        self.module.to(self.execution.device)

    def _finish_epoch(self, trainer: Trainer, module: LightningModule) -> None:
        loss = trainer.callback_metrics.get("train_loss")
        if (
            loss is None
            or not torch.isfinite(loss).all()
            or any(not torch.isfinite(p).all() for p in module.parameters())
        ):
            raise ValueError(
                "enhanced recurrent training produced a nonfinite loss or weight"
            )
        self.epochs_trained += 1

    def _resume_training(
        self, trainer: Trainer, module: LightningModule, another_epoch: bool
    ) -> None:
        module._trainer = trainer
        module.train()

    def retain_best_state(self) -> None:
        """Copy current weights into separately owned in-memory CPU tensors."""
        self.best_state = {
            name: value.detach().cpu().clone()
            for name, value in self.module.state_dict().items()
        }

    def fourth_step_nll(
        self, split: TemporalSplit, seed: np.random.SeedSequence
    ) -> np.ndarray:
        """Score fourth-step original-unit Gaussian mixtures at each real cutoff."""
        losses = []
        for target, cutoff, child in zip(
            split.validation_targets,
            split.validation_cutoffs,
            seed.spawn(len(split.validation_targets)),
            strict=True,
        ):
            contexts = tuple(prepare_prefix(h, cutoff) for h in self.histories)
            _, mu, sigma = recurrent_rollout(
                self.module,
                contexts,
                self.transforms,
                self.scales,
                self.settings.n_samples,
                child,
                self.execution,
            )
            truth = np.array(
                [h.raw_values[np.searchsorted(h.dates, target)] for h in self.histories]
            )
            losses.append(gaussian_mixture_nll(truth, mu[:, -1], sigma[:, -1]))
        return np.stack(losses, axis=1)

    def fit_full(self, epochs: int, seed: np.random.SeedSequence) -> None:
        """Rebuild every fitted object from raw full history, then train exactly epochs."""
        if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 1:
            raise ValueError("full refit requires a positive integer epoch count")
        cutoff = self.histories[0].dates[-1] + np.timedelta64(2, "D")
        self._build(cutoff, seed)
        self._fit_epochs(epochs)

    def predict_samples(self, seed: np.random.SeedSequence) -> np.ndarray:
        """Return finite original-unit paths with shape [nodes, four, n_samples]."""
        paths, _, _ = recurrent_rollout(
            self.module,
            self.prefixes,
            self.transforms,
            self.scales,
            self.settings.n_samples,
            seed,
            self.execution,
        )
        return paths


def make_enhanced_candidate(
    histories: tuple[PreparedHistory, ...],
    settings: V2ModelSettings,
    width: int,
    seed: np.random.SeedSequence,
    execution: NeuralExecution,
) -> EnhancedCandidate:
    """Construct an inner-fit candidate from its reserved training seed child.

    Args:
        histories: Protected raw outer histories with availability and forecast dates.
        settings: Registered settings (reduced sample budgets allowed in fixtures).
        width: Architecture width in the configured ascending candidate grid.
        seed: Candidate-training child; never pass the outer origin seed here.
        execution: Explicit neural device and full-precision contract.

    Returns:
        A candidate supporting selection, full refit, and seeded native prediction.
    """
    return EnhancedCandidate(histories, settings, width, seed, execution)


def _enhanced_split(histories: tuple[PreparedHistory, ...]) -> TemporalSplit:
    if not histories:
        raise ValueError("enhanced forecast requires nonempty histories")
    first = histories[0]
    common_length = min(len(h.dates) for h in histories)
    if not common_length or any(
        h.observed is None or h.forecast_dates is None for h in histories
    ):
        raise ValueError("enhanced candidate requires nonempty raw metadata")
    # Align only the shared holdout calendar. Keep each node's complete
    # available prefix for age, scaling, and its own number of windows.
    dates = first.dates[-common_length:]
    cutoffs = first.forecast_dates[-common_length:]
    if any(
        not np.array_equal(h.dates[-common_length:], dates)
        or not np.array_equal(h.forecast_dates[-common_length:], cutoffs)
        for h in histories
    ):
        raise ValueError("enhanced histories must share the trailing weekly calendar")
    split = neural_split(
        dates, np.stack([h.observed[-common_length:] for h in histories]), cutoffs
    )
    if split is None:
        raise ValueError("enhanced candidate requires an eligible neural split")
    return split


def _fit_diagnostics(candidate: EnhancedCandidate) -> dict:
    return {
        "scales": candidate.scales.copy(),
        "covariates": tuple(
            {"first_date": t.first_date, "mean": t.mean.copy(), "scale": t.scale.copy()}
            for t in candidate.transforms
        ),
    }


def enhanced_deepar_prediction(
    histories: tuple[PreparedHistory, ...],
    settings: V2ModelSettings,
    seed: np.random.SeedSequence,
    execution: NeuralExecution,
) -> NativePrediction:
    """Select a recurrent width/epoch and forecast from an independent full refit.

    Args:
        histories: Raw available prefixes in fixed hierarchy vocabulary order.
            Synthetic fixtures may use fewer nodes than the registered 57.
        settings: Model defaults or explicitly reduced development budgets.
        seed: Reserved model/origin seed, allocated once into 2*N+2 children.
        execution: Explicit neural device and full-precision contract.

    Returns:
        Protected native paths and candidate, refit, stream, and warning evidence.
        Diagnostic forecast_date is the shared last observation plus two days,
        the registered issuance cutoff for the next fourth-step target.

    Raises:
        ValueError: If histories, widths, split, or prediction payload are invalid.
        Exception: Candidate failures abort; the same exception carries immutable
            forecast_diagnostics including all warnings observed through failure.
    """
    split = _enhanced_split(histories)
    widths = tuple(sorted(settings.enhanced_hidden_dims))
    if not widths or len(set(widths)) != len(widths):
        raise ValueError("enhanced widths must be nonempty and unique")
    streams = neural_seed_streams(seed, len(widths))
    origin = histories[0].dates[-1] + np.timedelta64(2, "D")
    collector = ForecastWarnings("deepar_enhanced", origin)
    diagnostics = {
        "model": "deepar_enhanced",
        "forecast_date": origin,
        "streams": {
            key: tuple(child.spawn_key for child in children)
            for key, children in streams.items()
        },
        "tuning": [],
    }
    candidates, inner = [], []
    try:
        for width, training_child in zip(widths, streams["training"], strict=True):
            with collector.capture(width, "construction"):
                candidate = make_enhanced_candidate(
                    histories, settings, width, training_child, execution
                )
            candidates.append(candidate)
            inner.append(_fit_diagnostics(candidate))
        selected = select_neural(
            tuple(
                WarningCandidate(c, w, collector) for c, w in zip(candidates, widths)
            ),
            split,
            seed,
            max_epochs=settings.neural_max_epochs,
            patience=settings.neural_patience,
            streams=streams,
        )
        diagnostics["tuning"] = [
            {**row, "width": widths[index], **inner[index]}
            for index, row in enumerate(selected.tuning.to_dict("records"))
        ]
        winner = candidates[selected.candidate]
        width = widths[selected.candidate]
        diagnostics["final_fit"] = {
            "width": width,
            "epochs": winner.epochs_trained,
            "best_epoch": selected.best_epoch,
            "best_nll": selected.best_nll,
            **_fit_diagnostics(winner),
        }
        with collector.capture(width, "prediction"):
            samples = winner.predict_samples(streams["prediction"][0])
        if np.shape(samples) != (len(histories), 4, settings.n_samples):
            raise ValueError(
                "enhanced prediction must match nodes, four steps, and sample budget"
            )
        return NativePrediction(
            tuple(h.node for h in histories),
            settings.ahead,
            "samples",
            samples=samples,
            diagnostics={**diagnostics, **collector.diagnostics()},
        )
    except Exception as error:
        collector.attach(error, diagnostics)
        raise
