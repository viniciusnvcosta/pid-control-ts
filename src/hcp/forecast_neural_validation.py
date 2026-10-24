# ABOUTME: Scores original-unit fourth-step Gaussian densities for neural selection.
# ABOUTME: Owns stopping and selected-epoch refits with independent in-memory candidate states.
"""Factory-independent neural selection without checkpoint files or numeric I/O."""

from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd
from pytorch_lightning import Callback, LightningModule, Trainer
from scipy.special import logsumexp

from hcp.forecast_validation import TemporalSplit, neural_seed_streams


def gaussian_mixture_nll(
    y: np.ndarray, mu: np.ndarray, sigma: np.ndarray
) -> np.ndarray:
    """Evaluate an equally weighted conditional Gaussian mixture stably.

    Args:
        y: Observed original-unit targets with shape ``[...]``.
        mu: Original-unit conditional locations with shape ``[..., paths]``.
        sigma: Positive original-unit scales matching ``mu``. Direct Gaussian
            forecasts use one path; autoregression uses 500 simulated histories.

    Returns:
        Original-unit negative log density with the shape of ``y``.

    Raises:
        ValueError: If parameters, shapes, or resulting losses are invalid.
    """
    truth, location, scale = (
        np.asarray(value, dtype=float) for value in (y, mu, sigma)
    )
    if (
        location.ndim != truth.ndim + 1
        or location.shape[:-1] != truth.shape
        or location.shape != scale.shape
        or location.shape[-1] == 0
        or not all(np.isfinite(value).all() for value in (truth, location, scale))
        or np.any(scale <= 0)
    ):
        raise ValueError(
            "Gaussian mixture requires finite aligned positive-scale paths"
        )
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        z = (truth[..., None] - location) / scale
        log_density = -0.5 * z**2 - np.log(scale) - 0.5 * np.log(2 * np.pi)
        nll = np.log(location.shape[-1]) - logsumexp(log_density, axis=-1)
    if not np.isfinite(nll).all():
        raise ValueError("Gaussian mixture produced a nonfinite NLL")
    return nll


@dataclass(frozen=True)
class SelectionResult:
    """Selected candidate index, one-based epoch, and original-unit NLL.

    The owned tuning table contains one row per candidate in input order, with
    ``candidate``, ``best_epoch``, ``best_nll``, and ``epochs_run`` columns.
    """

    candidate: int
    best_epoch: int
    best_nll: float
    tuning: pd.DataFrame

    def __post_init__(self) -> None:
        object.__setattr__(self, "tuning", self.tuning.copy(deep=True))


class NeuralCandidate(Protocol):
    """Stateful architecture whose raw outer history survives inner selection.

    Construction owns inner preprocessing and its dedicated training child.
    Candidates must retain raw observations, availability, and forecast dates for
    rebuilding independently at final refit; never use held-out truth for inner
    scales, standardization, or training sampling weights.
    """

    def fit_selection(
        self,
        split: TemporalSplit,
        validation_seed: np.random.SeedSequence,
        max_epochs: int,
        patience: int,
        *,
        capture: "PhaseCapture | None" = None,
    ) -> "CandidateTrainingResult":
        """Train once, retaining the earliest strict improvement and patience stop."""
        ...

    def fourth_step_nll(
        self, split: TemporalSplit, seed: np.random.SeedSequence
    ) -> np.ndarray:
        """Return finite original-unit ``[node, nine origins]`` fourth-step NLL.

        Contexts include only observations known by each validation cutoff.
        Autoregressive densities mix 500 conditional Gaussian paths; direct
        Transformer parameters use one Gaussian, with inference dropout off.
        """
        ...

    def retain_best_state(self) -> None:
        """Own independent in-memory copies of current best model weights."""
        ...

    def fit_full(self, epochs: int, seed: np.random.SeedSequence) -> None:
        """Reinitialize and train the full outer prefix for exactly ``epochs``.

        Independently rebuild interpolation, scales, features, sampling weights,
        and architecture from raw outer metadata. Reuse no inner weights/scales
        or validation stopping criterion, including retained best weights.
        """
        ...


class NeuralLifecycle(NeuralCandidate, Protocol):
    """Internal candidate contract required by the single-fit Trainer lifecycle."""

    def _finish_epoch(self, trainer: Trainer, module: LightningModule) -> None:
        """Validate and record one completed training epoch."""
        ...

    def _resume_training(
        self, trainer: Trainer, module: LightningModule, another_epoch: bool
    ) -> None:
        """Restore training context after nested validation prediction."""
        ...

    def _fit_epochs(
        self, epochs: int, selection: "CandidateSelection | None" = None
    ) -> None:
        """Fit once with the optional per-epoch selection callback."""
        ...


def _validate_streams(
    streams: Mapping[str, tuple[np.random.SeedSequence, ...]], count: int
) -> None:
    if not isinstance(streams, Mapping):
        raise TypeError("neural streams must be a mapping")
    for key, length in (
        ("training", count),
        ("validation", count),
        ("refit", 1),
        ("prediction", 1),
    ):
        group = streams.get(key)
        if (
            not isinstance(group, tuple)
            or len(group) != length
            or not all(isinstance(child, np.random.SeedSequence) for child in group)
        ):
            raise ValueError(
                f"neural streams require {length} SeedSequence children for {key}"
            )


def select_neural(
    candidates: tuple[NeuralCandidate, ...],
    split: TemporalSplit,
    seed: np.random.SeedSequence,
    *,
    max_epochs: int = 100,
    patience: int = 10,
    streams: Mapping[str, tuple[np.random.SeedSequence, ...]] | None = None,
) -> SelectionResult:
    """Select fourth-step NLL and refit the winner using a fresh full prefix.

    Args:
        candidates: Nonempty architectures in ascending width order; exact ties
            keep the earlier candidate. Constructors own training streams.
        split: Nine observed fourth-step targets inside the twelve-week holdout.
        seed: Model/origin seed; allocated exactly once when streams is absent.
        max_epochs: Epoch cap, at most the registered 100 (fixtures may reduce).
        patience: Consecutive nonimproving epochs, at most the registered ten.
        streams: Optional allocation from ``neural_seed_streams(seed, N)``. Real
            adapters allocate once before constructing candidates with training
            children, pass the mapping here, then retain its prediction child.

    Returns:
        Best candidate, earliest best epoch, and each candidate's tuning result.
        Only the selected candidate is mutated by the final full-prefix refit.

    Raises:
        TypeError: If supplied streams are not a mapping.
        ValueError: If budgets, streams, split, or finite NLL shapes are invalid.
            Candidate fit/scoring failures propagate and abort the entire origin.
    """
    for name, value, limit in (
        ("max_epochs", max_epochs, 100),
        ("patience", patience, 10),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, np.integer))
            or not 1 <= value <= limit
        ):
            raise ValueError(f"{name} must be an integer between one and {limit}")
    if not candidates:
        raise ValueError("neural selection requires at least one candidate")
    if split.validation_targets.shape != (9,) or split.validation_cutoffs.shape != (9,):
        raise ValueError(
            "neural selection requires nine fourth-step validation origins"
        )
    allocated = (
        neural_seed_streams(seed, len(candidates)) if streams is None else streams
    )
    _validate_streams(allocated, len(candidates))
    records = []
    expected_shape = None
    for index, candidate in enumerate(candidates):
        result = candidate.fit_selection(
            split, allocated["validation"][index], max_epochs, patience
        )
        if expected_shape is None:
            expected_shape = result.loss_shape
        if result.loss_shape != expected_shape:
            raise ValueError(
                "fourth-step NLL shape must retain the same nodes and origins"
            )
        records.append(
            {
                "candidate": index,
                "best_epoch": result.best_epoch,
                "best_nll": result.best_nll,
                "epochs_run": result.epochs_run,
            }
        )
    selected = min(range(len(records)), key=lambda index: records[index]["best_nll"])
    winner = records[selected]
    candidates[selected].fit_full(winner["best_epoch"], allocated["refit"][0])
    return SelectionResult(
        selected, winner["best_epoch"], winner["best_nll"], pd.DataFrame(records)
    )


@dataclass(frozen=True)
class CandidateTrainingResult:
    """One candidate's earliest best score and completed selection epoch count."""

    best_epoch: int
    best_nll: float
    epochs_run: int
    loss_shape: tuple[int, int]


PhaseCapture = Callable[[str], AbstractContextManager[None]]


class CandidateSelection:
    """Keep the original fourth-step objective, seed reset, and stopping policy."""

    def __init__(
        self,
        candidate: NeuralCandidate,
        split: TemporalSplit,
        seed: np.random.SeedSequence,
        patience: int,
        capture: PhaseCapture | None = None,
    ) -> None:
        self.candidate, self.split, self.seed = candidate, split, seed
        self.patience, self.capture = patience, capture or (lambda phase: nullcontext())
        self.best_nll, self.best_epoch, self.epochs_run = np.inf, 0, 0
        self.unimproved, self.loss_shape = 0, (0, 0)

    def record_epoch(self) -> bool:
        """Score a completed epoch and return whether patience was exhausted."""
        self.epochs_run += 1
        with self.capture("validation"):
            losses = np.asarray(
                self.candidate.fourth_step_nll(
                    self.split, np.random.SeedSequence(**self.seed.state)
                ),
                dtype=float,
            )
        if losses.ndim != 2 or losses.shape[0] == 0 or losses.shape[1] != 9:
            raise ValueError("fourth-step NLL shape must be [nodes, nine origins]")
        if self.epochs_run == 1:
            self.loss_shape = losses.shape
        if losses.shape != self.loss_shape:
            raise ValueError(
                "fourth-step NLL shape must retain the same nodes and origins"
            )
        if not np.isfinite(losses).all():
            raise ValueError("all candidate fourth-step NLL values must be finite")
        score = float(np.sum(losses / losses.size))
        if not np.isfinite(score):
            raise ValueError("mean candidate fourth-step NLL must be finite")
        if score < self.best_nll:
            self.best_nll, self.best_epoch, self.unimproved = score, self.epochs_run, 0
            with self.capture("retention"):
                self.candidate.retain_best_state()
        else:
            self.unimproved += 1
        return self.unimproved >= self.patience

    def result(self) -> CandidateTrainingResult:
        """Return the independently owned numeric summary after training."""
        return CandidateTrainingResult(
            self.best_epoch, self.best_nll, self.epochs_run, self.loss_shape
        )


class NeuralEpochCallback(Callback):
    """Validate and stop inside one fit, restoring the active training context."""

    def __init__(
        self,
        candidate: NeuralLifecycle,
        max_epochs: int,
        selection: CandidateSelection | None = None,
    ) -> None:
        self.candidate, self.max_epochs, self.selection = (
            candidate,
            max_epochs,
            selection,
        )

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Check training, score the epoch, and restore state after nested prediction."""
        self.candidate._finish_epoch(trainer, pl_module)
        if self.selection is not None:
            trainer.should_stop = self.selection.record_epoch()
        self.candidate._resume_training(
            trainer,
            pl_module,
            trainer.current_epoch + 1 < self.max_epochs and not trainer.should_stop,
        )


def fit_candidate_selection(
    candidate: NeuralLifecycle,
    split: TemporalSplit,
    seed: np.random.SeedSequence,
    max_epochs: int,
    patience: int,
    capture: PhaseCapture | None = None,
) -> CandidateTrainingResult:
    """Run one architecture lifecycle with phase-specific warning collection."""
    phases = capture or (lambda phase: nullcontext())
    selection = CandidateSelection(candidate, split, seed, patience, phases)
    with phases("training"):
        candidate._fit_epochs(max_epochs, selection)
    return selection.result()
