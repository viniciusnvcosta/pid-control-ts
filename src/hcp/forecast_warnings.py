# ABOUTME: Preserves phase-specific forecast warnings with source locations and counts.
# ABOUTME: Keeps completed and failing phase evidence without changing exception identity.
"""Scoped in-memory warning provenance for neural forecast adapters."""

import logging
import warnings
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager

import numpy as np

from hcp.forecast_neural_validation import CandidateTrainingResult, NeuralCandidate
from hcp.forecast_types import _frozen_diagnostics
from hcp.forecast_validation import TemporalSplit


class ForecastWarnings:
    """Collect deduplicated warnings for one global model and outer origin."""

    def __init__(self, model: str, forecast_date: np.datetime64):
        self.model, self.forecast_date = model, forecast_date
        self._records: dict[tuple, dict] = {}

    @contextmanager
    def capture(self, candidate: int, phase: str) -> Iterator[None]:
        """Capture every occurrence, including those emitted before a failure.

        Args:
            candidate: Architecture width (or another explicit numeric identity).
            phase: Construction, training, validation, retention, refit, or prediction.

        Yields:
            Control to the phase; original exceptions propagate unmodified.
        """
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                yield
            finally:
                for item in caught:
                    key = (
                        item.category.__name__,
                        str(item.message),
                        item.filename,
                        item.lineno,
                        self.model,
                        self.forecast_date,
                        candidate,
                        phase,
                    )
                    if key not in self._records:
                        self._records[key] = dict(
                            zip(
                                (
                                    "category",
                                    "message",
                                    "filename",
                                    "lineno",
                                    "model",
                                    "forecast_date",
                                    "candidate",
                                    "phase",
                                ),
                                key,
                                strict=True,
                            ),
                            node=None,
                            count=0,
                        )
                    self._records[key]["count"] += 1

    def diagnostics(self) -> dict:
        """Return independent records and legacy-compatible category totals."""
        counts = Counter()
        for record in self._records.values():
            counts[record["category"]] += record["count"]
        return {
            "warnings": tuple(dict(row) for row in self._records.values()),
            "warning_counts": dict(sorted(counts.items())),
        }

    def attach(self, error: Exception, diagnostics: dict) -> None:
        """Attach an immutable snapshot before the caller re-raises the same error."""
        error.forecast_diagnostics = _frozen_diagnostics(
            {**diagnostics, **self.diagnostics()}
        )


class WarningCandidate:
    """Wrap only selection phases, retaining the real candidate and its state."""

    def __init__(
        self, candidate: NeuralCandidate, width: int, collector: ForecastWarnings
    ):
        self.candidate, self.width, self.collector = candidate, width, collector

    def fit_selection(
        self,
        split: TemporalSplit,
        validation_seed: np.random.SeedSequence,
        max_epochs: int,
        patience: int,
    ) -> CandidateTrainingResult:
        """Capture training separately from nested validation and retention phases."""
        return self.candidate.fit_selection(
            split,
            validation_seed,
            max_epochs,
            patience,
            capture=lambda phase: self.collector.capture(self.width, phase),
        )

    def fourth_step_nll(
        self, split: TemporalSplit, seed: np.random.SeedSequence
    ) -> np.ndarray:
        """Capture validation warnings while preserving the reset path seed."""
        with self.collector.capture(self.width, "validation"):
            return self.candidate.fourth_step_nll(split, seed)

    def retain_best_state(self) -> None:
        """Capture warnings while retaining the candidate's independent weights."""
        with self.collector.capture(self.width, "retention"):
            self.candidate.retain_best_state()

    def fit_full(self, epochs: int, seed: np.random.SeedSequence) -> None:
        """Capture all selected full-prefix refit warnings under the refit phase."""
        with self.collector.capture(self.width, "refit"):
            self.candidate.fit_full(epochs, seed)


@contextmanager
def quiet_neural_logs() -> Iterator[None]:
    """Suppress scoped Lightning INFO startup logs while retaining Python warnings."""
    prefix = "pytorch_lightning"
    names = [prefix] + [
        name
        for name in logging.root.manager.loggerDict
        if name.startswith(prefix + ".")
    ]
    loggers = [logging.getLogger(name) for name in names]
    levels = [logger.level for logger in loggers]
    try:
        for logger in loggers:
            logger.setLevel(max(logger.level, logging.WARNING))
        yield
    finally:
        for logger, level in zip(loggers, levels, strict=True):
            logger.setLevel(level)
