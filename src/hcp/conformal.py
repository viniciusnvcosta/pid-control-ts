# ABOUTME: Conformal controllers for the US interval: raw base forecaster, PI, and PID with Theta scorecaster.
# ABOUTME: Thin adapter over core.methods using the paper's cqr-asymmetric scores, alpha/2 per side (D4).
"""Conformal stage (decision D4).

Mirrors ``tests/base_test.py`` for ``score_function_name = "cqr-asymmetric"``: scores are
``[q_lo - y, y - q_hi]``, each side runs its controller at ``alpha / 2``, and the set is
``[q_lo - offset_lo, q_hi + offset_hi]``. The controllers come from ``core.methods``
unchanged. ``pid_theta`` runs inside a throw-away working directory, because core caches
Theta scorecasts under ``./.cache/scorecaster/`` and would otherwise reuse stale ones.
"""

import contextlib
import tempfile
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from core.methods import quantile_integrator_log, quantile_integrator_log_scorecaster


@dataclass(frozen=True)
class ControllerConfig:
    """Controller hyperparameters; defaults are the paper's COVID configs."""

    alpha: float = 0.2
    lr: float = 0.1
    Csat: float = 2.0
    KI: float = 1000.0
    T_burnin: int = 5
    ahead: int = 4


def _raw(scores: np.ndarray, _cfg: ControllerConfig, _upper: bool) -> np.ndarray:
    return np.zeros(len(scores))


def _pi(scores: np.ndarray, cfg: ControllerConfig, _upper: bool) -> np.ndarray:
    return quantile_integrator_log(
        scores, cfg.alpha / 2, cfg.lr, cfg.Csat, cfg.KI, cfg.ahead, cfg.T_burnin
    )["q"]


def _pid_theta(scores: np.ndarray, cfg: ControllerConfig, upper: bool) -> np.ndarray:
    no_scorecasts = pd.DataFrame(index=np.arange(len(scores)))
    with tempfile.TemporaryDirectory() as scratch, contextlib.chdir(scratch):
        result = quantile_integrator_log_scorecaster(
            scores,
            cfg.alpha / 2,
            cfg.lr,
            no_scorecasts,
            cfg.T_burnin,
            cfg.Csat,
            cfg.KI,
            upper,
            cfg.ahead,
            config_name="pid_theta",
        )
    return result["q"]


CONTROLLERS: dict[str, Callable[[np.ndarray, ControllerConfig, bool], np.ndarray]] = {
    "raw": _raw,
    "pi": _pi,
    "pid_theta": _pid_theta,
}


def cqr_scores(y: np.ndarray, q_lo: np.ndarray, q_hi: np.ndarray) -> np.ndarray:
    """Asymmetric CQR scores ``[T, 2]``: ``[q_lo - y, y - q_hi]``."""
    return np.stack([q_lo - y, y - q_hi], axis=1)


def conformalize(
    y: np.ndarray,
    q_lo: np.ndarray,
    q_hi: np.ndarray,
    controller: str,
    cfg: ControllerConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the conformalized interval ``(lo, hi)`` for one series.

    Raises:
        ValueError: On an unknown controller or a non-finite input.
    """
    if controller not in CONTROLLERS:
        raise ValueError(f"unknown controller: {controller}")
    if not (
        np.isfinite(y).all() and np.isfinite(q_lo).all() and np.isfinite(q_hi).all()
    ):
        raise ValueError("truth and base quantiles must be finite")
    scores = cqr_scores(y, q_lo, q_hi)
    offset_lo = CONTROLLERS[controller](scores[:, 0], cfg, False)
    offset_hi = CONTROLLERS[controller](scores[:, 1], cfg, True)
    return q_lo - offset_lo, q_hi + offset_hi
