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
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from core.methods import quantile_integrator_log, quantile_integrator_log_scorecaster
from hcp.forecast_types import PreparedHistory


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


def _pi_norm(scores: np.ndarray, cfg: ControllerConfig, upper: bool) -> np.ndarray:
    return _pi(scores, replace(cfg, KI=1.0), upper)


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
    "pi_norm": _pi_norm,
    "pid_theta": _pid_theta,
}


def cqr_scores(y: np.ndarray, q_lo: np.ndarray, q_hi: np.ndarray) -> np.ndarray:
    """Asymmetric CQR scores ``[T, 2]``: ``[q_lo - y, y - q_hi]``."""
    return np.stack([q_lo - y, y - q_hi], axis=1)


def fixed_pi_scales(histories: tuple[PreparedHistory, ...]) -> np.ndarray:
    """Compute fixed node scales from prepared pre-evaluation histories.

    Args:
        histories: Node-ordered training prefixes prepared at the first common
            evaluated forecast origin using its actual forecast-date cutoff.
            Callers reuse these scales across models, arms, sides, and weeks.

    Returns:
        Node scales ``max(mean(abs(prefix)), 1)``; evaluation truth is not used.

    Raises:
        ValueError: If a training prefix is empty or nonfinite.
    """
    scales = []
    for history in histories:
        if history.values.size == 0 or not np.isfinite(history.values).all():
            raise ValueError("PI training prefixes must be nonempty and finite")
        scales.append(max(float(np.abs(history.values).mean()), 1.0))
    return np.asarray(scales)


def conformalize(
    y: np.ndarray,
    q_lo: np.ndarray,
    q_hi: np.ndarray,
    controller: str,
    cfg: ControllerConfig,
    *,
    scale: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the conformalized interval ``(lo, hi)`` for one series.

    Args:
        y: Observed truth for the controller sequence.
        q_lo: Base lower quantiles.
        q_hi: Base upper quantiles.
        controller: Registered controller key.
        cfg: Controller settings; ``pi_norm`` always uses ``KI=1``.
        scale: Fixed pre-evaluation node scale, required only for ``pi_norm``.

    Raises:
        ValueError: On an unknown controller, nonfinite input, or invalid PI scale.
    """
    if controller not in CONTROLLERS:
        raise ValueError(f"unknown controller: {controller}")
    if not (
        np.isfinite(y).all() and np.isfinite(q_lo).all() and np.isfinite(q_hi).all()
    ):
        raise ValueError("truth and base quantiles must be finite")
    offset_scale = 1.0
    if controller == "pi_norm":
        if scale is None or not np.isfinite(scale) or scale <= 0:
            raise ValueError("pi_norm requires a finite positive explicit scale")
        offset_scale = scale
    scores = cqr_scores(y, q_lo, q_hi)
    offset_lo = offset_scale * CONTROLLERS[controller](
        scores[:, 0] / offset_scale, cfg, False
    )
    offset_hi = offset_scale * CONTROLLERS[controller](
        scores[:, 1] / offset_scale, cfg, True
    )
    return q_lo - offset_lo, q_hi + offset_hi
