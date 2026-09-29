# ABOUTME: Unit tests for the conformal adapter over core.methods (raw, PI, PID with Theta scorecaster).
# ABOUTME: Pins parity with the paper harness wiring (tests/base_test.py) and the paper's COVID score pickles.
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core.methods import quantile_integrator_log
from hcp.conformal import CONTROLLERS, ControllerConfig, conformalize, cqr_scores
from tests.helpers import PICKLE_STATES, PICKLES


def _narrow_series(n: int = 300, seed: int = 0) -> tuple[np.ndarray, ...]:
    y = np.random.default_rng(seed).normal(size=n)
    return y, np.full(n, -0.3), np.full(n, 0.3)


def test_registry_names() -> None:
    assert set(CONTROLLERS) == {"raw", "pi", "pid_theta"}


def test_cqr_scores_follow_paper_definition() -> None:
    y = np.array([1.0, 5.0])
    scores = cqr_scores(y, np.array([2.0, 1.0]), np.array([3.0, 4.0]))
    np.testing.assert_array_equal(scores, [[1.0, -2.0], [-4.0, 1.0]])


@pytest.mark.parametrize("state", PICKLE_STATES)
def test_cqr_scores_match_paper_pickles(state: str) -> None:
    frame = pd.read_pickle(PICKLES / f"{state}_proc_4wkdeaths.pkl").pivot(
        index="timestamp", columns="variable", values="target"
    )
    y = frame["y"].astype(float).to_numpy()
    keep = np.isfinite(y)
    forecasts = np.stack(frame["forecasts"].to_numpy())[keep]
    paper = np.stack(frame["scores"].to_numpy())[keep].astype(float)
    np.testing.assert_allclose(
        cqr_scores(y[keep], forecasts[:, 0], forecasts[:, 1]), paper
    )


def test_raw_returns_base_interval() -> None:
    y, lo, hi = _narrow_series()
    out_lo, out_hi = conformalize(y, lo, hi, "raw", ControllerConfig())
    np.testing.assert_array_equal(out_lo, lo)
    np.testing.assert_array_equal(out_hi, hi)


def test_pi_matches_paper_harness_wiring() -> None:
    y, lo, hi = _narrow_series()
    cfg = ControllerConfig(lr=0.1)
    scores = cqr_scores(y, lo, hi)
    kwargs = {
        "Csat": cfg.Csat,
        "KI": cfg.KI,
        "T_burnin": cfg.T_burnin,
        "data": None,
        "seasonal_period": None,
        "config_name": "parity",
        "ahead": cfg.ahead,
    }
    q_lo = quantile_integrator_log(
        scores[:, 0], cfg.alpha / 2, cfg.lr, upper=False, **kwargs
    )
    q_hi = quantile_integrator_log(
        scores[:, 1], cfg.alpha / 2, cfg.lr, upper=True, **kwargs
    )
    out_lo, out_hi = conformalize(y, lo, hi, "pi", cfg)
    np.testing.assert_array_equal(out_lo, lo - q_lo["q"])
    np.testing.assert_array_equal(out_hi, hi + q_hi["q"])


def test_pi_restores_coverage_of_a_too_narrow_interval() -> None:
    y, lo, hi = _narrow_series(n=400)
    out_lo, out_hi = conformalize(y, lo, hi, "pi", ControllerConfig(lr=0.1))
    raw = ((lo <= y) & (y <= hi)).mean()
    adjusted = ((out_lo <= y) & (y <= out_hi))[100:].mean()
    assert raw < 0.3
    assert 0.7 <= adjusted <= 0.9


def test_non_finite_truth_raises() -> None:
    y, lo, hi = _narrow_series()
    y[3] = np.nan
    with pytest.raises(ValueError, match="finite"):
        conformalize(y, lo, hi, "pi", ControllerConfig())


def test_unknown_controller_raises() -> None:
    y, lo, hi = _narrow_series()
    with pytest.raises(ValueError, match="unknown controller"):
        conformalize(y, lo, hi, "nope", ControllerConfig())


def test_pid_theta_is_deterministic_and_leaves_no_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    y, lo, hi = _narrow_series(n=60)
    cfg = ControllerConfig(lr=0.1)
    first = conformalize(y, lo, hi, "pid_theta", cfg)
    second = conformalize(y, lo, hi, "pid_theta", cfg)
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])
    assert not (tmp_path / ".cache").exists()
    assert not np.allclose(first[1], conformalize(y, lo, hi, "pi", cfg)[1])
