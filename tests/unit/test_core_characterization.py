# ABOUTME: Characterization tests pinning the upstream conformal controllers in core/methods.py.
# ABOUTME: Golden outputs were recorded before the NumPy 2 np.infty -> np.inf fix; they must never change.
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core.methods import (
    aci,
    quantile,
    quantile_integrator_log,
    quantile_integrator_log_scorecaster,
)

GOLDEN = np.load(Path(__file__).resolve().parents[1] / "fixtures" / "core_golden.npz")


def test_quantile_tracker_matches_golden() -> None:
    q = quantile(GOLDEN["scores"], 0.1, 0.1, 4, T_burnin=5)["q"]
    np.testing.assert_array_equal(q, GOLDEN["quantile"])


def test_pi_controller_matches_golden() -> None:
    q = quantile_integrator_log(GOLDEN["scores"], 0.1, 0.1, 2.0, 1000.0, 4, 5)["q"]
    np.testing.assert_array_equal(q, GOLDEN["integrator"])


def test_saturated_integrator_matches_golden() -> None:
    q = quantile_integrator_log(GOLDEN["saturating"], 0.1, 0.0, 0.1, 1.0, 4, 5)["q"]
    assert np.isinf(GOLDEN["saturated"]).any()
    np.testing.assert_array_equal(q, GOLDEN["saturated"])


def test_aci_matches_golden() -> None:
    q = aci(GOLDEN["scores"], 0.1, 0.01, window_length=100000, T_burnin=5, ahead=4)["q"]
    np.testing.assert_array_equal(q, GOLDEN["aci"])


def test_theta_scorecaster_ahead1_matches_golden(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    scores = GOLDEN["scores"][:80]
    q = quantile_integrator_log_scorecaster(
        scores,
        0.1,
        0.1,
        pd.DataFrame(index=np.arange(len(scores))),
        5,
        2.0,
        1000.0,
        True,
        1,
        config_name="golden_ahead1",
    )["q"]
    assert np.isfinite(q).all()
    assert len(q) == len(scores)
    np.testing.assert_array_equal(q, GOLDEN["scorecaster_ahead1"])


def test_theta_scorecaster_ahead4_matches_golden(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    scores = GOLDEN["scores"][:80]
    q = quantile_integrator_log_scorecaster(
        scores,
        0.1,
        0.1,
        pd.DataFrame(index=np.arange(len(scores))),
        5,
        2.0,
        1000.0,
        True,
        4,
        config_name="golden_ahead4",
    )["q"]
    assert np.isfinite(q).all()
    assert len(q) == len(scores)
    np.testing.assert_array_equal(q, GOLDEN["scorecaster_ahead4"])


def test_theta_scorecaster_ahead1_applies_the_scorecast(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    scores = GOLDEN["scores"][:80]
    with_scorecast = quantile_integrator_log_scorecaster(
        scores,
        0.1,
        0.1,
        pd.DataFrame(index=np.arange(len(scores))),
        5,
        2.0,
        1000.0,
        True,
        1,
        config_name="golden_ahead1_semantic",
    )["q"]
    without_scorecast = quantile_integrator_log(scores, 0.1, 0.1, 2.0, 1000.0, 1, 5)[
        "q"
    ]
    assert not np.allclose(with_scorecast, without_scorecast)
