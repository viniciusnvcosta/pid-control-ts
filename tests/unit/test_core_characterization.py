# ABOUTME: Characterization tests pinning the upstream conformal controllers in core/methods.py.
# ABOUTME: Golden outputs were recorded before the NumPy 2 np.infty -> np.inf fix; they must never change.
from pathlib import Path

import numpy as np

from core.methods import aci, quantile, quantile_integrator_log

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
