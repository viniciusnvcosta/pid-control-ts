# ABOUTME: Pins the calling convention the paper harness (tests/base_test.py) relies on in core/.
# ABOUTME: Methods take fn(scores, alpha, lr, **config) with extra keys, Csat by name, and data=None.
from pathlib import Path

import numpy as np
import pytest

from core.methods import (
    aci,
    aci_clipped,
    quantile,
    quantile_integrator_log,
    trailing_window,
)
from core.quantile import weighted_conformal
from core.synthetic_scores import generate_scores

GOLDEN = np.load(Path(__file__).resolve().parents[1] / "fixtures" / "core_golden.npz")

# Keys base_test.py adds to every method's YAML block before calling fn(..., **kwargs).
HARNESS_KWARGS = {
    "lrs": [0.1],
    "T_burnin": 5,
    "data": None,
    "seasonal_period": None,
    "config_name": "contract",
    "ahead": 4,
    "upper": True,
}


@pytest.mark.parametrize(
    ("fn", "lr", "method_config"),
    [
        (trailing_window, None, {"weight_length": 10}),
        (aci, 0.01, {"window_length": 100000}),
        (aci_clipped, 0.01, {"window_length": 100000}),
        (quantile, 0.1, {}),
        (quantile_integrator_log, 0.1, {"Csat": 2.0, "KI": 1000.0}),
    ],
)
def test_methods_accept_the_harness_kwargs(fn, lr, method_config) -> None:
    kwargs = {**HARNESS_KWARGS, **method_config}
    q = fn(GOLDEN["scores"], 0.1, lr, **kwargs)["q"]
    assert len(q) == len(GOLDEN["scores"])


def test_pi_controller_by_keyword_matches_golden() -> None:
    q = quantile_integrator_log(
        GOLDEN["scores"], 0.1, 0.1, Csat=2.0, KI=1000.0, ahead=4, T_burnin=5, data=None
    )["q"]
    np.testing.assert_array_equal(q, GOLDEN["integrator"])


def test_generate_scores_accepts_the_harness_sequence_block() -> None:
    np.random.seed(0)
    scores = generate_scores(
        category="linear",
        start_point=0.5,
        end_point=0.5,
        length=50,
        sigma=0.1,
        dataset="linear-0.5-0.5-50-0.1",
    )
    assert scores.shape == (50,)


def test_weighted_conformal_runs_with_normalized_weights() -> None:
    rng = np.random.default_rng(0)
    y = rng.normal(size=60)
    result = weighted_conformal(
        {
            "Yhat_test": np.zeros(60),
            "Y_test": y,
            "fixed_weights": np.ones(20),
            "alpha": 0.1,
            "T_burnin": 20,
        }
    )
    assert result["qhats"].shape == (40,)
    assert np.isfinite(result["qhats"]).all()
