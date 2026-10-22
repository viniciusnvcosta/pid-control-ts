# ABOUTME: Behavioral tests for fixed training-prefix scales and normalized PI.
# ABOUTME: Checks core parity, unit equivariance, explicit scales, and empty bounds.
import numpy as np
import pytest

from core.methods import quantile_integrator_log
from hcp import conformal
from hcp.forecast_types import PreparedHistory


def _series() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = np.tile([0.0, 1.0, 4.0, 2.0, 8.0], 12)
    return y, np.full(len(y), 1.0), np.full(len(y), 3.0)


def test_normalized_offsets_match_core_with_unit_integrator_gain() -> None:
    y, lo, hi = _series()
    cfg = conformal.ControllerConfig(KI=99.0)
    scale = 4.0
    out_lo, out_hi = conformal.conformalize(y, lo, hi, "pi_norm", cfg, scale=scale)
    expected_lo = (
        scale * quantile_integrator_log((lo - y) / scale, 0.1, 0.1, 2.0, 1.0, 4, 5)["q"]
    )
    expected_hi = (
        scale * quantile_integrator_log((y - hi) / scale, 0.1, 0.1, 2.0, 1.0, 4, 5)["q"]
    )
    np.testing.assert_allclose(out_lo, lo - expected_lo)
    np.testing.assert_allclose(out_hi, hi + expected_hi)
    assert cfg.KI == 99.0


def test_normalized_pi_is_equivariant_to_change_of_units() -> None:
    y, lo, hi = _series()
    cfg = conformal.ControllerConfig()
    original = conformal.conformalize(y, lo, hi, "pi_norm", cfg, scale=4.0)
    rescaled = conformal.conformalize(7 * y, 7 * lo, 7 * hi, "pi_norm", cfg, scale=28.0)
    np.testing.assert_allclose(rescaled, 7 * np.asarray(original))


@pytest.mark.parametrize("scale", [None, 0.0, -1.0, np.nan, np.inf])
def test_normalized_pi_requires_finite_positive_explicit_scale(
    scale: float | None,
) -> None:
    y, lo, hi = _series()
    with pytest.raises(ValueError, match="scale"):
        conformal.conformalize(
            y, lo, hi, "pi_norm", conformal.ControllerConfig(), scale=scale
        )


def test_raw_remains_identity_without_scale() -> None:
    y, lo, hi = _series()
    actual = conformal.conformalize(y, lo, hi, "raw", conformal.ControllerConfig())
    np.testing.assert_array_equal(actual, (lo, hi))


def test_fixed_scales_use_only_owned_pre_evaluation_prefixes() -> None:
    observations = np.array([-2.0, 4.0, 900.0, 1000.0])
    dates = np.arange("2020-01-01", "2020-01-05", dtype="datetime64[D]")
    histories = (
        PreparedHistory("us", dates[:2], observations[:2]),
        PreparedHistory("small", dates[:2], np.array([0.0, 0.5])),
    )
    observations[:] = 1e9
    # Only the caller's prepared first-evaluation prefixes enter this interface.
    np.testing.assert_array_equal(conformal.fixed_pi_scales(histories), [3.0, 1.0])


@pytest.mark.parametrize(
    "values", [np.array([]), np.array([np.nan]), np.array([np.inf])]
)
def test_fixed_scales_reject_missing_or_nonfinite_prefixes(values: np.ndarray) -> None:
    history = PreparedHistory("us", np.arange(len(values)), values)
    with pytest.raises(ValueError, match="finite|empty"):
        conformal.fixed_pi_scales((history,))


def test_normalized_pi_preserves_empty_final_sets() -> None:
    y = np.zeros(200)
    lo = np.full(200, -0.01)
    hi = np.full(200, 0.01)
    actual = conformal.conformalize(
        y, lo, hi, "pi_norm", conformal.ControllerConfig(), scale=1.0
    )
    assert np.any(actual[0] > actual[1])
