# ABOUTME: Tests direct local quantiles against causal feature and tuning regressions.
# ABOUTME: Keeps solver fault/crossing doubles narrow and uses real HiGHS otherwise.
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from hcp.forecast_types import PreparedHistory, V2ModelSettings
from hcp.forecast_validation import InsufficientHistoryError


def _history(n: int = 60, values: np.ndarray | None = None) -> PreparedHistory:
    dates = pd.date_range("2020-01-04", periods=n, freq="7D").to_numpy()
    raw = np.arange(n, dtype=float) if values is None else values
    return PreparedHistory(
        "us", dates, raw, raw, np.isfinite(raw), dates - np.timedelta64(26, "D")
    )


def _predict(history: PreparedHistory):
    from hcp.forecast_quantile import qr_prediction

    return qr_prediction((history,), V2ModelSettings(), np.random.SeedSequence(123))


def test_features_end_four_weeks_before_response() -> None:
    from hcp.forecast_quantile import direct_examples

    history = _history()
    features, responses = direct_examples(history, history.forecast_dates[30])
    np.testing.assert_array_equal(features[0], np.arange(11.0, -1, -1))
    np.testing.assert_array_equal(features[-1], np.arange(22.0, 10, -1))
    np.testing.assert_array_equal(responses, np.arange(15.0, 27))
    assert features.shape == (12, 12)


def test_examples_use_observed_truth_and_independent_feature_prefixes() -> None:
    from hcp.forecast_quantile import direct_examples

    history = _history()
    raw, observed = history.raw_values.copy(), history.observed.copy()
    raw[10], raw[20], observed[20] = np.nan, 99999, False
    observed[10] = False
    history = replace(history, raw_values=raw, observed=observed, values=np.zeros(60))
    features, responses = direct_examples(history, history.forecast_dates[30])
    assert 20 not in responses
    # response15 has endpoint11, so its gap10 interpolates from9 and11 only.
    np.testing.assert_array_equal(features[0], np.arange(11.0, -1, -1))
    assert 24 not in responses  # Unobserved endpoint20 is structurally unusable.
    changed = raw.copy()
    changed[27:] += 100000
    changed_features, changed_responses = direct_examples(
        replace(history, raw_values=changed), history.forecast_dates[30]
    )
    np.testing.assert_array_equal(features, changed_features)
    np.testing.assert_array_equal(responses, changed_responses)


def test_alpha_is_shared_across_quantiles_and_ties_choose_larger() -> None:
    prediction = _predict(_history(values=np.full(60, -5.0)))
    tuning = prediction.diagnostics["tuning"][0]
    assert tuning["chosen_alpha"] == 1.0
    assert dict(tuning["candidate_losses"]) == {
        0.001: 0.0,
        0.01: 0.0,
        0.1: 0.0,
        1.0: 0.0,
    }
    np.testing.assert_array_equal(prediction.quantiles, [[-5.0, -5.0, -5.0]])
    assert prediction.kind == "quantiles"
    assert prediction.samples is None


def test_holdout_scale_excludes_validation_values() -> None:
    history = _history()
    raw = history.raw_values.copy()
    raw[45:] += 10000
    prediction = _predict(replace(history, raw_values=raw, values=raw))
    tuning = prediction.diagnostics["tuning"][0]
    assert tuning["training_scale"] == 22.0  # Mean0..44, excludes purged45..47 too.
    assert tuning["training_start"] == history.dates[0]
    assert tuning["training_end"] == history.dates[44]
    assert tuning["training_response_end"] == history.dates[44]
    assert tuning["refit_scale"] == pytest.approx(float(np.mean(np.abs(raw))))
    assert tuning["validation_targets"] == 12


def test_solver_failure_does_not_drop_candidate(monkeypatch) -> None:
    from hcp import forecast_quantile

    original = forecast_quantile.QuantileRegressor.fit

    def fail_candidate(model, features, responses):
        if model.alpha == 0.01:
            raise RuntimeError("eligible HiGHS failure")
        return original(model, features, responses)

    monkeypatch.setattr(forecast_quantile.QuantileRegressor, "fit", fail_candidate)
    with pytest.raises(RuntimeError, match="eligible HiGHS failure"):
        _predict(_history())


def test_crossing_repair_preserves_negative_forecasts(monkeypatch) -> None:
    from hcp import forecast_quantile

    class CrossingSolver:
        def __init__(self, *, quantile, alpha, solver, fit_intercept):
            self.quantile = quantile
            assert solver == "highs" and fit_intercept

        def fit(self, features, responses):
            return self

        def predict(self, features):
            return np.full(
                len(features), {0.1: -1.0, 0.5: -3.0, 0.9: -2.0}[self.quantile]
            )

    monkeypatch.setattr(forecast_quantile, "QuantileRegressor", CrossingSolver)
    prediction = _predict(_history(values=np.full(60, -10.0)))
    np.testing.assert_array_equal(prediction.quantiles, [[-30.0, -20.0, -10.0]])
    repairs = prediction.diagnostics["quantile_repairs"]
    assert len(repairs) == 1
    assert repairs[0]["node"] == "us"
    assert repairs[0]["levels"] == (0.1, 0.5, 0.9)
    np.testing.assert_array_equal(repairs[0]["before"], [-10.0, -30.0, -20.0])
    np.testing.assert_array_equal(repairs[0]["after"], [-30.0, -20.0, -10.0])
    assert prediction.diagnostics["crossing_count"] == 1
    assert prediction.diagnostics["root_crossing_count"] == 1
    assert prediction.diagnostics["max_absolute_adjustment"] == 20.0
    assert prediction.diagnostics["mean_absolute_adjustment"] == pytest.approx(40 / 3)


def test_alpha_selection_scores_unrepaired_quantiles(monkeypatch) -> None:
    from hcp import forecast_quantile

    class SelectionSolver:
        def __init__(self, *, quantile, alpha, solver, fit_intercept):
            self.quantile, self.alpha = quantile, alpha

        def fit(self, features, responses):
            return self

        def predict(self, features):
            values = [0.1, 0, -0.1] if self.alpha == 0.001 else [-0.5, 0, 0.5]
            return np.full(len(features), values[(0.1, 0.5, 0.9).index(self.quantile)])

    monkeypatch.setattr(forecast_quantile, "QuantileRegressor", SelectionSolver)
    prediction = _predict(_history(values=np.zeros(60)))
    tuning = prediction.diagnostics["tuning"][0]
    assert tuning["chosen_alpha"] == 1.0
    assert tuning["candidate_losses"][0.001] == pytest.approx(0.06)
    assert tuning["candidate_losses"][1.0] == pytest.approx(1 / 30)


def test_ineligible_history_has_distinct_error_but_malformed_metadata_aborts() -> None:
    with pytest.raises(InsufficientHistoryError):
        _predict(_history(41))
    history = _history()
    with pytest.raises(ValueError) as error:
        _predict(
            replace(
                history, forecast_dates=history.forecast_dates + np.timedelta64(1, "D")
            )
        )
    assert not isinstance(error.value, InsufficientHistoryError)


def test_solver_nonconvergence_warning_aborts_even_with_finite_coefficients(
    monkeypatch,
) -> None:
    import warnings

    from sklearn.exceptions import ConvergenceWarning

    from hcp import forecast_quantile

    original = forecast_quantile.QuantileRegressor.fit

    def unfinished_fit(model, features, responses):
        result = original(model, features, responses)
        warnings.warn("iteration limit reached", ConvergenceWarning)
        return result

    monkeypatch.setattr(forecast_quantile.QuantileRegressor, "fit", unfinished_fit)
    with pytest.raises(ConvergenceWarning, match="iteration limit"):
        _predict(_history())


def test_roundoff_crossing_is_repaired_without_counting_a_d17_crossing(
    monkeypatch,
) -> None:
    from hcp import forecast_quantile

    def almost_ordered_predict(model, features):
        values = {0.1: -10.0, 0.5: -10.0 - 1e-10, 0.9: -5.0}
        return np.full(len(features), values[model.quantile])

    monkeypatch.setattr(
        forecast_quantile.QuantileRegressor, "predict", almost_ordered_predict
    )
    prediction = _predict(_history(values=np.zeros(60)))
    assert prediction.diagnostics["crossing_count"] == 0
    assert prediction.diagnostics["adjusted_cell_count"] == 1
    assert len(prediction.diagnostics["quantile_repairs"]) == 1
    assert np.all(np.diff(prediction.quantiles, axis=-1) >= 0)


@pytest.mark.parametrize("raw", [None, np.zeros(40)])
def test_malformed_raw_metadata_cannot_be_classified_as_structural(raw) -> None:
    with pytest.raises(ValueError) as error:
        _predict(replace(_history(41), raw_values=raw))
    assert not isinstance(error.value, InsufficientHistoryError)


def test_all_nodes_are_preflighted_before_any_candidate_fit(monkeypatch) -> None:
    from hcp import forecast_quantile

    def unexpected_fit(model, features, responses):
        raise AssertionError("candidate fit before structural preflight completed")

    monkeypatch.setattr(forecast_quantile.QuantileRegressor, "fit", unexpected_fit)
    with pytest.raises(InsufficientHistoryError):
        forecast_quantile.qr_prediction(
            (_history(), replace(_history(41), node="ca")),
            V2ModelSettings(),
            np.random.SeedSequence(123),
        )


def test_training_response_bounds_exclude_pairs_with_unobserved_feature_endpoint() -> (
    None
):
    history = _history()
    raw = history.raw_values.copy()
    raw[40] = (
        np.nan
    )  # Response44 is observed but its feature endpoint40 is unavailable.
    prediction = _predict(replace(history, raw_values=raw, observed=np.isfinite(raw)))
    tuning = prediction.diagnostics["tuning"][0]
    assert tuning["training_end"] == history.dates[44]
    assert tuning["training_response_end"] == history.dates[43]


def test_each_candidate_is_fit_once_with_fixed_purged_scale(monkeypatch) -> None:
    from hcp import forecast_quantile

    fits = []
    original = forecast_quantile.QuantileRegressor.fit

    def capture_fit(model, features, responses):
        fits.append((model.alpha, model.quantile, features.copy(), responses.copy()))
        return original(model, features, responses)

    monkeypatch.setattr(forecast_quantile.QuantileRegressor, "fit", capture_fit)
    _predict(_history())
    # Four candidates x three levels, then only the three selected full refits.
    assert len(fits) == 15
    for alpha, quantile, features, responses in fits[:12]:
        np.testing.assert_array_equal(responses, np.arange(15.0, 45) / 22)
        np.testing.assert_array_equal(features[0], np.arange(11.0, -1, -1) / 22)
        np.testing.assert_array_equal(features[-1], np.arange(40.0, 28, -1) / 22)
    for alpha, quantile, features, responses in fits[12:]:
        np.testing.assert_array_equal(responses, np.arange(15.0, 60) / 29.5)
