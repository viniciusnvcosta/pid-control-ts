# ABOUTME: Exercises the real Darts Transformer on reduced synthetic causal prefixes.
# ABOUTME: Protects optimizer continuity, fourth-step density, seeded paths and no-write execution.
import importlib
from dataclasses import replace

import numpy as np
import pytest
import torch

from hcp.forecast_types import PreparedHistory, V2ModelSettings, prediction_quantiles
from hcp.forecast_validation import neural_split, prepare_prefix

pytestmark = pytest.mark.integration


def _histories(future_shift=0.0):
    dates = np.datetime64("2020-01-04") + np.arange(44) * np.timedelta64(7, "D")
    result = []
    for i in range(2):
        values = -np.arange(44, dtype=float) * (i + 1)
        values[40:] += future_shift
        raw = PreparedHistory(
            str(i),
            dates,
            values,
            values,
            np.ones(44, bool),
            dates - np.timedelta64(26, "D"),
        )
        result.append(prepare_prefix(raw, dates[39] + np.timedelta64(2, "D")))
    return tuple(result)


def _module():
    assert importlib.util.find_spec("hcp.forecast_neural") is not None, (
        "Transformer adapter missing"
    )
    return importlib.import_module("hcp.forecast_neural")


def test_real_transformer_repeatability_future_invariance_and_fourth_step(
    tmp_path, monkeypatch
):
    neural = _module()
    monkeypatch.chdir(tmp_path)
    settings = replace(
        V2ModelSettings(), neural_max_epochs=1, neural_patience=1, n_samples=8
    )
    results = [
        neural.transformer_prediction(
            _histories(shift), settings, np.random.SeedSequence(81)
        )
        for shift in (0.0, 0.0, 1e6)
    ]
    first = results[0]
    assert first.samples.shape == (2, 4, 8) and not first.samples.flags.writeable
    assert np.std(first.samples[0, -1]) > 0 and np.any(first.samples < 0)
    for result in results:
        np.testing.assert_array_equal(first.samples, result.samples)
        np.testing.assert_array_equal(
            result.diagnostics["tuning"][0]["scales"], [13.5, 27.0]
        )
        np.testing.assert_array_equal(
            result.diagnostics["final_fit"]["scales"], [19.5, 39.0]
        )
        assert result.diagnostics["final_fit"]["epochs"] == 1
        rows = result.diagnostics["warnings"]
        assert all(row["filename"] and row["lineno"] > 0 for row in rows)
        assert sum(row["count"] for row in rows) == sum(
            result.diagnostics["warning_counts"].values()
        )
    np.testing.assert_array_equal(
        prediction_quantiles(first),
        np.quantile(first.samples[:, 3], [0.1, 0.5, 0.9], axis=-1).T,
    )
    assert list(tmp_path.iterdir()) == []


def test_real_two_epochs_preserve_adam_and_best_weights_and_score_direct_gaussian():
    neural = _module()
    data = _histories()
    candidate = neural.TransformerCandidate(
        data, V2ModelSettings(), np.random.SeedSequence(82)
    )
    candidate.train_epoch()
    optimizer = candidate.model.model.configure_optimizers()
    before = [value["step"].item() for value in optimizer.state.values()]
    candidate.retain_best_state()
    state = {key: value.clone() for key, value in candidate.best_state.items()}
    candidate.train_epoch()
    assert candidate.epochs_trained == 2
    assert candidate.model.model.configure_optimizers() is optimizer
    after = [value["step"].item() for value in optimizer.state.values()]
    assert all(right == left + 1 for left, right in zip(before, after, strict=True))
    assert any(
        not torch.equal(state[key], value)
        for key, value in candidate.model.model.state_dict().items()
    )
    for key, value in state.items():
        torch.testing.assert_close(value, candidate.best_state[key])
    split = neural_split(
        data[0].dates, np.stack([h.observed for h in data]), data[0].forecast_dates
    )
    losses = candidate.fourth_step_nll(split, np.random.SeedSequence(83))
    contexts = [prepare_prefix(h, split.validation_cutoffs[0]) for h in data]
    import pandas as pd
    from darts import TimeSeries

    series = [
        TimeSeries.from_times_and_values(pd.DatetimeIndex(h.dates), h.values / scale)
        for h, scale in zip(contexts, candidate.scales, strict=True)
    ]
    forecasts = candidate.model.predict(
        4,
        series=series,
        num_samples=1,
        predict_likelihood_parameters=True,
        mc_dropout=False,
        verbose=False,
        dataloader_kwargs={"num_workers": 0, "pin_memory": False},
        random_state=9,
    )
    for i, forecast in enumerate(forecasts):
        mu, sigma = forecast.all_values()[3, :, 0] * candidate.scales[i]
        truth = data[i].raw_values[31]
        expected = (
            0.5 * ((truth - mu) / sigma) ** 2 + np.log(sigma) + 0.5 * np.log(2 * np.pi)
        )
        np.testing.assert_allclose(losses[i, 0], expected)
    changed = candidate.fourth_step_nll(split, np.random.SeedSequence(84))
    np.testing.assert_array_equal(losses, changed)  # no sampling or inference dropout
