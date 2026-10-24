# ABOUTME: Exercises the real Darts Transformer on reduced synthetic causal prefixes.
# ABOUTME: Protects optimizer continuity, fourth-step density, seeded paths and no-write execution.
import importlib
from dataclasses import replace

import numpy as np
import pytest
import torch

from hcp.benchmark_batch import NeuralExecution
from hcp.forecast_types import PreparedHistory, V2ModelSettings, prediction_quantiles
from hcp.forecast_validation import neural_split, prepare_prefix

pytestmark = pytest.mark.integration


def _histories(future_shift=0.0):
    dates = np.datetime64("2020-01-04") + np.arange(44) * np.timedelta64(7, "D")
    result = []
    for i in range(2):
        values = -np.arange(44, dtype=float) * (i + 1)
        values[40:] += future_shift
        observed = np.arange(44) < 40
        result.append(
            PreparedHistory(
                str(i),
                dates,
                values,
                values,
                observed,
                dates - np.timedelta64(26, "D"),
            )
        )
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
    changed_histories = _histories(1e6)
    assert np.all(
        changed_histories[0].raw_values[40:] != _histories()[0].raw_values[40:]
    )
    assert not changed_histories[0].observed[40:].any()
    results = [
        neural.transformer_prediction(
            _histories(shift),
            settings,
            np.random.SeedSequence(81),
            NeuralExecution.cpu(),
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
        data, V2ModelSettings(), np.random.SeedSequence(82), NeuralExecution.cpu()
    )
    data = candidate.histories
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
        TimeSeries.from_times_and_values(
            pd.DatetimeIndex(h.dates), (h.values / scale).astype(np.float32)
        )
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


def test_real_cuda_transformer_repeatability_and_evidence(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    neural = _module()
    execution = NeuralExecution(
        "cuda:0", "NVIDIA GeForce RTX 4080", ":4096:8", False, "32-true"
    )
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    devices = []
    forward = neural._PersistentTransformerModule.forward

    def record_forward(module, inputs, *args, **kwargs):
        devices.append(str(inputs[0].device))
        return forward(module, inputs, *args, **kwargs)

    monkeypatch.setattr(neural._PersistentTransformerModule, "forward", record_forward)
    settings = replace(
        V2ModelSettings(), neural_max_epochs=1, neural_patience=1, n_samples=8
    )
    outputs = [
        neural.transformer_prediction(
            _histories(), settings, np.random.SeedSequence(81), execution
        )
        for _ in range(2)
    ]
    left, right = outputs
    np.testing.assert_array_equal(left.samples, right.samples)
    assert left.diagnostics["streams"] == right.diagnostics["streams"]
    assert left.diagnostics["warning_counts"] == right.diagnostics["warning_counts"]
    for a, b in zip(
        left.diagnostics["tuning"], right.diagnostics["tuning"], strict=True
    ):
        assert a.keys() == b.keys()
        for key in a:
            np.testing.assert_array_equal(a[key], b[key])
    assert devices and set(devices) == {"cuda:0"}


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_one_lifecycle_matches_three_epoch_cpu_characterization_and_refit(
    monkeypatch, caplog, device
):
    import copy
    import hashlib
    import logging

    from pytorch_lightning import Trainer

    neural = _module()
    if device == "cuda:0" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    execution = (
        NeuralExecution.cpu()
        if device == "cpu"
        else NeuralExecution(
            "cuda:0", "NVIDIA GeForce RTX 4080", ":4096:8", False, "32-true"
        )
    )
    if device == "cuda:0":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    settings = replace(V2ModelSettings(), n_samples=8)
    fits = []
    original_fit = Trainer.fit

    def record_fit(trainer, *args, **kwargs):
        fits.append(trainer)
        return original_fit(trainer, *args, **kwargs)

    monkeypatch.setattr(Trainer, "fit", record_fit)

    def build():
        return neural.TransformerCandidate(
            _histories(), settings, np.random.SeedSequence(82), execution
        )

    baseline = build()
    split = baseline.split
    losses, states, optimizers = [], [], []
    for _ in range(3):
        baseline.train_epoch()
        losses.append(baseline.fourth_step_nll(split, np.random.SeedSequence(83)))
        states.append(copy.deepcopy(baseline.model.model.state_dict()))
        optimizers.append(
            copy.deepcopy(baseline.model.model.configure_optimizers().state_dict())
        )
    if device == "cpu":
        # Frozen hashes captured directly from da54c5f before the lifecycle edits.
        assert [
            hashlib.sha256(
                b"".join(state[k].cpu().numpy().tobytes() for k in sorted(state))
            ).hexdigest()
            for state in states
        ] == [
            "c9d309bde9fd36767e4db016657066d5f1e4204af9d36feba23abb3d02d7a1d8",
            "41e0053e19c5fdb2c7faa98956a104299f0e90974abaffcde1bee20053fe3f13",
            "a4fff15b2d926624131509109c3cf059f601e5e7d6d2be205abf488ff82f91b0",
        ]
        assert [float(np.mean(x)) for x in losses] == [
            6.777089701471386,
            5.089570571643926,
            4.566897099806251,
        ]

    candidate = build()
    original_score = candidate.fourth_step_nll
    observed, optimizer_ids = [], []

    def score(split, seed):
        index = len(observed)
        actual = original_score(split, seed)
        np.testing.assert_array_equal(actual, losses[index])
        torch.testing.assert_close(
            candidate.model.model.state_dict(), states[index], rtol=0, atol=0
        )
        optimizer = candidate.model.model.configure_optimizers()
        optimizer_ids.append(id(optimizer))
        torch.testing.assert_close(
            optimizer.state_dict(),
            optimizers[index],
            rtol=0,
            atol=0,
            check_device=False,
        )
        observed.append(actual)
        return actual

    monkeypatch.setattr(candidate, "fourth_step_nll", score)
    caplog.clear()
    start = len(fits)
    with caplog.at_level(logging.INFO):
        result = candidate.fit_selection(split, np.random.SeedSequence(83), 3, 3)
    assert result.best_epoch == result.epochs_run == 3
    assert len(fits) - start == 1 and len(set(optimizer_ids)) == 1
    assert len(observed) == 3
    torch.testing.assert_close(
        candidate.best_state, states[-1], rtol=0, atol=0, check_device=False
    )
    # Startup output must stay bounded even though fourth-step prediction runs each epoch.
    assert sum("GPU available:" in record.message for record in caplog.records) <= 2

    baseline._build(
        baseline.histories[0].dates[-1] + np.timedelta64(2, "D"),
        np.random.SeedSequence(84),
    )
    for _ in range(3):
        baseline.train_epoch()
    start = len(fits)
    candidate.fit_full(3, np.random.SeedSequence(84))
    assert len(fits) - start == 1
    torch.testing.assert_close(
        candidate.model.model.state_dict(),
        baseline.model.model.state_dict(),
        rtol=0,
        atol=0,
    )
    for model in (baseline, candidate):
        model._samples = model.model.predict(
            4,
            series=model.series,
            num_samples=8,
            mc_dropout=False,
            verbose=False,
            dataloader_kwargs={"num_workers": 0, "pin_memory": False},
            random_state=85,
        )
    for a, b in zip(baseline._samples, candidate._samples, strict=True):
        np.testing.assert_array_equal(a.all_values(), b.all_values())
