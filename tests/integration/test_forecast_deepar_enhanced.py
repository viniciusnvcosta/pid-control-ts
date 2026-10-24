# ABOUTME: Exercises selected enhanced DeepAR with real Darts on synthetic 57-node data.
# ABOUTME: Checks repeatability, causal outer preparation, native scales, and warning evidence.
"""Reduced development budgets never alter the production defaults."""

import json
from dataclasses import replace

import numpy as np
import pytest

from hcp.benchmark_batch import NeuralExecution
from hcp.forecast_deepar import enhanced_deepar_prediction
from hcp.forecast_types import PreparedHistory, V2ModelSettings
from hcp.forecast_validation import prepare_prefix

pytestmark = pytest.mark.integration


def _available_histories(future_shift=0.0):
    dates = np.datetime64("2020-01-04") + np.arange(44) * np.timedelta64(7, "D")
    result = []
    for node in range(57):
        values = -np.arange(44, dtype=float) * (node + 1)
        values[40:] += future_shift
        raw = PreparedHistory(
            str(node),
            dates,
            values,
            values,
            np.ones(44, bool),
            dates - np.timedelta64(26, "D"),
        )
        result.append(prepare_prefix(raw, dates[39] + np.timedelta64(2, "D")))
    return tuple(result)


def test_real_57_node_paths_repeat_and_ignore_unavailable_future(record_property):
    settings = replace(
        V2ModelSettings(), neural_max_epochs=1, neural_patience=1, n_samples=8
    )
    predictions = [
        enhanced_deepar_prediction(
            _available_histories(shift),
            settings,
            np.random.SeedSequence(71),
            NeuralExecution.cpu(),
        )
        for shift in (0.0, 0.0, 1000000.0)
    ]
    first = predictions[0]
    assert first.nodes == tuple(str(i) for i in range(57))
    assert first.samples.shape == (57, 4, 8)
    assert np.isfinite(first.samples).all() and not first.samples.flags.writeable
    assert np.std(first.samples[0, -1]) > 0
    assert np.any(first.samples < 0)  # native Gaussian paths are never clipped
    for prediction in predictions:
        np.testing.assert_array_equal(first.samples, prediction.samples)
        tuning = prediction.diagnostics["tuning"]
        assert [row["width"] for row in tuning] == [40, 80, 120]
        for row in tuning:
            # Inner 28 weeks: mean 0..27 = 13.5; full 40 weeks: mean = 19.5.
            np.testing.assert_array_equal(row["scales"], 1.0 + 13.5 * np.arange(1, 58))
            assert row["best_epoch"] == row["epochs_run"] == 1
            assert np.isfinite(row["best_nll"])
        fit = prediction.diagnostics["final_fit"]
        np.testing.assert_array_equal(fit["scales"], 1.0 + 19.5 * np.arange(1, 58))
        assert fit["epochs"] == 1
        for covariance in fit["covariates"]:
            assert covariance["mean"][0] == 19.5
        rows = prediction.diagnostics["warnings"]
        assert rows and all(row["filename"] and row["lineno"] > 0 for row in rows)
        assert all(
            row["node"] is None and row["model"] == "deepar_enhanced" for row in rows
        )
        assert sum(row["count"] for row in rows) == sum(
            prediction.diagnostics["warning_counts"].values()
        )
        assert {row["phase"] for row in rows} >= {"training", "refit"}
    record_property(
        "warning_evidence",
        json.dumps(
            [
                [dict(row) for row in prediction.diagnostics["warnings"]]
                for prediction in predictions
            ],
            default=str,
        ),
    )


def test_real_cuda_enhanced_repeatability_and_evidence(monkeypatch):
    import torch

    from hcp import forecast_deepar

    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    execution = NeuralExecution(
        "cuda:0", "NVIDIA GeForce RTX 4080", ":4096:8", False, "32-true"
    )
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    devices = []
    forward = forecast_deepar.EnhancedRNNModule.forward

    def record_forward(module, inputs, *args, **kwargs):
        devices.append(str(inputs[0].device))
        return forward(module, inputs, *args, **kwargs)

    monkeypatch.setattr(forecast_deepar.EnhancedRNNModule, "forward", record_forward)
    settings = replace(
        V2ModelSettings(), neural_max_epochs=1, neural_patience=1, n_samples=8
    )
    outputs = [
        enhanced_deepar_prediction(
            _available_histories()[:2], settings, np.random.SeedSequence(71), execution
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
            if key == "covariates":
                for ca, cb in zip(a[key], b[key], strict=True):
                    for field in ca:
                        np.testing.assert_array_equal(ca[field], cb[field])
            else:
                np.testing.assert_array_equal(a[key], b[key])
    assert devices and set(devices) == {"cuda:0"}


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_one_lifecycle_preserves_cpu_epoch_weights_adam_sampler_and_full_refit(
    monkeypatch, device
):
    import copy
    import hashlib

    import torch
    from pytorch_lightning import Trainer

    from hcp.forecast_deepar import EnhancedCandidate, _enhanced_split

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

    def build():
        return EnhancedCandidate(
            _available_histories()[:2],
            settings,
            40,
            np.random.SeedSequence(82),
            execution,
        )

    baseline = build()
    split = _enhanced_split(baseline.histories)
    losses, states, optimizers, samplers = [], [], [], []
    for _ in range(3):
        baseline.train_epoch()
        losses.append(baseline.fourth_step_nll(split, np.random.SeedSequence(83)))
        states.append(copy.deepcopy(baseline.module.state_dict()))
        optimizers.append(
            copy.deepcopy(baseline.module.configure_optimizers().state_dict())
        )
        samplers.append(baseline.loader.sampler.generator.get_state().clone())
    if device == "cpu":
        # Frozen hashes captured directly from da54c5f before the lifecycle edits.
        assert [
            hashlib.sha256(
                b"".join(state[k].cpu().numpy().tobytes() for k in sorted(state))
            ).hexdigest()
            for state in states
        ] == [
            "772d6c8016ecef34c7099d7054aff146ada4828b5adbcb75a9b324c5e2788d2f",
            "e67a81ea5a9a001b3142d68ba5e6fd3575afa3bb275a5c5491b7b8ceeaaa5aff",
            "bee9fd55a93db8e59ad53b2fea8a05b6b486af3b7825aa59041074c9024c9e26",
        ]
        assert [float(np.mean(x)) for x in losses] == [
            10.01195852606616,
            9.815003573747745,
            9.622106724686086,
        ]
    candidate = build()
    original_score = candidate.fourth_step_nll
    observed, optimizer_ids = [], []

    def score(split, seed):
        index = len(observed)
        actual = original_score(split, seed)
        np.testing.assert_array_equal(actual, losses[index])
        torch.testing.assert_close(
            candidate.module.state_dict(), states[index], rtol=0, atol=0
        )
        optimizer = candidate.module.configure_optimizers()
        optimizer_ids.append(id(optimizer))
        torch.testing.assert_close(
            optimizer.state_dict(),
            optimizers[index],
            rtol=0,
            atol=0,
            check_device=False,
        )
        assert torch.equal(
            candidate.loader.sampler.generator.get_state(), samplers[index]
        )
        observed.append(actual)
        return actual

    monkeypatch.setattr(candidate, "fourth_step_nll", score)
    fits = []
    original_fit = Trainer.fit

    def record_fit(trainer, *args, **kwargs):
        fits.append(trainer)
        return original_fit(trainer, *args, **kwargs)

    monkeypatch.setattr(Trainer, "fit", record_fit)
    result = candidate.fit_selection(split, np.random.SeedSequence(83), 3, 3)
    assert result.best_epoch == result.epochs_run == 3
    assert len(fits) == 1 and len(set(optimizer_ids)) == 1
    torch.testing.assert_close(
        candidate.best_state, states[-1], rtol=0, atol=0, check_device=False
    )
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
        candidate.module.state_dict(), baseline.module.state_dict(), rtol=0, atol=0
    )
    np.testing.assert_array_equal(
        candidate.predict_samples(np.random.SeedSequence(85)),
        baseline.predict_samples(np.random.SeedSequence(85)),
    )
