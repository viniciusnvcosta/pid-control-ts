# ABOUTME: Checks Transformer configuration, causal scale restoration and epoch selection.
# ABOUTME: Uses an inexpensive backend at the actual Darts fit/predict boundary.
import importlib
import warnings
from dataclasses import replace
from typing import ClassVar

import numpy as np
import pytest
import torch
from darts import TimeSeries
from darts.utils.likelihood_models import GaussianLikelihood

from hcp.forecast_types import PreparedHistory, V2ModelSettings
from hcp.forecast_validation import neural_split


def histories():
    dates = np.datetime64("2020-01-04") + np.arange(40) * np.timedelta64(7, "D")
    return tuple(
        PreparedHistory(
            str(i),
            dates,
            values,
            values,
            np.ones(40, bool),
            dates - np.timedelta64(26, "D"),
        )
        for i, values in enumerate((np.arange(40, dtype=float), np.zeros(40)))
    )


def module():
    if importlib.util.find_spec("hcp.forecast_neural") is None:
        return None
    return importlib.import_module("hcp.forecast_neural")


class Backend:
    instances: ClassVar[list["Backend"]] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.model = torch.nn.Linear(1, 1)
        self.fits, self.predictions = [], []
        self.trainer = type(
            "Trainer", (), {"callback_metrics": {"train_loss": torch.tensor(1.0)}}
        )()
        self.instances.append(self)

    def fit(self, series, **kwargs):
        self.fits.append((series, kwargs))
        with torch.no_grad():
            self.model.weight.add_(1)
        warnings.warn("synthetic fit warning", UserWarning)
        return self

    def predict(
        self, n, *, series, num_samples=1, predict_likelihood_parameters=False, **kwargs
    ):
        self.predictions.append(
            (series, num_samples, predict_likelihood_parameters, kwargs)
        )
        assert n == 4
        if predict_likelihood_parameters:
            values = np.array([[100.0, 0.01], [200.0, 0.02], [300.0, 0.03], [2.0, 0.5]])
        else:
            values = np.broadcast_to(
                np.arange(-2.0, 2.0)[:, None, None], (4, 1, num_samples)
            )
        return [TimeSeries.from_values(values) for _ in series]


@pytest.fixture
def backend(monkeypatch):
    neural = module()
    if neural is None:
        return None
    Backend.instances = []
    monkeypatch.setattr(neural, "TransformerModel", Backend)
    return neural


def test_registered_constructor_scaling_selection_refit_and_streams(backend):
    assert backend is not None, "Transformer adapter is missing"
    settings = replace(
        V2ModelSettings(), n_samples=5, neural_max_epochs=5, neural_patience=1
    )
    seed = np.random.SeedSequence(71)
    prediction = backend.transformer_prediction(histories(), settings, seed)
    inner, full = Backend.instances
    for model in (inner, full):
        expected = {
            "input_chunk_length": 12,
            "output_chunk_length": 4,
            "d_model": 32,
            "nhead": 4,
            "num_encoder_layers": 2,
            "num_decoder_layers": 2,
            "dim_feedforward": 128,
            "dropout": 0.1,
            "activation": "relu",
            "norm_type": "LayerNorm",
            "batch_size": 32,
            "n_epochs": 1,
            "save_checkpoints": False,
            "log_tensorboard": False,
        }
        for key, value in expected.items():
            assert model.kwargs[key] == value
        assert isinstance(model.kwargs["likelihood"], GaussianLikelihood)
        assert model.kwargs["optimizer_cls"] is torch.optim.Adam
        assert model.kwargs["optimizer_kwargs"] == {"lr": 0.001}
        assert (
            model.kwargs.get("custom_encoder") is None
            and model.kwargs.get("custom_decoder") is None
        )
        assert model.kwargs.get("add_encoders") is None
        for _, kwargs in model.fits:
            assert kwargs["epochs"] == 1 and "val_series" not in kwargs
            assert kwargs["dataloader_kwargs"]["pin_memory"] is False
            assert kwargs["dataloader_kwargs"]["num_workers"] == 0
        for _, _, _, kwargs in model.predictions:
            assert kwargs["mc_dropout"] is False
    assert len(inner.fits) == 2 and len(full.fits) == 1
    np.testing.assert_allclose(inner.fits[0][0][0].values()[:, 0], np.arange(28) / 13.5)
    np.testing.assert_allclose(full.fits[0][0][0].values()[:, 0], np.arange(40) / 19.5)
    assert prediction.nodes == ("0", "1") and prediction.samples.shape == (2, 4, 5)
    np.testing.assert_array_equal(
        prediction.samples[0, :, 0], [-39.0, -19.5, 0.0, 19.5]
    )
    np.testing.assert_array_equal(prediction.samples[1, :, 0], [-2.0, -1.0, 0.0, 1.0])
    assert not prediction.samples.flags.writeable
    assert seed.n_children_spawned == 4
    assert prediction.diagnostics["streams"] == {
        "training": ((0,),),
        "validation": ((1,),),
        "refit": ((2,),),
        "prediction": ((3,),),
    }
    root_children = np.random.SeedSequence(71).spawn(4)
    assert inner.kwargs["random_state"] == int(root_children[0].generate_state(1)[0])
    assert full.kwargs["random_state"] == int(root_children[2].generate_state(1)[0])
    assert full.predictions[0][3]["random_state"] == int(
        root_children[3].generate_state(1)[0]
    )
    tuning = prediction.diagnostics["tuning"][0]
    assert tuning["best_epoch"] == 1 and tuning["epochs_run"] == 2
    np.testing.assert_array_equal(tuning["scales"], [13.5, 1.0])
    np.testing.assert_array_equal(
        prediction.diagnostics["final_fit"]["scales"], [19.5, 1.0]
    )
    rows = prediction.diagnostics["warnings"]
    assert {r["phase"] for r in rows} == {"training", "refit"}
    assert all(
        r["model"] == "transformer" and r["candidate"] == 32 and r["lineno"] > 0
        for r in rows
    )
    assert prediction.diagnostics["warning_counts"] == {"UserWarning": 3}


def test_direct_nll_uses_fourth_step_original_units_causal_context_and_owned_best(
    backend,
):
    assert backend is not None, "Transformer adapter is missing"
    data = histories()
    split = neural_split(
        data[0].dates, np.stack([h.observed for h in data]), data[0].forecast_dates
    )
    candidate = backend.TransformerCandidate(
        data, V2ModelSettings(), np.random.SeedSequence(4)
    )
    candidate.train_epoch()
    candidate.retain_best_state()
    retained = {key: val.clone() for key, val in candidate.best_state.items()}
    candidate.train_epoch()
    for key, value in retained.items():
        torch.testing.assert_close(candidate.best_state[key], value)
    assert not torch.equal(candidate.model.model.weight, candidate.best_state["weight"])
    losses = candidate.fourth_step_nll(split, np.random.SeedSequence(9))
    expected0 = (
        0.5 * ((np.arange(31.0, 40.0) - 27.0) / 6.75) ** 2
        + np.log(6.75)
        + 0.5 * np.log(2 * np.pi)
    )
    np.testing.assert_allclose(losses[0], expected0)
    np.testing.assert_allclose(losses[1], 8 + np.log(0.5) + 0.5 * np.log(2 * np.pi))
    for index, (series, samples, parameters, _) in enumerate(
        candidate.model.predictions
    ):
        assert samples == 1 and parameters
        assert len(series[0]) == 28 + index
        assert series[0].end_time().to_datetime64() < split.validation_cutoffs[index]


def test_failure_preserves_phase_warning_evidence(backend, monkeypatch):
    assert backend is not None, "Transformer adapter is missing"
    error = RuntimeError("fit failed")

    def fail(*args, **kwargs):
        warnings.warn("before failure", RuntimeWarning)
        raise error

    monkeypatch.setattr(Backend, "fit", fail)
    with pytest.raises(RuntimeError) as caught:
        backend.transformer_prediction(
            histories(), V2ModelSettings(), np.random.SeedSequence(8)
        )
    assert caught.value is error
    assert error.forecast_diagnostics["warnings"][0]["phase"] == "training"
    assert error.forecast_diagnostics["warning_counts"]["RuntimeWarning"] == 1
