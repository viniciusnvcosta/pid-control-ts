# ABOUTME: Checks outer enhanced DeepAR lifecycle and exact reserved stream use.
# ABOUTME: Uses a deterministic cheap training boundary to test real selection and warnings.
"""Adapter contracts without expensive recurrent training."""

import warnings
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from hcp import forecast_deepar as adapter
from hcp.benchmark_batch import NeuralExecution
from hcp.forecast_types import PreparedHistory, V2ModelSettings


def histories():
    dates = np.datetime64("2020-01-04") + np.arange(40) * np.timedelta64(7, "D")
    return tuple(
        PreparedHistory(
            str(i), dates, x, x, np.ones(40, bool), dates - np.timedelta64(26, "D")
        )
        for i, x in enumerate((np.arange(40.0), np.arange(40.0) * 3))
    )


class TrainingBoundary:
    def __init__(self, hs, settings, width, child, events, fail):
        self.histories, self.settings, self.width = hs, settings, width
        self.events, self.fail = events, fail
        self.epochs_trained = 0
        self.scales = np.array([14.5, 41.5])
        self.transforms = tuple(
            SimpleNamespace(
                first_date=h.dates[0],
                mean=np.array([13.5, 14.5]),
                scale=np.array([8.0, 8.0]),
            )
            for h in hs
        )
        events.append(("create", width, child.spawn_key, settings.n_samples))
        self.emit("construction")

    def emit(self, phase):
        for _ in range(2):
            warnings.warn(f"{phase} evidence", UserWarning)

    def train_epoch(self):
        self.epochs_trained += 1
        self.events.append(("train", self.width, self.epochs_trained))
        self.emit("training")
        if self.fail == self.width:
            raise RuntimeError("candidate failed")

    def fourth_step_nll(self, split, child):
        self.events.append(
            ("validate", self.width, child.spawn_key, child.n_children_spawned)
        )
        child.spawn(9)
        # Width 80 wins; epoch 2 beats epoch 1, then ties forever.
        score = {40: 5.0, 80: 3.0, 120: 4.0}[self.width] - min(self.epochs_trained, 2)
        return np.full((2, 9), score)

    def retain_best_state(self):
        self.events.append(("retain", self.width, self.epochs_trained))

    def fit_full(self, epochs, child):
        self.events.append(
            (
                "refit",
                self.width,
                epochs,
                child.spawn_key,
                tuple(len(h.values) for h in self.histories),
            )
        )
        self.scales = np.array([20.5, 59.5])
        self.epochs_trained = epochs
        self.emit("refit")

    def predict_samples(self, child):
        self.events.append(("predict", self.width, child.spawn_key))
        self.emit("prediction")
        return np.full(
            (2, 4, self.settings.n_samples), self.width + self.epochs_trained
        )


def install_boundary(monkeypatch, fail=None):
    events = []

    def factory(hs, settings, width, child, execution):
        return TrainingBoundary(hs, settings, width, child, events, fail)

    monkeypatch.setattr(adapter, "make_enhanced_candidate", factory)
    return events


def predict(hs, settings, seed):
    implementation = getattr(adapter, "enhanced_deepar_prediction", None)
    assert callable(implementation), "enhanced outer adapter must exist"
    return implementation(hs, settings, seed, NeuralExecution.cpu())


def test_default_budget_selects_epoch_and_full_refit_with_exact_streams(monkeypatch):
    events = install_boundary(monkeypatch)
    seed = np.random.SeedSequence(51, spawn_key=(0, 6, 7))
    result = predict(histories(), V2ModelSettings(), seed)
    assert [e[1] for e in events if e[0] == "create"] == [40, 80, 120]
    assert [e[2] for e in events if e[0] == "create"] == [
        (0, 6, 7, 0),
        (0, 6, 7, 1),
        (0, 6, 7, 2),
    ]
    assert seed.n_children_spawned == 8
    for width, key in [(40, 3), (80, 4), (120, 5)]:
        validation = [e for e in events if e[:2] == ("validate", width)]
        assert len(validation) == 12  # epoch 2 best, ten nonimprovements
        assert all(e[2:] == ((0, 6, 7, key), 0) for e in validation)
    assert [e for e in events if e[0] == "refit"] == [
        ("refit", 80, 2, (0, 6, 7, 6), (40, 40))
    ]
    assert events[-1] == ("predict", 80, (0, 6, 7, 7))
    assert result.nodes == ("0", "1") and result.samples.shape == (2, 4, 500)
    np.testing.assert_array_equal(result.samples, 82.0)
    assert not result.samples.flags.writeable
    diagnostics = result.diagnostics
    assert [r["width"] for r in diagnostics["tuning"]] == [40, 80, 120]
    assert [r["best_epoch"] for r in diagnostics["tuning"]] == [2, 2, 2]
    assert diagnostics["final_fit"]["width"] == 80
    assert diagnostics["final_fit"]["epochs"] == 2
    np.testing.assert_array_equal(diagnostics["final_fit"]["scales"], [20.5, 59.5])
    np.testing.assert_array_equal(diagnostics["tuning"][0]["scales"], [14.5, 41.5])
    assert diagnostics["warning_counts"] == {"UserWarning": 82}
    for row in diagnostics["warnings"]:
        assert row["model"] == "deepar_enhanced" and row["node"] is None
        assert row["forecast_date"] == histories()[0].dates[-1] + np.timedelta64(2, "D")
        assert row["filename"].endswith("test_forecast_deepar_selection.py")
        assert row["lineno"] > 0 and row["count"] >= 2
    assert {r["phase"] for r in diagnostics["warnings"]} == {
        "construction",
        "training",
        "refit",
        "prediction",
    }
    assert diagnostics["streams"]["prediction"] == ((0, 6, 7, 7),)


def test_epoch_cap_is_forwarded_and_width_grid_sorted(monkeypatch):
    events = install_boundary(monkeypatch)
    settings = replace(
        V2ModelSettings(),
        enhanced_hidden_dims=(120, 40, 80),
        neural_max_epochs=1,
        n_samples=7,
    )
    result = predict(histories(), settings, np.random.SeedSequence(4))
    assert [e[1] for e in events if e[0] == "create"] == [40, 80, 120]
    assert [r["epochs_run"] for r in result.diagnostics["tuning"]] == [1, 1, 1]
    assert result.samples.shape == (2, 4, 7)
    assert result.diagnostics["final_fit"]["epochs"] == 1


def test_failed_candidate_keeps_prior_and_failing_warning_records(monkeypatch):
    events = install_boundary(monkeypatch, fail=80)
    filters = list(warnings.filters)
    with pytest.raises(RuntimeError, match="candidate failed") as caught:
        predict(histories(), V2ModelSettings(), np.random.SeedSequence(5))
    assert warnings.filters == filters
    diagnostics = caught.value.forecast_diagnostics
    assert diagnostics["warning_counts"] == {"UserWarning": 32}
    rows = diagnostics["warnings"]
    assert sum(r["count"] for r in rows) == 32
    assert any(
        r["candidate"] == 40 and r["phase"] == "training" and r["count"] == 24
        for r in rows
    )
    assert any(
        r["candidate"] == 80 and r["phase"] == "training" and r["count"] == 2
        for r in rows
    )
    assert not any(e[0] in ("refit", "predict") for e in events)
    with pytest.raises(TypeError):
        diagnostics["warnings"][0]["count"] = 0


def test_default_maximum_runs_100_improving_epochs(monkeypatch):
    events = install_boundary(monkeypatch)

    def improving(self, split, child):
        return np.full((2, 9), 1000.0 - self.epochs_trained)

    monkeypatch.setattr(TrainingBoundary, "fourth_step_nll", improving)
    result = predict(histories(), V2ModelSettings(), np.random.SeedSequence(10))
    assert [r["epochs_run"] for r in result.diagnostics["tuning"]] == [100, 100, 100]
    assert result.diagnostics["final_fit"]["width"] == 40  # exact tie
    assert [e for e in events if e[0] == "refit"] == [
        ("refit", 40, 100, (6,), (40, 40))
    ]


def test_constructor_failure_preserves_original_exception_and_warning(monkeypatch):
    failure = LookupError("construction failed")

    def failing_factory(hs, settings, width, child, execution):
        warnings.warn("constructor warning", FutureWarning)
        raise failure

    monkeypatch.setattr(adapter, "make_enhanced_candidate", failing_factory)
    with pytest.raises(LookupError) as caught:
        predict(histories(), V2ModelSettings(), np.random.SeedSequence(9))
    assert caught.value is failure
    (row,) = failure.forecast_diagnostics["warnings"]
    assert (row["candidate"], row["phase"], row["count"]) == (40, "construction", 1)
    assert failure.forecast_diagnostics["warning_counts"] == {"FutureWarning": 1}


def test_ineligible_split_rejects_before_candidate_training(monkeypatch):
    events = install_boundary(monkeypatch)
    hs = histories()
    observed = hs[0].observed.copy()
    observed[-1] = False
    invalid = (replace(hs[0], observed=observed), hs[1])
    seed = np.random.SeedSequence(6)
    with pytest.raises(ValueError, match="eligible neural split"):
        predict(invalid, V2ModelSettings(), seed)
    assert events == [] and seed.n_children_spawned == 0


@pytest.mark.parametrize(
    "method,phase",
    [
        ("fourth_step_nll", "validation"),
        ("retain_best_state", "retention"),
        ("fit_full", "refit"),
        ("predict_samples", "prediction"),
    ],
)
def test_failure_in_each_later_phase_retains_warning(monkeypatch, method, phase):
    install_boundary(monkeypatch)
    failure = ArithmeticError("phase failed")

    def failing_phase(self, *args):
        self.emit("later phase")
        raise failure

    monkeypatch.setattr(TrainingBoundary, method, failing_phase)
    with pytest.raises(ArithmeticError) as caught:
        predict(histories(), V2ModelSettings(), np.random.SeedSequence(9))
    assert caught.value is failure
    rows = caught.value.forecast_diagnostics["warnings"]
    assert rows[-1]["phase"] == phase and rows[-1]["count"] == 2
    assert rows[0]["phase"] == "construction"


def test_wrong_backend_sample_budget_aborts_with_warning_evidence(monkeypatch):
    install_boundary(monkeypatch)

    def wrong_budget(self, seed):
        self.emit("malformed prediction")
        return np.ones((2, 4, 3))

    monkeypatch.setattr(TrainingBoundary, "predict_samples", wrong_budget)
    with pytest.raises(ValueError, match="sample budget") as caught:
        predict(histories(), V2ModelSettings(), np.random.SeedSequence(3))
    assert caught.value.forecast_diagnostics["warnings"][-1]["phase"] == "prediction"


def test_prediction_mode_cannot_change_another_candidates_loss_modules():
    first = adapter.EnhancedRNNModule(n_nodes=2, hidden_dim=40)
    second = adapter.EnhancedRNNModule(n_nodes=2, hidden_dim=40)
    second.train()
    first.eval()  # rollout switches only its own model to inference
    assert second.criterion.training, (
        "candidate criteria must not share Darts' mutable default"
    )
    assert second.train_criterion.training and second.val_criterion.training
    later = adapter.EnhancedRNNModule(n_nodes=2, hidden_dim=40)
    assert later.criterion.training
    assert later.train_criterion.training and later.val_criterion.training
