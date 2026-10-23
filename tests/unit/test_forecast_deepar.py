# ABOUTME: Behavioral contracts for the embedded recurrent model and training sampler.
# ABOUTME: Exercises actual Darts training, causal preparation, and Gaussian rollouts.
"""Synthetic enhanced DeepAR tests, with reduced path budgets."""

import importlib
import importlib.util
import warnings
from dataclasses import replace

import numpy as np
import pytest
import torch

from hcp.benchmark_batch import NeuralExecution
from hcp.forecast_types import PreparedHistory, V2ModelSettings
from hcp.forecast_validation import neural_split


@pytest.fixture(scope="module")
def _dependency_warnings():
    records = []
    yield records
    assert len(records) == 32
    messages = {str(warning.message) for warning in records}
    assert messages == {
        "Can't initialize NVML",
        "You defined a `validation_step` but have no `val_dataloader`. Skipping val loop.",
        "`isinstance(treespec, LeafSpec)` is deprecated, use `isinstance(treespec, TreeSpec) and treespec.is_leaf()` instead.",
        "The 'train_dataloader' does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` to `num_workers=7` in the `DataLoader` to improve performance.",
    }


@pytest.fixture(autouse=True)
def capture_direct_dependency_warnings(_dependency_warnings):
    """Capture direct candidate warnings in each test without global filters."""
    with warnings.catch_warnings(record=True) as captured:
        yield
    _dependency_warnings.extend(captured)


def components():
    assert importlib.util.find_spec("hcp.forecast_deepar") is not None, (
        "enhanced module missing"
    )
    return importlib.import_module("hcp.forecast_deepar")


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
        for i, values in enumerate(
            (np.arange(40, dtype=float), np.arange(40, dtype=float) * 3)
        )
    )


def candidate(seed=7):
    return components().make_enhanced_candidate(
        histories(),
        replace(V2ModelSettings(), n_samples=8),
        40,
        np.random.SeedSequence(seed),
        NeuralExecution.cpu(),
    )


@pytest.mark.parametrize("width", [40, 80, 120])
def test_node_embedding_affects_gaussian_parameters_and_receives_gradient(width):
    module = components().EnhancedRNNModule(n_nodes=2, hidden_dim=width)
    x = torch.ones(2, 12, 3)
    out, _ = module((x, None, torch.tensor([[[0.0]], [[1.0]]]), None))
    assert out.shape == (2, 12, 1, 2)
    assert not torch.allclose(out[0], out[1])
    out.square().sum().backward()
    assert module.embedding.weight.grad.shape == (2, 20)
    assert torch.all(module.embedding.weight.grad.abs().sum(dim=1) > 0)
    assert module.rnn.num_layers == 3


def test_sampler_node_mass_not_multiplied_by_window_count():
    components()
    data = importlib.import_module("hcp.forecast_deepar_data")
    counts = np.array([1000, 9000])
    sampler = data.weighted_window_sampler(
        counts, np.array([1.0, 3.0]), np.random.SeedSequence(21)
    )
    draws = np.array(list(sampler))
    assert len(draws) == 10000 and sampler.replacement and sampler.generator is not None
    # Binomial standard error sqrt(.25*.75/10000)=.00433; .025 exceeds 5 sigma.
    assert abs(np.mean(draws < 1000) - 0.25) < 0.025
    assert len(np.unique(draws)) < len(draws)
    np.testing.assert_array_equal(
        draws,
        list(
            data.weighted_window_sampler(
                counts, np.array([1.0, 3.0]), np.random.SeedSequence(21)
            )
        ),
    )


def test_covariates_use_first_observation_and_training_only_statistics():
    components()
    data = importlib.import_module("hcp.forecast_deepar_data")
    original = histories()[0]
    observed = original.observed.copy()
    observed[:2] = False
    raw = original.raw_values.copy()
    raw[:2] = np.nan
    history = replace(original, observed=observed, raw_values=raw)
    cutoff = original.dates[5] + np.timedelta64(2, "D")
    future = original.dates[6:8]
    (values,) = data.enhanced_covariates((history,), cutoff, future)
    # Four train ages 0,1,2,3: mean 1.5; population std sqrt(1.25).
    np.testing.assert_allclose(values[:, 0], (np.arange(6) - 1.5) / np.sqrt(1.25))
    np.testing.assert_allclose(values[:4].mean(axis=0), 0, atol=1e-7)
    np.testing.assert_allclose(values[:4].std(axis=0), 1)


def test_actual_training_consumes_sampler_and_advances_weights_and_optimizer_twice():
    model = candidate()
    before = {k: v.clone() for k, v in model.module.state_dict().items()}
    sampler_before = model.loader.sampler.generator.get_state().clone()
    model.train_epoch()
    assert not torch.equal(sampler_before, model.loader.sampler.generator.get_state())
    first = {k: v.clone() for k, v in model.module.state_dict().items()}
    assert any(not torch.equal(before[k], first[k]) for k in before)
    optimizer = model.module.configure_optimizers()
    steps = [int(s["step"]) for s in optimizer.state.values()]
    model.retain_best_state()
    model.train_epoch()
    assert model.module.configure_optimizers() is optimizer
    assert all(
        int(s["step"]) > step for s, step in zip(optimizer.state.values(), steps)
    )
    assert any(
        not torch.equal(first[k], v) for k, v in model.module.state_dict().items()
    )
    for k, v in model.best_state.items():
        assert (
            torch.equal(v, first[k])
            and v.data_ptr() != model.module.state_dict()[k].data_ptr()
        )
    assert model.loader.num_workers == 0 and not model.loader.pin_memory


def test_causal_scales_validation_paths_and_fresh_full_refit():
    model = candidate()
    np.testing.assert_allclose(model.scales, [14.5, 41.5])  # first 28 values only
    model.train_epoch()
    hs = histories()
    split = neural_split(
        hs[0].dates, np.stack([h.observed for h in hs]), hs[0].forecast_dates
    )
    losses = model.fourth_step_nll(split, np.random.SeedSequence(31))
    assert losses.shape == (2, 9) and np.isfinite(losses).all()
    np.testing.assert_array_equal(
        losses, model.fourth_step_nll(split, np.random.SeedSequence(31))
    )
    old_module, old_optimizer = model.module, model.module.configure_optimizers()
    model.fit_full(1, np.random.SeedSequence(40))
    assert (
        model.module is not old_module
        and model.module.configure_optimizers() is not old_optimizer
    )
    np.testing.assert_allclose(model.scales, [20.5, 59.5])
    paths = model.predict_samples(np.random.SeedSequence(50))
    assert paths.shape == (2, 4, 8) and np.isfinite(paths).all()
    np.testing.assert_array_equal(
        paths, model.predict_samples(np.random.SeedSequence(50))
    )


def test_gaussian_location_scale_and_draws_restore_identical_native_units():
    model = candidate()
    rollout = importlib.import_module("hcp.forecast_deepar_rollout").recurrent_rollout
    args = (model.module, model.prefixes, model.transforms, model.scales, 8)
    ordinary = rollout(*args, np.random.SeedSequence(91), NeuralExecution.cpu())
    doubled_contexts = tuple(replace(h, values=h.values * 2) for h in model.prefixes)
    doubled = rollout(
        model.module,
        doubled_contexts,
        model.transforms,
        model.scales * 2,
        8,
        np.random.SeedSequence(91),
        NeuralExecution.cpu(),
    )
    for actual, original in zip(doubled, ordinary, strict=True):
        np.testing.assert_allclose(actual, original * 2, rtol=0, atol=0)
        assert not actual.flags.writeable


def test_unavailable_values_cannot_change_inner_preprocessing_or_first_origin_rollout():
    model = candidate()
    hs = histories()
    changed = []
    for h in hs:
        raw = h.raw_values.copy()
        raw[28:31] += 1000000
        raw[32:] += 1000000
        # Incorrectly full-imputed model-ready values must be ignored entirely.
        changed.append(replace(h, raw_values=raw, values=np.full(40, -999.0)))
    other = components().make_enhanced_candidate(
        tuple(changed),
        model.settings,
        40,
        np.random.SeedSequence(7),
        NeuralExecution.cpu(),
    )
    np.testing.assert_array_equal(model.scales, other.scales)
    for a, b in zip(model.dataset.targets, other.dataset.targets, strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    for a, b in zip(model.transforms, other.transforms, strict=True):
        np.testing.assert_array_equal(a.mean, b.mean)
        np.testing.assert_array_equal(a.scale, b.scale)
    split = neural_split(
        hs[0].dates, np.stack([h.observed for h in hs]), hs[0].forecast_dates
    )
    ordinary = model.fourth_step_nll(split, np.random.SeedSequence(4))
    changed_nll = other.fourth_step_nll(split, np.random.SeedSequence(4))
    np.testing.assert_array_equal(ordinary[:, 0], changed_nll[:, 0])
    assert not np.array_equal(ordinary[:, 1:], changed_nll[:, 1:])


def test_refit_seed_reproduces_weights_independently_of_inner_training():
    first, second = candidate(51), candidate(82)
    first.train_epoch()
    first.retain_best_state()
    for model in (first, second):
        model.fit_full(2, np.random.SeedSequence(90))
        assert model.epochs_trained == 2
    for name, tensor in first.module.state_dict().items():
        torch.testing.assert_close(
            tensor, second.module.state_dict()[name], rtol=0, atol=0
        )


def test_train_after_validation_restores_training_mode_before_trainer(monkeypatch):
    model = candidate()
    model.predict_samples(np.random.SeedSequence(3))
    real_trainer = components().Trainer
    seen = []

    class TrainerBoundary(real_trainer):
        def fit(self, module, *args, **kwargs):
            seen.append(module.training)
            return super().fit(module, *args, **kwargs)

    monkeypatch.setattr(components(), "Trainer", TrainerBoundary)
    model.train_epoch()
    assert seen == [True]


def test_unequal_node_starts_keep_full_training_prefixes_and_correct_sampler_mass():
    first, second = histories()
    shortened = replace(
        second,
        dates=second.dates[5:],
        values=second.values[5:],
        raw_values=second.raw_values[5:],
        observed=second.observed[5:],
        forecast_dates=second.forecast_dates[5:],
    )
    model = components().make_enhanced_candidate(
        (first, shortened),
        replace(V2ModelSettings(), n_samples=8),
        40,
        np.random.SeedSequence(9),
        NeuralExecution.cpu(),
    )
    assert [len(h.values) for h in model.prefixes] == [28, 23]
    np.testing.assert_array_equal(model.dataset.window_counts, [12, 7])
    np.testing.assert_allclose(model.scales, [14.5, 49.0])
    weights = model.loader.sampler.weights.numpy()
    np.testing.assert_allclose([weights[:12].sum(), weights[12:].sum()], [14.5, 49.0])
    assert model.transforms[0].first_date == first.dates[0]
    assert model.transforms[1].first_date == second.dates[5]
    past, _, _, future_cov, _, weight, target = model.dataset[0]
    np.testing.assert_allclose(past[:, 0], np.arange(16) / 14.5)
    np.testing.assert_allclose(target[:, 0], np.arange(1, 17) / 14.5)
    np.testing.assert_allclose(
        future_cov[:, 0], (np.arange(1, 17) - 13.5) / np.std(np.arange(28)), rtol=1e-6
    )
    assert weight is None
    model.train_epoch()
