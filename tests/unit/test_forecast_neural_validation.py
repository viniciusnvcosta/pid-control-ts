# ABOUTME: Exercises original-unit Gaussian mixture scoring and neural selection behavior.
# ABOUTME: Uses deterministic candidates to expose stopping, refit, and random-stream errors.
"""Behavioral tests for factory-independent neural validation."""

import numpy as np
import pytest


def test_constant_gaussian_mixture_matches_analytic_density() -> None:
    from hcp.forecast_neural_validation import gaussian_mixture_nll

    actual = gaussian_mixture_nll(
        np.array([[2.0, 4.0]]),
        np.full((1, 2, 500), 2.0),
        np.full((1, 2, 500), 2.0),
    )
    # log(2 sqrt(2 pi)) plus one half of the squared standardized residual.
    np.testing.assert_allclose(actual, [[1.612085713764618, 2.112085713764618]])


def test_mixture_uses_log_sum_exp_for_extreme_density() -> None:
    from hcp.forecast_neural_validation import gaussian_mixture_nll

    actual = gaussian_mixture_nll(
        np.array([100.0]), np.array([[0.0, 100.0]]), np.ones((1, 2))
    )
    np.testing.assert_allclose(actual, [1.612085713764618])
    far = gaussian_mixture_nll(np.array([1e100]), np.zeros((1, 2)), np.ones((1, 2)))
    np.testing.assert_allclose(far, [5e199])


@pytest.mark.parametrize(
    "y, mu, sigma",
    [
        ([0.0], [[0.0]], [[0.0]]),
        ([0.0], [[0.0]], [[-1.0]]),
        ([0.0], [[np.nan]], [[1.0]]),
        ([np.inf], [[0.0]], [[1.0]]),
        ([0.0], [[]], [[]]),
        ([0.0], [[0.0, 1.0]], [[1.0]]),
        ([0.0, 1.0], [[0.0]], [[1.0]]),
    ],
)
def test_mixture_rejects_invalid_gaussian_inputs(y, mu, sigma) -> None:
    from hcp.forecast_neural_validation import gaussian_mixture_nll

    with pytest.raises(ValueError):
        gaussian_mixture_nll(np.array(y), np.array(mu), np.array(sigma))


class Candidate:
    """Small trainable state machine standing in for slow neural architectures."""

    def __init__(self, losses, *, training_seed=None, one_step_loss=0.0):
        self.losses = losses
        self.one_step_loss = one_step_loss
        self.weights = np.array([0.0])
        self.best_weights = None
        self.epoch = 0
        self.saved_epochs = []
        self.validation_draws = []
        self.validation_child_draws = []
        self.refit = None
        self.rng = np.random.default_rng(training_seed)

    def fit_selection(
        self, split, validation_seed, max_epochs, patience, *, capture=None
    ):
        from contextlib import nullcontext

        from hcp.forecast_neural_validation import CandidateSelection

        phases = capture or (lambda phase: nullcontext())
        selection = CandidateSelection(self, split, validation_seed, patience, phases)
        for _ in range(max_epochs):
            with phases("training"):
                self.train_epoch()
            if selection.record_epoch():
                break
        return selection.result()

    def train_epoch(self):
        self.epoch += 1
        self.weights[:] = self.epoch
        self.rng.normal()

    def fourth_step_nll(self, split, seed):
        self.validation_draws.append(np.random.default_rng(seed).normal(size=5))
        self.validation_child_draws.append(
            np.random.default_rng(seed.spawn(1)[0]).normal(size=5)
        )
        loss = self.losses[min(self.epoch - 1, len(self.losses) - 1)]
        return (
            np.full((2, len(split.validation_targets)), loss)
            if np.ndim(loss) == 0
            else loss
        )

    def retain_best_state(self):
        self.best_weights = self.weights.copy()
        self.saved_epochs.append(self.epoch)

    def fit_full(self, epochs, seed):
        self.refit = (epochs, seed.spawn_key)
        self.weights = np.array([-float(epochs)])


def split_fixture():
    from hcp.forecast_validation import neural_split

    dates = np.arange(np.datetime64("2020-01-04"), np.datetime64("2021-07-17"), 7)
    return neural_split(
        dates, np.ones((2, len(dates)), dtype=bool), dates - np.timedelta64(26, "D")
    )


def test_strict_improvement_keeps_epoch_two_and_stops_after_ten_ties() -> None:
    from hcp.forecast_neural_validation import select_neural

    candidate = Candidate([3.0, 2.0, 2.0])
    result = select_neural((candidate,), split_fixture(), np.random.SeedSequence(7))
    assert (result.candidate, result.best_epoch, result.best_nll) == (0, 2, 2.0)
    assert candidate.epoch == 12
    assert candidate.saved_epochs == [1, 2]
    np.testing.assert_array_equal(candidate.best_weights, [2.0])
    np.testing.assert_array_equal(candidate.weights, [-2.0])
    assert candidate.refit == (2, (2,))
    assert result.tuning.iloc[0].to_dict() == {
        "candidate": 0,
        "best_epoch": 2,
        "best_nll": 2.0,
        "epochs_run": 12,
    }


def test_equal_candidates_keep_smaller_width_and_use_fourth_step_loss() -> None:
    from hcp.forecast_neural_validation import select_neural

    candidates = (
        Candidate([4.0], one_step_loss=-1000.0),
        Candidate([2.0], one_step_loss=1000.0),
        Candidate([2.0], one_step_loss=-2000.0),
    )
    result = select_neural(
        candidates, split_fixture(), np.random.SeedSequence(9), max_epochs=2
    )
    assert result.candidate == 1
    assert result.best_nll == 2.0
    assert result.tuning.best_epoch.tolist() == [1, 1, 1]
    assert result.tuning.best_nll.tolist() == [4.0, 2.0, 2.0]
    assert [candidate.refit for candidate in candidates] == [None, (1, (6,)), None]


def test_original_unit_losses_weight_nodes_and_origins_equally() -> None:
    from hcp.forecast_neural_validation import select_neural

    candidate = Candidate([np.array([[1.0] * 9, [3.0] * 8 + [21.0]])])
    result = select_neural(
        (candidate,), split_fixture(), np.random.SeedSequence(1), max_epochs=1
    )
    assert result.best_nll == 3.0


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_any_nonfinite_candidate_aborts_without_refit(bad) -> None:
    from hcp.forecast_neural_validation import select_neural

    candidates = (Candidate([1.0]), Candidate([bad]))
    with pytest.raises(ValueError, match="finite"):
        select_neural(
            candidates, split_fixture(), np.random.SeedSequence(2), max_epochs=2
        )
    assert all(candidate.refit is None for candidate in candidates)


def test_validation_stream_resets_without_mutating_preallocated_streams() -> None:
    from hcp.forecast_neural_validation import select_neural
    from hcp.forecast_validation import neural_seed_streams

    seed = np.random.SeedSequence(11)
    streams = neural_seed_streams(seed, 1)
    # A child already used for spawning must reset to that same starting state too.
    streams["validation"][0].spawn(2)
    candidate = Candidate([3.0, 2.0, 1.0], training_seed=streams["training"][0])
    before = {
        key: tuple(child.state for child in group) for key, group in streams.items()
    }
    select_neural((candidate,), split_fixture(), seed, max_epochs=3, streams=streams)
    assert seed.n_children_spawned == 4
    assert {
        key: tuple(child.state for child in group) for key, group in streams.items()
    } == before
    for draws in (candidate.validation_draws, candidate.validation_child_draws):
        assert len(draws) == 3
        for draw in draws[1:]:
            np.testing.assert_array_equal(draw, draws[0])


@pytest.mark.parametrize("epochs, bad", [(1, False), (5, False), (3, True)])
def test_prediction_stream_is_independent_of_training_length_and_failures(
    epochs, bad
) -> None:
    from hcp.forecast_neural_validation import select_neural
    from hcp.forecast_validation import neural_seed_streams

    seed = np.random.SeedSequence(17)
    streams = neural_seed_streams(seed, 1)
    candidate = Candidate([np.nan if bad else 1.0])
    if bad:
        with pytest.raises(ValueError):
            select_neural(
                (candidate,), split_fixture(), seed, max_epochs=epochs, streams=streams
            )
    else:
        select_neural(
            (candidate,), split_fixture(), seed, max_epochs=epochs, streams=streams
        )
    np.testing.assert_allclose(
        np.random.default_rng(streams["prediction"][0]).normal(size=3),
        [0.06808493, -0.75642129, 1.54658726],
    )


@pytest.mark.parametrize("bad_loss", [np.ones(9), np.ones((2, 1)), np.ones((0, 9))])
def test_objective_requires_one_loss_per_node_and_nine_fourth_step_origins(
    bad_loss,
) -> None:
    from hcp.forecast_neural_validation import select_neural

    with pytest.raises(ValueError, match="shape"):
        select_neural(
            (Candidate([bad_loss]),), split_fixture(), np.random.SeedSequence(4)
        )


@pytest.mark.parametrize("change", ["missing", "count", "type"])
def test_invalid_preallocated_streams_fail_before_training(change) -> None:
    from hcp.forecast_neural_validation import select_neural
    from hcp.forecast_validation import neural_seed_streams

    seed = np.random.SeedSequence(2)
    streams = neural_seed_streams(seed, 1)
    if change == "missing":
        del streams["prediction"]
    elif change == "count":
        streams["training"] = ()
    else:
        streams["validation"] = (123,)
    candidate = Candidate([1.0])
    with pytest.raises(ValueError, match="streams"):
        select_neural((candidate,), split_fixture(), seed, streams=streams)
    assert candidate.epoch == 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_epochs": 0},
        {"max_epochs": 101},
        {"max_epochs": True},
        {"patience": 0},
        {"patience": 11},
    ],
)
def test_invalid_training_budget_fails_before_training(kwargs) -> None:
    from hcp.forecast_neural_validation import select_neural

    candidate = Candidate([1.0])
    with pytest.raises(ValueError):
        select_neural(
            (candidate,), split_fixture(), np.random.SeedSequence(5), **kwargs
        )
    assert candidate.epoch == 0


def test_candidate_fit_failure_aborts_instead_of_removing_candidate() -> None:
    from hcp.forecast_neural_validation import select_neural

    class FailedFit(Candidate):
        def train_epoch(self):
            raise RuntimeError("fit failed")

    candidates = (Candidate([1.0]), FailedFit([2.0]))
    with pytest.raises(RuntimeError, match="fit failed"):
        select_neural(
            candidates, split_fixture(), np.random.SeedSequence(8), max_epochs=1
        )
    assert candidates[0].refit is None


def test_selection_obeys_epoch_budget_when_every_epoch_improves() -> None:
    from hcp.forecast_neural_validation import select_neural

    candidate = Candidate(list(range(101, 0, -1)))
    result = select_neural((candidate,), split_fixture(), np.random.SeedSequence(3))
    assert candidate.epoch == result.best_epoch == 100
    assert result.best_nll == 2.0
    assert candidate.refit == (100, (2,))


def test_preallocated_streams_must_be_a_mapping_before_training() -> None:
    from hcp.forecast_neural_validation import select_neural

    candidate = Candidate([1.0])
    with pytest.raises(TypeError, match="streams"):
        select_neural(
            (candidate,), split_fixture(), np.random.SeedSequence(2), streams=[]
        )
    assert candidate.epoch == 0
