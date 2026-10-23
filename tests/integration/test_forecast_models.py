# ABOUTME: Integration tests for real Darts probabilistic benchmark model adapters.
# ABOUTME: Checks shape, finiteness, seed determinism, stochasticity, and horizon-four sensitivity.
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from hcp import forecast_neural
from hcp.benchmark_batch import NeuralExecution
from hcp.forecast_models import (
    ForecastSettings,
    PreparedHistory,
    arima_samples,
    deepar_samples,
    sarima_samples,
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.filterwarnings(
        "ignore:Maximum Likelihood optimization failed to converge"
    ),
    pytest.mark.filterwarnings(
        "ignore:Non-stationary starting seasonal autoregressive"
    ),
    pytest.mark.filterwarnings("ignore:Can't initialize NVML"),
    pytest.mark.filterwarnings(
        "ignore:`isinstance\\(treespec, LeafSpec\\)` is deprecated"
    ),
    pytest.mark.filterwarnings("ignore:Total length of `list` across ranks is zero"),
    pytest.mark.filterwarnings("ignore:'pin_memory' argument is set as true"),
]


def _history(last_shift: float = 0.0) -> PreparedHistory:
    dates = pd.date_range("2020-01-04", periods=64, freq="7D").to_numpy()
    index = np.arange(64, dtype=float)
    values = 40.0 + 0.15 * index + 5.0 * np.sin(2.0 * np.pi * index / 13.0)
    values[-4:] += last_shift
    return PreparedHistory(node="aa", dates=dates, values=values)


@pytest.mark.parametrize("adapter", [arima_samples, sarima_samples])
def test_classical_adapter_is_seeded_probabilistic_and_uses_recent_history(
    adapter,
) -> None:
    settings = replace(ForecastSettings(), n_samples=7)

    first = adapter((_history(),), settings, np.random.SeedSequence(123))
    repeated = adapter((_history(),), settings, np.random.SeedSequence(123))
    changed_seed = adapter((_history(),), settings, np.random.SeedSequence(124))
    changed_tail = adapter(
        (_history(last_shift=20.0),), settings, np.random.SeedSequence(123)
    )

    assert first.shape == (1, 4, 7)
    assert np.isfinite(first).all()
    np.testing.assert_array_equal(first, repeated)
    assert not np.array_equal(first, changed_seed)
    assert not np.array_equal(first[:, -1, :], changed_tail[:, -1, :])


def test_sarima_accepts_fixed_annual_seasonal_order_with_64_weeks() -> None:
    settings = replace(ForecastSettings(), n_samples=7)

    samples = sarima_samples((_history(),), settings, np.random.SeedSequence(9))

    assert settings.sarima_seasonal_order == (1, 0, 0, 52)
    assert samples.shape == (1, 4, 7)


def test_deepar_is_repeatable_preserves_order_and_restores_scale() -> None:
    settings = replace(ForecastSettings(), n_samples=7)
    small = _history()
    large = PreparedHistory(
        node="large",
        dates=small.dates.copy(),
        values=small.values * 10.0,
    )

    first = deepar_samples(
        (large, small), settings, np.random.SeedSequence(321), NeuralExecution.cpu()
    )
    repeated = deepar_samples(
        (large, small), settings, np.random.SeedSequence(321), NeuralExecution.cpu()
    )

    assert first.shape == (2, 4, 7)
    assert np.isfinite(first).all()
    np.testing.assert_array_equal(first, repeated)
    assert np.mean(np.abs(first[0])) > 5.0 * np.mean(np.abs(first[1]))


def test_deepar_uses_cpu_deterministic_trainer_without_writes(
    tmp_path, monkeypatch
) -> None:
    settings = replace(ForecastSettings(), n_samples=7)
    constructor: dict[str, object] = {}
    real_rnn_model = forecast_neural.RNNModel

    def recording_rnn_model(*args, **kwargs):
        constructor.update(kwargs)
        return real_rnn_model(*args, **kwargs)

    monkeypatch.setattr(forecast_neural, "RNNModel", recording_rnn_model)
    monkeypatch.chdir(tmp_path)

    result = deepar_samples(
        (_history(),), settings, np.random.SeedSequence(7), NeuralExecution.cpu()
    )

    trainer = constructor["pl_trainer_kwargs"]
    assert trainer == {
        "accelerator": "cpu",
        "devices": 1,
        "precision": "32-true",
        "deterministic": True,
        "logger": False,
        "enable_checkpointing": False,
        "enable_progress_bar": False,
        "enable_model_summary": False,
    }
    assert constructor["save_checkpoints"] is False
    assert constructor["log_tensorboard"] is False
    assert result.shape == (1, 4, 7)
    assert list(tmp_path.iterdir()) == []


def test_real_cuda_deepar_repeatability_and_device(monkeypatch):
    import torch

    from hcp.forecast import FORECASTERS

    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    execution = NeuralExecution(
        "cuda:0", "NVIDIA GeForce RTX 4080", ":4096:8", False, "32-true"
    )
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    devices = []
    real = forecast_neural.RNNModel
    from darts.models.forecasting.rnn_model import _RNNModule

    forward = _RNNModule.forward

    def record_forward(module, inputs, *args, **kwargs):
        devices.append(str(inputs[0].device))
        return forward(module, inputs, *args, **kwargs)

    monkeypatch.setattr(_RNNModule, "forward", record_forward)

    def construct(**kwargs):
        trainer = kwargs["pl_trainer_kwargs"]
        assert trainer["accelerator"] == "gpu"
        assert trainer["devices"] == [0]
        assert trainer["precision"] == "32-true"
        assert trainer["deterministic"] is True
        model = real(**kwargs)
        fit = model.fit

        def record(*args, **kw):
            assert kw["dataloader_kwargs"] == {"num_workers": 0, "pin_memory": False}
            return fit(*args, **kw)

        model.fit = record
        return model

    monkeypatch.setattr(forecast_neural, "RNNModel", construct)
    settings = replace(ForecastSettings(), n_samples=8, deepar_n_epochs=1)
    import warnings
    from collections import Counter

    outputs, warning_counts, seeds = [], [], []
    for _ in range(2):
        seed = np.random.SeedSequence(89)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            outputs.append(
                FORECASTERS["deepar"]((_history(),), settings, seed, execution)
            )
        warning_counts.append(Counter(w.category.__name__ for w in caught))
        seeds.append(
            (seed.spawn_key, seed.n_children_spawned, seed.generate_state(1).tolist())
        )
    assert warning_counts[0] == warning_counts[1]
    assert seeds[0] == seeds[1]
    assert seeds[0][:2] == ((), 0)
    np.testing.assert_array_equal(*outputs)
    assert devices and set(devices) == {"cuda:0"}
