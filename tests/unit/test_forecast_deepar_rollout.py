# ABOUTME: Checks recurrent inference follows the model device and host conversion boundary.
# ABOUTME: Intercepts tensor construction to exercise CUDA routing without hardware.
from contextlib import nullcontext

import numpy as np
import torch
from darts.utils.likelihood_models import GaussianLikelihood

from hcp.benchmark_batch import NeuralExecution
from hcp.forecast_deepar_data import CovariateTransform
from hcp.forecast_deepar_rollout import recurrent_rollout
from tests.unit.test_forecast_deepar import histories


def test_rollout_uses_model_device_and_copies_to_cpu_before_numpy(monkeypatch):
    execution = NeuralExecution(
        "cuda:0", "NVIDIA GeForce RTX 4080", ":4096:8", False, "32-true"
    )
    requested, converted, scopes = [], [], []
    for name in ("tensor", "arange"):
        real = getattr(torch, name)

        def construct(*args, _real=real, **kwargs):
            requested.append(str(kwargs.pop("device", None)))
            return _real(*args, **kwargs)

        monkeypatch.setattr(torch, name, construct)
    real_cpu = torch.Tensor.cpu
    real_numpy = torch.Tensor.numpy
    copied = set()

    def cpu(tensor, *args, **kwargs):
        converted.append(True)
        result = real_cpu(tensor, *args, **kwargs)
        copied.add(id(result))
        return result

    def numpy(tensor, *args, **kwargs):
        assert id(tensor) in copied, "NumPy conversion must follow an explicit CPU copy"
        copied.remove(id(tensor))
        return real_numpy(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "cpu", cpu)
    monkeypatch.setattr(torch.Tensor, "numpy", numpy)
    monkeypatch.setattr(
        torch.random,
        "fork_rng",
        lambda *, devices: scopes.append(devices) or nullcontext(),
    )
    monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda seed: None)

    class Module:
        device = torch.device("cuda:0")
        likelihood = GaussianLikelihood()

        def eval(self):
            return self

        def __call__(self, inputs, hidden):
            x = inputs[0]
            return torch.zeros(x.shape[0], x.shape[1], 1, 2), None

    contexts = histories()
    recurrent_rollout(
        Module(),
        contexts,
        tuple(CovariateTransform.fit(h) for h in contexts),
        np.ones(2),
        4,
        np.random.SeedSequence(6),
        execution,
    )
    assert requested == ["cuda:0"] * 3
    assert scopes == [[0]]
    assert len(converted) == 12
