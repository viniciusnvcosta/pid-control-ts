# ABOUTME: Simulates recurrent Gaussian paths using Darts likelihood parameter machinery.
# ABOUTME: Returns conditional densities and draws in original units without teacher forcing.
"""Device-aware four-step autoregressive inference with fixed fit preprocessing."""

import numpy as np
import torch
from darts.models.forecasting.rnn_model import CustomRNNModule

from hcp.benchmark_batch import NeuralExecution
from hcp.forecast_deepar_data import CovariateTransform, seed_integer
from hcp.forecast_types import PreparedHistory


def recurrent_rollout(
    module: CustomRNNModule,
    contexts: tuple[PreparedHistory, ...],
    transforms: tuple[CovariateTransform, ...],
    scales: np.ndarray,
    n_paths: int,
    seed: np.random.SeedSequence,
    execution: NeuralExecution,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate paths and conditional Gaussian parameters at all four future steps.

    Args:
        module: Fitted Darts Gaussian recurrent module; inference disables dropout.
        contexts: Independently prepared raw prefixes at the actual forecast cutoff.
        transforms: Fixed inner-fit (or final-refit) calendar normalization per node.
        scales: Corresponding positive fit-only autoregressive scales.
        n_paths: Monte Carlo paths; 500 in the registered configuration.
        seed: Dedicated reset validation or prediction stream.
        execution: Explicit device contract whose CUDA RNG must be preserved.

    Returns:
        Three arrays [nodes, four, paths]: draws, locations, standard deviations,
        restored to original units. The fourth-step density conditions on three
        simulated future values, never on held-out truth.
    """
    if n_paths < 1:
        raise ValueError("rollout requires positive path count")
    histories, features = [], []
    for context, transform, scale in zip(contexts, transforms, scales, strict=True):
        if len(context.values) < 12:
            raise ValueError("rollout requires twelve known weekly observations")
        dates = context.dates[-12:]
        future = dates[-1] + np.arange(1, 5) * np.timedelta64(7, "D")
        # Each known target[t] is paired with the next date's covariate[t+1].
        features.append(transform.transform(np.concatenate((dates[1:], future))))
        histories.append(context.values[-12:] / scale)
    device = module.device
    inputs = torch.tensor(np.stack(histories), dtype=torch.float32, device=device)[
        :, :, None
    ]
    covariates = torch.tensor(np.stack(features), dtype=torch.float32, device=device)
    inputs = inputs.repeat_interleave(n_paths, dim=0)
    covariates = covariates.repeat_interleave(n_paths, dim=0)
    identities = torch.arange(
        len(contexts), dtype=torch.float32, device=device
    ).repeat_interleave(n_paths)[:, None, None]
    x = torch.cat((inputs, covariates[:, :12]), dim=2)
    draws, locations, deviations = [], [], []
    module.eval()
    with (
        torch.random.fork_rng(devices=[0] if execution.device == "cuda:0" else []),
        torch.no_grad(),
    ):
        torch.manual_seed(seed_integer(seed))
        hidden = None
        for step in range(4):
            output, hidden = module((x, None, identities, None), hidden)
            last = output[:, -1:, :, :]
            # Public Darts parameter mapping applies its exact positive-scale transform.
            parameters = module.likelihood.predict_likelihood_parameters(last)
            sampled = module.likelihood.sample(last)
            locations.append(parameters[:, 0, 0].detach().cpu().numpy())
            deviations.append(parameters[:, 0, 1].detach().cpu().numpy())
            draws.append(sampled[:, 0, 0].detach().cpu().numpy())
            if step < 3:
                x = torch.cat((sampled, covariates[:, 12 + step : 13 + step]), dim=2)
    result = tuple(
        np.stack(values, axis=1).reshape(len(contexts), n_paths, 4).transpose(0, 2, 1)
        * scales[:, None, None]
        for values in (draws, locations, deviations)
    )
    if not all(np.isfinite(array).all() for array in result) or np.any(result[2] <= 0):
        raise ValueError("enhanced recurrent rollout produced invalid Gaussian paths")
    for array in result:
        array.setflags(write=False)
    return result
