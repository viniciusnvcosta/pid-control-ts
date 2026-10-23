# ABOUTME: Runs every v2 forecast adapter on a reduced real-data development slice.
# ABOUTME: Checks cutoff alignment and bundle shapes without evaluating registered contrasts or publishing results.
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from hcp.benchmark_data import ActualPanel, canonical_actuals, reference_bundle
from hcp.data import FORECASTER, file_sha256, read_forecasts
from hcp.forecast import ForecastConfig, generate_model_bundles
from hcp.forecast_types import V2ModelSettings
from tests.helpers import REAL_DATA, REAL_SHA256

pytestmark = [
    pytest.mark.integration,
    pytest.mark.filterwarnings(
        "ignore:Maximum Likelihood optimization failed to converge"
    ),
    pytest.mark.filterwarnings("ignore:Non-stationary starting autoregressive"),
]

MODELS = (
    "arima",
    "sarima",
    "deepar",
    "prophet",
    "transformer",
    "qr_l1",
    "deepar_enhanced",
)


def test_all_v2_adapters_integrate_with_reduced_real_data_and_preserve_cutoff(
    tmp_path: Path,
) -> None:
    """Exercise all adapters with one real origin and development-only budgets."""
    assert file_sha256(REAL_DATA) == REAL_SHA256
    frame = read_forecasts(REAL_DATA, REAL_SHA256)
    complete = canonical_actuals(frame)
    reference = reference_bundle(frame, complete, FORECASTER)

    # Keep the US and one endpoint leaf so each fitting path sees actual data while
    # the development check remains much smaller than the registered 57-node run.
    node_indices = [complete.nodes.index(node) for node in ("us", "ca")]
    actuals = ActualPanel(
        complete.dates,
        tuple(complete.nodes[index] for index in node_indices),
        complete.y[node_indices],
        complete.observed[node_indices],
    )
    target = reference.dates[-1:]
    cutoff = reference.forecast_dates[-1:]
    output_dir = tmp_path / "results"
    development_run_id = f"task14-development-{tmp_path.name}"

    settings = replace(
        V2ModelSettings(),
        n_samples=8,
        deepar_n_epochs=1,
        neural_max_epochs=1,
        neural_patience=1,
    )
    config = ForecastConfig(
        models=MODELS,
        min_train_weeks=60,
        root_seed=20261021,
        settings=settings,
        protocol_version=2,
        v2_settings=settings,
    )
    generated = generate_model_bundles(actuals, target, cutoff, config)

    assert set(generated.bundles) == set(MODELS)
    assert generated.diagnostics["models"].keys() == set(MODELS)
    for model, bundle in generated.bundles.items():
        assert bundle.nodes == actuals.nodes
        np.testing.assert_array_equal(bundle.dates, target)
        np.testing.assert_array_equal(bundle.forecast_dates, cutoff)
        assert bundle.q.shape == (2, 1, 3)
        assert np.isfinite(bundle.q).all(), model
    assert all(target[0] > cutoff[0] for _ in MODELS)
    assert not (output_dir / development_run_id).exists()
    assert not output_dir.exists()
