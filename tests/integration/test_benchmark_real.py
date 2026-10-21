# ABOUTME: Reduced real-data integration anchor for the probabilistic-model benchmark pipeline.
# ABOUTME: Fits one late ARIMA origin across all 57 nodes without computing registered outcomes or writing results.
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from hcp.benchmark import BenchmarkConfig
from hcp.benchmark_data import canonical_actuals, reference_bundle
from hcp.conformal import ControllerConfig
from hcp.data import FORECASTER, file_sha256, read_forecasts
from hcp.forecast import FORECASTERS, ForecastConfig, generate_model_bundles
from hcp.forecast_models import ForecastSettings
from hcp.panel import PanelConfig
from hcp.reconcile import reconcile_bundle
from tests.helpers import REAL_DATA, REAL_SHA256

pytestmark = [
    pytest.mark.integration,
    pytest.mark.filterwarnings(
        "ignore:Maximum Likelihood optimization failed to converge"
    ),
    pytest.mark.filterwarnings("ignore:Non-stationary starting autoregressive"),
]


def test_reduced_real_benchmark_preserves_cutoff_alignment_and_coherence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    results = Path("results")
    before = set(results.rglob("*")) if results.exists() else set()
    assert file_sha256(REAL_DATA) == REAL_SHA256
    frame = read_forecasts(REAL_DATA, REAL_SHA256)
    actuals = canonical_actuals(frame)
    reference = reference_bundle(frame, actuals, FORECASTER)
    targets = reference.dates[-1:]
    origins = reference.forecast_dates[-1:]
    settings = replace(ForecastSettings(), n_samples=5)
    forecast_cfg = ForecastConfig(
        models=("arima",),
        min_train_weeks=60,
        root_seed=20261021,
        settings=settings,
    )
    panel_cfg = PanelConfig(
        controller=ControllerConfig(ahead=4),
        lr_grid=(0.1,),
        reconcilers=("none", "bottom_up", "mint_shrink", "mint_median"),
        controllers=("raw", "pi"),
        mint_warmup=20,
        rolling_window=10,
        blocks=(4,),
        n_boot=5,
        root_seed=20261021,
        primary_forecaster="covidhub",
    )
    benchmark_cfg = BenchmarkConfig(
        panel=panel_cfg,
        models=("arima",),
        reference_model="covidhub",
        min_eval_weeks=1,
    )
    histories_seen = []
    real_arima = FORECASTERS["arima"]

    def recording_arima(histories, adapter_settings, seed):
        histories_seen.extend(histories)
        return real_arima(histories, adapter_settings, seed)

    monkeypatch.setitem(FORECASTERS, "arima", recording_arima)
    generated = generate_model_bundles(actuals, targets, origins, forecast_cfg)
    bundle = generated.bundles["arima"]

    assert benchmark_cfg.panel.blocks == (4,)
    assert benchmark_cfg.panel.n_boot == 5
    assert settings.ahead == 4
    assert len(bundle.nodes) == 57
    assert bundle.nodes == reference.nodes == actuals.nodes
    np.testing.assert_array_equal(bundle.dates, targets)
    np.testing.assert_array_equal(bundle.forecast_dates, origins)
    assert bundle.q.shape == (57, 1, 3)
    assert np.isfinite(bundle.q).all()
    assert len(histories_seen) == 57
    assert all(history.dates[-1] <= origins[0] for history in histories_seen)
    assert all(not np.any(history.dates > origins[0]) for history in histories_seen)
    assert targets[0] > origins[0]

    reconciled = reconcile_bundle(bundle, "bottom_up", warmup=20).q
    np.testing.assert_allclose(reconciled[0], reconciled[1:].sum(axis=0))
    after = set(results.rglob("*")) if results.exists() else set()
    assert after == before
